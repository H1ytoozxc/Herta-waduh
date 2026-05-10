//! Voice pipeline orchestrator.
//!
//! The pipeline wires together [`AudioInput`](crate::audio::AudioInput),
//! [`SttEngine`](crate::stt::SttEngine), [`LlmProvider`](crate::llm::LlmProvider),
//! [`TtsEngine`](crate::tts::TtsEngine), [`AudioOutput`](crate::audio::AudioOutput),
//! and [`Memory`](crate::memory::Memory) behind a single async API.
//!
//! Stages are independent so a text-only deployment can skip audio/STT/TTS
//! by constructing a pipeline with only the LLM and memory.

use crate::{
    DialogContext, HertaError, HertaResult, Message, Role,
    audio::{AudioOutput, Utterance},
    llm::{LlmProvider, LlmResponse},
    memory::Memory,
    stt::{SttEngine, Transcript},
    tts::{SpeakOptions, SynthesizedAudio, TtsEngine},
};
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::{Instrument, info_span};

/// Events emitted during pipeline execution. Subscribers receive these via
/// [`VoicePipeline::subscribe`] and can render them in a CLI, dashboard, or
/// log stream.
#[derive(Debug, Clone)]
pub enum PipelineEvent {
    /// A user utterance was transcribed.
    UserTranscribed(Transcript),
    /// A direct (text) user turn started.
    UserText(String),
    /// The LLM produced a response.
    AssistantReply(LlmResponse),
    /// TTS finished synthesizing audio.
    ///
    /// `bytes` is the size of the PCM buffer that was synthesized.
    TtsSynthesized {
        /// Size of the synthesized PCM buffer in bytes.
        bytes: usize,
    },
    /// A recoverable error happened; pipeline continues running.
    Warning(String),
}

/// Static configuration for a [`VoicePipeline`].
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum conversation context (user+assistant messages) to load.
    pub context_messages: usize,
    /// Locked persona/system prefix prepended to every request.
    pub locked_prefix: Vec<Message>,
    /// Tenant id for multi-tenant deployments (optional).
    pub tenant_id: Option<String>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            context_messages: 12,
            locked_prefix: Vec::new(),
            tenant_id: None,
        }
    }
}

/// High-level voice pipeline that owns references to every stage.
///
/// The pipeline is `Clone`-cheap: all components are held behind `Arc`, so
/// spawning multiple worker tasks is a single allocation.
#[derive(Clone)]
pub struct VoicePipeline {
    llm: Arc<dyn LlmProvider>,
    memory: Arc<dyn Memory>,
    stt: Option<Arc<dyn SttEngine>>,
    tts: Option<Arc<dyn TtsEngine>>,
    output: Option<Arc<dyn AudioOutput>>,
    config: PipelineConfig,
    events: tokio::sync::broadcast::Sender<PipelineEvent>,
}

impl std::fmt::Debug for VoicePipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoicePipeline")
            .field("llm", &self.llm.name())
            .field("memory", &self.memory.name())
            .field("stt", &self.stt.as_ref().map(|s| s.name()))
            .field("tts", &self.tts.as_ref().map(|s| s.name()))
            .field("output", &self.output.as_ref().map(|s| s.name()))
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// Builder for [`VoicePipeline`]. Required: `llm`, `memory`. Optional: rest.
#[derive(Default)]
pub struct VoicePipelineBuilder {
    llm: Option<Arc<dyn LlmProvider>>,
    memory: Option<Arc<dyn Memory>>,
    stt: Option<Arc<dyn SttEngine>>,
    tts: Option<Arc<dyn TtsEngine>>,
    output: Option<Arc<dyn AudioOutput>>,
    config: PipelineConfig,
}

impl VoicePipelineBuilder {
    /// Set the LLM provider.
    #[must_use]
    pub fn llm(mut self, llm: Arc<dyn LlmProvider>) -> Self {
        self.llm = Some(llm);
        self
    }
    /// Set the memory backend.
    #[must_use]
    pub fn memory(mut self, memory: Arc<dyn Memory>) -> Self {
        self.memory = Some(memory);
        self
    }
    /// Optional STT engine (required for voice mode).
    #[must_use]
    pub fn stt(mut self, stt: Arc<dyn SttEngine>) -> Self {
        self.stt = Some(stt);
        self
    }
    /// Optional TTS engine.
    #[must_use]
    pub fn tts(mut self, tts: Arc<dyn TtsEngine>) -> Self {
        self.tts = Some(tts);
        self
    }
    /// Optional audio output.
    #[must_use]
    pub fn output(mut self, output: Arc<dyn AudioOutput>) -> Self {
        self.output = Some(output);
        self
    }
    /// Override pipeline config.
    #[must_use]
    pub fn config(mut self, config: PipelineConfig) -> Self {
        self.config = config;
        self
    }

    /// Finalize and return a [`VoicePipeline`].
    pub fn build(self) -> HertaResult<VoicePipeline> {
        let llm = self
            .llm
            .ok_or_else(|| HertaError::config("pipeline missing LLM provider"))?;
        let memory = self
            .memory
            .ok_or_else(|| HertaError::config("pipeline missing memory backend"))?;
        let (tx, _rx) = tokio::sync::broadcast::channel(64);
        Ok(VoicePipeline {
            llm,
            memory,
            stt: self.stt,
            tts: self.tts,
            output: self.output,
            config: self.config,
            events: tx,
        })
    }
}

impl VoicePipeline {
    /// Start building a new pipeline.
    pub fn builder() -> VoicePipelineBuilder {
        VoicePipelineBuilder::default()
    }

    /// Subscribe to pipeline events (fan-out; missed events when lagged).
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<PipelineEvent> {
        self.events.subscribe()
    }

    /// Run a single text turn end-to-end: load context → LLM → append → TTS.
    ///
    /// Returns the assistant response text.
    #[tracing::instrument(level = "info", skip(self, user_text, cancel), fields(len = user_text.len()))]
    pub async fn run_text_turn(
        &self,
        user_text: &str,
        cancel: Option<&CancellationToken>,
    ) -> HertaResult<String> {
        check_cancel(cancel)?;
        let user_text = user_text.trim();
        if user_text.is_empty() {
            return Err(HertaError::invalid("user_text is empty"));
        }
        let _ = self
            .events
            .send(PipelineEvent::UserText(user_text.to_string()));

        let history = self
            .memory
            .load_context(self.config.context_messages)
            .await
            .unwrap_or_default();
        let mut ctx = DialogContext::new(user_text);
        ctx.locked_prefix = self.config.locked_prefix.clone();
        ctx.history = history;
        if let Some(tenant) = &self.config.tenant_id {
            ctx.tenant_id = Some(tenant.clone());
        }

        check_cancel(cancel)?;
        let response = self
            .llm
            .generate(&ctx)
            .instrument(info_span!("llm.generate", provider = self.llm.name()))
            .await?;

        let _ = self
            .events
            .send(PipelineEvent::AssistantReply(response.clone()));

        if let Err(err) = self
            .memory
            .append_turn(&ctx.user_utterance, &response.text)
            .await
        {
            tracing::warn!(error = %err, "failed to persist dialogue memory");
            let _ = self
                .events
                .send(PipelineEvent::Warning(format!("memory: {err}")));
        }

        self.maybe_speak(&response.text, cancel).await?;
        Ok(response.text)
    }

    /// Run a single voice turn: transcribe utterance → LLM → TTS.
    #[tracing::instrument(level = "info", skip(self, utterance, cancel), fields(ms = utterance.duration_ms))]
    pub async fn run_voice_turn(
        &self,
        utterance: &Utterance,
        cancel: Option<&CancellationToken>,
    ) -> HertaResult<Option<String>> {
        let stt = self.stt.as_ref().ok_or_else(|| {
            HertaError::config("voice turn requested but no STT engine configured")
        })?;

        check_cancel(cancel)?;
        let transcript = stt
            .transcribe(utterance)
            .instrument(info_span!("stt.transcribe", provider = stt.name()))
            .await?;

        if !transcript.is_meaningful() {
            let _ = self.events.send(PipelineEvent::UserTranscribed(transcript));
            return Ok(None);
        }
        let text = transcript.text.clone();
        let _ = self.events.send(PipelineEvent::UserTranscribed(transcript));
        let reply = self.run_text_turn(&text, cancel).await?;
        Ok(Some(reply))
    }

    async fn maybe_speak(
        &self,
        text: &str,
        cancel: Option<&CancellationToken>,
    ) -> HertaResult<()> {
        let Some(tts) = self.tts.as_ref() else {
            return Ok(());
        };
        check_cancel(cancel)?;
        let SynthesizedAudio {
            pcm,
            format,
            provider: _,
            latency: _,
        } = tts
            .synthesize(text, &SpeakOptions::default())
            .instrument(info_span!("tts.synthesize", provider = tts.name()))
            .await?;

        let _ = self.events.send(PipelineEvent::TtsSynthesized {
            bytes: pcm.len(),
        });

        if let Some(output) = self.output.as_ref() {
            check_cancel(cancel)?;
            output
                .play(pcm, format)
                .instrument(info_span!("audio.play", backend = output.name()))
                .await?;
        }
        Ok(())
    }

    /// The primary LLM provider name, for diagnostics.
    pub fn llm_name(&self) -> &'static str {
        self.llm.name()
    }

    /// Construct a snapshot of the dialogue messages that would be sent on
    /// the next turn. Useful for diagnostics and debugging.
    pub async fn preview_context(
        &self,
        user_utterance: &str,
    ) -> HertaResult<Vec<Message>> {
        let history = self
            .memory
            .load_context(self.config.context_messages)
            .await?;
        let ctx = DialogContext {
            correlation_id: uuid::Uuid::new_v4(),
            tenant_id: self.config.tenant_id.clone(),
            locked_prefix: self.config.locked_prefix.clone(),
            history,
            user_utterance: user_utterance.to_string(),
            language: None,
        };
        Ok(ctx.messages())
    }

    /// Return `true` if a locked prefix system message matches `role` exactly.
    pub fn has_locked_role(&self, role: Role) -> bool {
        self.config.locked_prefix.iter().any(|m| m.role == role)
    }
}

fn check_cancel(cancel: Option<&CancellationToken>) -> HertaResult<()> {
    if let Some(token) = cancel
        && token.is_cancelled()
    {
        return Err(HertaError::Cancelled);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        audio::{AudioFormat, Utterance},
        mocks::{EchoLlm, FixedStt, InMemoryMemory, RecordingAudioOutput, SilentTts},
    };
    use bytes::Bytes;
    use std::sync::Arc;

    fn test_pipeline() -> (VoicePipeline, Arc<RecordingAudioOutput>, Arc<InMemoryMemory>) {
        let memory = Arc::new(InMemoryMemory::default());
        let output = Arc::new(RecordingAudioOutput::default());
        let pipeline = VoicePipeline::builder()
            .llm(Arc::new(EchoLlm::new("test-model")))
            .memory(memory.clone())
            .stt(Arc::new(FixedStt::new("hello from test")))
            .tts(Arc::new(SilentTts))
            .output(output.clone())
            .config(PipelineConfig::default())
            .build()
            .unwrap();
        (pipeline, output, memory)
    }

    #[tokio::test]
    async fn text_turn_generates_and_stores() {
        let (pipeline, output, memory) = test_pipeline();
        let reply = pipeline.run_text_turn("hi", None).await.unwrap();
        assert_eq!(reply, "echo: hi");
        assert_eq!(memory.load_context(10).await.unwrap().len(), 2);
        assert_eq!(output.recorded().len(), 1);
    }

    #[tokio::test]
    async fn text_turn_requires_nonempty_input() {
        let (pipeline, _, _) = test_pipeline();
        assert!(pipeline.run_text_turn("   ", None).await.is_err());
    }

    #[tokio::test]
    async fn voice_turn_flows_stt_into_llm() {
        let (pipeline, output, memory) = test_pipeline();
        let utt = Utterance {
            pcm: Bytes::new(),
            format: AudioFormat::default(),
            duration_ms: 500,
        };
        let reply = pipeline.run_voice_turn(&utt, None).await.unwrap();
        assert_eq!(reply.as_deref(), Some("echo: hello from test"));
        assert_eq!(memory.load_context(10).await.unwrap().len(), 2);
        assert!(!output.recorded().is_empty());
    }

    #[tokio::test]
    async fn voice_turn_without_stt_is_configuration_error() {
        let pipeline = VoicePipeline::builder()
            .llm(Arc::new(EchoLlm::new("m")))
            .memory(Arc::new(InMemoryMemory::default()))
            .build()
            .unwrap();
        let utt = Utterance {
            pcm: Bytes::new(),
            format: AudioFormat::default(),
            duration_ms: 100,
        };
        let err = pipeline.run_voice_turn(&utt, None).await.unwrap_err();
        assert!(matches!(err, HertaError::Config(_)));
    }

    #[tokio::test]
    async fn cancellation_short_circuits() {
        let (pipeline, _, _) = test_pipeline();
        let token = CancellationToken::new();
        token.cancel();
        let err = pipeline.run_text_turn("hi", Some(&token)).await.unwrap_err();
        assert!(matches!(err, HertaError::Cancelled));
    }

    #[tokio::test]
    async fn preview_context_reflects_locked_prefix() {
        let pipeline = VoicePipeline::builder()
            .llm(Arc::new(EchoLlm::new("m")))
            .memory(Arc::new(InMemoryMemory::default()))
            .config(PipelineConfig {
                context_messages: 4,
                locked_prefix: vec![Message::system("persona")],
                tenant_id: None,
            })
            .build()
            .unwrap();
        let msgs = pipeline.preview_context("hello").await.unwrap();
        assert_eq!(msgs.first().unwrap().role, Role::System);
        assert_eq!(msgs.last().unwrap().content, "hello");
    }
}
