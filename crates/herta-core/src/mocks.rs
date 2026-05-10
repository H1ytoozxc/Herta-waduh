//! Mock implementations of the core traits, intended for tests and local dev.
//!
//! These mocks are deliberately simple but sufficient for end-to-end pipeline
//! tests that exercise the orchestration logic without network or hardware.

use crate::{
    DialogContext, HertaResult, Message,
    audio::{AudioChunk, AudioFormat, AudioInput, AudioOutput, Utterance},
    llm::{LlmProvider, LlmResponse, TokenUsage},
    memory::Memory,
    stt::{SttEngine, Transcript},
    tts::{SpeakOptions, SynthesizedAudio, TtsEngine},
};
use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::Mutex;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;

/// An LLM provider that returns a deterministic echo of the user's utterance,
/// suitable for integration tests. Records every call for later assertion.
#[derive(Debug, Clone)]
pub struct EchoLlm {
    model: String,
    calls: Arc<Mutex<Vec<DialogContext>>>,
}

impl EchoLlm {
    /// Build an echo LLM with the given model label.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            calls: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Return recorded calls captured during tests.
    pub fn calls(&self) -> Vec<DialogContext> {
        self.calls.lock().clone()
    }
}

#[async_trait]
impl LlmProvider for EchoLlm {
    fn name(&self) -> &'static str {
        "mock-echo"
    }

    async fn generate(&self, ctx: &DialogContext) -> HertaResult<LlmResponse> {
        let start = Instant::now();
        self.calls.lock().push(ctx.clone());
        Ok(LlmResponse {
            text: format!("echo: {}", ctx.user_utterance),
            provider: "mock-echo".into(),
            model: self.model.clone(),
            latency: start.elapsed(),
            usage: Some(TokenUsage::default()),
        })
    }
}

/// STT engine that returns a fixed transcript regardless of input.
#[derive(Debug, Clone)]
pub struct FixedStt {
    text: String,
}

impl FixedStt {
    /// Build an STT that always returns `text`.
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

#[async_trait]
impl SttEngine for FixedStt {
    fn name(&self) -> &'static str {
        "mock-stt"
    }

    async fn transcribe(&self, _utterance: &Utterance) -> HertaResult<Transcript> {
        Ok(Transcript {
            text: self.text.clone(),
            language: None,
            confidence: Some(1.0),
            latency: Duration::from_millis(0),
            provider: "mock-stt".into(),
        })
    }
}

/// TTS engine that returns a zero-filled PCM buffer proportional to text length.
#[derive(Debug, Clone, Default)]
pub struct SilentTts;

#[async_trait]
impl TtsEngine for SilentTts {
    fn name(&self) -> &'static str {
        "mock-tts"
    }

    async fn synthesize(
        &self,
        text: &str,
        _options: &SpeakOptions,
    ) -> HertaResult<SynthesizedAudio> {
        // 16 kHz mono f32 at ~100ms per word, approximated.
        let word_count = text.split_whitespace().count().max(1);
        let samples_per_word = 1_600; // 100ms @ 16k
        let total_samples = word_count * samples_per_word;
        let buf = vec![0u8; total_samples * 4];
        Ok(SynthesizedAudio {
            pcm: Bytes::from(buf),
            format: AudioFormat::default(),
            provider: "mock-tts".into(),
            latency: Duration::from_millis(0),
        })
    }
}

/// Audio input driven by a bounded MPSC channel. Use [`ChannelAudioInput::sender`]
/// to push chunks from tests.
pub struct ChannelAudioInput {
    rx: tokio::sync::Mutex<mpsc::Receiver<AudioChunk>>,
    tx: mpsc::Sender<AudioChunk>,
}

impl ChannelAudioInput {
    /// Build a new channel-backed input.
    pub fn new(buffer: usize) -> Self {
        let (tx, rx) = mpsc::channel(buffer.max(1));
        Self {
            rx: tokio::sync::Mutex::new(rx),
            tx,
        }
    }
    /// Clone of the sender, for tests.
    pub fn sender(&self) -> mpsc::Sender<AudioChunk> {
        self.tx.clone()
    }
}

impl std::fmt::Debug for ChannelAudioInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelAudioInput").finish()
    }
}

#[async_trait]
impl AudioInput for ChannelAudioInput {
    fn name(&self) -> &'static str {
        "mock-audio-in"
    }
    async fn start(&self) -> HertaResult<()> {
        Ok(())
    }
    async fn stop(&self) -> HertaResult<()> {
        Ok(())
    }
    async fn next_chunk(&self) -> HertaResult<Option<AudioChunk>> {
        Ok(self.rx.lock().await.recv().await)
    }
}

/// Audio output that records every buffer played for later inspection.
#[derive(Debug, Clone, Default)]
pub struct RecordingAudioOutput {
    buffers: Arc<Mutex<Vec<Bytes>>>,
}

impl RecordingAudioOutput {
    /// Retrieve the recorded buffers.
    pub fn recorded(&self) -> Vec<Bytes> {
        self.buffers.lock().clone()
    }
}

#[async_trait]
impl AudioOutput for RecordingAudioOutput {
    fn name(&self) -> &'static str {
        "mock-audio-out"
    }
    async fn play(&self, pcm: Bytes, _format: AudioFormat) -> HertaResult<()> {
        self.buffers.lock().push(pcm);
        Ok(())
    }
    async fn play_test_tone(&self) -> HertaResult<()> {
        self.buffers.lock().push(Bytes::new());
        Ok(())
    }
}

/// In-memory `Memory` backend used for tests and ephemeral environments.
#[derive(Debug, Default, Clone)]
pub struct InMemoryMemory {
    messages: Arc<Mutex<Vec<Message>>>,
}

#[async_trait]
impl Memory for InMemoryMemory {
    fn name(&self) -> &'static str {
        "mock-memory"
    }
    async fn load_context(&self, max_messages: usize) -> HertaResult<Vec<Message>> {
        Ok(crate::memory::window_messages(
            &self.messages.lock(),
            max_messages,
        ))
    }
    async fn append_turn(
        &self,
        user_text: &str,
        assistant_text: &str,
    ) -> HertaResult<()> {
        let user_text = user_text.trim();
        let assistant_text = assistant_text.trim();
        if user_text.is_empty() || assistant_text.is_empty() {
            return Ok(());
        }
        let mut lock = self.messages.lock();
        lock.push(Message::user(user_text));
        lock.push(Message::assistant(assistant_text));
        Ok(())
    }
    async fn clear(&self) -> HertaResult<()> {
        self.messages.lock().clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn echo_llm_records_calls() {
        let llm = EchoLlm::new("m");
        let ctx = DialogContext::new("hi");
        let r = llm.generate(&ctx).await.unwrap();
        assert_eq!(r.text, "echo: hi");
        assert_eq!(llm.calls().len(), 1);
    }

    #[tokio::test]
    async fn in_memory_memory_roundtrip() {
        let mem = InMemoryMemory::default();
        mem.append_turn("hello", "world").await.unwrap();
        mem.append_turn("foo", "bar").await.unwrap();
        let ctx = mem.load_context(10).await.unwrap();
        assert_eq!(ctx.len(), 4);
        assert_eq!(ctx[0].content, "hello");
        mem.clear().await.unwrap();
        assert!(mem.load_context(10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn in_memory_memory_skips_empty_turns() {
        let mem = InMemoryMemory::default();
        mem.append_turn("", "only-assistant").await.unwrap();
        mem.append_turn("only-user", "").await.unwrap();
        assert!(mem.load_context(10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn silent_tts_returns_nonempty_buffer() {
        let tts = SilentTts;
        let out = tts.synthesize("hello world", &SpeakOptions::default()).await.unwrap();
        assert!(!out.pcm.is_empty());
    }

    #[tokio::test]
    async fn recording_output_captures_plays() {
        let out = RecordingAudioOutput::default();
        out.play(Bytes::from_static(b"abc"), AudioFormat::default())
            .await
            .unwrap();
        assert_eq!(out.recorded().len(), 1);
    }
}
