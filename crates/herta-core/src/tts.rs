//! Text-to-speech abstraction.

use crate::{HertaResult, audio::AudioFormat};
use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Synthesized PCM audio produced by a [`TtsEngine`].
#[derive(Debug, Clone)]
pub struct SynthesizedAudio {
    /// Raw PCM bytes (linear).
    pub pcm: Bytes,
    /// Format description of the produced PCM.
    pub format: AudioFormat,
    /// Backend identifier.
    pub provider: String,
    /// Total synthesis latency.
    pub latency: Duration,
}

/// Optional per-call synthesis parameters. Fields are hints — backends may
/// silently ignore unsupported ones.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeakOptions {
    /// Override the default voice id.
    pub voice: Option<String>,
    /// Override speaking rate (e.g. `"-6%"` for SSML-like backends).
    pub rate: Option<String>,
    /// Override volume (e.g. `"+0%"`).
    pub volume: Option<String>,
    /// Override pitch (e.g. `"+8Hz"`).
    pub pitch: Option<String>,
    /// BCP-47 language hint.
    pub language: Option<String>,
}

/// TTS engine contract.
#[async_trait]
pub trait TtsEngine: Send + Sync + 'static {
    /// Short, stable backend name.
    fn name(&self) -> &'static str;

    /// Best-effort warm-up.
    async fn warm_up(&self) -> HertaResult<bool> {
        Ok(true)
    }

    /// Synthesize the given text to PCM.
    async fn synthesize(
        &self,
        text: &str,
        options: &SpeakOptions,
    ) -> HertaResult<SynthesizedAudio>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speak_options_default_is_empty() {
        let o = SpeakOptions::default();
        assert!(o.voice.is_none());
        assert!(o.rate.is_none());
        assert!(o.volume.is_none());
        assert!(o.pitch.is_none());
        assert!(o.language.is_none());
    }
}
