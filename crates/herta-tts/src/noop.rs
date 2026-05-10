//! No-op TTS engine, useful for headless deployments and tests.

use async_trait::async_trait;
use bytes::Bytes;
use herta_core::{
    HertaResult,
    audio::AudioFormat,
    tts::{SpeakOptions, SynthesizedAudio, TtsEngine},
};
use std::time::Duration;

/// A TTS engine that accepts any input and returns an empty audio buffer.
/// Intended for headless environments where audio output is disabled.
#[derive(Debug, Clone, Default)]
pub struct NoopTts;

#[async_trait]
impl TtsEngine for NoopTts {
    fn name(&self) -> &'static str {
        "noop"
    }

    async fn synthesize(
        &self,
        _text: &str,
        _options: &SpeakOptions,
    ) -> HertaResult<SynthesizedAudio> {
        Ok(SynthesizedAudio {
            pcm: Bytes::new(),
            format: AudioFormat::default(),
            provider: "noop".into(),
            latency: Duration::from_millis(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn noop_returns_empty_pcm() {
        let t = NoopTts;
        let out = t.synthesize("anything", &SpeakOptions::default()).await.unwrap();
        assert!(out.pcm.is_empty());
        assert_eq!(out.provider, "noop");
    }
}
