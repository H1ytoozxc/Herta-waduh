//! Edge TTS adapter.
//!
//! This calls out to an external `edge-tts` (or `edge-tts-cli`) binary that
//! reads text from stdin and writes audio bytes (WAV/PCM) to stdout. This
//! keeps the Rust side free of the Microsoft websocket stack while retaining
//! parity with the Python implementation.
//!
//! Contract:
//! - `stdin`: UTF-8 text
//! - `stdout`: PCM little-endian (16kHz, 16-bit mono by default)
//! - non-zero exit → error

use crate::subprocess::{SubprocessTts, SubprocessTtsConfig};
use async_trait::async_trait;
use herta_config::schema::EdgeTtsConfig;
use herta_core::{
    HertaError, HertaResult,
    audio::{AudioFormat, SampleFormat},
    tts::{SpeakOptions, SynthesizedAudio, TtsEngine},
};
use std::path::PathBuf;

const PROVIDER: &str = "edge_tts";

/// Edge TTS engine (subprocess-backed).
#[derive(Debug)]
pub struct EdgeTts {
    inner: SubprocessTts,
    cfg: EdgeTtsConfig,
}

impl EdgeTts {
    /// Build a new Edge TTS engine from configuration.
    pub fn new(cfg: EdgeTtsConfig) -> HertaResult<Self> {
        let binary = std::env::var("EDGE_TTS_BINARY")
            .map_or_else(|_| PathBuf::from("edge-tts"), PathBuf::from);

        let sub_cfg = SubprocessTtsConfig {
            provider: PROVIDER,
            binary,
            fixed_args: vec![
                "--text-stdin".into(),
                "--voice".into(),
                cfg.voice.clone(),
                "--rate".into(),
                cfg.rate.clone(),
                "--volume".into(),
                cfg.volume.clone(),
                "--pitch".into(),
                cfg.pitch.clone(),
            ],
            output_format: AudioFormat {
                sample_rate: 24_000,
                channels: 1,
                sample_format: SampleFormat::I16,
            },
        };
        Ok(Self {
            inner: SubprocessTts::new(sub_cfg),
            cfg,
        })
    }

    /// Access the underlying voice id (for diagnostics).
    pub fn voice(&self) -> &str {
        &self.cfg.voice
    }
}

#[async_trait]
impl TtsEngine for EdgeTts {
    fn name(&self) -> &'static str {
        PROVIDER
    }

    async fn warm_up(&self) -> HertaResult<bool> {
        self.inner.warm_up_inner().await
    }

    async fn synthesize(
        &self,
        text: &str,
        options: &SpeakOptions,
    ) -> HertaResult<SynthesizedAudio> {
        if text.trim().is_empty() {
            return Err(HertaError::invalid("empty TTS input"));
        }
        self.inner.synthesize_inner(text, options).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn empty_text_rejected() {
        let engine = EdgeTts::new(EdgeTtsConfig::default()).unwrap();
        assert!(matches!(
            engine.synthesize("   ", &SpeakOptions::default()).await,
            Err(HertaError::InvalidInput(_))
        ));
    }

    #[test]
    fn voice_getter_returns_configured_voice() {
        let cfg = EdgeTtsConfig {
            voice: "ru-RU-DmitryNeural".into(),
            ..EdgeTtsConfig::default()
        };
        let engine = EdgeTts::new(cfg).unwrap();
        assert_eq!(engine.voice(), "ru-RU-DmitryNeural");
    }
}
