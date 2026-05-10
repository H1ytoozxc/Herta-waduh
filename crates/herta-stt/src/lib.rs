//! # herta-stt
//!
//! Speech-to-text providers for The Herta Voice Assistant.
//!
//! Providers:
//!
//! - [`google_ai::GoogleAiStt`] — Gemini-based transcription via AI Studio.
//! - [`fallback::FallbackStt`] — wraps a primary + secondary engine; used to
//!   degrade from cloud to local Whisper on failure.
//! - [`local::LocalWhisperStt`] — stub adapter that invokes a configurable
//!   external STT binary (e.g. a bundled `whisper.cpp` / faster-whisper
//!   service). Real weight loading is not in-process.

#![warn(missing_docs)]

pub mod fallback;
#[cfg(feature = "google-ai")]
pub mod google_ai;
#[cfg(feature = "whisper-local")]
pub mod local;
mod wav;

use herta_config::schema::{Config, SttProviderKind};
use herta_core::{HertaResult, stt::SttEngine};
use std::sync::Arc;

/// Build the configured STT engine.
pub fn build_from_config(cfg: &Config) -> HertaResult<Arc<dyn SttEngine>> {
    match cfg.stt_provider {
        SttProviderKind::Whisper => {
            #[cfg(feature = "whisper-local")]
            {
                Ok(Arc::new(local::LocalWhisperStt::new(cfg.stt.clone())?))
            }
            #[cfg(not(feature = "whisper-local"))]
            {
                Err(HertaError::config(
                    "whisper STT requires the 'whisper-local' feature",
                ))
            }
        }
        SttProviderKind::GoogleAi => {
            #[cfg(feature = "google-ai")]
            {
                let primary: Arc<dyn SttEngine> = Arc::new(google_ai::GoogleAiStt::new(
                    cfg.google_stt.clone(),
                    cfg.audio.sample_rate,
                )?);
                if cfg.google_stt.fallback_to_whisper {
                    #[cfg(feature = "whisper-local")]
                    {
                        let secondary: Arc<dyn SttEngine> =
                            Arc::new(local::LocalWhisperStt::new(cfg.stt.clone())?);
                        return Ok(Arc::new(fallback::FallbackStt::new(primary, secondary)));
                    }
                }
                Ok(primary)
            }
            #[cfg(not(feature = "google-ai"))]
            {
                Err(HertaError::config(
                    "google-ai STT requires the 'google-ai' feature",
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use herta_config::schema::Config;

    #[test]
    fn default_config_builds_an_engine() {
        let cfg = Config::default();
        // With default features (`google-ai`), `whisper` provider is not available.
        // The default LLM config has stt_provider=Whisper which requires the
        // `whisper-local` feature. Assert the expected error shape.
        let result = build_from_config(&cfg);
        #[cfg(feature = "whisper-local")]
        assert!(result.is_ok());
        #[cfg(not(feature = "whisper-local"))]
        assert!(matches!(result, Err(HertaError::Config(_))));
    }
}
