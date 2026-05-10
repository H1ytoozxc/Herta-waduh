//! # herta-tts
//!
//! Text-to-speech providers for The Herta Voice Assistant.
//!
//! Providers:
//!
//! - [`edge::EdgeTts`] — Microsoft Edge voice via a subprocess contract.
//! - [`subprocess::SubprocessTts`] — generic subprocess TTS adapter.
//! - [`noop::NoopTts`] — disabled-but-valid engine for headless deployments.

#![warn(missing_docs)]

pub mod edge;
pub mod noop;
pub mod subprocess;

use herta_config::schema::Config;
use herta_core::{HertaResult, tts::TtsEngine};
use std::sync::Arc;

/// Build the configured TTS engine based on top-level config.
///
/// Returns `Ok(None)` when TTS is disabled.
pub fn build_from_config(cfg: &Config) -> HertaResult<Option<Arc<dyn TtsEngine>>> {
    if !cfg.tts.enabled {
        return Ok(None);
    }
    Ok(Some(Arc::new(edge::EdgeTts::new(cfg.tts.clone())?)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_returns_none() {
        let mut cfg = Config::default();
        cfg.tts.enabled = false;
        assert!(build_from_config(&cfg).unwrap().is_none());
    }
}
