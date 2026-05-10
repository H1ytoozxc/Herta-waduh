//! # herta-audio
//!
//! Cross-platform audio I/O for The Herta Voice Assistant.
//!
//! - [`cpal_backend::CpalAudioInput`] / [`cpal_backend::CpalAudioOutput`] —
//!   production backend (feature `cpal-backend`, enabled by default).
//! - [`null_backend::NullAudioOutput`] — discards audio (headless mode).
//! - [`vad::EnergyVad`] — lightweight energy-based VAD segmenter. A more
//!   accurate neural VAD (e.g. Silero) can be swapped in later without
//!   touching call-sites, since [`Vad`] is a trait.
//!
//! All audio types are re-exported from [`herta_core::audio`].

#![warn(missing_docs)]

pub mod null_backend;
pub mod tone;
pub mod vad;

#[cfg(feature = "cpal-backend")]
pub mod cpal_backend;

pub use herta_core::audio::{
    AudioChunk, AudioFormat, AudioInput, AudioOutput, DeviceEnumerator, DeviceId, DeviceInfo,
    SampleFormat, Utterance,
};
pub use vad::{EnergyVad, Vad, VadDecision};

#[cfg(feature = "cpal-backend")]
pub use cpal_backend::CpalDeviceEnumerator;

use herta_config::schema::{AudioInputConfig, AudioOutputConfig, Config};
use herta_core::HertaResult;
use std::sync::Arc;

/// Factory that chooses the best available input backend.
///
/// When the `cpal-backend` feature is enabled, CPAL is used; otherwise an
/// error is returned instructing the user to enable the feature.
pub fn build_input(cfg: &AudioInputConfig) -> HertaResult<Arc<dyn AudioInput>> {
    #[cfg(feature = "cpal-backend")]
    {
        Ok(Arc::new(cpal_backend::CpalAudioInput::new(cfg.clone())?))
    }
    #[cfg(not(feature = "cpal-backend"))]
    {
        let _ = cfg;
        Err(herta_core::HertaError::config(
            "no audio input backend enabled (enable 'cpal-backend')",
        ))
    }
}

/// Factory that chooses the best available output backend.
pub fn build_output(cfg: &AudioOutputConfig) -> HertaResult<Arc<dyn AudioOutput>> {
    #[cfg(feature = "cpal-backend")]
    {
        Ok(Arc::new(cpal_backend::CpalAudioOutput::new(cfg.clone())?))
    }
    #[cfg(not(feature = "cpal-backend"))]
    {
        let _ = cfg;
        Ok(Arc::new(null_backend::NullAudioOutput::default()))
    }
}

/// Convenience: build both input and output from a top-level [`Config`].
pub fn build_all(
    cfg: &Config,
) -> HertaResult<(Arc<dyn AudioInput>, Arc<dyn AudioOutput>)> {
    let input = build_input(&cfg.audio)?;
    let output = build_output(&cfg.audio_output)?;
    Ok((input, output))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_builds_even_without_cpal() {
        let cfg = AudioOutputConfig::default();
        let _ = build_output(&cfg);
    }
}
