//! Test-tone generation helpers.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]

use herta_core::audio::{AudioFormat, SampleFormat};
use std::f32::consts::TAU;

/// Generate a pure sine-wave tone as interleaved PCM bytes matching `format`.
///
/// The waveform is scaled by `volume` (clamped to `[0.0, 1.0]`).
pub fn generate_tone(
    format: AudioFormat,
    frequency_hz: f32,
    duration_seconds: f32,
    volume: f32,
) -> Vec<u8> {
    let volume = volume.clamp(0.0, 1.0);
    let total_samples = (f32::from(format.channels.max(1))
        * format.sample_rate as f32
        * duration_seconds)
        .round() as usize;

    match format.sample_format {
        SampleFormat::F32 => {
            let mut buf = Vec::with_capacity(total_samples * 4);
            for i in 0..total_samples {
                let t = i as f32
                    / (format.sample_rate as f32 * f32::from(format.channels.max(1)));
                let sample = (TAU * frequency_hz * t).sin() * volume;
                buf.extend_from_slice(&sample.to_le_bytes());
            }
            buf
        }
        SampleFormat::I16 => {
            let mut buf = Vec::with_capacity(total_samples * 2);
            for i in 0..total_samples {
                let t = i as f32
                    / (format.sample_rate as f32 * f32::from(format.channels.max(1)));
                let sample = (TAU * frequency_hz * t).sin() * volume;
                let scaled = (sample * f32::from(i16::MAX)) as i16;
                buf.extend_from_slice(&scaled.to_le_bytes());
            }
            buf
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tone_length_matches_expected_sample_count() {
        let format = AudioFormat {
            sample_rate: 8_000,
            channels: 1,
            sample_format: SampleFormat::I16,
        };
        let bytes = generate_tone(format, 440.0, 0.25, 0.5);
        assert_eq!(bytes.len(), 8_000 * 2 / 4);
    }

    #[test]
    fn tone_nonzero_at_nonzero_volume() {
        let format = AudioFormat::default();
        let bytes = generate_tone(format, 440.0, 0.1, 0.2);
        assert!(bytes.iter().any(|b| *b != 0));
    }

    #[test]
    fn tone_is_silent_at_zero_volume() {
        let format = AudioFormat {
            sample_rate: 8_000,
            channels: 1,
            sample_format: SampleFormat::I16,
        };
        let bytes = generate_tone(format, 440.0, 0.1, 0.0);
        assert!(bytes.iter().all(|b| *b == 0));
    }
}
