//! Audio input/output abstractions.
//!
//! The real audio backend lives in the `herta-audio` crate; this module
//! declares the wire-level types and traits that downstream code consumes.

use crate::HertaResult;
use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Linear-PCM sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SampleFormat {
    /// 32-bit floating-point samples in `[-1.0, 1.0]`.
    F32,
    /// 16-bit signed integer samples.
    I16,
}

impl fmt::Display for SampleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F32 => "f32",
            Self::I16 => "i16",
        })
    }
}

/// Audio buffer description attached to every chunk/utterance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioFormat {
    /// Sample rate in Hz (e.g. 16000, 44100).
    pub sample_rate: u32,
    /// Channel count (e.g. 1 for mono).
    pub channels: u16,
    /// Sample encoding.
    pub sample_format: SampleFormat,
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            channels: 1,
            sample_format: SampleFormat::F32,
        }
    }
}

/// A short block of audio produced by the input source.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Raw interleaved PCM bytes.
    pub pcm: Bytes,
    /// Format description.
    pub format: AudioFormat,
    /// Capture timestamp in milliseconds since epoch.
    pub captured_at_ms: i64,
}

/// A complete utterance assembled by VAD from a sequence of chunks.
#[derive(Debug, Clone)]
pub struct Utterance {
    /// Concatenated PCM for the utterance.
    pub pcm: Bytes,
    /// Format description.
    pub format: AudioFormat,
    /// Duration of the utterance in milliseconds.
    pub duration_ms: u64,
}

impl Utterance {
    /// Build an utterance from a list of chunks. All chunks must share the
    /// same format; the first chunk's format is used as the reference.
    pub fn from_chunks(chunks: &[AudioChunk]) -> Self {
        let format = chunks
            .first()
            .map(|c| c.format)
            .unwrap_or_default();
        let mut buffer: Vec<u8> =
            Vec::with_capacity(chunks.iter().map(|c| c.pcm.len()).sum());
        for c in chunks {
            buffer.extend_from_slice(&c.pcm);
        }
        let bytes_per_sample = match format.sample_format {
            SampleFormat::F32 => 4,
            SampleFormat::I16 => 2,
        };
        let total_samples =
            buffer.len() / (bytes_per_sample * usize::from(format.channels.max(1)));
        let duration_ms = (total_samples as u64) * 1_000 / u64::from(format.sample_rate.max(1));

        Self {
            pcm: Bytes::from(buffer),
            format,
            duration_ms,
        }
    }
}

/// Opaque identifier for an audio device enumerated by the backend.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceId {
    /// Use the system default device.
    Default,
    /// Select a device by index reported from enumeration.
    Index(u32),
    /// Select a device by a substring of its display name (case-insensitive).
    Name(String),
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => f.write_str("<default>"),
            Self::Index(i) => write!(f, "<#{i}>"),
            Self::Name(n) => write!(f, "<{n}>"),
        }
    }
}

/// Descriptor returned when enumerating devices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Backend-assigned stable index (may vary across boots).
    pub index: u32,
    /// Human-readable device name.
    pub name: String,
    /// Maximum number of input channels.
    pub max_input_channels: u16,
    /// Maximum number of output channels.
    pub max_output_channels: u16,
    /// Preferred sample rate if the backend reports one.
    pub default_sample_rate: Option<u32>,
}

/// Audio input trait.
#[async_trait]
pub trait AudioInput: Send + Sync + 'static {
    /// Short, stable backend name for metrics/logs.
    fn name(&self) -> &'static str;

    /// Start capturing. Implementations must be idempotent.
    async fn start(&self) -> HertaResult<()>;

    /// Stop capturing and release the underlying device.
    async fn stop(&self) -> HertaResult<()>;

    /// Await the next chunk. Should return `Ok(None)` on clean shutdown.
    async fn next_chunk(&self) -> HertaResult<Option<AudioChunk>>;
}

/// Audio output trait.
#[async_trait]
pub trait AudioOutput: Send + Sync + 'static {
    /// Short, stable backend name.
    fn name(&self) -> &'static str;

    /// Play a pre-rendered PCM buffer synchronously (awaits completion).
    async fn play(&self, pcm: Bytes, format: AudioFormat) -> HertaResult<()>;

    /// Play a brief calibration tone; intended for diagnostics.
    async fn play_test_tone(&self) -> HertaResult<()>;
}

/// Device enumeration trait, implemented by each backend.
pub trait DeviceEnumerator: Send + Sync + 'static {
    /// List all known input devices.
    fn list_input_devices(&self) -> HertaResult<Vec<DeviceInfo>>;
    /// List all known output devices.
    fn list_output_devices(&self) -> HertaResult<Vec<DeviceInfo>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[test]
    fn utterance_duration_is_computed_from_format() {
        // 16 kHz mono f32, 16000 samples == 1 second == 64_000 bytes.
        let chunk = AudioChunk {
            pcm: Bytes::from(vec![0u8; 64_000]),
            format: AudioFormat::default(),
            captured_at_ms: 0,
        };
        let utt = Utterance::from_chunks(&[chunk]);
        assert_eq!(utt.duration_ms, 1_000);
    }

    #[test]
    fn empty_utterance_has_zero_duration() {
        let utt = Utterance::from_chunks(&[]);
        assert_eq!(utt.duration_ms, 0);
        assert!(utt.pcm.is_empty());
    }

    #[test]
    fn device_id_display_is_human_readable() {
        assert_eq!(DeviceId::Default.to_string(), "<default>");
        assert_eq!(DeviceId::Index(3).to_string(), "<#3>");
        assert_eq!(DeviceId::Name("fifine".into()).to_string(), "<fifine>");
    }

    #[test]
    fn i16_format_duration_is_computed_correctly() {
        // 16 kHz mono i16, 16000 samples == 1s == 32_000 bytes.
        let chunk = AudioChunk {
            pcm: Bytes::from(vec![0u8; 32_000]),
            format: AudioFormat {
                sample_rate: 16_000,
                channels: 1,
                sample_format: SampleFormat::I16,
            },
            captured_at_ms: 0,
        };
        let utt = Utterance::from_chunks(&[chunk]);
        assert_eq!(utt.duration_ms, 1_000);
    }
}
