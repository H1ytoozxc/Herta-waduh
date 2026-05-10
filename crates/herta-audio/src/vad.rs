//! Voice-activity-detection (VAD) abstractions.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]
//!
//! The default implementation ([`EnergyVad`]) is a simple energy-threshold
//! detector with hysteresis. A more accurate neural detector (Silero) can be
//! added later behind the same [`Vad`] trait.

use herta_config::schema::VadConfig;
use herta_core::{
    HertaResult,
    audio::{AudioChunk, SampleFormat, Utterance},
};

/// Verdict for a single chunk of audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadDecision {
    /// Chunk contains speech; the utterance is still ongoing.
    Speaking,
    /// Chunk is silence; utterance may end if silence threshold reached.
    Silent,
    /// Utterance finalized (returned instead of chunk decision).
    UtteranceReady,
}

/// Voice-activity detector / segmenter.
pub trait Vad: Send + Sync + 'static {
    /// Feed a chunk into the detector.
    ///
    /// Returns `Ok(Some(utt))` when a complete utterance has been assembled;
    /// otherwise `Ok(None)` and the caller should keep feeding chunks.
    fn process_chunk(&mut self, chunk: AudioChunk) -> HertaResult<Option<Utterance>>;

    /// Reset the internal state, discarding any buffered audio.
    fn reset(&mut self);
}

/// Simple energy-based VAD with hysteresis, calibrated by [`VadConfig`].
///
/// This detector is not state-of-the-art but is dependency-free, deterministic,
/// and sufficient for the default `Fifine`-style microphone scenarios.
#[derive(Debug)]
pub struct EnergyVad {
    cfg: VadConfig,
    buffer: Vec<AudioChunk>,
    silent_ms: u32,
    total_ms: u32,
    in_speech: bool,
}

impl EnergyVad {
    /// Build a new energy VAD.
    pub fn new(cfg: VadConfig) -> Self {
        Self {
            cfg,
            buffer: Vec::with_capacity(64),
            silent_ms: 0,
            total_ms: 0,
            in_speech: false,
        }
    }

    fn rms_of_chunk(chunk: &AudioChunk) -> f32 {
        match chunk.format.sample_format {
            SampleFormat::F32 => {
                if chunk.pcm.len() < 4 {
                    return 0.0;
                }
                let samples = chunk.pcm.len() / 4;
                let mut sum = 0.0f64;
                for i in 0..samples {
                    let s = i * 4;
                    let v = f32::from_le_bytes([
                        chunk.pcm[s],
                        chunk.pcm[s + 1],
                        chunk.pcm[s + 2],
                        chunk.pcm[s + 3],
                    ]);
                    sum += (v as f64) * (v as f64);
                }
                (sum / samples as f64).sqrt() as f32
            }
            SampleFormat::I16 => {
                if chunk.pcm.len() < 2 {
                    return 0.0;
                }
                let samples = chunk.pcm.len() / 2;
                let mut sum = 0.0f64;
                for i in 0..samples {
                    let s = i * 2;
                    let v = i16::from_le_bytes([chunk.pcm[s], chunk.pcm[s + 1]]);
                    let norm = v as f64 / i16::MAX as f64;
                    sum += norm * norm;
                }
                (sum / samples as f64).sqrt() as f32
            }
        }
    }

    fn chunk_duration_ms(chunk: &AudioChunk) -> u32 {
        let bps = match chunk.format.sample_format {
            SampleFormat::F32 => 4,
            SampleFormat::I16 => 2,
        };
        let total = chunk.pcm.len() / (bps * usize::from(chunk.format.channels.max(1)));
        ((total as u64) * 1_000 / u64::from(chunk.format.sample_rate.max(1))) as u32
    }
}

impl Vad for EnergyVad {
    fn process_chunk(&mut self, chunk: AudioChunk) -> HertaResult<Option<Utterance>> {
        let rms = Self::rms_of_chunk(&chunk);
        let chunk_ms = Self::chunk_duration_ms(&chunk);
        let is_speech = rms >= self.cfg.threshold;

        self.total_ms = self.total_ms.saturating_add(chunk_ms);

        if is_speech {
            self.in_speech = true;
            self.silent_ms = 0;
            self.buffer.push(chunk);
        } else if self.in_speech {
            self.silent_ms = self.silent_ms.saturating_add(chunk_ms);
            self.buffer.push(chunk);
        }

        let max_ms = (self.cfg.max_utterance_seconds * 1_000.0) as u32;
        let should_finalize = self.in_speech
            && (self.silent_ms >= self.cfg.min_silence_duration_ms
                || self.total_ms >= max_ms);

        if should_finalize {
            let utt = Utterance::from_chunks(&self.buffer);
            let min_ms = self.cfg.min_utterance_duration_ms;
            self.reset();
            if utt.duration_ms >= u64::from(min_ms) {
                return Ok(Some(utt));
            }
        }
        Ok(None)
    }

    fn reset(&mut self) {
        self.buffer.clear();
        self.silent_ms = 0;
        self.total_ms = 0;
        self.in_speech = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use herta_core::audio::AudioFormat;

    fn chunk(rms_target: f32, ms: u32) -> AudioChunk {
        // generate a tone chunk with approximate RMS magnitude.
        let sample_rate = 16_000u32;
        let samples = ((sample_rate as u64) * u64::from(ms) / 1_000) as usize;
        let mut pcm = Vec::with_capacity(samples * 4);
        for i in 0..samples {
            let v = if (i % 2) == 0 { rms_target } else { -rms_target };
            pcm.extend_from_slice(&v.to_le_bytes());
        }
        AudioChunk {
            pcm: Bytes::from(pcm),
            format: AudioFormat {
                sample_rate,
                channels: 1,
                sample_format: SampleFormat::F32,
            },
            captured_at_ms: 0,
        }
    }

    #[test]
    fn rms_of_silence_is_zero() {
        let ch = AudioChunk {
            pcm: Bytes::from(vec![0u8; 4 * 100]),
            format: AudioFormat::default(),
            captured_at_ms: 0,
        };
        assert_eq!(EnergyVad::rms_of_chunk(&ch), 0.0);
    }

    #[test]
    fn finalizes_after_silence() {
        let cfg = VadConfig {
            threshold: 0.1,
            min_silence_duration_ms: 300,
            speech_pad_ms: 0,
            min_utterance_duration_ms: 100,
            max_utterance_seconds: 10.0,
        };
        let mut vad = EnergyVad::new(cfg);
        assert!(vad.process_chunk(chunk(0.5, 500)).unwrap().is_none());
        assert!(vad.process_chunk(chunk(0.0, 200)).unwrap().is_none());
        let r = vad.process_chunk(chunk(0.0, 200)).unwrap();
        assert!(r.is_some());
        let utt = r.unwrap();
        assert!(utt.duration_ms >= 100);
    }

    #[test]
    fn skips_utterances_below_min_duration() {
        let cfg = VadConfig {
            threshold: 0.1,
            min_silence_duration_ms: 100,
            speech_pad_ms: 0,
            min_utterance_duration_ms: 600,
            max_utterance_seconds: 10.0,
        };
        let mut vad = EnergyVad::new(cfg);
        vad.process_chunk(chunk(0.5, 100)).unwrap();
        let r = vad.process_chunk(chunk(0.0, 300)).unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn finalizes_when_max_duration_reached() {
        let cfg = VadConfig {
            threshold: 0.1,
            min_silence_duration_ms: 10_000,
            speech_pad_ms: 0,
            min_utterance_duration_ms: 100,
            max_utterance_seconds: 0.5,
        };
        let mut vad = EnergyVad::new(cfg);
        // continuous speech; forced finalize when total ms >= 500ms
        assert!(vad.process_chunk(chunk(0.5, 200)).unwrap().is_none());
        let r = vad.process_chunk(chunk(0.5, 400)).unwrap();
        assert!(r.is_some());
    }
}
