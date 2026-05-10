//! Minimal WAV (RIFF/WAVE) encoder used to wrap raw PCM utterances before
//! uploading to cloud STT services. Supports mono/stereo, 16-bit PCM or
//! 32-bit float IEEE, little-endian.

use herta_core::{
    HertaError, HertaResult,
    audio::{AudioFormat, SampleFormat, Utterance},
};

/// Encode an utterance to a WAV byte buffer.
pub(crate) fn encode_wav(utterance: &Utterance) -> HertaResult<Vec<u8>> {
    encode_pcm(&utterance.pcm, utterance.format)
}

/// Encode raw PCM (little-endian interleaved) to a WAV byte buffer.
pub(crate) fn encode_pcm(pcm: &[u8], format: AudioFormat) -> HertaResult<Vec<u8>> {
    if format.channels == 0 || format.sample_rate == 0 {
        return Err(HertaError::audio("invalid audio format for WAV encoding"));
    }
    let (audio_format_tag, bits_per_sample) = match format.sample_format {
        SampleFormat::I16 => (1u16, 16u16),
        SampleFormat::F32 => (3u16, 32u16),
    };
    let byte_rate = format.sample_rate
        * u32::from(format.channels)
        * u32::from(bits_per_sample)
        / 8;
    let block_align = format.channels * bits_per_sample / 8;
    let data_len = u32::try_from(pcm.len())
        .map_err(|_| HertaError::audio("PCM buffer too large for WAV encoding"))?;
    let chunk_size = 36u32
        .checked_add(data_len)
        .ok_or_else(|| HertaError::audio("WAV chunk size overflow"))?;

    let mut buf = Vec::with_capacity(44 + pcm.len());
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&chunk_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&audio_format_tag.to_le_bytes());
    buf.extend_from_slice(&format.channels.to_le_bytes());
    buf.extend_from_slice(&format.sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());

    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    buf.extend_from_slice(pcm);

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use herta_core::audio::AudioFormat;

    #[test]
    fn encodes_valid_riff_header() {
        let format = AudioFormat {
            sample_rate: 16_000,
            channels: 1,
            sample_format: SampleFormat::I16,
        };
        let pcm = vec![0u8; 32];
        let wav = encode_pcm(&pcm, format).unwrap();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
        // total length = header (44) + data
        assert_eq!(wav.len(), 44 + pcm.len());
    }

    #[test]
    fn rejects_zero_sample_rate() {
        let format = AudioFormat {
            sample_rate: 0,
            channels: 1,
            sample_format: SampleFormat::F32,
        };
        assert!(encode_pcm(&[], format).is_err());
    }

    #[test]
    fn encodes_utterance_header_matches() {
        let format = AudioFormat {
            sample_rate: 16_000,
            channels: 1,
            sample_format: SampleFormat::F32,
        };
        let utt = Utterance {
            pcm: Bytes::from(vec![0u8; 64]),
            format,
            duration_ms: 1,
        };
        let wav = encode_wav(&utt).unwrap();
        // bits per sample for f32 = 32
        let bps = u16::from_le_bytes([wav[34], wav[35]]);
        assert_eq!(bps, 32);
    }
}
