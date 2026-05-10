//! Null audio output: discards everything. Used in headless / test contexts
//! where a live audio device is not available.

use async_trait::async_trait;
use bytes::Bytes;
use herta_core::{
    HertaResult,
    audio::{AudioFormat, AudioOutput},
};
use parking_lot::Mutex;

/// Null audio output that counts the number of buffers it has discarded.
#[derive(Debug, Default)]
pub struct NullAudioOutput {
    plays: Mutex<u64>,
}

impl NullAudioOutput {
    /// Number of buffers that were handed to `play` since construction.
    pub fn play_count(&self) -> u64 {
        *self.plays.lock()
    }
}

#[async_trait]
impl AudioOutput for NullAudioOutput {
    fn name(&self) -> &'static str {
        "null"
    }

    async fn play(&self, _pcm: Bytes, _format: AudioFormat) -> HertaResult<()> {
        *self.plays.lock() += 1;
        Ok(())
    }

    async fn play_test_tone(&self) -> HertaResult<()> {
        *self.plays.lock() += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn counts_plays() {
        let out = NullAudioOutput::default();
        out.play(Bytes::new(), AudioFormat::default()).await.unwrap();
        out.play(Bytes::new(), AudioFormat::default()).await.unwrap();
        assert_eq!(out.play_count(), 2);
    }
}
