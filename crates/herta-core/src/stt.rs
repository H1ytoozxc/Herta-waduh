//! Speech-to-text abstraction.

use crate::{HertaResult, audio::Utterance};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// A transcription result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    /// Full transcribed text. May be empty when confidence is too low.
    pub text: String,
    /// Optional language code detected by the engine.
    pub language: Option<String>,
    /// Confidence in [0.0, 1.0] where supported.
    pub confidence: Option<f32>,
    /// Engine latency for the call.
    pub latency: Duration,
    /// Engine identifier for metrics.
    pub provider: String,
}

impl Transcript {
    /// Returns `true` when the transcript produced usable text.
    pub fn is_meaningful(&self) -> bool {
        !self.text.trim().is_empty()
    }
}

/// STT engine contract.
#[async_trait]
pub trait SttEngine: Send + Sync + 'static {
    /// Short, stable backend name for metrics/logs.
    fn name(&self) -> &'static str;

    /// Human-readable description of the active device / target.
    fn active_device(&self) -> String {
        self.name().to_string()
    }

    /// Best-effort warm-up (e.g. download weights, open session).
    async fn warm_up(&self) -> HertaResult<bool> {
        Ok(true)
    }

    /// Transcribe a single utterance.
    async fn transcribe(&self, utterance: &Utterance) -> HertaResult<Transcript>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meaningful_trims_whitespace() {
        let t = Transcript {
            text: "   \n\t  ".into(),
            language: None,
            confidence: None,
            latency: Duration::from_millis(0),
            provider: "x".into(),
        };
        assert!(!t.is_meaningful());

        let t = Transcript {
            text: "hello".into(),
            ..t
        };
        assert!(t.is_meaningful());
    }
}
