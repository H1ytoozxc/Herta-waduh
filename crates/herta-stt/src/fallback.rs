//! Fallback STT wrapper that routes requests through a primary engine and
//! falls back to a secondary one on error.
//!
//! The fallback engine is lazily consulted; if the primary never fails, the
//! secondary is never called (which may matter when the secondary is
//! resource-intensive to initialize).

use async_trait::async_trait;
use herta_core::{
    HertaResult,
    audio::Utterance,
    stt::{SttEngine, Transcript},
};
use std::sync::Arc;

/// Fallback-aware STT engine.
pub struct FallbackStt {
    primary: Arc<dyn SttEngine>,
    secondary: Arc<dyn SttEngine>,
}

impl std::fmt::Debug for FallbackStt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FallbackStt")
            .field("primary", &self.primary.name())
            .field("secondary", &self.secondary.name())
            .finish()
    }
}

impl FallbackStt {
    /// Wrap two engines.
    pub fn new(primary: Arc<dyn SttEngine>, secondary: Arc<dyn SttEngine>) -> Self {
        Self { primary, secondary }
    }
}

#[async_trait]
impl SttEngine for FallbackStt {
    fn name(&self) -> &'static str {
        "fallback-stt"
    }

    fn active_device(&self) -> String {
        format!(
            "{}+{}",
            self.primary.active_device(),
            self.secondary.active_device()
        )
    }

    async fn warm_up(&self) -> HertaResult<bool> {
        let p = self.primary.warm_up().await.unwrap_or(false);
        let s = self.secondary.warm_up().await.unwrap_or(false);
        Ok(p || s)
    }

    async fn transcribe(&self, utterance: &Utterance) -> HertaResult<Transcript> {
        match self.primary.transcribe(utterance).await {
            Ok(t) => Ok(t),
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    primary = self.primary.name(),
                    secondary = self.secondary.name(),
                    "primary STT failed, trying fallback"
                );
                self.secondary.transcribe(utterance).await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use bytes::Bytes;
    use herta_core::{
        HertaError,
        audio::{AudioFormat, Utterance},
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    #[derive(Default)]
    struct AlwaysFailStt {
        called: AtomicBool,
    }

    #[async_trait]
    impl SttEngine for AlwaysFailStt {
        fn name(&self) -> &'static str {
            "fail"
        }
        async fn transcribe(&self, _u: &Utterance) -> HertaResult<Transcript> {
            self.called.store(true, Ordering::SeqCst);
            Err(HertaError::transport("nope"))
        }
    }

    struct AlwaysOkStt;

    #[async_trait]
    impl SttEngine for AlwaysOkStt {
        fn name(&self) -> &'static str {
            "ok"
        }
        async fn transcribe(&self, _u: &Utterance) -> HertaResult<Transcript> {
            Ok(Transcript {
                text: "fallback-result".into(),
                language: None,
                confidence: None,
                latency: Duration::from_millis(0),
                provider: "ok".into(),
            })
        }
    }

    fn test_utterance() -> Utterance {
        Utterance {
            pcm: Bytes::new(),
            format: AudioFormat::default(),
            duration_ms: 100,
        }
    }

    #[tokio::test]
    async fn falls_back_on_primary_failure() {
        let primary = Arc::new(AlwaysFailStt::default());
        let secondary = Arc::new(AlwaysOkStt);
        let engine = FallbackStt::new(primary.clone(), secondary);
        let t = engine.transcribe(&test_utterance()).await.unwrap();
        assert_eq!(t.text, "fallback-result");
        assert!(primary.called.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn primary_success_is_returned() {
        let primary = Arc::new(AlwaysOkStt);
        let secondary = Arc::new(AlwaysFailStt::default());
        let engine = FallbackStt::new(primary, secondary.clone());
        let t = engine.transcribe(&test_utterance()).await.unwrap();
        assert_eq!(t.text, "fallback-result");
        assert!(!secondary.called.load(Ordering::SeqCst));
    }
}
