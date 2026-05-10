//! Exponential-backoff retry helper shared by every provider.
//!
//! The goal of this module is to concentrate retry policy in one place so
//! every provider (LLM, STT, TTS, memory) obeys the same semantics:
//!
//! - Only retry errors where [`crate::HertaError::is_retryable`] is `true`.
//! - Respect a configurable `max_attempts` (1 == no retry).
//! - Use exponential backoff with full jitter.
//! - Stop immediately on caller-initiated cancellation.

use crate::{HertaError, HertaResult};
use std::future::Future;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Configurable retry policy.
#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    /// Maximum number of attempts, including the first try. `1` disables retries.
    pub max_attempts: u32,
    /// Initial backoff delay.
    pub initial_backoff: Duration,
    /// Maximum backoff delay per step.
    pub max_backoff: Duration,
    /// Multiplier applied to the backoff on each failure.
    pub multiplier: f64,
    /// Jitter factor in `[0.0, 1.0]`. `0.0` disables jitter.
    pub jitter: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(250),
            max_backoff: Duration::from_secs(5),
            multiplier: 2.0,
            jitter: 0.2,
        }
    }
}

impl RetryPolicy {
    /// Convenience constructor: disable retries (single attempt only).
    pub fn no_retry() -> Self {
        Self {
            max_attempts: 1,
            ..Self::default()
        }
    }

    fn delay_for(&self, attempt: u32) -> Duration {
            let base = self.initial_backoff.as_secs_f64()
            * self.multiplier.powi(i32::try_from(attempt.saturating_sub(1)).unwrap_or(i32::MAX));
        let capped = base.min(self.max_backoff.as_secs_f64());

        // Deterministic pseudo-jitter (hash attempt count) so tests stay
        // reproducible without pulling in a PRNG dependency.
        let jitter_ratio = if self.jitter > 0.0 {
            let seed = f64::from(attempt.wrapping_mul(2_654_435_761)) / f64::from(u32::MAX);
            1.0 - self.jitter + seed * self.jitter * 2.0
        } else {
            1.0
        };
        Duration::from_secs_f64((capped * jitter_ratio).max(0.0))
    }
}

/// Run `op` with the configured retry policy, aborting on cancellation.
pub async fn with_retry<T, F, Fut>(
    policy: RetryPolicy,
    cancel: Option<&CancellationToken>,
    mut op: F,
) -> HertaResult<T>
where
    F: FnMut() -> Fut + Send,
    Fut: Future<Output = HertaResult<T>> + Send,
{
    let max = policy.max_attempts.max(1);
    let mut last_error: Option<HertaError> = None;

    for attempt in 1..=max {
        if let Some(token) = cancel
            && token.is_cancelled()
        {
            return Err(HertaError::Cancelled);
        }

        match op().await {
            Ok(value) => return Ok(value),
            Err(err) => {
                if !err.is_retryable() || attempt == max {
                    return Err(err);
                }
                tracing::debug!(
                    target: "herta.retry",
                    attempt,
                    max,
                    kind = err.kind(),
                    "retrying after retryable failure"
                );
                last_error = Some(err);

                let sleep = policy.delay_for(attempt);
                match cancel {
                    Some(token) => {
                        tokio::select! {
                            () = tokio::time::sleep(sleep) => {}
                            () = token.cancelled() => return Err(HertaError::Cancelled),
                        }
                    }
                    None => tokio::time::sleep(sleep).await,
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| HertaError::internal("retry loop exhausted")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    #[tokio::test]
    async fn succeeds_on_first_attempt() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = calls.clone();
        let out: HertaResult<u32> = with_retry(RetryPolicy::default(), None, move || {
            let c = c.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(42)
            }
        })
        .await;
        assert_eq!(out.unwrap(), 42);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retries_retryable_failures() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = calls.clone();
        let policy = RetryPolicy {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(0),
            max_backoff: Duration::from_millis(0),
            multiplier: 1.0,
            jitter: 0.0,
        };
        let out: HertaResult<u32> = with_retry(policy, None, move || {
            let c = c.clone();
            async move {
                let n = c.fetch_add(1, Ordering::SeqCst) + 1;
                if n < 3 {
                    Err(HertaError::transport("flaky"))
                } else {
                    Ok(n)
                }
            }
        })
        .await;
        assert_eq!(out.unwrap(), 3);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn does_not_retry_non_retryable() {
        let calls = Arc::new(AtomicU32::new(0));
        let c = calls.clone();
        let out: HertaResult<u32> = with_retry(RetryPolicy::default(), None, move || {
            let c = c.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                Err(HertaError::config("bad"))
            }
        })
        .await;
        assert!(out.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cancellation_aborts_immediately() {
        let token = CancellationToken::new();
        token.cancel();
        let out: HertaResult<u32> =
            with_retry(RetryPolicy::default(), Some(&token), || async {
                Ok::<u32, HertaError>(1)
            })
            .await;
        assert!(matches!(out, Err(HertaError::Cancelled)));
    }
}
