//! Shared helpers for LLM providers: HTTP client construction, timeouts, and
//! HTTP-status → `HertaError` mapping.

use herta_core::{HertaError, HertaResult, retry::RetryPolicy};
use reqwest::{Client, ClientBuilder, Response, StatusCode};
use std::time::Duration;

/// Default user-agent string sent with every provider request.
pub const USER_AGENT: &str = concat!("the-herta/", env!("CARGO_PKG_VERSION"));

/// Build a shared HTTP client with sensible production defaults.
pub fn build_http_client(timeout: Duration) -> HertaResult<Client> {
    ClientBuilder::new()
        .user_agent(USER_AGENT)
        .timeout(timeout)
        .connect_timeout(Duration::from_secs(10))
        .pool_max_idle_per_host(8)
        .pool_idle_timeout(Duration::from_mins(1))
        .http1_title_case_headers()
        .build()
        .map_err(|e| HertaError::transport(format!("failed to build HTTP client: {e}")))
}

/// Translate a retry count into a [`RetryPolicy`].
pub fn retry_from_attempts(attempts: u32) -> RetryPolicy {
    if attempts == 0 {
        RetryPolicy::no_retry()
    } else {
        RetryPolicy {
            max_attempts: attempts.saturating_add(1),
            ..RetryPolicy::default()
        }
    }
}

/// Convert an HTTP status + body into a `HertaError`.
pub fn map_status_error(
    provider: &'static str,
    operation: &'static str,
    status: StatusCode,
    body_snippet: &str,
    retry_after: Option<Duration>,
) -> HertaError {
    let snippet = truncate_body(body_snippet);
    match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
            HertaError::Auth(format!("{provider}:{operation}: {status}: {snippet}"))
        }
        StatusCode::NOT_FOUND => HertaError::NotFound(format!(
            "{provider}:{operation}: {status}: {snippet}"
        )),
        StatusCode::REQUEST_TIMEOUT | StatusCode::GATEWAY_TIMEOUT => {
            HertaError::Timeout(retry_after.unwrap_or(Duration::from_secs(10)))
        }
        StatusCode::TOO_MANY_REQUESTS | StatusCode::CONFLICT => {
            HertaError::RateLimited { retry_after }
        }
        s if s.is_server_error() => HertaError::Unavailable(format!(
            "{provider}:{operation}: {status}: {snippet}"
        )),
        _ => HertaError::provider(
            provider,
            operation,
            format!("{status}: {snippet}"),
        ),
    }
}

/// Extract a `Retry-After` hint from a response.
pub fn retry_after_hint(resp: &Response) -> Option<Duration> {
    resp.headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(Duration::from_secs)
}

/// Truncate long bodies so they can be embedded in an error string.
pub fn truncate_body(body: &str) -> String {
    const MAX: usize = 256;
    let trimmed = body.trim();
    if trimmed.len() <= MAX {
        trimmed.to_string()
    } else {
        let mut s = String::with_capacity(MAX + 3);
        s.push_str(&trimmed[..MAX]);
        s.push_str("...");
        s
    }
}

/// Wrap a reqwest error as a provider transport error.
pub fn to_transport_error(e: &reqwest::Error) -> HertaError {
    if e.is_timeout() {
        HertaError::Timeout(Duration::from_secs(0))
    } else if e.is_connect() {
        HertaError::transport(format!("connect error: {e}"))
    } else {
        HertaError::transport(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_rate_limit_returns_rate_limited() {
        let err = map_status_error(
            "ollama", "generate", StatusCode::TOO_MANY_REQUESTS,
            "too many", Some(Duration::from_secs(5)),
        );
        assert!(matches!(err, HertaError::RateLimited { retry_after: Some(_) }));
    }

    #[test]
    fn map_unauthorized_returns_auth() {
        let err = map_status_error(
            "google_ai", "generate", StatusCode::UNAUTHORIZED, "bad key", None,
        );
        assert!(matches!(err, HertaError::Auth(_)));
    }

    #[test]
    fn map_5xx_returns_unavailable() {
        let err = map_status_error(
            "deepseek", "generate", StatusCode::INTERNAL_SERVER_ERROR, "oops", None,
        );
        assert!(matches!(err, HertaError::Unavailable(_)));
        assert!(err.is_retryable());
    }

    #[test]
    fn truncate_caps_length() {
        let long = "a".repeat(500);
        assert!(truncate_body(&long).len() <= 259);
    }

    #[test]
    fn retry_from_attempts_handles_zero() {
        let p = retry_from_attempts(0);
        assert_eq!(p.max_attempts, 1);
    }

    #[test]
    fn retry_from_attempts_adds_one() {
        let p = retry_from_attempts(3);
        assert_eq!(p.max_attempts, 4);
    }
}
