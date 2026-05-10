//! Unified error taxonomy for the entire Herta workspace.

use std::time::Duration;
use thiserror::Error;

/// Result alias for operations producing a [`HertaError`].
pub type HertaResult<T> = Result<T, HertaError>;

/// Top-level error type shared across every crate in the workspace.
///
/// Errors are deliberately coarse-grained at the boundary and carry rich
/// contextual information (provider name, operation, correlation ids) so they
/// can be turned into structured log events and Prometheus labels without
/// leaking private data.
#[derive(Debug, Error)]
pub enum HertaError {
    /// Configuration was invalid or missing required fields.
    #[error("configuration error: {0}")]
    Config(String),

    /// An upstream provider (LLM, STT, TTS, cloud API) returned an error.
    #[error("provider '{provider}' failed during '{operation}': {source}")]
    Provider {
        /// Provider identifier, e.g. `"ollama"`, `"deepseek"`, `"google_ai"`.
        provider: &'static str,
        /// Short label describing the attempted operation.
        operation: &'static str,
        /// Underlying source error rendered as a string to stay `Send + Sync`.
        #[source]
        source: BoxError,
    },

    /// Network, transport, or HTTP-level error.
    #[error("transport error: {0}")]
    Transport(String),

    /// Authentication or authorization failure (missing/invalid API key).
    #[error("authentication error: {0}")]
    Auth(String),

    /// Provider-level rate limiting; `retry_after` indicates a suggested wait.
    #[error("rate limited; retry after {retry_after:?}")]
    RateLimited {
        /// Optional hint for how long to back off.
        retry_after: Option<Duration>,
    },

    /// The operation did not complete within the allotted time.
    #[error("operation timed out after {0:?}")]
    Timeout(Duration),

    /// The operation was cancelled by the caller.
    #[error("operation cancelled")]
    Cancelled,

    /// An I/O error occurred (file system, audio device, etc.).
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization or deserialization error.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// The requested resource (model, device, file) was not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// Invalid or malformed input supplied by the caller.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Audio subsystem error (device, format, buffer, VAD).
    #[error("audio error: {0}")]
    Audio(String),

    /// Persistence layer error (JSON, `SQLite`, Sled).
    #[error("memory error: {0}")]
    Memory(String),

    /// The pipeline detected an unrecoverable state transition.
    #[error("pipeline error: {0}")]
    Pipeline(String),

    /// The component is temporarily unavailable; transient failures should
    /// be retried by callers with exponential backoff.
    #[error("unavailable: {0}")]
    Unavailable(String),

    /// Catch-all for conditions that don't map to a specific variant.
    #[error("internal error: {0}")]
    Internal(String),
}

/// Boxed, thread-safe error used as the `#[source]` for provider errors.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

impl HertaError {
    /// Build a [`HertaError::Provider`] from a provider id, operation label,
    /// and an arbitrary source error.
    pub fn provider<E>(provider: &'static str, operation: &'static str, source: E) -> Self
    where
        E: Into<BoxError>,
    {
        Self::Provider {
            provider,
            operation,
            source: source.into(),
        }
    }

    /// Construct a configuration error.
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config(message.into())
    }

    /// Construct a transport error.
    pub fn transport(message: impl Into<String>) -> Self {
        Self::Transport(message.into())
    }

    /// Construct an invalid-input error.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }

    /// Construct an internal/unknown error.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }

    /// Construct an audio subsystem error.
    pub fn audio(message: impl Into<String>) -> Self {
        Self::Audio(message.into())
    }

    /// Construct a memory subsystem error.
    pub fn memory(message: impl Into<String>) -> Self {
        Self::Memory(message.into())
    }

    /// Returns `true` when the error represents a transient condition that a
    /// caller is permitted to retry with backoff.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Transport(_)
                | Self::RateLimited { .. }
                | Self::Timeout(_)
                | Self::Unavailable(_)
        )
    }

    /// Short, stable label suitable for use as a metrics dimension.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Config(_) => "config",
            Self::Provider { .. } => "provider",
            Self::Transport(_) => "transport",
            Self::Auth(_) => "auth",
            Self::RateLimited { .. } => "rate_limited",
            Self::Timeout(_) => "timeout",
            Self::Cancelled => "cancelled",
            Self::Io(_) => "io",
            Self::Serialization(_) => "serialization",
            Self::NotFound(_) => "not_found",
            Self::InvalidInput(_) => "invalid_input",
            Self::Audio(_) => "audio",
            Self::Memory(_) => "memory",
            Self::Pipeline(_) => "pipeline",
            Self::Unavailable(_) => "unavailable",
            Self::Internal(_) => "internal",
        }
    }
}

impl From<serde_json::Error> for HertaError {
    fn from(e: serde_json::Error) -> Self {
        Self::Serialization(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_classification() {
        assert!(HertaError::transport("boom").is_retryable());
        assert!(HertaError::Timeout(Duration::from_secs(1)).is_retryable());
        assert!(HertaError::RateLimited { retry_after: None }.is_retryable());
        assert!(!HertaError::config("bad").is_retryable());
        assert!(!HertaError::invalid("nope").is_retryable());
    }

    #[test]
    fn provider_error_preserves_source() {
        let err = HertaError::provider("ollama", "generate", "upstream 500");
        match err {
            HertaError::Provider {
                provider,
                operation,
                ..
            } => {
                assert_eq!(provider, "ollama");
                assert_eq!(operation, "generate");
            }
            _ => panic!("expected provider variant"),
        }
    }

    #[test]
    fn kind_labels_are_stable() {
        assert_eq!(HertaError::config("x").kind(), "config");
        assert_eq!(HertaError::audio("x").kind(), "audio");
        assert_eq!(HertaError::memory("x").kind(), "memory");
        assert_eq!(HertaError::Cancelled.kind(), "cancelled");
    }
}
