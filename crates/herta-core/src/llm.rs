//! LLM provider abstraction.
//!
//! Implementations live in the `herta-llm` crate; this module only defines
//! the trait and supporting value types.

use crate::{DialogContext, HertaResult};
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Opaque token emitted during streaming responses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Delta of text produced since the last chunk.
    pub delta: String,
    /// Whether this chunk is the terminal chunk of the response.
    pub finished: bool,
}

/// A complete LLM response with optional usage metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmResponse {
    /// Final text produced by the model.
    pub text: String,
    /// Provider that produced the response.
    pub provider: String,
    /// Model identifier.
    pub model: String,
    /// Total wall-clock latency to produce the response.
    pub latency: Duration,
    /// Optional usage information (prompt/completion tokens).
    pub usage: Option<TokenUsage>,
}

/// Token-usage statistics, when the provider reports them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    /// Input/prompt tokens.
    pub prompt_tokens: u64,
    /// Output/completion tokens.
    pub completion_tokens: u64,
    /// Total tokens charged.
    pub total_tokens: u64,
}

/// LLM provider contract.
///
/// Implementations must be safe to call concurrently from many tasks and
/// must handle cancellation via their own internal timeouts/abort signals;
/// the higher-level pipeline additionally wraps provider calls in a
/// `CancellationToken`.
#[async_trait]
pub trait LlmProvider: Send + Sync + 'static {
    /// Short, stable name for log/metric labels, e.g. `"ollama"`.
    fn name(&self) -> &'static str;

    /// Best-effort warm-up. Implementations should return `Ok(true)` when
    /// the backend is verified ready, `Ok(false)` if warm-up was attempted
    /// but inconclusive, and an error only for hard failures.
    async fn warm_up(&self) -> HertaResult<bool> {
        Ok(true)
    }

    /// Produce a single complete response for the given context.
    async fn generate(&self, ctx: &DialogContext) -> HertaResult<LlmResponse>;

    /// Optional streaming response. Default implementation falls back to
    /// [`Self::generate`] and yields a single terminal chunk.
    async fn stream<'a>(
        &'a self,
        ctx: &'a DialogContext,
    ) -> HertaResult<BoxStream<'a, HertaResult<StreamChunk>>> {
        let response = self.generate(ctx).await?;
        let chunk = StreamChunk {
            delta: response.text,
            finished: true,
        };
        Ok(Box::pin(futures::stream::iter(vec![Ok(chunk)])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mocks::EchoLlm;
    use futures::StreamExt;

    #[tokio::test]
    async fn default_stream_falls_back_to_generate() {
        let llm = EchoLlm::new("echoing");
        let ctx = DialogContext::new("ping");
        let mut s = llm.stream(&ctx).await.expect("stream ok");
        let chunk = s.next().await.expect("at least one chunk").unwrap();
        assert!(chunk.finished);
        assert!(chunk.delta.contains("ping"));
    }

    #[tokio::test]
    async fn generate_returns_provider_metadata() {
        let llm = EchoLlm::new("test");
        let ctx = DialogContext::new("hi");
        let resp = llm.generate(&ctx).await.unwrap();
        assert_eq!(resp.provider, "mock-echo");
        assert!(!resp.model.is_empty());
    }
}
