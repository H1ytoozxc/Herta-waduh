//! Ollama LLM provider (local HTTP, `/api/chat` endpoint).

use crate::common::{
    build_http_client, map_status_error, retry_after_hint, to_transport_error,
};
use async_trait::async_trait;
use herta_config::schema::OllamaConfig;
use herta_core::{
    DialogContext, HertaError, HertaResult, Role,
    llm::{LlmProvider, LlmResponse, TokenUsage},
    retry::{RetryPolicy, with_retry},
};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use url::Url;

const PROVIDER: &str = "ollama";

/// Ollama LLM provider.
#[derive(Debug, Clone)]
pub struct OllamaLlm {
    cfg: OllamaConfig,
    chat_url: Url,
    tags_url: Url,
    http: Client,
    retry: RetryPolicy,
}

impl OllamaLlm {
    /// Build a new Ollama provider from configuration.
    pub fn new(cfg: OllamaConfig) -> HertaResult<Self> {
        let base =
            Url::parse(&cfg.host).map_err(|e| HertaError::config(format!("bad OLLAMA_HOST: {e}")))?;
        let chat_url = base
            .join("api/chat")
            .map_err(|e| HertaError::config(format!("join: {e}")))?;
        let tags_url = base
            .join("api/tags")
            .map_err(|e| HertaError::config(format!("join: {e}")))?;
        let timeout = Duration::from_secs_f64(cfg.timeout_seconds.max(1.0));
        let http = build_http_client(timeout)?;
        Ok(Self {
            cfg,
            chat_url,
            tags_url,
            http,
            retry: RetryPolicy::default(),
        })
    }
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    stream: bool,
    keep_alive: &'a str,
    options: ChatOptions,
    think: bool,
}

#[derive(Debug, Serialize)]
struct ChatOptions {
    temperature: f32,
    num_ctx: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_gpu: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[serde(default)]
    message: Option<OutgoingMessage>,
    #[serde(default)]
    done: Option<bool>,
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    eval_count: Option<u64>,
    #[serde(default)]
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OutgoingMessage {
    #[serde(default)]
    content: String,
}

fn role_to_str(r: Role) -> &'static str {
    match r {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

#[async_trait]
impl LlmProvider for OllamaLlm {
    fn name(&self) -> &'static str {
        PROVIDER
    }

    #[tracing::instrument(level = "info", skip(self), fields(provider = PROVIDER, model = %self.cfg.model))]
    async fn warm_up(&self) -> HertaResult<bool> {
        // /api/tags is cheap; if it responds 200 we assume the daemon is up.
        let resp = self.http.get(self.tags_url.clone()).send().await.map_err(|e| to_transport_error(&e))?;
        Ok(resp.status() == StatusCode::OK)
    }

    #[tracing::instrument(level = "info", skip(self, ctx), fields(
        provider = PROVIDER,
        model = %self.cfg.model,
        correlation_id = %ctx.correlation_id
    ))]
    async fn generate(&self, ctx: &DialogContext) -> HertaResult<LlmResponse> {
        let messages_owned = ctx.messages();
        let messages: Vec<ChatMessage<'_>> = messages_owned
            .iter()
            .map(|m| ChatMessage {
                role: role_to_str(m.role),
                content: m.content.as_str(),
            })
            .collect();

        let body = ChatRequest {
            model: &self.cfg.model,
            messages,
            stream: false,
            keep_alive: &self.cfg.keep_alive,
            options: ChatOptions {
                temperature: self.cfg.temperature,
                num_ctx: self.cfg.num_ctx,
                num_gpu: self.cfg.num_gpu,
            },
            think: self.cfg.think,
        };

        let http = &self.http;
        let url = self.chat_url.clone();
        let started = Instant::now();

        let response: ChatResponse = with_retry(self.retry, None, || {
            let http = http.clone();
            let url = url.clone();
            let body_ref = &body;
            async move {
                let resp = http
                    .post(url)
                    .json(body_ref)
                    .send()
                    .await
                    .map_err(|e| to_transport_error(&e))?;

                let status = resp.status();
                if !status.is_success() {
                    let retry_after = retry_after_hint(&resp);
                    let body = resp.text().await.unwrap_or_default();
                    return Err(map_status_error(
                        PROVIDER, "generate", status, &body, retry_after,
                    ));
                }
                let parsed = resp
                    .json::<ChatResponse>()
                    .await
                    .map_err(|e| HertaError::provider(PROVIDER, "decode", e))?;
                Ok(parsed)
            }
        })
        .await?;

        let text = response
            .message
            .map(|m| m.content)
            .unwrap_or_default()
            .trim()
            .to_string();

        if text.is_empty() && response.done != Some(true) {
            return Err(HertaError::provider(
                PROVIDER,
                "generate",
                "empty response",
            ));
        }

        let usage = match (response.prompt_eval_count, response.eval_count) {
            (Some(p), Some(c)) => Some(TokenUsage {
                prompt_tokens: p,
                completion_tokens: c,
                total_tokens: p + c,
            }),
            _ => None,
        };

        Ok(LlmResponse {
            text,
            provider: PROVIDER.into(),
            model: response.model.unwrap_or_else(|| self.cfg.model.clone()),
            latency: started.elapsed(),
            usage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg(host: &str) -> OllamaConfig {
        OllamaConfig {
            host: host.into(),
            model: "qwen3:4b".into(),
            timeout_seconds: 5.0,
            keep_alive: "10m".into(),
            think: false,
            temperature: 0.5,
            num_ctx: 1024,
            num_gpu: None,
        }
    }

    #[tokio::test]
    async fn warm_up_succeeds_on_200() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
            .mount(&server)
            .await;

        let llm = OllamaLlm::new(cfg(&server.uri())).unwrap();
        assert!(llm.warm_up().await.unwrap());
    }

    #[tokio::test]
    async fn generate_parses_chat_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "qwen3:4b",
                "message": { "role": "assistant", "content": "hello back" },
                "done": true,
                "prompt_eval_count": 3,
                "eval_count": 2
            })))
            .mount(&server)
            .await;

        let llm = OllamaLlm::new(cfg(&server.uri())).unwrap();
        let ctx = DialogContext::new("hi");
        let resp = llm.generate(&ctx).await.unwrap();
        assert_eq!(resp.text, "hello back");
        assert_eq!(resp.model, "qwen3:4b");
        assert_eq!(resp.usage.unwrap().total_tokens, 5);
    }

    #[tokio::test]
    async fn generate_maps_500_to_unavailable() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
            .mount(&server)
            .await;
        let mut c = cfg(&server.uri());
        c.timeout_seconds = 2.0;
        let mut llm = OllamaLlm::new(c).unwrap();
        llm.retry = RetryPolicy::no_retry();
        let ctx = DialogContext::new("hi");
        let err = llm.generate(&ctx).await.unwrap_err();
        assert!(matches!(err, HertaError::Unavailable(_)));
    }

    #[tokio::test]
    async fn generate_maps_401_to_auth() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(401).set_body_string("nope"))
            .mount(&server)
            .await;
        let mut llm = OllamaLlm::new(cfg(&server.uri())).unwrap();
        llm.retry = RetryPolicy::no_retry();
        let ctx = DialogContext::new("hi");
        let err = llm.generate(&ctx).await.unwrap_err();
        assert!(matches!(err, HertaError::Auth(_)));
    }
}
