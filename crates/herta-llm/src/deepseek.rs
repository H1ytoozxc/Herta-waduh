//! `DeepSeek` LLM provider (OpenAI-compatible chat/completions).

#![allow(clippy::struct_field_names)]

use crate::common::{
    build_http_client, map_status_error, retry_after_hint, retry_from_attempts,
    to_transport_error,
};
use async_trait::async_trait;
use herta_config::schema::DeepSeekConfig;
use herta_core::{
    DialogContext, HertaError, HertaResult, Role,
    llm::{LlmProvider, LlmResponse, TokenUsage},
    retry::with_retry,
};
use reqwest::{Client, header::{AUTHORIZATION, HeaderMap, HeaderValue}};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use url::Url;

const PROVIDER: &str = "deepseek";

/// `DeepSeek` provider.
pub struct DeepSeekLlm {
    cfg: DeepSeekConfig,
    url: Url,
    http: Client,
}

impl std::fmt::Debug for DeepSeekLlm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeepSeekLlm")
            .field("base_url", &self.cfg.base_url)
            .field("model", &self.cfg.model)
            .finish_non_exhaustive()
    }
}

impl DeepSeekLlm {
    /// Build a new `DeepSeek` provider.
    pub fn new(cfg: DeepSeekConfig) -> HertaResult<Self> {
        let base = Url::parse(&cfg.base_url)
            .map_err(|e| HertaError::config(format!("bad DEEPSEEK_BASE_URL: {e}")))?;
        let url = base
            .join("v1/chat/completions")
            .map_err(|e| HertaError::config(format!("join: {e}")))?;
        let timeout = Duration::from_secs_f64(cfg.timeout_seconds.max(1.0));
        let http = build_http_client(timeout)?;
        Ok(Self { cfg, url, http })
    }

    fn auth_headers(&self) -> HertaResult<HeaderMap> {
        let mut headers = HeaderMap::new();
        let key = self
            .cfg
            .api_key
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| HertaError::Auth("DEEPSEEK_API_KEY is missing".into()))?;
        let value = format!("Bearer {key}");
        let mut v = HeaderValue::from_str(&value)
            .map_err(|e| HertaError::config(format!("invalid auth header: {e}")))?;
        v.set_sensitive(true);
        headers.insert(AUTHORIZATION, v);
        Ok(headers)
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
    temperature: f32,
    max_tokens: u32,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[serde(default)]
    model: Option<String>,
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: OutgoingMessage,
}

#[derive(Debug, Deserialize)]
struct OutgoingMessage {
    #[serde(default)]
    content: String,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
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
impl LlmProvider for DeepSeekLlm {
    fn name(&self) -> &'static str {
        PROVIDER
    }

    #[tracing::instrument(level = "info", skip(self), fields(provider = PROVIDER, model = %self.cfg.model))]
    async fn warm_up(&self) -> HertaResult<bool> {
        // DeepSeek has no cheap probe; verify credentials are set.
        if self.cfg.api_key.as_deref().unwrap_or_default().is_empty() {
            return Ok(false);
        }
        Ok(true)
    }

    #[tracing::instrument(level = "info", skip(self, ctx), fields(
        provider = PROVIDER,
        model = %self.cfg.model,
        correlation_id = %ctx.correlation_id
    ))]
    async fn generate(&self, ctx: &DialogContext) -> HertaResult<LlmResponse> {
        let headers = self.auth_headers()?;
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
            temperature: self.cfg.temperature,
            max_tokens: self.cfg.max_tokens,
            stream: false,
        };

        let http = &self.http;
        let url = self.url.clone();
        let started = Instant::now();
        let policy = retry_from_attempts(self.cfg.retry_attempts);

        let response: ChatResponse = with_retry(policy, None, || {
            let http = http.clone();
            let url = url.clone();
            let headers = headers.clone();
            let body_ref = &body;
            async move {
                let resp = http
                    .post(url)
                    .headers(headers)
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
                resp.json::<ChatResponse>()
                    .await
                    .map_err(|e| HertaError::provider(PROVIDER, "decode", e))
            }
        })
        .await?;

        let text = response
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .unwrap_or_default()
            .trim()
            .to_string();

        if text.is_empty() {
            return Err(HertaError::provider(PROVIDER, "generate", "empty response"));
        }

        let usage = response.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

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
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg(base: &str) -> DeepSeekConfig {
        DeepSeekConfig {
            api_key: Some("test-key".into()),
            base_url: base.into(),
            model: "deepseek-chat".into(),
            timeout_seconds: 5.0,
            temperature: 0.5,
            max_tokens: 64,
            retry_attempts: 0,
            rate_limit_retries: 0,
        }
    }

    #[tokio::test]
    async fn auth_header_is_set() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "deepseek-chat",
                "choices": [{ "message": { "role": "assistant", "content": "ok" } }],
                "usage": { "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2 }
            })))
            .mount(&server)
            .await;

        let llm = DeepSeekLlm::new(cfg(&server.uri())).unwrap();
        let ctx = DialogContext::new("hi");
        let r = llm.generate(&ctx).await.unwrap();
        assert_eq!(r.text, "ok");
    }

    #[tokio::test]
    async fn missing_key_returns_auth_error() {
        let mut c = cfg("http://example.invalid");
        c.api_key = None;
        let llm = DeepSeekLlm::new(c).unwrap();
        let ctx = DialogContext::new("hi");
        let err = llm.generate(&ctx).await.unwrap_err();
        assert!(matches!(err, HertaError::Auth(_)));
    }

    #[tokio::test]
    async fn warm_up_false_without_key() {
        let mut c = cfg("http://example.invalid");
        c.api_key = None;
        let llm = DeepSeekLlm::new(c).unwrap();
        assert!(!llm.warm_up().await.unwrap());
    }
}
