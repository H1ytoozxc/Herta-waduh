//! Google AI Studio (Gemini / Gemma) LLM provider.
//!
//! Uses the `v1beta/models/{model}:generateContent` endpoint with an API key.

#![allow(clippy::struct_field_names)]

use crate::common::{
    build_http_client, map_status_error, retry_after_hint, retry_from_attempts,
    to_transport_error,
};
use async_trait::async_trait;
use herta_config::schema::GoogleAiConfig;
use herta_core::{
    DialogContext, HertaError, HertaResult, Role,
    llm::{LlmProvider, LlmResponse, TokenUsage},
    retry::with_retry,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use url::Url;

const PROVIDER: &str = "google_ai";

/// Google AI Studio LLM provider.
pub struct GoogleAiLlm {
    cfg: GoogleAiConfig,
    base: Url,
    http: Client,
}

impl std::fmt::Debug for GoogleAiLlm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoogleAiLlm")
            .field("base_url", &self.cfg.base_url)
            .field("model", &self.cfg.model)
            .finish_non_exhaustive()
    }
}

impl GoogleAiLlm {
    /// Build a new Google AI provider.
    pub fn new(cfg: GoogleAiConfig) -> HertaResult<Self> {
        let base = Url::parse(&cfg.base_url)
            .map_err(|e| HertaError::config(format!("bad GOOGLE_AI_BASE_URL: {e}")))?;
        let timeout = Duration::from_secs_f64(cfg.timeout_seconds.max(1.0));
        let http = build_http_client(timeout)?;
        Ok(Self { cfg, base, http })
    }

    fn endpoint(&self, model: &str) -> HertaResult<Url> {
        let mut path = self.base.clone();
        {
            let mut segments = path
                .path_segments_mut()
                .map_err(|()| HertaError::config("base URL cannot be a base"))?;
            segments.push("models");
            segments.push(&format!("{model}:generateContent"));
        }
        Ok(path)
    }

    fn api_key(&self) -> HertaResult<&str> {
        self.cfg
            .api_key
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| HertaError::Auth("GOOGLE_AI_API_KEY is missing".into()))
    }
}

#[derive(Debug, Serialize)]
struct GenerateBody<'a> {
    contents: Vec<Content<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<Content<'a>>,
    generation_config: GenerationConfig,
}

#[derive(Debug, Serialize)]
struct Content<'a> {
    role: &'a str,
    parts: Vec<Part<'a>>,
}

#[derive(Debug, Serialize)]
struct Part<'a> {
    text: &'a str,
}

#[derive(Debug, Serialize)]
struct GenerationConfig {
    temperature: f32,
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct GenerateResponse {
    #[serde(default)]
    candidates: Vec<Candidate>,
    #[serde(default, rename = "usageMetadata")]
    usage: Option<UsageMetadata>,
    #[serde(default, rename = "modelVersion")]
    model_version: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    #[serde(default)]
    content: Option<CandidateContent>,
}

#[derive(Debug, Deserialize)]
struct CandidateContent {
    #[serde(default)]
    parts: Vec<CandidatePart>,
}

#[derive(Debug, Deserialize)]
struct CandidatePart {
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
struct UsageMetadata {
    #[serde(default, rename = "promptTokenCount")]
    prompt_token_count: u64,
    #[serde(default, rename = "candidatesTokenCount")]
    candidates_token_count: u64,
    #[serde(default, rename = "totalTokenCount")]
    total_token_count: u64,
}

fn role_to_str(r: Role) -> &'static str {
    match r {
        Role::User | Role::System | Role::Tool => "user",
        Role::Assistant => "model",
    }
}

fn build_contents(ctx: &DialogContext) -> (Vec<Content<'_>>, Option<Content<'_>>) {
    // The Google Generate API distinguishes `system_instruction` from turns.
    let mut system_parts: Vec<&str> = ctx
        .locked_prefix
        .iter()
        .filter(|m| m.role == Role::System)
        .map(|m| m.content.as_str())
        .collect();
    system_parts.extend(
        ctx.history
            .iter()
            .filter(|m| m.role == Role::System)
            .map(|m| m.content.as_str()),
    );

    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(Content {
            role: "user",
            parts: system_parts.into_iter().map(|t| Part { text: t }).collect(),
        })
    };

    let mut contents = Vec::new();
    for m in ctx
        .locked_prefix
        .iter()
        .chain(ctx.history.iter())
        .filter(|m| m.role != Role::System)
    {
        contents.push(Content {
            role: role_to_str(m.role),
            parts: vec![Part {
                text: m.content.as_str(),
            }],
        });
    }
    contents.push(Content {
        role: "user",
        parts: vec![Part {
            text: ctx.user_utterance.as_str(),
        }],
    });

    (contents, system_instruction)
}

#[async_trait]
impl LlmProvider for GoogleAiLlm {
    fn name(&self) -> &'static str {
        PROVIDER
    }

    async fn warm_up(&self) -> HertaResult<bool> {
        Ok(self.cfg.api_key.as_deref().is_some_and(|s| !s.is_empty()))
    }

    #[tracing::instrument(level = "info", skip(self, ctx), fields(
        provider = PROVIDER,
        model = %self.cfg.model,
        correlation_id = %ctx.correlation_id
    ))]
    async fn generate(&self, ctx: &DialogContext) -> HertaResult<LlmResponse> {
        let api_key = self.api_key()?.to_string();
        let url = self.endpoint(&self.cfg.model)?;

        let (contents, system_instruction_owned) = build_contents(ctx);
        let system_instruction = if self.cfg.system_instruction_enabled {
            system_instruction_owned
        } else {
            None
        };
        let body = GenerateBody {
            contents,
            system_instruction,
            generation_config: GenerationConfig {
                temperature: self.cfg.temperature,
                max_output_tokens: self.cfg.max_tokens,
            },
        };

        let http = &self.http;
        let started = Instant::now();
        let policy = retry_from_attempts(self.cfg.retry_attempts);

        let response: GenerateResponse = with_retry(policy, None, || {
            let http = http.clone();
            let url = url.clone();
            let api_key = api_key.clone();
            let body_ref = &body;
            async move {
                let resp = http
                    .post(url)
                    .query(&[("key", api_key.as_str())])
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
                resp.json::<GenerateResponse>()
                    .await
                    .map_err(|e| HertaError::provider(PROVIDER, "decode", e))
            }
        })
        .await?;

        let text = response
            .candidates
            .into_iter()
            .next()
            .and_then(|c| c.content)
            .map(|c| {
                c.parts
                    .into_iter()
                    .map(|p| p.text)
                    .collect::<String>()
            })
            .unwrap_or_default()
            .trim()
            .to_string();

        if text.is_empty() {
            return Err(HertaError::provider(PROVIDER, "generate", "empty response"));
        }

        let usage = response.usage.map(|u| TokenUsage {
            prompt_tokens: u.prompt_token_count,
            completion_tokens: u.candidates_token_count,
            total_tokens: u.total_token_count,
        });

        Ok(LlmResponse {
            text,
            provider: PROVIDER.into(),
            model: response.model_version.unwrap_or_else(|| self.cfg.model.clone()),
            latency: started.elapsed(),
            usage,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path_regex, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg(base: &str) -> GoogleAiConfig {
        GoogleAiConfig {
            api_key: Some("key-123".into()),
            base_url: base.into(),
            model: "gemma-3-27b-it".into(),
            fallback_model: None,
            timeout_seconds: 5.0,
            temperature: 0.5,
            max_tokens: 64,
            retry_attempts: 0,
            rate_limit_retries: 0,
            system_instruction_enabled: false,
        }
    }

    #[tokio::test]
    async fn builds_contents_with_system_locked_prefix() {
        let mut ctx = DialogContext::new("hello");
        ctx.locked_prefix = vec![herta_core::Message::system("persona")];
        let (contents, sys) = build_contents(&ctx);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, "user");
        assert!(sys.is_some());
    }

    #[tokio::test]
    async fn generate_decodes_response() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"/models/.+:generateContent"))
            .and(query_param("key", "key-123"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": [{
                    "content": { "parts": [{ "text": "привет" }] }
                }],
                "usageMetadata": {
                    "promptTokenCount": 2,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 3
                },
                "modelVersion": "gemma-3-27b-it"
            })))
            .mount(&server)
            .await;

        let llm = GoogleAiLlm::new(cfg(&server.uri())).unwrap();
        let ctx = DialogContext::new("hi");
        let r = llm.generate(&ctx).await.unwrap();
        assert_eq!(r.text, "привет");
        assert_eq!(r.usage.unwrap().total_tokens, 3);
    }

    #[tokio::test]
    async fn missing_key_errors() {
        let mut c = cfg("http://example.invalid");
        c.api_key = None;
        let llm = GoogleAiLlm::new(c).unwrap();
        let ctx = DialogContext::new("hi");
        assert!(matches!(
            llm.generate(&ctx).await.unwrap_err(),
            HertaError::Auth(_)
        ));
    }
}
