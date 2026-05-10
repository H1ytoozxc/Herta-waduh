//! Google AI Studio (Gemini) STT provider.
//!
//! Wraps the utterance PCM in a WAV container, base64-encodes it, and sends
//! the blob as `inlineData` to `generateContent`. Returns the extracted text.

use crate::wav;
use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD as B64};
use herta_config::schema::GoogleSttConfig;
use herta_core::{
    HertaError, HertaResult,
    audio::Utterance,
    retry::{RetryPolicy, with_retry},
    stt::{SttEngine, Transcript},
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use url::Url;

const PROVIDER: &str = "google_ai_stt";

/// Google AI STT provider.
pub struct GoogleAiStt {
    cfg: GoogleSttConfig,
    base: Url,
    http: Client,
    input_sample_rate: u32,
}

impl std::fmt::Debug for GoogleAiStt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoogleAiStt")
            .field("model", &self.cfg.model)
            .field("base_url", &self.cfg.base_url)
            .finish_non_exhaustive()
    }
}

impl GoogleAiStt {
    /// Build a new provider.
    pub fn new(cfg: GoogleSttConfig, input_sample_rate: u32) -> HertaResult<Self> {
        let base = Url::parse(&cfg.base_url)
            .map_err(|e| HertaError::config(format!("bad GOOGLE_STT_BASE_URL: {e}")))?;
        let timeout = Duration::from_secs_f64(cfg.timeout_seconds.max(1.0));
        let http = reqwest::ClientBuilder::new()
            .user_agent(concat!("the-herta/", env!("CARGO_PKG_VERSION")))
            .timeout(timeout)
            .build()
            .map_err(|e| HertaError::transport(format!("build http: {e}")))?;
        Ok(Self {
            cfg,
            base,
            http,
            input_sample_rate,
        })
    }

    fn endpoint(&self) -> HertaResult<Url> {
        let mut path = self.base.clone();
        {
            let mut segments = path
                .path_segments_mut()
                .map_err(|()| HertaError::config("base URL cannot be a base"))?;
            segments.push("models");
            segments.push(&format!("{}:generateContent", self.cfg.model));
        }
        Ok(path)
    }

    fn api_key(&self) -> HertaResult<&str> {
        self.cfg
            .api_key
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| HertaError::Auth("GOOGLE_STT_API_KEY / GEMINI_API_KEY missing".into()))
    }
}

#[derive(Debug, Serialize)]
struct Body<'a> {
    contents: Vec<Content<'a>>,
    generation_config: GenerationConfig,
}

#[derive(Debug, Serialize)]
struct Content<'a> {
    role: &'a str,
    parts: Vec<Part<'a>>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum Part<'a> {
    Text {
        text: &'a str,
    },
    Inline {
        #[serde(rename = "inlineData")]
        inline_data: InlineData,
    },
}

#[derive(Debug, Serialize)]
struct InlineData {
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct GenerationConfig {
    temperature: f32,
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct Response {
    #[serde(default)]
    candidates: Vec<Candidate>,
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

fn build_prompt(language_hint: Option<&str>) -> String {
    let lang = language_hint.unwrap_or("auto-detect");
    format!(
        "Transcribe the attached audio verbatim into plain UTF-8 text. \
         Return only the transcription, without commentary or timestamps. \
         Language hint: {lang}."
    )
}

#[async_trait]
impl SttEngine for GoogleAiStt {
    fn name(&self) -> &'static str {
        PROVIDER
    }

    fn active_device(&self) -> String {
        format!("google_ai:{}", self.cfg.model)
    }

    async fn warm_up(&self) -> HertaResult<bool> {
        Ok(self.cfg.api_key.as_deref().is_some_and(|s| !s.is_empty()))
    }

    #[tracing::instrument(level = "info", skip(self, utterance), fields(
        provider = PROVIDER,
        model = %self.cfg.model,
        ms = utterance.duration_ms
    ))]
    async fn transcribe(&self, utterance: &Utterance) -> HertaResult<Transcript> {
        if utterance.pcm.is_empty() {
            return Err(HertaError::invalid("empty utterance"));
        }
        let api_key = self.api_key()?.to_string();
        let url = self.endpoint()?;
        let wav_bytes = wav::encode_wav(utterance)?;
        let data = B64.encode(&wav_bytes);

        let prompt = build_prompt(self.cfg.language_hint.as_deref());
        let _ = self.input_sample_rate; // reserved for future resampling checks.

        let body = Body {
            contents: vec![Content {
                role: "user",
                parts: vec![
                    Part::Text { text: &prompt },
                    Part::Inline {
                        inline_data: InlineData {
                            mime_type: "audio/wav".into(),
                            data,
                        },
                    },
                ],
            }],
            generation_config: GenerationConfig {
                temperature: 0.0,
                max_output_tokens: 512,
            },
        };

        let http = &self.http;
        let started = Instant::now();
        let policy = RetryPolicy {
            max_attempts: self.cfg.retry_attempts.saturating_add(1).max(1),
            ..RetryPolicy::default()
        };

        let response: Response = with_retry(policy, None, || {
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
                    .map_err(|e| HertaError::transport(e.to_string()))?;
                let status = resp.status();
                if !status.is_success() {
                    let body = resp.text().await.unwrap_or_default();
                    return Err(match status.as_u16() {
                        401 | 403 => HertaError::Auth(body),
                        429 => HertaError::RateLimited { retry_after: None },
                        500..=599 => HertaError::Unavailable(body),
                        _ => HertaError::provider(PROVIDER, "transcribe", body),
                    });
                }
                resp.json::<Response>()
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

        Ok(Transcript {
            text,
            language: self.cfg.language_hint.clone(),
            confidence: None,
            latency: started.elapsed(),
            provider: PROVIDER.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use herta_core::audio::{AudioFormat, SampleFormat};
    use wiremock::matchers::{method, path_regex, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg(base: &str) -> GoogleSttConfig {
        GoogleSttConfig {
            api_key: Some("stt-key".into()),
            base_url: base.into(),
            model: "gemini-2.5-flash".into(),
            timeout_seconds: 5.0,
            retry_attempts: 0,
            rate_limit_retries: 0,
            language_hint: Some("ru".into()),
            fallback_to_whisper: false,
        }
    }

    fn one_sec_f32_silence() -> Utterance {
        Utterance {
            pcm: Bytes::from(vec![0u8; 16_000 * 4]),
            format: AudioFormat {
                sample_rate: 16_000,
                channels: 1,
                sample_format: SampleFormat::F32,
            },
            duration_ms: 1_000,
        }
    }

    #[tokio::test]
    async fn transcribe_extracts_text() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path_regex(r"/models/.+:generateContent"))
            .and(query_param("key", "stt-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": [{
                    "content": { "parts": [{ "text": "привет мир" }] }
                }]
            })))
            .mount(&server)
            .await;

        let engine = GoogleAiStt::new(cfg(&server.uri()), 16_000).unwrap();
        let t = engine.transcribe(&one_sec_f32_silence()).await.unwrap();
        assert_eq!(t.text, "привет мир");
        assert_eq!(t.provider, PROVIDER);
    }

    #[tokio::test]
    async fn empty_utterance_rejected() {
        let engine = GoogleAiStt::new(cfg("http://example.invalid"), 16_000).unwrap();
        let u = Utterance {
            pcm: Bytes::new(),
            format: AudioFormat::default(),
            duration_ms: 0,
        };
        assert!(engine.transcribe(&u).await.is_err());
    }

    #[tokio::test]
    async fn missing_key_errors() {
        let mut c = cfg("http://example.invalid");
        c.api_key = None;
        let engine = GoogleAiStt::new(c, 16_000).unwrap();
        let err = engine
            .transcribe(&one_sec_f32_silence())
            .await
            .unwrap_err();
        assert!(matches!(err, HertaError::Auth(_)));
    }
}
