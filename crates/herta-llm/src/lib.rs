//! # herta-llm
//!
//! LLM providers for The Herta Voice Assistant.
//!
//! Each provider implements [`herta_core::llm::LlmProvider`] and is gated
//! behind a Cargo feature so builds can opt in only to what they need.
//!
//! Providers:
//!
//! - [`ollama::OllamaLlm`] — local HTTP, default for dev.
//! - [`deepseek::DeepSeekLlm`] — OpenAI-compatible cloud API.
//! - [`google_ai::GoogleAiLlm`] — Gemini / Gemma via Google AI Studio.
//!
//! All providers share the [`common`] HTTP client configuration, retry policy
//! translation, and error mapping helpers.

#![warn(missing_docs)]

pub mod common;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "deepseek")]
pub mod deepseek;

#[cfg(feature = "google-ai")]
pub mod google_ai;

use herta_config::schema::{Config, LlmProviderKind};
use herta_core::{HertaResult, llm::LlmProvider};
use std::sync::Arc;

/// Build the configured LLM provider.
///
/// Returns an `Arc<dyn LlmProvider>` so the caller can freely share it across
/// tasks. If the selected provider's Cargo feature is disabled, returns a
/// `HertaError::Config` describing the mismatch.
pub fn build_from_config(cfg: &Config) -> HertaResult<Arc<dyn LlmProvider>> {
    match cfg.llm_provider {
        #[cfg(feature = "ollama")]
        LlmProviderKind::Ollama => Ok(Arc::new(ollama::OllamaLlm::new(cfg.ollama.clone())?)),
        #[cfg(not(feature = "ollama"))]
        LlmProviderKind::Ollama => Err(HertaError::config(
            "'ollama' provider requires the 'ollama' cargo feature",
        )),

        #[cfg(feature = "deepseek")]
        LlmProviderKind::DeepSeek => Ok(Arc::new(deepseek::DeepSeekLlm::new(cfg.deepseek.clone())?)),
        #[cfg(not(feature = "deepseek"))]
        LlmProviderKind::DeepSeek => Err(HertaError::config(
            "'deepseek' provider requires the 'deepseek' cargo feature",
        )),

        #[cfg(feature = "google-ai")]
        LlmProviderKind::GoogleAi => {
            Ok(Arc::new(google_ai::GoogleAiLlm::new(cfg.google_ai.clone())?))
        }
        #[cfg(not(feature = "google-ai"))]
        LlmProviderKind::GoogleAi => Err(HertaError::config(
            "'google_ai' provider requires the 'google-ai' cargo feature",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_selects_provider_from_config() {
        let cfg = Config::default();
        let provider = build_from_config(&cfg).unwrap();
        assert_eq!(provider.name(), "ollama");
    }
}
