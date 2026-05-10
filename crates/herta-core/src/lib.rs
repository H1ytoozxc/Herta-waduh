//! # herta-core
//!
//! Foundational abstractions for The Herta Voice Assistant.
//!
//! This crate defines the stable, provider-agnostic contracts that every other
//! crate in the workspace depends on:
//!
//! - [`llm::LlmProvider`] — chat-style LLM interface with optional streaming.
//! - [`stt::SttEngine`] — speech-to-text interface.
//! - [`tts::TtsEngine`] — text-to-speech interface.
//! - [`audio::AudioInput`] / [`audio::AudioOutput`] — audio I/O abstractions.
//! - [`memory::Memory`] — pluggable dialogue memory backend.
//! - [`pipeline::VoicePipeline`] — event-driven voice pipeline orchestrator.
//!
//! All traits are `async` (via [`async_trait`]) and return a unified
//! [`HertaResult`] using the [`HertaError`] error taxonomy.
//!
//! ## Design principles
//!
//! - **Separation of contracts from implementations** — this crate declares
//!   interfaces only; concrete providers live in `herta-llm`, `herta-stt`,
//!   `herta-tts`, `herta-audio`, and `herta-memory`.
//! - **Cancellation safety** — long-running operations accept a
//!   [`CancellationToken`] so they can be interrupted cleanly.
//! - **Observability by construction** — every public entry point carries a
//!   [`tracing`] span and returns structured errors.
//! - **Deterministic, testable** — mock implementations are provided under
//!   [`mocks`] for integration testing without hardware or network.

#![warn(missing_docs)]

pub mod audio;
pub mod context;
pub mod error;
pub mod health;
pub mod llm;
pub mod memory;
pub mod mocks;
pub mod pipeline;
pub mod retry;
pub mod stt;
pub mod tts;

pub use context::{DialogContext, Message, Role};
pub use error::{HertaError, HertaResult};
pub use health::{HealthReport, HealthState, HealthStatus};

/// Re-exports commonly useful types from external crates so downstream code
/// can avoid pulling them in directly.
pub mod prelude {
    pub use crate::{
        DialogContext, HertaError, HertaResult, Message, Role,
        audio::{AudioChunk, AudioInput, AudioOutput, Utterance},
        llm::{LlmProvider, LlmResponse},
        memory::Memory,
        pipeline::{PipelineEvent, VoicePipeline},
        stt::SttEngine,
        tts::{SynthesizedAudio, TtsEngine},
    };
    pub use async_trait::async_trait;
    pub use tokio_util::sync::CancellationToken;
}
