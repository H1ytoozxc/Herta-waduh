//! # herta-observability
//!
//! Structured logging, metrics, and health endpoints for The Herta.
//!
//! - [`init_tracing`] configures a `tracing` subscriber with either a pretty,
//!   JSON, or compact format depending on the active [`TelemetryConfig`].
//! - [`metrics`] exposes a Prometheus text exporter guarded by feature `prometheus`.
//! - [`server`] provides an `axum`-based `/healthz`, `/ready`, and `/metrics`
//!   HTTP surface that can be spawned alongside the main pipeline.
//!
//! Integration is optional: the CLI can run without any of these, they are
//! always best-effort, and failures here never take down the main pipeline.

#![warn(missing_docs)]

pub mod metrics;
pub mod server;
pub mod tracing_init;

pub use server::{HealthProbe, ObservabilityServer};
pub use tracing_init::{init_tracing, TracingGuard};
