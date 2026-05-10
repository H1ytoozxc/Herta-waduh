//! Observability HTTP surface: `/healthz`, `/ready`, `/metrics`.
//!
//! The server runs on a configurable socket address, outside the main Tokio
//! task so it can keep answering probes while the voice pipeline is busy.

use async_trait::async_trait;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
};
use herta_config::schema::ServerConfig;
use herta_core::{HealthReport, HealthState, HealthStatus};
use serde::Serialize;
use std::{net::SocketAddr, sync::Arc};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

/// Trait implemented by any component that can contribute a health status.
#[async_trait]
pub trait HealthProbe: Send + Sync + 'static {
    /// Run the health check; must not block for more than a few seconds.
    async fn probe(&self) -> HealthStatus;
}

/// The embedded HTTP server providing health and metrics endpoints.
pub struct ObservabilityServer {
    config: ServerConfig,
    probes: Vec<Arc<dyn HealthProbe>>,
}

impl ObservabilityServer {
    /// Build a new observability server (does not bind yet).
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config,
            probes: Vec::new(),
        }
    }

    /// Register a health probe.
    #[must_use]
    pub fn with_probe(mut self, probe: Arc<dyn HealthProbe>) -> Self {
        self.probes.push(probe);
        self
    }

    /// Run the server until `cancel` is signalled. Returns when the listener
    /// stops accepting new connections.
    pub async fn run(self, cancel: CancellationToken) -> std::io::Result<()> {
        let addr: SocketAddr = self
            .config
            .bind
            .parse()
            .unwrap_or_else(|_| "0.0.0.0:9090".parse().expect("valid default"));
        let state = Arc::new(AppState {
            probes: self.probes,
            health_path: self.config.health_path.clone(),
            metrics_path: self.config.metrics_path.clone(),
            readiness_path: self.config.readiness_path.clone(),
        });

        let app = build_router(state.clone());
        let listener = TcpListener::bind(addr).await?;
        tracing::info!(%addr, "observability server listening");

        axum::serve(listener, app.into_make_service())
            .with_graceful_shutdown(async move { cancel.cancelled().await })
            .await
    }
}

struct AppState {
    probes: Vec<Arc<dyn HealthProbe>>,
    health_path: String,
    metrics_path: String,
    readiness_path: String,
}

fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route(&state.health_path.clone(), get(health))
        .route(&state.readiness_path.clone(), get(ready))
        .route(&state.metrics_path.clone(), get(metrics))
        .route("/", get(root))
        .with_state(state)
}

#[derive(Serialize)]
struct Root<'a> {
    service: &'a str,
    status: &'a str,
}

async fn root() -> impl IntoResponse {
    Json(Root {
        service: "the-herta",
        status: "running",
    })
}

async fn health(State(state): State<Arc<AppState>>) -> Response {
    let statuses = gather(&state.probes).await;
    let report = HealthReport::from_components(statuses);
    let code = match report.state {
        HealthState::Healthy | HealthState::Degraded => StatusCode::OK,
        HealthState::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
    };
    (code, Json(report)).into_response()
}

async fn ready(State(state): State<Arc<AppState>>) -> Response {
    let statuses = gather(&state.probes).await;
    let report = HealthReport::from_components(statuses);
    let code = match report.state {
        HealthState::Healthy => StatusCode::OK,
        HealthState::Degraded | HealthState::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
    };
    (code, Json(report)).into_response()
}

async fn metrics(State(_state): State<Arc<AppState>>) -> Response {
    let body = crate::metrics::render();
    ([("content-type", "text/plain; version=0.0.4")], body).into_response()
}

async fn gather(probes: &[Arc<dyn HealthProbe>]) -> Vec<HealthStatus> {
    let mut out = Vec::with_capacity(probes.len());
    for p in probes {
        out.push(p.probe().await);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct AlwaysHealthy;
    #[async_trait]
    impl HealthProbe for AlwaysHealthy {
        async fn probe(&self) -> HealthStatus {
            HealthStatus {
                component: "test".into(),
                state: HealthState::Healthy,
                detail: None,
            }
        }
    }

    struct AlwaysDown;
    #[async_trait]
    impl HealthProbe for AlwaysDown {
        async fn probe(&self) -> HealthStatus {
            HealthStatus {
                component: "bad".into(),
                state: HealthState::Unhealthy,
                detail: Some("down".into()),
            }
        }
    }

    #[tokio::test]
    async fn gather_collects_statuses() {
        let probes: Vec<Arc<dyn HealthProbe>> =
            vec![Arc::new(AlwaysHealthy), Arc::new(AlwaysDown)];
        let statuses = gather(&probes).await;
        assert_eq!(statuses.len(), 2);
    }

    #[tokio::test]
    async fn server_stops_on_cancel() {
        let cfg = ServerConfig {
            bind: "127.0.0.1:0".into(),
            ..ServerConfig::default()
        };
        let server = ObservabilityServer::new(cfg);
        let token = CancellationToken::new();
        let token2 = token.clone();
        let handle = tokio::spawn(async move { server.run(token2).await });
        token.cancel();
        let _ = handle.await.unwrap();
    }
}
