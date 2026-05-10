//! Tracing / logging subscriber initialization.

use herta_config::schema::{LogFormat, TelemetryConfig};
use std::io;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, Registry, fmt, prelude::*};

/// RAII guard returned by [`init_tracing`] that keeps the log worker alive.
///
/// Drop this at process shutdown (e.g. at the end of `main`) so buffered logs
/// are flushed.
pub struct TracingGuard {
    _worker: Option<WorkerGuard>,
}

/// Initialize a tracing subscriber based on the telemetry configuration.
///
/// Call once at process start. Subsequent calls log a warning and are
/// treated as a no-op; this allows idempotent initialization from tests.
pub fn init_tracing(cfg: &TelemetryConfig, default_level: &str) -> TracingGuard {
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(default_level))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let (non_blocking, worker) = tracing_appender::non_blocking(io::stderr());

    let subscriber = Registry::default().with(filter);
    let base = match cfg.log_format {
        LogFormat::Pretty => {
            let layer = fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(true)
                .with_target(true);
            subscriber.with(layer).try_init()
        }
        LogFormat::Compact => {
            let layer = fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false)
                .with_target(false)
                .compact();
            subscriber.with(layer).try_init()
        }
        LogFormat::Json => {
            let layer = fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false)
                .with_target(true)
                .json();
            subscriber.with(layer).try_init()
        }
    };

    if let Err(e) = base {
        eprintln!("tracing already initialized: {e}");
        return TracingGuard { _worker: None };
    }

    TracingGuard {
        _worker: Some(worker),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_twice_does_not_panic() {
        let cfg = TelemetryConfig::default();
        let _g1 = init_tracing(&cfg, "warn");
        let _g2 = init_tracing(&cfg, "warn");
    }
}
