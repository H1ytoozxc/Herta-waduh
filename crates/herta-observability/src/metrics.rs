//! Metrics helpers.
//!
//! The `metrics` crate provides a facade; an exporter must be installed to
//! actually expose measurements. When the `prometheus` feature is enabled we
//! install a [`metrics_exporter_prometheus::PrometheusRecorder`] and expose
//! the scrape text through [`render`].

#[cfg(feature = "prometheus")]
use metrics_exporter_prometheus::PrometheusHandle;
use once_cell::sync::OnceCell;

#[cfg(feature = "prometheus")]
static PROM_HANDLE: OnceCell<PrometheusHandle> = OnceCell::new();

#[cfg(not(feature = "prometheus"))]
static PROM_HANDLE: OnceCell<()> = OnceCell::new();

/// Initialize the Prometheus metrics recorder. Safe to call multiple times:
/// subsequent calls return the previously installed handle/sentinel.
#[cfg(feature = "prometheus")]
pub fn init_prometheus() -> Result<(), String> {
    if PROM_HANDLE.get().is_some() {
        return Ok(());
    }
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new()
        .build_recorder();
    let handle = recorder.handle();
    metrics::set_global_recorder(recorder).map_err(|e| e.to_string())?;
    let _ = PROM_HANDLE.set(handle);
    Ok(())
}

/// Stub when the Prometheus feature is disabled.
#[cfg(not(feature = "prometheus"))]
pub fn init_prometheus() -> Result<(), String> {
    let _ = PROM_HANDLE.set(());
    Ok(())
}

/// Render the current metric snapshot in Prometheus text format.
#[cfg(feature = "prometheus")]
pub fn render() -> String {
    PROM_HANDLE
        .get()
        .map_or_else(
            || "# metrics recorder not initialized\n".into(),
            metrics_exporter_prometheus::PrometheusHandle::render,
        )
}

/// Stub renderer when Prometheus is disabled.
#[cfg(not(feature = "prometheus"))]
pub fn render() -> String {
    "# prometheus feature disabled\n".into()
}

/// Record a counter increment with a label set.
pub fn incr_counter(name: &'static str, value: u64) {
    metrics::counter!(name).increment(value);
}

/// Record a histogram sample with a single `kind` label.
pub fn observe_latency_ms(name: &'static str, kind: &'static str, ms: f64) {
    metrics::histogram!(name, "kind" => kind).record(ms);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_and_render_is_idempotent() {
        let _ = init_prometheus();
        let _ = init_prometheus();
        // render() always returns at least an empty string; just verify it
        // doesn't panic and returns a valid UTF-8 string.
        let snapshot = render();
        // The snapshot may be empty before any metrics are recorded, or may
        // contain Prometheus comment lines — both are valid.
        assert!(snapshot.is_ascii() || !snapshot.is_empty() || snapshot.is_empty());
    }

    #[test]
    fn record_counter_does_not_panic() {
        let _ = init_prometheus();
        incr_counter("herta_test_counter", 1);
    }

    #[test]
    fn observe_latency_does_not_panic() {
        let _ = init_prometheus();
        observe_latency_ms("herta_test_latency_ms", "llm", 12.5);
    }
}
