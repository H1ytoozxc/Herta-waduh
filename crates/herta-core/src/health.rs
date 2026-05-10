//! Health / readiness reporting primitives.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use time::OffsetDateTime;

/// Top-level health state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthState {
    /// Service is fully functional.
    Healthy,
    /// Service is running but degraded (e.g. fallback provider in use).
    Degraded,
    /// Service is unable to serve requests.
    Unhealthy,
}

impl HealthState {
    /// Return a stable short identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Unhealthy => "unhealthy",
        }
    }

    /// Merge two states, taking the worse one.
    pub fn worst(a: Self, b: Self) -> Self {
        match (a, b) {
            (Self::Unhealthy, _) | (_, Self::Unhealthy) => Self::Unhealthy,
            (Self::Degraded, _) | (_, Self::Degraded) => Self::Degraded,
            _ => Self::Healthy,
        }
    }
}

/// Health status for a single component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Component name (e.g. `"llm.ollama"`, `"memory.json"`).
    pub component: String,
    /// State of this component.
    pub state: HealthState,
    /// Optional human-readable detail for operators.
    pub detail: Option<String>,
}

/// Aggregate report returned by the `/healthz` endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Overall state (worst of all components).
    pub state: HealthState,
    /// Report generation timestamp (RFC 3339).
    pub timestamp: String,
    /// Per-component statuses keyed by component name.
    pub components: BTreeMap<String, HealthStatus>,
}

impl HealthReport {
    /// Build a report from a list of component statuses.
    pub fn from_components(components: Vec<HealthStatus>) -> Self {
        let mut map = BTreeMap::new();
        let mut overall = HealthState::Healthy;
        for status in components {
            overall = HealthState::worst(overall, status.state);
            map.insert(status.component.clone(), status);
        }
        Self {
            state: overall,
            timestamp: OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap_or_default(),
            components: map,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worst_is_unhealthy_dominant() {
        assert_eq!(
            HealthState::worst(HealthState::Healthy, HealthState::Unhealthy),
            HealthState::Unhealthy
        );
        assert_eq!(
            HealthState::worst(HealthState::Degraded, HealthState::Unhealthy),
            HealthState::Unhealthy
        );
    }

    #[test]
    fn worst_is_degraded_over_healthy() {
        assert_eq!(
            HealthState::worst(HealthState::Healthy, HealthState::Degraded),
            HealthState::Degraded
        );
    }

    #[test]
    fn report_rolls_up_state() {
        let report = HealthReport::from_components(vec![
            HealthStatus {
                component: "llm".into(),
                state: HealthState::Healthy,
                detail: None,
            },
            HealthStatus {
                component: "memory".into(),
                state: HealthState::Degraded,
                detail: Some("using fallback".into()),
            },
        ]);
        assert_eq!(report.state, HealthState::Degraded);
        assert_eq!(report.components.len(), 2);
    }

    #[test]
    fn state_labels_are_stable() {
        assert_eq!(HealthState::Healthy.as_str(), "healthy");
        assert_eq!(HealthState::Degraded.as_str(), "degraded");
        assert_eq!(HealthState::Unhealthy.as_str(), "unhealthy");
    }
}
