//! # herta-memory
//!
//! Pluggable dialogue memory backends.
//!
//! Currently provided:
//!
//! - [`JsonMemory`] — atomic-write JSON file on disk (default, always available).
//! - [`SqliteMemory`] — `SQLite` via `rusqlite` (feature `sqlite-backend`).
//! - [`SledMemory`] — embedded KV store (feature `sled-backend`).
//!
//! All backends implement [`herta_core::memory::Memory`] and share the same
//! windowing semantics defined in [`herta_core::memory::window_messages`].

#![warn(missing_docs)]

pub mod json;
#[cfg(feature = "sled-backend")]
pub mod sled_backend;
#[cfg(feature = "sqlite-backend")]
pub mod sqlite;

pub use json::JsonMemory;
#[cfg(feature = "sled-backend")]
pub use sled_backend::SledMemory;
#[cfg(feature = "sqlite-backend")]
pub use sqlite::SqliteMemory;

use herta_config::schema::{MemoryBackendKind, MemoryConfig};
use herta_core::{HertaResult, memory::Memory};
use std::sync::Arc;

/// Factory that constructs the configured memory backend.
///
/// Returns `None` when memory is disabled in configuration so callers can
/// skip persistence entirely. Returns a [`herta_core::mocks::InMemoryMemory`]
/// when `MemoryBackendKind::InMemory` is selected (useful for tests).
pub fn build_from_config(cfg: &MemoryConfig) -> HertaResult<Option<Arc<dyn Memory>>> {
    if !cfg.enabled {
        return Ok(None);
    }
    let backend: Arc<dyn Memory> = match cfg.backend {
        MemoryBackendKind::Json => Arc::new(JsonMemory::new(cfg.clone())?),
        MemoryBackendKind::InMemory => {
            Arc::new(herta_core::mocks::InMemoryMemory::default())
        }
        #[cfg(feature = "sqlite-backend")]
        MemoryBackendKind::Sqlite => Arc::new(SqliteMemory::new(cfg.clone())?),
        #[cfg(not(feature = "sqlite-backend"))]
        MemoryBackendKind::Sqlite => {
            return Err(HertaError::config(
                "sqlite memory backend selected but 'sqlite-backend' feature is disabled",
            ));
        }
        #[cfg(feature = "sled-backend")]
        MemoryBackendKind::Sled => Arc::new(SledMemory::new(cfg.clone())?),
        #[cfg(not(feature = "sled-backend"))]
        MemoryBackendKind::Sled => {
            return Err(HertaError::config(
                "sled memory backend selected but 'sled-backend' feature is disabled",
            ));
        }
    };
    Ok(Some(backend))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn disabled_config_returns_none() {
        let cfg = MemoryConfig {
            enabled: false,
            ..MemoryConfig::default()
        };
        assert!(build_from_config(&cfg).unwrap().is_none());
    }

    #[tokio::test]
    async fn json_backend_via_factory() {
        let dir = tempdir().unwrap();
        let cfg = MemoryConfig {
            path: dir.path().join("mem.json"),
            backend: MemoryBackendKind::Json,
            ..MemoryConfig::default()
        };
        let mem = build_from_config(&cfg).unwrap().unwrap();
        mem.append_turn("u", "a").await.unwrap();
        let ctx = mem.load_context(10).await.unwrap();
        assert_eq!(ctx.len(), 2);
    }

    #[tokio::test]
    async fn in_memory_via_factory() {
        let cfg = MemoryConfig {
            backend: MemoryBackendKind::InMemory,
            ..MemoryConfig::default()
        };
        let mem = build_from_config(&cfg).unwrap().unwrap();
        mem.append_turn("u", "a").await.unwrap();
        assert_eq!(mem.load_context(10).await.unwrap().len(), 2);
    }
}
