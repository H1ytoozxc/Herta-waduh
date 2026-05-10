//! Sled (embedded KV) memory backend (feature-gated behind `sled-backend`).
//!
//! Layout: a single sled `Tree` stores big-endian u64 keys to message JSON.
//! Turns are appended sequentially, so iteration order matches insertion order.

use async_trait::async_trait;
use herta_config::schema::MemoryConfig;
use herta_core::{
    HertaError, HertaResult, Message, Role,
    memory::{Memory, window_messages},
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredMessage {
    role: String,
    content: String,
}

impl StoredMessage {
    fn into_message(self) -> Option<Message> {
        let role = match self.role.as_str() {
            "user" => Role::User,
            "assistant" => Role::Assistant,
            "system" => Role::System,
            "tool" => Role::Tool,
            _ => return None,
        };
        let content = self.content.trim().to_string();
        if content.is_empty() {
            return None;
        }
        Some(Message { role, content })
    }
}

/// Sled-backed memory.
pub struct SledMemory {
    cfg: MemoryConfig,
    db: Arc<sled::Db>,
    tree: Arc<sled::Tree>,
}

impl std::fmt::Debug for SledMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SledMemory")
            .field("path", &self.cfg.path)
            .finish_non_exhaustive()
    }
}

impl SledMemory {
    /// Open a sled database at the configured path.
    pub fn new(cfg: MemoryConfig) -> HertaResult<Self> {
        let path = resolve_path(&cfg.path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                HertaError::memory(format!(
                    "failed to create dir '{}': {e}",
                    parent.display()
                ))
            })?;
        }
        let db = sled::open(&path)
            .map_err(|e| HertaError::memory(format!("sled open: {e}")))?;
        let tree = db
            .open_tree("messages")
            .map_err(|e| HertaError::memory(format!("sled tree: {e}")))?;
        Ok(Self {
            cfg,
            db: Arc::new(db),
            tree: Arc::new(tree),
        })
    }

    fn next_id(&self) -> HertaResult<u64> {
        self.db
            .generate_id()
            .map_err(|e| HertaError::memory(format!("sled gen id: {e}")))
    }

    fn append_record(&self, role: &str, content: &str) -> HertaResult<()> {
        let id = self.next_id()?;
        let key = id.to_be_bytes();
        let payload = StoredMessage {
            role: role.to_string(),
            content: content.to_string(),
        };
        let bytes = serde_json::to_vec(&payload)
            .map_err(|e| HertaError::memory(format!("serialize: {e}")))?;
        self.tree
            .insert(key, bytes)
            .map_err(|e| HertaError::memory(format!("sled insert: {e}")))?;
        self.trim_to_max()?;
        self.tree
            .flush()
            .map_err(|e| HertaError::memory(format!("sled flush: {e}")))?;
        Ok(())
    }

    fn trim_to_max(&self) -> HertaResult<()> {
        if self.cfg.max_messages == 0 {
            return Ok(());
        }
        let len = self.tree.len();
        if len <= self.cfg.max_messages {
            return Ok(());
        }
        let to_remove = len - self.cfg.max_messages;
        let mut removed = 0;
        let keys: Vec<sled::IVec> = self
            .tree
            .iter()
            .keys()
            .filter_map(Result::ok)
            .take(to_remove)
            .collect();
        for k in keys {
            self.tree
                .remove(&k)
                .map_err(|e| HertaError::memory(format!("sled remove: {e}")))?;
            removed += 1;
            if removed >= to_remove {
                break;
            }
        }
        Ok(())
    }
}

fn resolve_path(raw: &std::path::Path) -> PathBuf {
    if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(raw)
    }
}

#[async_trait]
impl Memory for SledMemory {
    fn name(&self) -> &'static str {
        "sled"
    }

    async fn load_context(&self, max_messages: usize) -> HertaResult<Vec<Message>> {
        let tree = self.tree.clone();
        let all = tokio::task::spawn_blocking(move || {
            let mut out = Vec::new();
            for item in tree.iter() {
                let (_k, v) = item.map_err(|e| HertaError::memory(format!("iter: {e}")))?;
                let stored: StoredMessage = serde_json::from_slice(&v)
                    .map_err(|e| HertaError::memory(format!("decode: {e}")))?;
                if let Some(m) = stored.into_message() {
                    out.push(m);
                }
            }
            Ok::<_, HertaError>(out)
        })
        .await
        .map_err(|e| HertaError::memory(format!("join: {e}")))??;
        Ok(window_messages(&all, max_messages))
    }

    async fn append_turn(
        &self,
        user_text: &str,
        assistant_text: &str,
    ) -> HertaResult<()> {
        let user_text = user_text.trim().to_string();
        let assistant_text = assistant_text.trim().to_string();
        if user_text.is_empty() || assistant_text.is_empty() {
            return Ok(());
        }
        let tree = self.tree.clone();
        let db = self.db.clone();
        let cfg = self.cfg.clone();
        tokio::task::spawn_blocking(move || -> HertaResult<()> {
            let mem = SledMemory { cfg, db, tree };
            mem.append_record("user", &user_text)?;
            mem.append_record("assistant", &assistant_text)
        })
        .await
        .map_err(|e| HertaError::memory(format!("join: {e}")))?
    }

    async fn clear(&self) -> HertaResult<()> {
        let tree = self.tree.clone();
        tokio::task::spawn_blocking(move || {
            tree.clear()
                .map_err(|e| HertaError::memory(format!("sled clear: {e}")))
        })
        .await
        .map_err(|e| HertaError::memory(format!("join: {e}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn cfg(dir: &std::path::Path) -> MemoryConfig {
        MemoryConfig {
            enabled: true,
            backend: herta_config::schema::MemoryBackendKind::Sled,
            path: dir.join("mem.sled"),
            max_messages: 6,
            context_messages: 6,
        }
    }

    #[tokio::test]
    async fn sled_roundtrip() {
        let dir = tempdir().unwrap();
        let mem = SledMemory::new(cfg(dir.path())).unwrap();
        mem.append_turn("hi", "yo").await.unwrap();
        let out = mem.load_context(10).await.unwrap();
        assert_eq!(out.len(), 2);
    }
}
