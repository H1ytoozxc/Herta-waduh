//! JSON-file memory backend.
//!
//! Stores messages in a single JSON document `{ "version": 1, "messages": [...] }`.
//! Writes are atomic: new content is written to a sibling tempfile and then
//! renamed over the target path.

use std::fmt::Write as _;
use async_trait::async_trait;
use herta_config::schema::MemoryConfig;
use herta_core::{
    HertaError, HertaResult, Message, Role,
    memory::{Memory, window_messages},
};
use serde::{Deserialize, Serialize};
use std::{
    io::Write,
    path::{Path, PathBuf},
};

const MEMORY_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Payload {
    #[serde(default = "default_version")]
    version: u32,
    #[serde(default)]
    messages: Vec<PersistedMessage>,
}

fn default_version() -> u32 {
    MEMORY_VERSION
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedMessage {
    role: String,
    content: String,
}

impl From<&Message> for PersistedMessage {
    fn from(m: &Message) -> Self {
        Self {
            role: m.role.to_string(),
            content: m.content.clone(),
        }
    }
}

impl PersistedMessage {
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

/// JSON-file memory backend.
#[derive(Debug)]
pub struct JsonMemory {
    cfg: MemoryConfig,
    path: PathBuf,
}

impl JsonMemory {
    /// Build a new JSON memory at the configured path.
    pub fn new(cfg: MemoryConfig) -> HertaResult<Self> {
        let path = resolve_path(&cfg.path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                HertaError::memory(format!(
                    "failed to create memory dir '{}': {e}",
                    parent.display()
                ))
            })?;
        }
        Ok(Self { cfg, path })
    }

    /// Path actually used on disk, after resolution.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

fn resolve_path(raw: &Path) -> PathBuf {
    if raw.is_absolute() {
        raw.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(raw)
    }
}

fn tempfile_path(path: &Path) -> PathBuf {
    let pid = std::process::id();
    let mut file_name = path
        .file_name()
        .map_or_else(|| "memory.json".into(), |s| s.to_string_lossy().into_owned());
    let _ = write!(file_name, ".{pid}.tmp");
    path.with_file_name(file_name)
}

fn read_payload_at(path: &Path) -> HertaResult<Payload> {
    if !path.exists() {
        return Ok(Payload {
            version: MEMORY_VERSION,
            messages: Vec::new(),
        });
    }
    let bytes = std::fs::read(path).map_err(|e| {
        HertaError::memory(format!("failed to read memory '{}': {e}", path.display()))
    })?;
    if bytes.is_empty() {
        return Ok(Payload {
            version: MEMORY_VERSION,
            messages: Vec::new(),
        });
    }
    serde_json::from_slice::<Payload>(&bytes)
        .map_err(|e| HertaError::memory(format!("memory file is not valid JSON: {e}")))
}

fn sanitize(raw: Vec<PersistedMessage>) -> Vec<Message> {
    raw.into_iter().filter_map(PersistedMessage::into_message).collect()
}

fn write_messages_to(
    path: &Path,
    cfg: &MemoryConfig,
    messages: &[Message],
) -> HertaResult<()> {
    let cap = cfg.max_messages;
    let trimmed: Vec<&Message> = if cap == 0 {
        Vec::new()
    } else if messages.len() > cap {
        messages[messages.len() - cap..].iter().collect()
    } else {
        messages.iter().collect()
    };
    let persisted: Vec<PersistedMessage> =
        trimmed.into_iter().map(PersistedMessage::from).collect();
    let payload = Payload {
        version: MEMORY_VERSION,
        messages: persisted,
    };
    let serialized = serde_json::to_vec_pretty(&payload)
        .map_err(|e| HertaError::memory(format!("serialize: {e}")))?;

    let dir = path.parent().ok_or_else(|| {
        HertaError::memory(format!("memory path has no parent: {}", path.display()))
    })?;
    std::fs::create_dir_all(dir).map_err(|e| {
        HertaError::memory(format!(
            "failed to create memory dir '{}': {e}",
            dir.display()
        ))
    })?;
    let tmp = tempfile_path(path);
    {
        let mut file = std::fs::File::create(&tmp).map_err(|e| {
            HertaError::memory(format!("failed to open temp '{}': {e}", tmp.display()))
        })?;
        file.write_all(&serialized)
            .map_err(|e| HertaError::memory(format!("write tempfile: {e}")))?;
        file.sync_all().ok();
    }
    std::fs::rename(&tmp, path)
        .map_err(|e| HertaError::memory(format!("rename tempfile: {e}")))
}

#[async_trait]
impl Memory for JsonMemory {
    fn name(&self) -> &'static str {
        "json"
    }

    async fn load_context(&self, max_messages: usize) -> HertaResult<Vec<Message>> {
        let path = self.path.clone();
        let messages = tokio::task::spawn_blocking(move || {
            let payload = read_payload_at(&path)?;
            Ok::<_, HertaError>(sanitize(payload.messages))
        })
        .await
        .map_err(|e| HertaError::memory(format!("join: {e}")))??;
        tracing::debug!(path = %self.path.display(), count = messages.len(), "loaded memory payload");
        Ok(window_messages(&messages, max_messages))
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
        let path = self.path.clone();
        let cfg = self.cfg.clone();
        tokio::task::spawn_blocking(move || -> HertaResult<()> {
            let payload = read_payload_at(&path)?;
            let mut messages = sanitize(payload.messages);
            messages.push(Message::user(user_text));
            messages.push(Message::assistant(assistant_text));
            write_messages_to(&path, &cfg, &messages)
        })
        .await
        .map_err(|e| HertaError::memory(format!("join: {e}")))?
    }

    async fn clear(&self) -> HertaResult<()> {
        let path = self.path.clone();
        let cfg = self.cfg.clone();
        tokio::task::spawn_blocking(move || write_messages_to(&path, &cfg, &[]))
            .await
            .map_err(|e| HertaError::memory(format!("join: {e}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn cfg_in(dir: &Path) -> MemoryConfig {
        MemoryConfig {
            enabled: true,
            backend: herta_config::schema::MemoryBackendKind::Json,
            path: dir.join("mem.json"),
            max_messages: 10,
            context_messages: 10,
        }
    }

    #[tokio::test]
    async fn roundtrip_persists_messages() {
        let dir = tempdir().unwrap();
        let mem = JsonMemory::new(cfg_in(dir.path())).unwrap();
        mem.append_turn("hello", "hi there").await.unwrap();
        let out = mem.load_context(10).await.unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, Role::User);
        assert_eq!(out[1].role, Role::Assistant);
    }

    #[tokio::test]
    async fn clear_erases_file() {
        let dir = tempdir().unwrap();
        let mem = JsonMemory::new(cfg_in(dir.path())).unwrap();
        mem.append_turn("a", "b").await.unwrap();
        mem.clear().await.unwrap();
        assert!(mem.load_context(10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn trimming_bounded_by_max_messages() {
        let dir = tempdir().unwrap();
        let mut cfg = cfg_in(dir.path());
        cfg.max_messages = 4;
        let mem = JsonMemory::new(cfg).unwrap();
        for i in 0..6 {
            mem.append_turn(&format!("u{i}"), &format!("a{i}")).await.unwrap();
        }
        let out = mem.load_context(100).await.unwrap();
        assert!(out.len() <= 4);
    }

    #[tokio::test]
    async fn empty_turns_are_skipped() {
        let dir = tempdir().unwrap();
        let mem = JsonMemory::new(cfg_in(dir.path())).unwrap();
        mem.append_turn("   ", "non-empty").await.unwrap();
        mem.append_turn("non-empty", "   ").await.unwrap();
        assert!(mem.load_context(10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn missing_file_returns_empty() {
        let dir = tempdir().unwrap();
        let mem = JsonMemory::new(cfg_in(dir.path())).unwrap();
        assert!(mem.load_context(10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn garbage_file_returns_memory_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("mem.json");
        std::fs::write(&path, "not-json").unwrap();
        let cfg = MemoryConfig {
            path,
            ..cfg_in(dir.path())
        };
        let mem = JsonMemory::new(cfg).unwrap();
        let err = mem.load_context(10).await.unwrap_err();
        assert!(matches!(err, HertaError::Memory(_)));
    }
}
