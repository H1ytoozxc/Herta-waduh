//! `SQLite` memory backend (feature-gated behind `sqlite-backend`).
//!
//! Schema (auto-created on first open):
//!
//! ```sql
//! CREATE TABLE IF NOT EXISTS messages (
//!   id      INTEGER PRIMARY KEY AUTOINCREMENT,
//!   role    TEXT NOT NULL,
//!   content TEXT NOT NULL,
//!   ts      INTEGER NOT NULL
//! );
//! ```

use async_trait::async_trait;
use herta_config::schema::MemoryConfig;
use herta_core::{
    HertaError, HertaResult, Message, Role,
    memory::{Memory, window_messages},
};
use parking_lot::Mutex;
use rusqlite::{Connection, params};
use std::path::PathBuf;
use std::sync::Arc;

/// `SQLite`-backed memory. One connection serialized via a `parking_lot` mutex —
/// sufficient for a voice assistant that produces one turn per second at most.
pub struct SqliteMemory {
    cfg: MemoryConfig,
    conn: Arc<Mutex<Connection>>,
}

impl std::fmt::Debug for SqliteMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteMemory")
            .field("path", &self.cfg.path)
            .field("max_messages", &self.cfg.max_messages)
            .finish_non_exhaustive()
    }
}

impl SqliteMemory {
    /// Open (or create) the `SQLite` database at the configured path.
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
        let conn = Connection::open(&path)
            .map_err(|e| HertaError::memory(format!("open sqlite: {e}")))?;
        conn.execute_batch(
            r"
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS messages (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                role    TEXT NOT NULL,
                content TEXT NOT NULL,
                ts      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_ts_idx ON messages (ts);
            ",
        )
        .map_err(|e| HertaError::memory(format!("init schema: {e}")))?;

        Ok(Self {
            cfg,
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn load_all(&self) -> HertaResult<Vec<Message>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT role, content FROM messages ORDER BY id ASC",
            )
            .map_err(|e| HertaError::memory(format!("prepare: {e}")))?;
        let rows = stmt
            .query_map([], |row| {
                let role: String = row.get(0)?;
                let content: String = row.get(1)?;
                Ok((role, content))
            })
            .map_err(|e| HertaError::memory(format!("query: {e}")))?;
        let mut out = Vec::new();
        for row in rows {
            let (role, content) =
                row.map_err(|e| HertaError::memory(format!("row: {e}")))?;
            let Some(role) = parse_role(&role) else { continue };
            let content = content.trim().to_string();
            if content.is_empty() {
                continue;
            }
            out.push(Message { role, content });
        }
        Ok(out)
    }

    fn append(&self, user: &str, assistant: &str) -> HertaResult<()> {
        let ts = time::OffsetDateTime::now_utc().unix_timestamp();
        let mut conn = self.conn.lock();
        let tx = conn
            .transaction()
            .map_err(|e| HertaError::memory(format!("begin: {e}")))?;
        tx.execute(
            "INSERT INTO messages (role, content, ts) VALUES (?1, ?2, ?3)",
            params!["user", user, ts],
        )
        .map_err(|e| HertaError::memory(format!("insert user: {e}")))?;
        tx.execute(
            "INSERT INTO messages (role, content, ts) VALUES (?1, ?2, ?3)",
            params!["assistant", assistant, ts],
        )
        .map_err(|e| HertaError::memory(format!("insert assistant: {e}")))?;
        // Trim beyond max_messages.
        if self.cfg.max_messages > 0 {
            tx.execute(
                "DELETE FROM messages WHERE id IN (
                    SELECT id FROM messages
                    ORDER BY id DESC
                    LIMIT -1 OFFSET ?1
                 )",
                params![i64::try_from(self.cfg.max_messages).unwrap_or(i64::MAX)],
            )
            .map_err(|e| HertaError::memory(format!("trim: {e}")))?;
        }
        tx.commit()
            .map_err(|e| HertaError::memory(format!("commit: {e}")))?;
        Ok(())
    }

    fn clear_all(&self) -> HertaResult<()> {
        let conn = self.conn.lock();
        conn.execute("DELETE FROM messages", [])
            .map_err(|e| HertaError::memory(format!("delete: {e}")))?;
        Ok(())
    }
}

fn parse_role(s: &str) -> Option<Role> {
    match s {
        "user" => Some(Role::User),
        "assistant" => Some(Role::Assistant),
        "system" => Some(Role::System),
        "tool" => Some(Role::Tool),
        _ => None,
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
impl Memory for SqliteMemory {
    fn name(&self) -> &'static str {
        "sqlite"
    }

    async fn load_context(&self, max_messages: usize) -> HertaResult<Vec<Message>> {
        let this = self.conn.clone();
        let cfg = self.cfg.clone();
        let all = tokio::task::spawn_blocking(move || {
            let mem = SqliteMemory { cfg, conn: this };
            mem.load_all()
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
        let this = self.conn.clone();
        let cfg = self.cfg.clone();
        tokio::task::spawn_blocking(move || {
            let mem = SqliteMemory { cfg, conn: this };
            mem.append(&user_text, &assistant_text)
        })
        .await
        .map_err(|e| HertaError::memory(format!("join: {e}")))?
    }

    async fn clear(&self) -> HertaResult<()> {
        let this = self.conn.clone();
        let cfg = self.cfg.clone();
        tokio::task::spawn_blocking(move || {
            let mem = SqliteMemory { cfg, conn: this };
            mem.clear_all()
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
            backend: herta_config::schema::MemoryBackendKind::Sqlite,
            path: dir.join("mem.sqlite"),
            max_messages: 8,
            context_messages: 8,
        }
    }

    #[tokio::test]
    async fn sqlite_roundtrip() {
        let dir = tempdir().unwrap();
        let mem = SqliteMemory::new(cfg(dir.path())).unwrap();
        mem.append_turn("hi", "hello").await.unwrap();
        let out = mem.load_context(10).await.unwrap();
        assert_eq!(out.len(), 2);
    }

    #[tokio::test]
    async fn sqlite_trims_to_max() {
        let dir = tempdir().unwrap();
        let mut c = cfg(dir.path());
        c.max_messages = 4;
        let mem = SqliteMemory::new(c).unwrap();
        for i in 0..5 {
            mem.append_turn(&format!("u{i}"), &format!("a{i}")).await.unwrap();
        }
        let out = mem.load_context(100).await.unwrap();
        assert!(out.len() <= 4);
    }

    #[tokio::test]
    async fn sqlite_clear_removes_all() {
        let dir = tempdir().unwrap();
        let mem = SqliteMemory::new(cfg(dir.path())).unwrap();
        mem.append_turn("a", "b").await.unwrap();
        mem.clear().await.unwrap();
        assert!(mem.load_context(10).await.unwrap().is_empty());
    }
}
