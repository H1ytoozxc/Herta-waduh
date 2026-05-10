//! Dialogue memory abstraction.
//!
//! Concrete backends (JSON file, `SQLite`, Sled) live in `herta-memory`.

use crate::{HertaResult, context::Message};
use async_trait::async_trait;

/// Persistent memory interface for dialogue history.
///
/// Implementations must be safe to call concurrently from multiple tasks.
/// Idempotent retrieval is required; writes may be eventually consistent
/// across backends.
#[async_trait]
pub trait Memory: Send + Sync + 'static {
    /// Backend name for metrics/logs.
    fn name(&self) -> &'static str;

    /// Load the rolling context window (user/assistant pairs only).
    async fn load_context(&self, max_messages: usize) -> HertaResult<Vec<Message>>;

    /// Append a completed turn (user + assistant message).
    async fn append_turn(
        &self,
        user_text: &str,
        assistant_text: &str,
    ) -> HertaResult<()>;

    /// Wipe all stored messages. Destructive; callers should confirm intent.
    async fn clear(&self) -> HertaResult<()>;
}

/// Pure windowing helper used by backends to ensure consistent trimming
/// semantics regardless of underlying storage.
///
/// Rules:
/// - Drops everything but the last `max_messages` entries.
/// - Skips leading `assistant` messages so the window always starts with a
///   `user` turn (or is empty).
pub fn window_messages(messages: &[Message], max_messages: usize) -> Vec<Message> {
    if max_messages == 0 || messages.is_empty() {
        return Vec::new();
    }
    let start = messages.len().saturating_sub(max_messages);
    let mut view = messages[start..].to_vec();
    while let Some(first) = view.first() {
        if matches!(first.role, crate::Role::Assistant) {
            view.remove(0);
        } else {
            break;
        }
    }
    view
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Role;
    use proptest::prelude::*;

    #[test]
    fn window_trims_to_last_n() {
        let msgs = vec![
            Message::user("a"),
            Message::assistant("b"),
            Message::user("c"),
            Message::assistant("d"),
            Message::user("e"),
            Message::assistant("f"),
        ];
        let out = window_messages(&msgs, 4);
        assert_eq!(out.len(), 4);
        assert_eq!(out.first().unwrap().role, Role::User);
    }

    #[test]
    fn window_skips_leading_assistant() {
        let msgs = vec![
            Message::assistant("orphan"),
            Message::user("u1"),
            Message::assistant("a1"),
        ];
        let out = window_messages(&msgs, 10);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, Role::User);
    }

    #[test]
    fn zero_window_returns_empty() {
        let msgs = vec![Message::user("u")];
        assert!(window_messages(&msgs, 0).is_empty());
    }

    proptest! {
        #[test]
        fn window_size_never_exceeds_max(
            n in 0usize..50,
            max in 0usize..50,
        ) {
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                v.push(if i % 2 == 0 { Message::user("u") } else { Message::assistant("a") });
            }
            let out = window_messages(&v, max);
            prop_assert!(out.len() <= max);
            prop_assert!(out.len() <= n);
            if let Some(first) = out.first() {
                prop_assert_eq!(first.role, Role::User);
            }
        }
    }
}
