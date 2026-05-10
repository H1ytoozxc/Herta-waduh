//! Dialogue context primitives.
//!
//! A [`DialogContext`] is a serializable view over the conversation that is
//! passed into [`crate::llm::LlmProvider::generate`]. It carries the current
//! utterance, the rolling message history (already trimmed by the memory
//! layer), and optional metadata such as tenant id and correlation id.

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

/// Chat role. Mirrors the OpenAI/Ollama convention so the mapping to external
/// APIs is trivial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Static system/persona instruction.
    System,
    /// The human end-user.
    User,
    /// The assistant (model output or replayed memory).
    Assistant,
    /// Tool/function output (reserved for future tool-calling support).
    Tool,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        })
    }
}

/// A single message in the conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Who authored the message.
    pub role: Role,
    /// Message content (UTF-8 text).
    pub content: String,
}

impl Message {
    /// Construct a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }
    /// Construct a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }
    /// Construct an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// A dialogue context delivered to the LLM per turn.
///
/// This type is intentionally small and `Clone`-cheap so it can be passed
/// across async boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogContext {
    /// Monotonically increasing correlation id used in log/metric labels and
    /// distributed tracing.
    pub correlation_id: Uuid,
    /// Optional tenant identifier (multi-tenant deployments).
    pub tenant_id: Option<String>,
    /// Locked persona / system prefix messages; never trimmed.
    pub locked_prefix: Vec<Message>,
    /// Rolling history, already trimmed to the configured window.
    pub history: Vec<Message>,
    /// Current user utterance.
    pub user_utterance: String,
    /// Optional detected language code (BCP-47, e.g. `"ru"`, `"en"`).
    pub language: Option<String>,
}

impl DialogContext {
    /// Build a new context for a fresh turn.
    pub fn new(user_utterance: impl Into<String>) -> Self {
        Self {
            correlation_id: Uuid::new_v4(),
            tenant_id: None,
            locked_prefix: Vec::new(),
            history: Vec::new(),
            user_utterance: user_utterance.into(),
            language: None,
        }
    }

    /// Return the full message list as the LLM should see it, without
    /// allocating unnecessarily when possible.
    pub fn messages(&self) -> Vec<Message> {
        let mut out =
            Vec::with_capacity(self.locked_prefix.len() + self.history.len() + 1);
        out.extend_from_slice(&self.locked_prefix);
        out.extend_from_slice(&self.history);
        out.push(Message::user(self.user_utterance.clone()));
        out
    }

    /// Attach a tenant id; returns `self` for builder-style use.
    #[must_use]
    pub fn with_tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// Attach a language hint; returns `self` for builder-style use.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_constructors() {
        assert_eq!(Message::user("hi").role, Role::User);
        assert_eq!(Message::system("s").role, Role::System);
        assert_eq!(Message::assistant("a").role, Role::Assistant);
    }

    #[test]
    fn messages_view_includes_prefix_history_and_utterance() {
        let ctx = DialogContext {
            correlation_id: Uuid::nil(),
            tenant_id: None,
            locked_prefix: vec![Message::system("persona")],
            history: vec![
                Message::user("prev"),
                Message::assistant("answer"),
            ],
            user_utterance: "current".into(),
            language: None,
        };
        let msgs = ctx.messages();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs.last().unwrap().content, "current");
    }

    #[test]
    fn role_display_is_stable() {
        assert_eq!(Role::User.to_string(), "user");
        assert_eq!(Role::Assistant.to_string(), "assistant");
        assert_eq!(Role::System.to_string(), "system");
        assert_eq!(Role::Tool.to_string(), "tool");
    }

    #[test]
    fn builder_style_attachments() {
        let ctx = DialogContext::new("hi")
            .with_tenant("acme")
            .with_language("ru");
        assert_eq!(ctx.tenant_id.as_deref(), Some("acme"));
        assert_eq!(ctx.language.as_deref(), Some("ru"));
    }
}
