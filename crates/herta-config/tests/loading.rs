//! Integration tests for configuration loading from environment variables.
#![allow(unsafe_code)]

use herta_config::{LoadOptions, schema::Config};
use tempfile::tempdir;

/// Serialize env mutations to avoid cross-test interference. The Rust test
/// harness uses threads by default, so without a lock one test's env changes
/// can leak into another's `load`.
fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

fn clear_herta_env() {
    // SAFETY: tests are serialized by env_lock.
    let keys: Vec<String> = std::env::vars()
        .map(|(k, _)| k)
        .filter(|k| k.starts_with("HERTA_"))
        .collect();
    for k in keys {
        unsafe {
            std::env::remove_var(k);
        }
    }
}

#[test]
fn defaults_are_valid_and_stable() {
    let _lock = env_lock();
    clear_herta_env();
    let cfg = Config::load_with(&LoadOptions {
        skip_dotenv: true,
        env_prefix: Some("HERTA_TEST_NO_MATCH".into()),
        ..LoadOptions::default()
    })
    .unwrap();
    assert_eq!(
        cfg.llm_provider,
        herta_config::schema::LlmProviderKind::Ollama
    );
    assert!(cfg.memory.enabled);
}

#[test]
fn file_source_overrides_defaults() {
    let _lock = env_lock();
    clear_herta_env();
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.yaml");
    std::fs::write(
        &path,
        "llm_provider: google_ai\nmemory:\n  max_messages: 33\n  context_messages: 5\n",
    )
    .unwrap();

    let cfg = Config::load_with(&LoadOptions {
        path: Some(path),
        skip_dotenv: true,
        env_prefix: Some("HERTA_TEST_NO_MATCH".into()),
    })
    .unwrap();
    assert_eq!(
        cfg.llm_provider,
        herta_config::schema::LlmProviderKind::GoogleAi
    );
    assert_eq!(cfg.memory.max_messages, 33);
    assert_eq!(cfg.memory.context_messages, 5);
}

#[test]
fn env_overrides_file() {
    let _lock = env_lock();
    clear_herta_env();
    let dir = tempdir().unwrap();
    let path = dir.path().join("config.yaml");
    std::fs::write(&path, "log_level: DEBUG\n").unwrap();

    unsafe {
        std::env::set_var("HERTA_LOG_LEVEL", "WARN");
        std::env::set_var("HERTA_MEMORY__MAX_MESSAGES", "42");
    }

    let cfg = Config::load_with(&LoadOptions {
        path: Some(path),
        skip_dotenv: true,
        env_prefix: Some("HERTA".into()),
    })
    .unwrap();
    assert_eq!(cfg.log_level, "WARN");
    assert_eq!(cfg.memory.max_messages, 42);

    unsafe {
        std::env::remove_var("HERTA_LOG_LEVEL");
        std::env::remove_var("HERTA_MEMORY__MAX_MESSAGES");
    }
}
