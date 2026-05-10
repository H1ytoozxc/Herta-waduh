//! # herta-config
//!
//! Centralized, strongly-typed configuration for The Herta Voice Assistant.
//!
//! Sources are merged with the following precedence (higher wins):
//!
//! 1. Environment variables (prefix `HERTA_`, nested with `__`).
//! 2. `HERTA_CONFIG_FILE` — explicit YAML/TOML path.
//! 3. `config.yaml` / `config.toml` in the current working directory.
//! 4. Compile-time defaults derived from [`Config::default`].
//!
//! `.env` files are loaded automatically on [`Config::load`] if present.
//!
//! Every secret field is `Option<String>` so empty/missing values can be
//! distinguished from intentionally blank ones, and log formatting elides
//! secret values.

#![warn(missing_docs)]

pub mod schema;

use crate::schema::Config;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors produced by configuration loading / validation.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// A value could not be parsed from its source.
    #[error("invalid configuration value: {0}")]
    Invalid(String),
    /// The configuration file could not be read or parsed.
    #[error("failed to read configuration file: {0}")]
    File(String),
    /// A validation rule was violated after merging sources.
    #[error("validation error: {0}")]
    Validation(String),
    /// Underlying `config` crate failure.
    #[error("config error: {0}")]
    Underlying(#[from] config::ConfigError),
}

/// Options controlling [`Config::load`].
#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    /// Explicit path to a YAML/TOML config file. Overrides env/auto-discovery.
    pub path: Option<PathBuf>,
    /// When `true`, do not attempt to load `.env` files.
    pub skip_dotenv: bool,
    /// Override the environment variable prefix (defaults to `HERTA`).
    pub env_prefix: Option<String>,
}

impl LoadOptions {
    /// Convenience: build options with an explicit path.
    pub fn with_path(path: impl Into<PathBuf>) -> Self {
        Self {
            path: Some(path.into()),
            ..Self::default()
        }
    }
}

impl Config {
    /// Load configuration using the default search order.
    pub fn load() -> Result<Self, ConfigError> {
        Self::load_with(&LoadOptions::default())
    }

    /// Load configuration with explicit options.
    pub fn load_with(opts: &LoadOptions) -> Result<Self, ConfigError> {
        if !opts.skip_dotenv {
            // Ignore missing `.env`; it's optional by design.
            let _ = dotenvy::dotenv();
        }

        let prefix = opts.env_prefix.as_deref().unwrap_or("HERTA");

        // Serialize compile-time defaults to JSON, then register that JSON
        // blob as the lowest-priority source. Every other source overrides it.
        let defaults_json = serde_json::to_string(&Self::default())
            .map_err(|e| ConfigError::Invalid(format!("serialize defaults: {e}")))?;
        let defaults_source =
            config::File::from_str(&defaults_json, config::FileFormat::Json);

        let mut builder = config::Config::builder().add_source(defaults_source);

        if let Some(path) = opts.path.as_deref() {
            builder = builder.add_source(file_source_required(path)?);
        } else if let Some(path) = auto_discover_file() {
            builder = builder.add_source(file_source_required(&path)?);
        }

        let env = config::Environment::with_prefix(prefix)
            .prefix_separator("_")
            .separator("__")
            .try_parsing(true)
            .ignore_empty(true);
        builder = builder.add_source(env);

        let cfg: Self = builder.build()?.try_deserialize()?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate invariants that can't be expressed in the type system.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.memory.max_messages == 0 {
            return Err(ConfigError::Validation(
                "memory.max_messages must be > 0".into(),
            ));
        }
        if self.memory.context_messages > self.memory.max_messages {
            return Err(ConfigError::Validation(
                "memory.context_messages cannot exceed memory.max_messages".into(),
            ));
        }
        if self.audio.sample_rate == 0 {
            return Err(ConfigError::Validation(
                "audio.sample_rate must be > 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.vad.threshold) {
            return Err(ConfigError::Validation(
                "vad.threshold must be within [0.0, 1.0]".into(),
            ));
        }
        Ok(())
    }
}

fn file_source_required(path: &Path) -> Result<config::File<config::FileSourceFile, config::FileFormat>, ConfigError> {
    let format = match path.extension().and_then(|s| s.to_str()) {
        Some("yaml" | "yml") => config::FileFormat::Yaml,
        Some("toml") => config::FileFormat::Toml,
        Some("json") => config::FileFormat::Json,
        _ => {
            return Err(ConfigError::File(format!(
                "unsupported config file extension: {}",
                path.display()
            )));
        }
    };
    Ok(config::File::new(
        path.to_string_lossy().as_ref(),
        format,
    )
    .required(false))
}

fn auto_discover_file() -> Option<PathBuf> {
    for candidate in ["config.yaml", "config.yml", "config.toml", "config.json"] {
        let path = PathBuf::from(candidate);
        if path.is_file() {
            return Some(path);
        }
    }
    None
}

/// Redact a secret-like string for logging, keeping only a short prefix.
pub fn redact_secret(value: &str) -> String {
    if value.is_empty() {
        return "<unset>".into();
    }
    let visible = value.chars().take(4).collect::<String>();
    format!("{visible}***<redacted:{}chars>", value.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn defaults_validate() {
        Config::default().validate().unwrap();
    }

    #[test]
    fn invalid_vad_threshold_rejected() {
        let mut cfg = Config::default();
        cfg.vad.threshold = 2.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn zero_memory_max_rejected() {
        let mut cfg = Config::default();
        cfg.memory.max_messages = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn context_over_max_rejected() {
        let mut cfg = Config::default();
        cfg.memory.context_messages = cfg.memory.max_messages + 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn redact_secret_preserves_prefix() {
        assert_eq!(redact_secret(""), "<unset>");
        assert!(redact_secret("hunter2-super-secret").starts_with("hunt***"));
    }

    #[test]
    fn loads_yaml_file_via_options() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.yaml");
        std::fs::write(
            &path,
            "llm_provider: ollama\nmemory:\n  max_messages: 50\n  context_messages: 10\n",
        )
        .unwrap();
        let cfg = Config::load_with(&LoadOptions {
            path: Some(path),
            skip_dotenv: true,
            env_prefix: Some("HERTA_TEST_UNSET".into()),
        })
        .unwrap();
        assert_eq!(cfg.memory.max_messages, 50);
        assert_eq!(cfg.memory.context_messages, 10);
    }
}
