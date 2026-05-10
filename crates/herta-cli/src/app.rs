//! CLI argument parsing and dispatch.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// The Herta — production voice assistant CLI.
#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "The Herta voice assistant (Rust edition)",
    long_about = None,
    propagate_version = true
)]
pub struct Cli {
    /// Path to a YAML/TOML configuration file.
    #[arg(long, env = "HERTA_CONFIG_FILE", value_name = "PATH", global = true)]
    pub config: Option<PathBuf>,

    /// Override the default log level (trace, debug, info, warn, error).
    #[arg(long, env = "HERTA_LOG_LEVEL", value_name = "LEVEL", global = true)]
    pub log_level: Option<String>,

    /// Emit structured JSON logs regardless of config setting.
    #[arg(long, global = true)]
    pub json_logs: bool,

    /// Do not bind the /healthz + /metrics server.
    #[arg(long, global = true)]
    pub no_server: bool,

    /// Run non-interactively (no REPL; useful in containers and CI).
    #[arg(long, global = true)]
    pub non_interactive: bool,

    /// The subcommand to execute.
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Start the voice pipeline (microphone → VAD → STT → LLM → TTS).
    Voice {
        /// Disable TTS playback for this run.
        #[arg(long)]
        no_tts: bool,
    },
    /// Run a single text turn and exit.
    Text {
        /// Prompt text.
        prompt: String,
        /// Disable TTS playback for this run.
        #[arg(long)]
        no_tts: bool,
    },
    /// Start an interactive text REPL.
    Repl {
        /// Disable TTS playback for this run.
        #[arg(long)]
        no_tts: bool,
    },
    /// Print the effective configuration (secrets redacted) and exit.
    ShowConfig {
        /// Output format: yaml (default), toml, or json.
        #[arg(long, default_value = "yaml")]
        format: String,
    },
    /// List available audio input devices.
    ListInputDevices,
    /// List available audio output devices.
    ListOutputDevices,
    /// Play a short test tone through the configured output device.
    OutputTest,
    /// Print a short TTS sentence through the configured voice.
    TtsTest {
        /// Override the spoken text.
        #[arg(long, default_value = "This is The Herta. Voice output test complete.")]
        text: String,
    },
    /// Run a production readiness self-check and exit (doctor).
    Doctor,
}
