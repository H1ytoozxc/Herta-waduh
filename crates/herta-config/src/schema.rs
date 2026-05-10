//! Strongly-typed configuration schema.
//!
//! The root [`Config`] mirrors the nested structure of the original
//! `config.py` from the Python codebase, translated into idiomatic Rust with
//! `serde`-based deserialization.

#![allow(missing_docs)] // Fields are self-documenting; module-level docs apply.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub log_level: String,
    pub llm_provider: LlmProviderKind,
    pub stt_provider: SttProviderKind,
    pub max_history_messages: u32,
    pub persona_rewrite_enabled: bool,
    pub telemetry: TelemetryConfig,
    pub server: ServerConfig,
    pub ollama: OllamaConfig,
    pub deepseek: DeepSeekConfig,
    pub google_ai: GoogleAiConfig,
    pub google_stt: GoogleSttConfig,
    pub tts: EdgeTtsConfig,
    pub audio: AudioInputConfig,
    pub audio_output: AudioOutputConfig,
    pub vad: VadConfig,
    pub stt: WhisperSttConfig,
    pub memory: MemoryConfig,
    pub system_actions: SystemActionsConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            log_level: "INFO".into(),
            llm_provider: LlmProviderKind::Ollama,
            stt_provider: SttProviderKind::Whisper,
            max_history_messages: 8,
            persona_rewrite_enabled: false,
            telemetry: TelemetryConfig::default(),
            server: ServerConfig::default(),
            ollama: OllamaConfig::default(),
            deepseek: DeepSeekConfig::default(),
            google_ai: GoogleAiConfig::default(),
            google_stt: GoogleSttConfig::default(),
            tts: EdgeTtsConfig::default(),
            audio: AudioInputConfig::default(),
            audio_output: AudioOutputConfig::default(),
            vad: VadConfig::default(),
            stt: WhisperSttConfig::default(),
            memory: MemoryConfig::default(),
            system_actions: SystemActionsConfig::default(),
        }
    }
}

/// Which LLM backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LlmProviderKind {
    Ollama,
    DeepSeek,
    GoogleAi,
}

/// Which STT backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SttProviderKind {
    Whisper,
    GoogleAi,
}

/// Which memory backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryBackendKind {
    Json,
    Sqlite,
    Sled,
    InMemory,
}

/// Telemetry / observability configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TelemetryConfig {
    pub log_format: LogFormat,
    pub metrics_enabled: bool,
    pub tracing_enabled: bool,
    pub otlp_endpoint: Option<String>,
    pub service_name: String,
    pub environment: String,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            log_format: LogFormat::Pretty,
            metrics_enabled: true,
            tracing_enabled: false,
            otlp_endpoint: None,
            service_name: "the-herta".into(),
            environment: "dev".into(),
        }
    }
}

/// Log output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    Pretty,
    Json,
    Compact,
}

/// Embedded HTTP server for /healthz and /metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub enabled: bool,
    pub bind: String,
    pub health_path: String,
    pub metrics_path: String,
    pub readiness_path: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bind: "0.0.0.0:9090".into(),
            health_path: "/healthz".into(),
            metrics_path: "/metrics".into(),
            readiness_path: "/ready".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OllamaConfig {
    pub host: String,
    pub model: String,
    pub timeout_seconds: f64,
    pub keep_alive: String,
    pub think: bool,
    pub temperature: f32,
    pub num_ctx: u32,
    pub num_gpu: Option<u32>,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            host: "http://127.0.0.1:11434".into(),
            model: "qwen3:4b".into(),
            timeout_seconds: 300.0,
            keep_alive: "10m".into(),
            think: false,
            temperature: 0.55,
            num_ctx: 2048,
            num_gpu: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DeepSeekConfig {
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
    pub timeout_seconds: f64,
    pub temperature: f32,
    pub max_tokens: u32,
    pub retry_attempts: u32,
    pub rate_limit_retries: u32,
}

impl Default for DeepSeekConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: "https://api.deepseek.com".into(),
            model: "deepseek-v4-flash".into(),
            timeout_seconds: 120.0,
            temperature: 0.55,
            max_tokens: 700,
            retry_attempts: 4,
            rate_limit_retries: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GoogleAiConfig {
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
    pub fallback_model: Option<String>,
    pub timeout_seconds: f64,
    pub temperature: f32,
    pub max_tokens: u32,
    pub retry_attempts: u32,
    pub rate_limit_retries: u32,
    pub system_instruction_enabled: bool,
}

impl Default for GoogleAiConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: "https://generativelanguage.googleapis.com/v1beta".into(),
            model: "gemma-3-27b-it".into(),
            fallback_model: None,
            timeout_seconds: 45.0,
            temperature: 0.55,
            max_tokens: 700,
            retry_attempts: 0,
            rate_limit_retries: 2,
            system_instruction_enabled: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GoogleSttConfig {
    pub api_key: Option<String>,
    pub base_url: String,
    pub model: String,
    pub timeout_seconds: f64,
    pub retry_attempts: u32,
    pub rate_limit_retries: u32,
    pub language_hint: Option<String>,
    pub fallback_to_whisper: bool,
}

impl Default for GoogleSttConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: "https://generativelanguage.googleapis.com/v1beta".into(),
            model: "gemini-2.5-flash".into(),
            timeout_seconds: 60.0,
            retry_attempts: 3,
            rate_limit_retries: 2,
            language_hint: Some("ru".into()),
            fallback_to_whisper: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EdgeTtsConfig {
    pub enabled: bool,
    pub prefer_local: bool,
    pub voice: String,
    pub rate: String,
    pub volume: String,
    pub pitch: String,
    pub sapi_voice: Option<String>,
    pub sapi_rate: i32,
    pub sapi_volume: i32,
    pub piper_model_path: Option<PathBuf>,
    pub piper_config_path: Option<PathBuf>,
    pub piper_use_cuda: bool,
}

impl Default for EdgeTtsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prefer_local: true,
            voice: "ru-RU-DariyaNeural".into(),
            rate: "-6%".into(),
            volume: "+0%".into(),
            pitch: "+8Hz".into(),
            sapi_voice: Some("Microsoft Irina Desktop - Russian".into()),
            sapi_rate: 0,
            sapi_volume: 100,
            piper_model_path: Some(PathBuf::from("models/piper/ru_RU-irina-medium.onnx")),
            piper_config_path: None,
            piper_use_cuda: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioInputConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub dtype: String,
    pub block_size: u32,
    pub device: Option<String>,
    pub queue_max_chunks: u32,
}

impl Default for AudioInputConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            channels: 1,
            dtype: "float32".into(),
            block_size: 512,
            device: None,
            queue_max_chunks: 128,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioOutputConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub dtype: String,
    pub device: Option<String>,
    pub tone_frequency_hz: f32,
    pub tone_duration_seconds: f32,
    pub tone_volume: f32,
}

impl Default for AudioOutputConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            channels: 2,
            dtype: "float32".into(),
            device: None,
            tone_frequency_hz: 523.25,
            tone_duration_seconds: 0.7,
            tone_volume: 0.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VadConfig {
    pub threshold: f32,
    pub min_silence_duration_ms: u32,
    pub speech_pad_ms: u32,
    pub min_utterance_duration_ms: u32,
    pub max_utterance_seconds: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_silence_duration_ms: 600,
            speech_pad_ms: 200,
            min_utterance_duration_ms: 450,
            max_utterance_seconds: 20.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WhisperSttConfig {
    pub model_size: String,
    pub device: String,
    pub compute_type: String,
    pub cpu_threads: u32,
    pub num_workers: u32,
    pub language: Option<String>,
    pub beam_size: u32,
    pub best_of: u32,
    pub no_speech_threshold: f32,
    pub log_prob_threshold: f32,
    pub compression_ratio_threshold: f32,
    pub min_peak_level: f32,
    pub min_rms_level: f32,
    pub normalize_audio: bool,
    pub local_files_only: bool,
    pub download_root: Option<PathBuf>,
    pub initial_prompt: Option<String>,
}

impl Default for WhisperSttConfig {
    fn default() -> Self {
        Self {
            model_size: "small".into(),
            device: "cpu".into(),
            compute_type: "int8".into(),
            cpu_threads: 4,
            num_workers: 1,
            language: None,
            beam_size: 5,
            best_of: 5,
            no_speech_threshold: 0.6,
            log_prob_threshold: -0.8,
            compression_ratio_threshold: 2.2,
            min_peak_level: 0.01,
            min_rms_level: 0.0015,
            normalize_audio: true,
            local_files_only: false,
            download_root: None,
            initial_prompt: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    pub enabled: bool,
    pub backend: MemoryBackendKind,
    pub path: PathBuf,
    pub max_messages: usize,
    pub context_messages: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            backend: MemoryBackendKind::Json,
            path: PathBuf::from("data/dialogue_memory.json"),
            max_messages: 80,
            context_messages: 12,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SystemActionsConfig {
    pub enabled: bool,
    pub document_dir: String,
    pub registry_path: PathBuf,
    pub browser_home_url: String,
    pub vscode_command: String,
    pub vscode_open_workspace: bool,
}

impl Default for SystemActionsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            document_dir: "desktop".into(),
            registry_path: PathBuf::from("data/system_actions_registry.json"),
            browser_home_url: "https://www.google.com".into(),
            vscode_command: "code".into(),
            vscode_open_workspace: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_round_trip_through_yaml() {
        let cfg = Config::default();
        let yaml = serde_yaml::to_string(&cfg).unwrap();
        let parsed: Config = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(parsed.llm_provider, LlmProviderKind::Ollama);
        assert_eq!(parsed.stt_provider, SttProviderKind::Whisper);
    }

    #[test]
    fn defaults_round_trip_through_toml() {
        let cfg = Config::default();
        let toml = toml::to_string(&cfg).unwrap();
        let parsed: Config = toml::from_str(&toml).unwrap();
        assert_eq!(parsed.memory.max_messages, 80);
    }

    #[test]
    fn enum_snake_case_in_yaml() {
        let mut cfg = Config::default();
        cfg.llm_provider = LlmProviderKind::GoogleAi;
        cfg.stt_provider = SttProviderKind::GoogleAi;
        let yaml = serde_yaml::to_string(&cfg).unwrap();
        assert!(yaml.contains("google_ai"));
    }
}
