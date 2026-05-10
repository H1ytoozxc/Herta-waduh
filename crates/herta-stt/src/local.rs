//! Local Whisper STT adapter.
//!
//! This is a lightweight, production-oriented placeholder: the Python
//! implementation used `faster-whisper` (`CTranslate2`) which requires a
//! sizable native dependency. Rather than bundling that at workspace build
//! time, we invoke an external binary (e.g. `whisper` / `whisper.cpp`) via
//! a subprocess. The subprocess contract is:
//!
//! - Read WAV bytes from stdin.
//! - Write the transcription as UTF-8 text on stdout.
//! - Exit code `0` on success.
//!
//! Any concrete engine that satisfies this contract can be swapped in via
//! the `WHISPER_BINARY` environment variable or the `stt.binary` config
//! field. If the configured binary is not present, `warm_up` returns
//! `Ok(false)` and `transcribe` returns `HertaError::Unavailable`.

use crate::wav;
use async_trait::async_trait;
use herta_config::schema::WhisperSttConfig;
use herta_core::{
    HertaError, HertaResult,
    audio::Utterance,
    stt::{SttEngine, Transcript},
};
use std::{
    path::PathBuf,
    process::Stdio,
    time::{Duration, Instant},
};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

const PROVIDER: &str = "whisper_local";

/// Local Whisper STT via an external binary.
#[derive(Debug, Clone)]
pub struct LocalWhisperStt {
    cfg: WhisperSttConfig,
    binary: PathBuf,
}

impl LocalWhisperStt {
    /// Build a new local STT engine.
    pub fn new(cfg: WhisperSttConfig) -> HertaResult<Self> {
        let binary = std::env::var("WHISPER_BINARY")
            .map_or_else(|_| PathBuf::from("whisper"), PathBuf::from);
        Ok(Self { cfg, binary })
    }

    fn args(&self) -> Vec<String> {
        let mut args = vec![
            "--model".into(),
            self.cfg.model_size.clone(),
            "--output".into(),
            "txt".into(),
            "--stdin".into(),
            "--stdout".into(),
        ];
        if let Some(lang) = &self.cfg.language {
            args.push("--language".into());
            args.push(lang.clone());
        }
        args.push("--beam-size".into());
        args.push(self.cfg.beam_size.to_string());
        args.push("--threads".into());
        args.push(self.cfg.cpu_threads.to_string());
        args
    }
}

#[async_trait]
impl SttEngine for LocalWhisperStt {
    fn name(&self) -> &'static str {
        PROVIDER
    }

    fn active_device(&self) -> String {
        format!(
            "whisper:{}:{}",
            self.cfg.model_size.trim(),
            self.cfg.device
        )
    }

    async fn warm_up(&self) -> HertaResult<bool> {
        let output = Command::new(&self.binary)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
            .await;
        Ok(output.is_ok_and(|o| o.status.success()))
    }

    #[tracing::instrument(level = "info", skip(self, utterance), fields(
        provider = PROVIDER,
        model = %self.cfg.model_size,
        ms = utterance.duration_ms
    ))]
    async fn transcribe(&self, utterance: &Utterance) -> HertaResult<Transcript> {
        let wav = wav::encode_wav(utterance)?;
        let mut child = Command::new(&self.binary)
            .args(self.args())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                HertaError::Unavailable(format!(
                    "whisper binary '{}' not executable: {e}",
                    self.binary.display()
                ))
            })?;

        let started = Instant::now();
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(&wav)
                .await
                .map_err(|e| HertaError::Audio(format!("stdin write: {e}")))?;
        }
        let output = child
            .wait_with_output()
            .await
            .map_err(|e| HertaError::Audio(format!("whisper wait: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(HertaError::provider(PROVIDER, "transcribe", stderr));
        }

        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(Transcript {
            text,
            language: self.cfg.language.clone(),
            confidence: None,
            latency: started.elapsed().max(Duration::from_millis(0)),
            provider: PROVIDER.into(),
        })
    }
}

#[cfg(test)]
#[allow(unsafe_code)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn warm_up_false_when_binary_missing() {
        // Override env to something non-existent.
        // Safety: test-only; no other threads read this env in this test.
        unsafe {
            std::env::set_var(
                "WHISPER_BINARY",
                "/nonexistent/whisper-xyz-herta-test-binary",
            );
        }
        let engine = LocalWhisperStt::new(WhisperSttConfig::default()).unwrap();
        assert!(!engine.warm_up().await.unwrap());
    }

    #[test]
    fn args_include_beam_and_threads() {
        unsafe {
            std::env::set_var(
                "WHISPER_BINARY",
                "/nonexistent/whisper-xyz-herta-test-binary",
            );
        }
        let engine = LocalWhisperStt::new(WhisperSttConfig::default()).unwrap();
        let a = engine.args();
        assert!(a.iter().any(|x| x == "--beam-size"));
        assert!(a.iter().any(|x| x == "--threads"));
    }
}
