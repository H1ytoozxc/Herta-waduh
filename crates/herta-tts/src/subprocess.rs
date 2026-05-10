//! Generic subprocess TTS adapter.
//!
//! Many TTS engines (edge-tts, piper, SAPI helpers) can be driven from a
//! small subprocess with a simple stdin-in / stdout-out contract. This type
//! centralizes the plumbing.

use bytes::Bytes;
use herta_core::{
    HertaError, HertaResult,
    audio::AudioFormat,
    tts::{SpeakOptions, SynthesizedAudio},
};
use std::{
    path::PathBuf,
    process::Stdio,
    time::{Duration, Instant},
};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

/// Reusable configuration for [`SubprocessTts`].
#[derive(Debug, Clone)]
pub struct SubprocessTtsConfig {
    /// Provider identifier for metrics / logs (static for lifetime).
    pub provider: &'static str,
    /// Path or command name of the subprocess binary.
    pub binary: PathBuf,
    /// Arguments passed before any per-call overrides.
    pub fixed_args: Vec<String>,
    /// Audio format the subprocess is expected to emit on stdout.
    pub output_format: AudioFormat,
}

/// Subprocess-backed TTS. Not exposed as a `TtsEngine` directly; wrappers
/// like [`crate::edge::EdgeTts`] delegate to it.
#[derive(Debug)]
pub struct SubprocessTts {
    cfg: SubprocessTtsConfig,
}

impl SubprocessTts {
    /// Build a new adapter.
    pub fn new(cfg: SubprocessTtsConfig) -> Self {
        Self { cfg }
    }

    /// Attempt a `--version` probe. Returns `Ok(false)` if the probe fails.
    pub async fn warm_up_inner(&self) -> HertaResult<bool> {
        let output = Command::new(&self.cfg.binary)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .await;
        Ok(output.is_ok_and(|o| o.status.success()))
    }

    /// Synthesize by spawning the configured binary and piping text.
    pub async fn synthesize_inner(
        &self,
        text: &str,
        options: &SpeakOptions,
    ) -> HertaResult<SynthesizedAudio> {
        let mut args = self.cfg.fixed_args.clone();
        if let Some(voice) = &options.voice {
            args.push("--voice".into());
            args.push(voice.clone());
        }
        if let Some(rate) = &options.rate {
            args.push("--rate".into());
            args.push(rate.clone());
        }

        let mut child = Command::new(&self.cfg.binary)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                HertaError::Unavailable(format!(
                    "TTS binary '{}' not executable: {e}",
                    self.cfg.binary.display()
                ))
            })?;
        let started = Instant::now();

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(text.as_bytes())
                .await
                .map_err(|e| HertaError::Audio(format!("stdin write: {e}")))?;
            stdin.shutdown().await.ok();
        }

        let output = child
            .wait_with_output()
            .await
            .map_err(|e| HertaError::Audio(format!("child wait: {e}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(HertaError::provider(
                self.cfg.provider,
                "synthesize",
                stderr,
            ));
        }

        Ok(SynthesizedAudio {
            pcm: Bytes::from(output.stdout),
            format: self.cfg.output_format,
            provider: self.cfg.provider.to_string(),
            latency: started.elapsed().max(Duration::from_millis(0)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use herta_core::audio::SampleFormat;

    fn cfg() -> SubprocessTtsConfig {
        SubprocessTtsConfig {
            provider: "test",
            binary: PathBuf::from("/nonexistent/tts-binary-herta-test"),
            fixed_args: vec!["--fake".into()],
            output_format: AudioFormat {
                sample_rate: 16_000,
                channels: 1,
                sample_format: SampleFormat::I16,
            },
        }
    }

    #[tokio::test]
    async fn missing_binary_surfaces_unavailable() {
        let t = SubprocessTts::new(cfg());
        let err = t
            .synthesize_inner("hello", &SpeakOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, HertaError::Unavailable(_)));
    }

    #[tokio::test]
    async fn warm_up_false_when_binary_missing() {
        let t = SubprocessTts::new(cfg());
        assert!(!t.warm_up_inner().await.unwrap());
    }
}
