//! CPAL-based production audio backend.
//!
//! This backend maps the abstract [`AudioInput`]/[`AudioOutput`] traits onto
//! the [`cpal`] crate and works on Windows (WASAPI), macOS (`CoreAudio`),
//! Linux (ALSA/Pulse), and additional backends supported by CPAL.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::needless_pass_by_value
)]

use async_trait::async_trait;
use bytes::Bytes;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use herta_config::schema::{AudioInputConfig, AudioOutputConfig};
use herta_core::{
    HertaError, HertaResult,
    audio::{
        AudioChunk, AudioFormat, AudioInput, AudioOutput, DeviceEnumerator, DeviceInfo,
        SampleFormat,
    },
};
use parking_lot::Mutex;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::SystemTime,
};
use tokio::sync::mpsc;

use crate::tone::generate_tone;

const CPAL_PROVIDER_IN: &str = "cpal-in";
const CPAL_PROVIDER_OUT: &str = "cpal-out";

/// CPAL-backed device enumerator.
#[derive(Debug, Clone, Default)]
pub struct CpalDeviceEnumerator;

impl DeviceEnumerator for CpalDeviceEnumerator {
    fn list_input_devices(&self) -> HertaResult<Vec<DeviceInfo>> {
        let host = cpal::default_host();
        let devices = host
            .input_devices()
            .map_err(|e| HertaError::audio(format!("enumerate input: {e}")))?;
        Ok(collect_devices(devices, true))
    }
    fn list_output_devices(&self) -> HertaResult<Vec<DeviceInfo>> {
        let host = cpal::default_host();
        let devices = host
            .output_devices()
            .map_err(|e| HertaError::audio(format!("enumerate output: {e}")))?;
        Ok(collect_devices(devices, false))
    }
}

fn collect_devices(
    iter: impl Iterator<Item = cpal::Device>,
    kind_is_input: bool,
) -> Vec<DeviceInfo> {
    iter.enumerate()
        .map(|(index, dev)| {
            let name = dev.name().unwrap_or_else(|_| "<unnamed>".into());
            let (max_in, max_out, rate) = if kind_is_input {
                let cfg = dev.default_input_config().ok();
                let channels = cfg.as_ref().map_or(0, cpal::SupportedStreamConfig::channels);
                let rate = cfg.as_ref().map(|c| c.sample_rate().0);
                (channels, 0, rate)
            } else {
                let cfg = dev.default_output_config().ok();
                let channels = cfg.as_ref().map_or(0, cpal::SupportedStreamConfig::channels);
                let rate = cfg.as_ref().map(|c| c.sample_rate().0);
                (0, channels, rate)
            };
            DeviceInfo {
                index: index as u32,
                name,
                max_input_channels: max_in,
                max_output_channels: max_out,
                default_sample_rate: rate,
            }
        })
        .collect()
}

/// Find a device by name substring; case-insensitive.
fn find_device_by_name(
    host: &cpal::Host,
    name_substr: &str,
    input: bool,
) -> HertaResult<Option<cpal::Device>> {
    let needle = name_substr.to_lowercase();
    let iter: Box<dyn Iterator<Item = cpal::Device>> = if input {
        Box::new(
            host.input_devices()
                .map_err(|e| HertaError::audio(format!("input devices: {e}")))?,
        )
    } else {
        Box::new(
            host.output_devices()
                .map_err(|e| HertaError::audio(format!("output devices: {e}")))?,
        )
    };
    for dev in iter {
        if let Ok(name) = dev.name()
            && name.to_lowercase().contains(&needle)
        {
            return Ok(Some(dev));
        }
    }
    Ok(None)
}

fn select_input_device(
    host: &cpal::Host,
    preferred: Option<&str>,
) -> HertaResult<cpal::Device> {
    if let Some(name) = preferred
        && !name.is_empty()
        && let Some(dev) = find_device_by_name(host, name, true)?
    {
        return Ok(dev);
    }
    host.default_input_device()
        .ok_or_else(|| HertaError::audio("no default audio input device available"))
}

fn select_output_device(
    host: &cpal::Host,
    preferred: Option<&str>,
) -> HertaResult<cpal::Device> {
    if let Some(name) = preferred
        && !name.is_empty()
        && let Some(dev) = find_device_by_name(host, name, false)?
    {
        return Ok(dev);
    }
    host.default_output_device()
        .ok_or_else(|| HertaError::audio("no default audio output device available"))
}

/// CPAL audio input.
pub struct CpalAudioInput {
    cfg: AudioInputConfig,
    sender: mpsc::Sender<AudioChunk>,
    receiver: Arc<tokio::sync::Mutex<mpsc::Receiver<AudioChunk>>>,
    running: Arc<AtomicBool>,
    thread: Mutex<Option<thread::JoinHandle<()>>>,
}

impl std::fmt::Debug for CpalAudioInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpalAudioInput")
            .field("cfg", &self.cfg)
            .field("running", &self.running.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

impl CpalAudioInput {
    /// Build a new CPAL audio input.
    pub fn new(cfg: AudioInputConfig) -> HertaResult<Self> {
        let (tx, rx) = mpsc::channel(cfg.queue_max_chunks.max(8) as usize);
        Ok(Self {
            cfg,
            sender: tx,
            receiver: Arc::new(tokio::sync::Mutex::new(rx)),
            running: Arc::new(AtomicBool::new(false)),
            thread: Mutex::new(None),
        })
    }
}

#[async_trait]
impl AudioInput for CpalAudioInput {
    fn name(&self) -> &'static str {
        CPAL_PROVIDER_IN
    }

    async fn start(&self) -> HertaResult<()> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        let cfg = self.cfg.clone();
        let sender = self.sender.clone();
        let running = self.running.clone();

        let handle = thread::Builder::new()
            .name("herta-audio-in".into())
            .spawn(move || {
                if let Err(err) = run_input_stream(cfg, sender, running.clone()) {
                    tracing::error!(error = %err, "audio input stream exited with error");
                }
                running.store(false, Ordering::SeqCst);
            })
            .map_err(|e| HertaError::audio(format!("spawn audio input thread: {e}")))?;

        *self.thread.lock() = Some(handle);
        Ok(())
    }

    async fn stop(&self) -> HertaResult<()> {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.thread.lock().take() {
            let _ = handle.join();
        }
        Ok(())
    }

    async fn next_chunk(&self) -> HertaResult<Option<AudioChunk>> {
        Ok(self.receiver.lock().await.recv().await)
    }
}

fn run_input_stream(
    cfg: AudioInputConfig,
    sender: mpsc::Sender<AudioChunk>,
    running: Arc<AtomicBool>,
) -> HertaResult<()> {
    let host = cpal::default_host();
    let device = select_input_device(&host, cfg.device.as_deref())?;
    let supported = device
        .default_input_config()
        .map_err(|e| HertaError::audio(format!("input cfg: {e}")))?;

    let target_rate = cpal::SampleRate(cfg.sample_rate);
    let stream_config = cpal::StreamConfig {
        channels: cfg.channels.max(1),
        sample_rate: target_rate,
        buffer_size: cpal::BufferSize::Fixed(cfg.block_size.max(128)),
    };

    let format = AudioFormat {
        sample_rate: cfg.sample_rate,
        channels: cfg.channels.max(1),
        sample_format: SampleFormat::F32,
    };

    let err_fn = |e| tracing::warn!("cpal input error: {e}");
    let running_clone = running.clone();

    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _info| {
                if !running_clone.load(Ordering::Relaxed) {
                    return;
                }
                let bytes = f32_slice_to_bytes(data);
                let chunk = AudioChunk {
                    pcm: Bytes::from(bytes),
                    format,
                    captured_at_ms: now_unix_ms(),
                };
                let _ = sender.try_send(chunk);
            },
            err_fn,
            None,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _info| {
                if !running_clone.load(Ordering::Relaxed) {
                    return;
                }
                let mut buf = Vec::with_capacity(data.len() * 4);
                for s in data {
                    let f = f32::from(*s) / f32::from(i16::MAX);
                    buf.extend_from_slice(&f.to_le_bytes());
                }
                let chunk = AudioChunk {
                    pcm: Bytes::from(buf),
                    format,
                    captured_at_ms: now_unix_ms(),
                };
                let _ = sender.try_send(chunk);
            },
            err_fn,
            None,
        ),
        other => {
            return Err(HertaError::audio(format!(
                "unsupported CPAL sample format: {other:?}"
            )));
        }
    }
    .map_err(|e| HertaError::audio(format!("build input stream: {e}")))?;

    stream
        .play()
        .map_err(|e| HertaError::audio(format!("start input stream: {e}")))?;

    while running.load(Ordering::Relaxed) {
        thread::sleep(std::time::Duration::from_millis(50));
    }
    drop(stream);
    Ok(())
}

fn f32_slice_to_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX))
}

/// CPAL audio output.
pub struct CpalAudioOutput {
    cfg: AudioOutputConfig,
}

impl std::fmt::Debug for CpalAudioOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpalAudioOutput").field("cfg", &self.cfg).finish()
    }
}

impl CpalAudioOutput {
    /// Build a new CPAL audio output.
    pub fn new(cfg: AudioOutputConfig) -> HertaResult<Self> {
        Ok(Self { cfg })
    }
}

#[async_trait]
impl AudioOutput for CpalAudioOutput {
    fn name(&self) -> &'static str {
        CPAL_PROVIDER_OUT
    }

    async fn play(&self, pcm: Bytes, format: AudioFormat) -> HertaResult<()> {
        let cfg = self.cfg.clone();
        tokio::task::spawn_blocking(move || play_blocking(&cfg, pcm, format))
            .await
            .map_err(|e| HertaError::audio(format!("join audio: {e}")))?
    }

    async fn play_test_tone(&self) -> HertaResult<()> {
        let format = AudioFormat {
            sample_rate: self.cfg.sample_rate,
            channels: self.cfg.channels.max(1),
            sample_format: SampleFormat::F32,
        };
        let pcm = generate_tone(
            format,
            self.cfg.tone_frequency_hz,
            self.cfg.tone_duration_seconds,
            self.cfg.tone_volume,
        );
        self.play(Bytes::from(pcm), format).await
    }
}

fn play_blocking(
    cfg: &AudioOutputConfig,
    pcm: Bytes,
    format: AudioFormat,
) -> HertaResult<()> {
    let host = cpal::default_host();
    let device = select_output_device(&host, cfg.device.as_deref())?;
    let supported = device
        .default_output_config()
        .map_err(|e| HertaError::audio(format!("output cfg: {e}")))?;

    let stream_config = cpal::StreamConfig {
        channels: supported.channels(),
        sample_rate: cpal::SampleRate(format.sample_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    let samples_per_channel = pcm_to_f32_samples(&pcm, format);
    let total_frames = samples_per_channel.len() / usize::from(format.channels.max(1));
    let done = Arc::new(AtomicBool::new(false));
    let done_clone = done.clone();
    let pos = Arc::new(Mutex::new(0usize));
    let pos_clone = pos.clone();
    let input_channels = usize::from(format.channels.max(1));
    let out_channels = usize::from(stream_config.channels.max(1));

    let err_fn = |e| tracing::warn!("cpal output error: {e}");

    let stream = device
        .build_output_stream(
            &stream_config,
            move |data: &mut [f32], _info| {
                let mut p = pos_clone.lock();
                let needed_frames = data.len() / out_channels;
                for frame in 0..needed_frames {
                    if *p >= total_frames {
                        for c in 0..out_channels {
                            data[frame * out_channels + c] = 0.0;
                        }
                    } else {
                        for c in 0..out_channels {
                            let src_ch = c.min(input_channels - 1);
                            let src_idx = *p * input_channels + src_ch;
                            data[frame * out_channels + c] =
                                samples_per_channel[src_idx];
                        }
                        *p += 1;
                    }
                }
                if *p >= total_frames {
                    done_clone.store(true, Ordering::SeqCst);
                }
            },
            err_fn,
            None,
        )
        .map_err(|e| HertaError::audio(format!("build output stream: {e}")))?;

    stream
        .play()
        .map_err(|e| HertaError::audio(format!("start output stream: {e}")))?;

    while !done.load(Ordering::Relaxed) {
        thread::sleep(std::time::Duration::from_millis(20));
    }
    // Give the device a moment to drain.
    thread::sleep(std::time::Duration::from_millis(50));
    drop(stream);
    Ok(())
}

fn pcm_to_f32_samples(pcm: &[u8], format: AudioFormat) -> Vec<f32> {
    match format.sample_format {
        SampleFormat::F32 => {
            let mut out = Vec::with_capacity(pcm.len() / 4);
            for chunk in pcm.chunks_exact(4) {
                out.push(f32::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                ]));
            }
            out
        }
        SampleFormat::I16 => {
            let mut out = Vec::with_capacity(pcm.len() / 2);
            for chunk in pcm.chunks_exact(2) {
                let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(f32::from(s) / f32::from(i16::MAX));
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerator_does_not_panic() {
        let enumerator = CpalDeviceEnumerator;
        let _ = enumerator.list_input_devices();
        let _ = enumerator.list_output_devices();
    }

    #[test]
    fn pcm_round_trip_f32() {
        let format = AudioFormat {
            sample_rate: 16_000,
            channels: 1,
            sample_format: SampleFormat::F32,
        };
        let samples = vec![0.1f32, -0.2, 0.3];
        let mut bytes = Vec::new();
        for s in &samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let decoded = pcm_to_f32_samples(&bytes, format);
        assert_eq!(decoded.len(), 3);
        assert!((decoded[0] - 0.1).abs() < 1e-6);
    }
}
