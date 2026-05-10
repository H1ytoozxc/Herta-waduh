//! Command implementations.

use crate::app::{Cli, Command};
use anyhow::{Context, Result, anyhow};
use herta_audio::{CpalDeviceEnumerator, DeviceEnumerator};
use herta_config::{LoadOptions, redact_secret, schema::Config};
use herta_core::{
    Message,
    pipeline::{PipelineConfig, VoicePipeline},
};
use herta_observability::{init_tracing, TracingGuard};
use std::{io::Write, sync::Arc};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio_util::sync::CancellationToken;

/// Entry point used by `main.rs`. Dispatches to the subcommand implementations.
pub async fn run(cli: Cli) -> Result<i32> {
    let mut cfg = load_config(&cli)?;
    apply_cli_overrides(&cli, &mut cfg);

    let _guard = init_tracing(&cfg.telemetry, &cfg.log_level);
    let _ = herta_observability::metrics::init_prometheus();

    // Kick off observability server (non-blocking) unless disabled.
    let cancel = CancellationToken::new();
    let server_handle = if cfg.server.enabled && !cli.no_server {
        let token = cancel.clone();
        let server = herta_observability::ObservabilityServer::new(cfg.server.clone());
        Some(tokio::spawn(async move {
            if let Err(e) = server.run(token).await {
                tracing::warn!(error = %e, "observability server exited");
            }
        }))
    } else {
        None
    };

    let exit = match cli.command {
        Command::ShowConfig { format } => show_config(&cfg, &format),
        Command::ListInputDevices => list_input_devices(),
        Command::ListOutputDevices => list_output_devices(),
        Command::OutputTest => output_test(&cfg).await,
        Command::TtsTest { text } => tts_test(&cfg, &text).await,
        Command::Text { prompt, no_tts } => run_text(&cfg, &prompt, no_tts).await,
        Command::Repl { no_tts } => run_repl(&cfg, no_tts, cancel.clone()).await,
        Command::Voice { no_tts } => run_voice(&cfg, no_tts, cancel.clone()).await,
        Command::Doctor => run_doctor(&cfg).await,
    };

    cancel.cancel();
    if let Some(h) = server_handle {
        let _ = h.await;
    }
    exit
}

fn load_config(cli: &Cli) -> Result<Config> {
    let options = LoadOptions {
        path: cli.config.clone(),
        skip_dotenv: false,
        env_prefix: None,
    };
    Config::load_with(&options).context("failed to load configuration")
}

fn apply_cli_overrides(cli: &Cli, cfg: &mut Config) {
    if let Some(level) = &cli.log_level {
    cfg.log_level.clone_from(level);
    }
    if cli.json_logs {
        cfg.telemetry.log_format = herta_config::schema::LogFormat::Json;
    }
    if cli.no_server {
        cfg.server.enabled = false;
    }
}

fn show_config(cfg: &Config, format: &str) -> Result<i32> {
    let mut cfg = cfg.clone();
    redact_in_place(&mut cfg);
    let out = match format {
        "yaml" | "yml" => serde_yaml::to_string(&cfg)?,
        "toml" => toml::to_string_pretty(&cfg)?,
        "json" => serde_json::to_string_pretty(&cfg)?,
        other => return Err(anyhow!("unsupported format: {other}")),
    };
    println!("{out}");
    Ok(0)
}

fn redact_in_place(cfg: &mut Config) {
    if let Some(key) = cfg.deepseek.api_key.as_mut() {
        *key = redact_secret(key);
    }
    if let Some(key) = cfg.google_ai.api_key.as_mut() {
        *key = redact_secret(key);
    }
    if let Some(key) = cfg.google_stt.api_key.as_mut() {
        *key = redact_secret(key);
    }
}

fn list_input_devices() -> Result<i32> {
    let enumerator = CpalDeviceEnumerator;
    let devices = enumerator
        .list_input_devices()
        .context("failed to enumerate input devices")?;
    if devices.is_empty() {
        eprintln!("No input devices found.");
        return Ok(1);
    }
    for d in devices {
        println!(
            "#{:<3} {:<48} max_in={:<2} rate={:?}",
            d.index, d.name, d.max_input_channels, d.default_sample_rate
        );
    }
    Ok(0)
}

fn list_output_devices() -> Result<i32> {
    let enumerator = CpalDeviceEnumerator;
    let devices = enumerator
        .list_output_devices()
        .context("failed to enumerate output devices")?;
    if devices.is_empty() {
        eprintln!("No output devices found.");
        return Ok(1);
    }
    for d in devices {
        println!(
            "#{:<3} {:<48} max_out={:<2} rate={:?}",
            d.index, d.name, d.max_output_channels, d.default_sample_rate
        );
    }
    Ok(0)
}

async fn output_test(cfg: &Config) -> Result<i32> {
    let output = herta_audio::build_output(&cfg.audio_output)?;
    output.play_test_tone().await?;
    println!("Output tone test complete.");
    Ok(0)
}

async fn tts_test(cfg: &Config, text: &str) -> Result<i32> {
    let tts = herta_tts::build_from_config(cfg)?
        .ok_or_else(|| anyhow!("TTS is disabled in configuration"))?;
    let output = herta_audio::build_output(&cfg.audio_output)?;
    let audio = tts
        .synthesize(text, &herta_core::tts::SpeakOptions::default())
        .await?;
    output.play(audio.pcm, audio.format).await?;
    println!("{text}");
    Ok(0)
}

fn build_pipeline(
    cfg: &Config,
    enable_tts: bool,
    enable_stt: bool,
) -> Result<VoicePipeline> {
    let llm = herta_llm::build_from_config(cfg)?;
    let memory = herta_memory::build_from_config(&cfg.memory)?
        .unwrap_or_else(|| Arc::new(herta_core::mocks::InMemoryMemory::default()));

    let mut builder = VoicePipeline::builder()
        .llm(llm)
        .memory(memory)
        .config(PipelineConfig {
            context_messages: cfg.memory.context_messages,
            locked_prefix: vec![Message::system(build_default_persona())],
            tenant_id: None,
        });

    if enable_tts && cfg.tts.enabled
        && let Some(tts) = herta_tts::build_from_config(cfg)?
    {
        builder = builder.tts(tts);
        let output = herta_audio::build_output(&cfg.audio_output)?;
        builder = builder.output(output);
    }
    if enable_stt {
        let stt = herta_stt::build_from_config(cfg)?;
        builder = builder.stt(stt);
    }
    Ok(builder.build()?)
}

fn build_default_persona() -> String {
    "You are The Herta, a calm and concise bilingual voice assistant. \
     Respond in the user's language. Be brief and helpful."
        .to_string()
}

async fn run_text(cfg: &Config, prompt: &str, no_tts: bool) -> Result<i32> {
    let pipeline = build_pipeline(cfg, !no_tts, false)?;
    let reply = pipeline.run_text_turn(prompt, None).await?;
    println!("{reply}");
    Ok(0)
}

async fn run_repl(cfg: &Config, no_tts: bool, cancel: CancellationToken) -> Result<i32> {
    let pipeline = build_pipeline(cfg, !no_tts, false)?;
    println!(
        "The Herta assistant ready (provider={}). Type a message or 'exit' to quit.",
        pipeline.llm_name()
    );

    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();

    loop {
        if cancel.is_cancelled() {
            break;
        }
        print!("you> ");
        std::io::stdout().flush().ok();
        line.clear();
        let n = match reader.read_line(&mut line).await {
            Ok(n) => n,
            Err(e) => {
                tracing::warn!(error = %e, "stdin read failed");
                break;
            }
        };
        if n == 0 {
            break; // EOF
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if matches!(input.to_lowercase().as_str(), "exit" | "quit" | "q") {
            break;
        }
        match pipeline.run_text_turn(input, Some(&cancel)).await {
            Ok(reply) => println!("herta> {reply}"),
            Err(err) => eprintln!("error: {err}"),
        }
    }
    Ok(0)
}

async fn run_voice(
    cfg: &Config,
    no_tts: bool,
    cancel: CancellationToken,
) -> Result<i32> {
    let pipeline = build_pipeline(cfg, !no_tts, true)?;

    let input = herta_audio::build_input(&cfg.audio)?;
    input.start().await?;

    let mut vad = herta_audio::EnergyVad::new(cfg.vad.clone());
    println!(
        "Voice mode ready (provider={}). Speak into the microphone. Press Ctrl+C to stop.",
        pipeline.llm_name()
    );

    let ctrl_c = {
        let token = cancel.clone();
        tokio::spawn(async move {
            let _ = tokio::signal::ctrl_c().await;
            token.cancel();
        })
    };

    loop {
        if cancel.is_cancelled() {
            break;
        }
        let chunk = tokio::select! {
            c = input.next_chunk() => c?,
            () = cancel.cancelled() => break,
        };
        let Some(chunk) = chunk else { break };
        if let Some(utt) = herta_audio::Vad::process_chunk(&mut vad, chunk)? {
            match pipeline.run_voice_turn(&utt, Some(&cancel)).await {
                Ok(Some(reply)) => println!("herta> {reply}"),
                Ok(None) => {}
                Err(err) => eprintln!("error: {err}"),
            }
        }
    }

    input.stop().await?;
    ctrl_c.abort();
    Ok(0)
}

async fn run_doctor(cfg: &Config) -> Result<i32> {
    let mut exit_code = 0;
    println!("== The Herta doctor ==");
    println!(
        "llm_provider = {:?}, stt_provider = {:?}",
        cfg.llm_provider, cfg.stt_provider
    );

    match herta_llm::build_from_config(cfg) {
        Ok(llm) => match llm.warm_up().await {
            Ok(true) => println!("llm: ok (provider={})", llm.name()),
            Ok(false) => {
                println!("llm: WARN (warm-up inconclusive)");
                exit_code = 2;
            }
            Err(err) => {
                println!("llm: FAIL ({err})");
                exit_code = 1;
            }
        },
        Err(err) => {
            println!("llm: FAIL ({err})");
            exit_code = 1;
        }
    }

    match herta_memory::build_from_config(&cfg.memory) {
        Ok(Some(mem)) => println!("memory: ok (backend={})", mem.name()),
        Ok(None) => println!("memory: disabled"),
        Err(err) => {
            println!("memory: FAIL ({err})");
            exit_code = 1;
        }
    }

    match herta_stt::build_from_config(cfg) {
        Ok(stt) => match stt.warm_up().await {
            Ok(true) => println!("stt: ok (provider={})", stt.name()),
            Ok(false) => println!("stt: WARN (warm-up inconclusive)"),
            Err(err) => println!("stt: WARN ({err})"),
        },
        Err(err) => println!("stt: skipped ({err})"),
    }

    match herta_tts::build_from_config(cfg) {
        Ok(Some(tts)) => match tts.warm_up().await {
            Ok(true) => println!("tts: ok (provider={})", tts.name()),
            Ok(false) => println!("tts: WARN (warm-up inconclusive)"),
            Err(err) => println!("tts: WARN ({err})"),
        },
        Ok(None) => println!("tts: disabled"),
        Err(err) => println!("tts: skipped ({err})"),
    }

    println!("exit = {exit_code}");
    Ok(exit_code)
}

/// Keep this alive so the tracing guard drops at the right time. Caller moves
/// it into their `main`.
#[allow(dead_code)]
struct _KeepAlive(TracingGuard);
