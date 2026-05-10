# The Herta Voice Assistant — Rust Edition

A production-grade, cloud-ready rewrite of
[The Herta Voice Assistant](../README.md) in Rust. The workspace lives under
`rust/` alongside the original Python implementation so you can migrate
incrementally without disrupting the current deployment.

This edition prioritizes reliability, observability, security, and
operability over feature novelty. It ships the same provider set
(Ollama, DeepSeek, Google AI Studio, Whisper-local, Edge-TTS, RVC-compatible
subprocess adapter) behind trait-based abstractions with mocks, tests, and
CI baked in.

## Highlights

- **Rust 2024 edition**, minimum toolchain 1.85.
- **Cargo workspace** with nine crates and clean module boundaries.
- **Async-first** (Tokio), cancellation-safe, structured error taxonomy.
- **Pluggable providers** via traits: `LlmProvider`, `SttEngine`, `TtsEngine`,
  `AudioInput`, `AudioOutput`, `Memory`.
- **Observability**: `tracing` (pretty/compact/JSON), Prometheus metrics,
  `/healthz` + `/ready` + `/metrics` endpoints via `axum`.
- **Configuration**: YAML/TOML + env (`HERTA_*__*` nested) + defaults, with
  validation and secret redaction.
- **Security**: secrets never hardcoded; redacted in log/config dumps;
  runs as non-root in Docker with `readOnlyRootFilesystem`.
- **Tests**: unit, integration, property, and CLI smoke tests. Deterministic
  mocks (`herta_core::mocks`) enable offline CI.
- **CI/CD**: multi-platform GitHub Actions (Linux/macOS/Windows), clippy,
  rustfmt, cargo-audit, docs, Docker build smoke test.
- **Deployment**: multi-stage Dockerfile, Kubernetes manifests with probes,
  PDB, non-root security context.

## Workspace layout

```
rust/
├── Cargo.toml                 # workspace manifest
├── config.yaml                # example configuration
├── crates/
│   ├── herta-core/            # traits, pipeline, errors, mocks
│   ├── herta-config/          # typed Config + env/file/defaults loader
│   ├── herta-memory/          # JSON / SQLite / Sled memory backends
│   ├── herta-llm/             # Ollama, DeepSeek, Google AI providers
│   ├── herta-stt/             # Google AI + fallback + local STT adapters
│   ├── herta-tts/             # Edge-TTS-compatible + subprocess + noop
│   ├── herta-audio/           # CPAL backend, VAD, tone generator
│   ├── herta-observability/   # tracing, metrics, health server
│   └── herta-cli/             # clap-based CLI + production entrypoint
├── infra/
│   ├── docker/Dockerfile      # multi-stage image
│   └── k8s/deployment.yaml    # Deployment + Service + PDB
└── .github/workflows/ci.yml   # CI pipeline
```

## Quick start

### Build & run locally

```sh
cd rust
cargo build --release
./target/release/herta show-config
./target/release/herta text "hello, who are you?" --no-tts
./target/release/herta repl --no-tts
```

### Voice mode

Voice mode needs a working audio input device and (optionally) a TTS binary.
Enable the CPAL backend (default) and run:

```sh
HERTA_LLM_PROVIDER=ollama \
HERTA_OLLAMA__MODEL=qwen3:4b \
./target/release/herta voice --no-tts
```

### Run in a container

```sh
docker build -f infra/docker/Dockerfile -t the-herta:dev .
docker run --rm -p 9090:9090 \
  -e HERTA_LLM_PROVIDER=ollama \
  -e HERTA_OLLAMA__HOST=http://host.docker.internal:11434 \
  the-herta:dev text "hello" --no-tts
```

### Deploy to Kubernetes

Edit `infra/k8s/deployment.yaml` to point at your image registry, then:

```sh
kubectl apply -f infra/k8s/deployment.yaml
kubectl -n the-herta port-forward svc/the-herta 9090:9090
curl http://localhost:9090/healthz
```

## CLI commands

| Command                | Description                                    |
|------------------------|------------------------------------------------|
| `voice`                | Live mic → VAD → STT → LLM → TTS loop          |
| `text <prompt>`        | Run a single turn and print the reply          |
| `repl`                 | Interactive text REPL                          |
| `show-config [--format]` | Print effective configuration (secrets redacted) |
| `list-input-devices`   | Enumerate audio input devices                  |
| `list-output-devices`  | Enumerate audio output devices                 |
| `output-test`          | Play a short tone through the output device    |
| `tts-test [--text]`    | Speak a short calibration sentence             |
| `doctor`               | Run a production readiness self-check          |

Global flags: `--config PATH`, `--log-level LEVEL`, `--json-logs`,
`--no-server`, `--non-interactive`.

## Configuration

The loader merges sources in this precedence (higher wins):

1. Environment variables prefixed with `HERTA_`. Nested fields use `__`
   separators, e.g. `HERTA_MEMORY__MAX_MESSAGES=100`.
2. A YAML/TOML/JSON config file (via `--config` / `HERTA_CONFIG_FILE`, or
   auto-discovered `config.yaml|yml|toml|json` in the working directory).
3. Compile-time defaults from [`herta_config::schema::Config`].

See [`config.yaml`](./config.yaml) for every available field with comments.

### Environment variable reference (common)

| Env var                                | Maps to                              |
|----------------------------------------|--------------------------------------|
| `HERTA_LOG_LEVEL`                      | `log_level`                          |
| `HERTA_LLM_PROVIDER`                   | `llm_provider`                       |
| `HERTA_STT_PROVIDER`                   | `stt_provider`                       |
| `HERTA_OLLAMA__HOST`                   | `ollama.host`                        |
| `HERTA_OLLAMA__MODEL`                  | `ollama.model`                       |
| `HERTA_DEEPSEEK__API_KEY`              | `deepseek.api_key` (secret)          |
| `HERTA_GOOGLE_AI__API_KEY`             | `google_ai.api_key` (secret)        |
| `HERTA_GOOGLE_STT__API_KEY`            | `google_stt.api_key` (secret)        |
| `HERTA_MEMORY__BACKEND`                | `memory.backend`                     |
| `HERTA_MEMORY__PATH`                   | `memory.path`                        |
| `HERTA_TELEMETRY__LOG_FORMAT`          | `telemetry.log_format`               |
| `HERTA_SERVER__BIND`                   | `server.bind`                        |

Secrets are *only* accepted via environment variables in production. Config
file secrets are tolerated for local dev but are **not** recommended.

## Architecture

```
 ┌──────────────┐     ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
 │ AudioInput   │ ──► │  VAD     │──► │  STT     │──► │  LLM     │──► │  TTS     │
 │ (cpal/mock)  │     │ Energy/  │    │ Google/  │    │ Ollama/  │    │ Edge/    │
 │              │     │  Silero  │    │ Whisper  │    │ DeepSeek │    │ subproc  │
 └──────────────┘     └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                           │               │
                                                           ▼               ▼
                                                      ┌──────────┐    ┌──────────┐
                                                      │ Memory   │    │AudioOutput│
                                                      │ JSON/    │    │  (cpal)  │
                                                      │ SQLite/  │    │          │
                                                      │ Sled     │    │          │
                                                      └──────────┘    └──────────┘
```

`VoicePipeline` wires everything together and emits `PipelineEvent`s on a
broadcast channel for dashboards, CLIs, or downstream hooks.

## Observability

- Structured logs (`tracing`), with per-turn correlation IDs and span context.
- Prometheus metrics via `metrics` facade; scrape at `/metrics`.
- Health probes at `/healthz` (liveness) and `/ready` (readiness). `/ready`
  returns 503 when any registered `HealthProbe` is degraded, so Kubernetes
  won't route traffic to a pod that has lost its LLM backend.

## Testing

```sh
cargo test --workspace --all-features
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo doc --workspace --no-deps --all-features
```

Every crate includes unit tests; `herta-core` + `herta-config` + `herta-cli`
also ship integration tests under `tests/`. Property tests cover
`window_messages` and retry backoff bounds. LLM/STT clients are covered with
`wiremock` so the suite runs without network access.

## Migration from the Python stack

The Rust edition is feature-parity for the text, Ollama, DeepSeek, Google AI,
memory, and Edge-TTS paths. The *Google Live* native-audio path and *Applio
RVC* worker are stubbed behind the `TtsEngine` / `AudioOutput` traits; port
them later by implementing those traits without touching the pipeline.

Recommended rollout:

1. **Phase 0 — shadow traffic.** Keep the Python process in production.
   Deploy the Rust `text` mode to staging with the same config file; compare
   replies for correctness.
2. **Phase 1 — text traffic in prod.** Route a fraction of text traffic to
   the Rust CLI behind a feature flag (e.g. tenant id). Watch metrics
   (`herta_llm_*`, `herta_memory_*`).
3. **Phase 2 — memory cutover.** Migrate JSON memory files in place (schema
   is identical; Rust reads existing `dialogue_memory.json`).
4. **Phase 3 — voice.** Enable voice mode on operator workstations; CPAL
   supports WASAPI/CoreAudio/ALSA.
5. **Phase 4 — retire Python.** Once SLOs hold for a burn-in period.

### Rollback plan

- Each binary is a single self-contained artifact. Rollback is a single
  `docker image pull <previous-tag>` or `kubectl rollout undo`.
- Memory schema is append-only JSON; the Python implementation can continue
  reading files the Rust edition wrote.
- Configuration format is backward-compatible: all Python env var names have
  direct Rust equivalents listed in the env-var reference above.

## Security

- `#![forbid(unsafe_code)]` is intentionally **not** global: CPAL requires
  FFI. Instead we enforce `unsafe_code = "deny"` as a workspace lint with
  documented, narrow exceptions where unavoidable.
- All HTTP clients use rustls, explicit timeouts, and pooled connections.
- Secrets are `Option<String>`, redacted in `show-config` output, and marked
  sensitive in HTTP headers.
- Docker image runs as UID 10001, with `readOnlyRootFilesystem`, no Linux
  capabilities, and a `seccompProfile: RuntimeDefault`.

## Contributing

Start with `cargo test -p herta-core` to iterate on the trait surface, then
branch out into the crate corresponding to your change. New providers should:

1. Implement the relevant trait (`LlmProvider`, `SttEngine`, `TtsEngine`, etc.).
2. Provide a `build_from_config` arm + feature flag.
3. Ship at least unit + `wiremock`-based integration tests.
4. Update the CLI `doctor` command if it introduces a new failure mode.

License: dual MIT / Apache-2.0.
