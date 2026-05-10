# The Herta — Developer Guide

This document is the internal map for contributors. It complements the
user-facing [README](./README.md) and focuses on architecture, conventions,
and how to extend the workspace.

## Crate responsibilities

| Crate                 | Responsibility                                         | Depends on                  |
|-----------------------|--------------------------------------------------------|-----------------------------|
| `herta-core`          | Traits, pipeline, error taxonomy, mocks                | (leaf crate, no workspace deps) |
| `herta-config`        | Strongly-typed `Config`, env/file/defaults loader      | external only               |
| `herta-memory`        | Memory backends (JSON, SQLite, Sled)                   | `herta-core`, `herta-config` |
| `herta-llm`           | LLM providers (Ollama, DeepSeek, Google AI)            | `herta-core`, `herta-config` |
| `herta-stt`           | STT providers + fallback wrapper                       | `herta-core`, `herta-config` |
| `herta-tts`           | TTS providers (Edge, subprocess, noop)                 | `herta-core`, `herta-config` |
| `herta-audio`         | CPAL input/output, VAD, tone helpers                   | `herta-core`, `herta-config` |
| `herta-observability` | Tracing, metrics, health/metrics HTTP server           | `herta-core`, `herta-config` |
| `herta-cli`           | Clap CLI + production entrypoint (`herta` binary)      | all of the above            |

Only `herta-cli` composes concrete providers. Every other crate is a library
and can be reused from another binary (bot, web service, sidecar) without
pulling in CLI machinery.

## Key traits

### `LlmProvider`

```rust
#[async_trait]
pub trait LlmProvider: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    async fn warm_up(&self) -> HertaResult<bool> { Ok(true) }
    async fn generate(&self, ctx: &DialogContext) -> HertaResult<LlmResponse>;
    async fn stream<'a>(&'a self, ctx: &'a DialogContext)
        -> HertaResult<BoxStream<'a, HertaResult<StreamChunk>>>;
}
```

- Must be safe to call concurrently.
- Errors are retryable iff `HertaError::is_retryable()` returns `true`.
- The default `stream` impl wraps `generate`, so streaming providers only
  need to override it if they genuinely push partial tokens.

### `SttEngine`, `TtsEngine`, `Memory`, `AudioInput`, `AudioOutput`

All follow the same pattern: `name()`, optional `warm_up()`, plus the
domain-specific methods. All take `&self` (never `&mut self`) so they compose
cleanly behind `Arc<dyn Trait>`.

### Error taxonomy (`HertaError`)

```rust
Config, Provider, Transport, Auth, RateLimited, Timeout, Cancelled,
Io, Serialization, NotFound, InvalidInput, Audio, Memory, Pipeline,
Unavailable, Internal
```

Rules:
- Every crate funnels upstream errors through `HertaError::provider(...)`.
- Use `HertaError::is_retryable()` as the single source of truth for retry
  decisions.
- Use `kind()` for metric labels; never serialize a full error message as a
  label (unbounded cardinality).

## Adding a new LLM provider

1. Create `crates/herta-llm/src/my_provider.rs`.
2. Define a config struct if the built-in schema doesn't fit; add it to
   `herta-config/src/schema.rs` under `Config`.
3. Implement `LlmProvider`. Use `common::build_http_client` and
   `common::map_status_error` for consistent behavior.
4. Add a Cargo feature in `herta-llm/Cargo.toml`:
   ```toml
   [features]
   default = ["ollama", "deepseek", "google-ai", "my-provider"]
   my-provider = []
   ```
5. Extend `herta-llm::build_from_config` with an arm that constructs your
   provider when `cfg.llm_provider` matches.
6. Add unit tests + a `wiremock`-based integration test.
7. Document the new provider in [`README.md`](./README.md) and add relevant
   env vars to the table.
8. Update the CLI `doctor` command if you introduce a new failure mode.

## Adding a new memory backend

1. Create `crates/herta-memory/src/my_backend.rs`.
2. Implement `Memory`. Use `herta_core::memory::window_messages` for trimming
   to preserve semantic parity.
3. Add a feature flag and an arm in `herta-memory::build_from_config`.
4. Tests must cover roundtrip, trimming, and `clear()`.

## Observability conventions

- Every public async method is annotated with `#[tracing::instrument(level = "info", skip(...))]`
  and carries `correlation_id`, `provider`, and `model` fields where relevant.
- Metrics names follow the `herta_<component>_<operation>_<unit>` convention
  (e.g. `herta_llm_generate_latency_ms`).
- Every transient upstream failure becomes a counter increment labeled with
  `kind` (from `HertaError::kind()`), not with the raw message.

## Testing tiers

| Tier              | Where                          | Criteria                                    |
|-------------------|--------------------------------|---------------------------------------------|
| Unit              | Inline `#[cfg(test)] mod tests`| Single function/module; no IO besides fs    |
| Integration       | `crates/<crate>/tests/*.rs`    | Uses `wiremock`/mocks; no live network      |
| CLI smoke         | `crates/herta-cli/tests/`      | Invokes the compiled binary via `assert_cmd`|
| Property          | `proptest!` blocks             | Invariants over randomized inputs           |

Runtime assumptions:
- Tests must be deterministic. When randomness is required, seed it.
- No test may require a running LLM daemon, audio device, or network.

## Release process

1. Bump `version` in `Cargo.toml` (`workspace.package.version`).
2. Update `CHANGELOG.md` (create if missing).
3. `cargo test --workspace --all-features`.
4. Build and tag the Docker image: `docker build -t ghcr.io/.../the-herta:<ver>`.
5. Tag the git commit and push: `git tag vX.Y.Z && git push --tags`.
6. CI builds multi-platform binaries; attach them to the GitHub release.

## Style

- `rustfmt` defaults (no overrides).
- `clippy::pedantic` is warn-level workspace-wide. Use `#[allow(...)]` with a
  comment when you must override it.
- Every public item has a `///` doc comment. `missing_docs = "warn"` is set
  at the workspace root.
- No `unwrap()` or `panic!` on error paths in non-test code. `expect("msg")`
  is acceptable only when the condition is truly unreachable and the message
  explains why.
