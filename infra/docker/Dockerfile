# Multi-stage Dockerfile for The Herta (Rust edition).
#
# Stage 1 — build: compile the release binary with cached cargo registry.
# Stage 2 — runtime: minimal Debian-slim image running as a non-root user.
#
# Build from the workspace root:
#
#   docker build -f infra/docker/Dockerfile -t the-herta:latest .
#
# The resulting image exposes the observability port 9090 and expects a
# config.yaml to be mounted at /etc/herta/config.yaml (or env vars set).

# -----------------------------------------------------------------------------
# Stage 1: builder
# -----------------------------------------------------------------------------
FROM rust:1.85-slim-bookworm AS builder

ARG CARGO_PROFILE=release

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        pkg-config \
        libssl-dev \
        build-essential \
        cmake \
        libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY Cargo.toml ./Cargo.toml
COPY crates ./crates

# Prime the build cache with dependency-only compile first.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/workspace/target \
    cargo fetch

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/workspace/target \
    cargo build --${CARGO_PROFILE} --bin herta \
    && cp target/${CARGO_PROFILE}/herta /workspace/herta

# -----------------------------------------------------------------------------
# Stage 2: runtime
# -----------------------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libssl3 \
        libasound2 \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home-dir /home/herta --shell /usr/sbin/nologin herta \
    && mkdir -p /home/herta /etc/herta /var/lib/herta \
    && chown -R herta:herta /home/herta /etc/herta /var/lib/herta

COPY --from=builder /workspace/herta /usr/local/bin/herta
COPY infra/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER herta
WORKDIR /home/herta

ENV HERTA_CONFIG_FILE=/etc/herta/config.yaml \
    HERTA_SERVER__BIND=0.0.0.0:9090 \
    RUST_BACKTRACE=1 \
    HERTA_TELEMETRY__LOG_FORMAT=json \
    HERTA_TELEMETRY__ENVIRONMENT=production

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD /usr/local/bin/herta --no-server doctor || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]
CMD ["repl", "--no-tts"]
