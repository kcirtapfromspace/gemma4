# Multi-stage build for Jetson Orin Nano (ARM64)
# Build context must be repo root (for data/eicr-samples access at compile time)
# Usage: docker build -f apps/demo-app-rs/Dockerfile .
FROM rust:1-bookworm AS builder

RUN apt-get update && apt-get install -y cmake g++ pkg-config libssl-dev protobuf-compiler

WORKDIR /app

# Copy Cargo files and build deps with a dummy main for layer caching
COPY apps/demo-app-rs/Cargo.toml apps/demo-app-rs/Cargo.lock* ./
RUN mkdir -p src && echo 'fn main(){}' > src/main.rs && cargo build --release 2>/dev/null || true
# Remove the dummy binary so cargo rebuilds with real source
RUN rm -f target/release/cliniq target/release/deps/cliniq-*

# Copy real source + data (needed by include_str! at compile time)
COPY apps/demo-app-rs/src/ src/
COPY apps/demo-app-rs/static/ static/
COPY data/eicr-samples/ ../data/eicr-samples/

RUN cargo build --release && cp target/release/cliniq /cliniq

# Runtime: debian slim (DuckDB needs glibc)
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /cliniq /cliniq
EXPOSE 8080 8081
ENTRYPOINT ["/cliniq"]
