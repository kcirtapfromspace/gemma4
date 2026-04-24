# NAS offload manifest — 2026-04-24

On this date the Mac Studio data volume was 97% full, blocking the
`apps/mobile/convert/merge_and_convert.py` pipeline (C16). The large
non-critical artifacts below were moved to the NAS share at
`/Users/thinkstudio/mnt/models/cliniq/` to free space.

## Still local (actively used)

| File | Size | Why it stays |
|---|---|---|
| `models/cliniq-compact-lora.gguf` | 48 MB | LoRA adapter — input to mobile conversion pipeline |
| `models/cliniq-gemma4-e2b-Q3_K_M.gguf` | 3.0 GB | C2 sweep winner, used by iOS `LlamaCppInferenceEngine` + llama-server |
| `models/cliniq-gemma4-e2b-Q3_K_M.gguf` (seeded in simulator) | 3.0 GB | Copy in the iPhone 17 Pro simulator app sandbox |
| `models/gemma-4-E2B-it-Q3_K_M.gguf` | 2.4 GB | Base Gemma 4 (non-fine-tuned), fallback + speculative-decoding draft candidate |

## Offloaded to NAS → `/Users/thinkstudio/mnt/models/cliniq/`

### `gguf-variants/` (14.9 GB)
Quantization sweep variants that C2's Pareto report flagged as non-winner:

| File | Size | C2 verdict |
|---|---|---|
| `cliniq-gemma4-e2b-Q2_K.gguf` | 2.8 GB | Quality collapse |
| `cliniq-gemma4-e2b-Q3_K_S.gguf` | 2.9 GB | Quality collapse |
| `cliniq-gemma4-e2b-IQ4_XS.gguf` | 3.1 GB | Not Pareto-winner; 1.29 tok/s vs 1.45 baseline |
| `cliniq-gemma4-e2b-Q4_K_M.gguf` | 3.2 GB | 8% slower than winner, same quality |
| `gemma-4-E2B-it-Q4_K_M.gguf` | 2.9 GB | Stock-base Q4 duplicate; Q3_K_M kept locally |

Regenerable via `llama-quantize models/cliniq-gemma4-e2b.gguf <out> <level>`
but the Q8_0 source has ALSO been offloaded (see below) so re-merge
from HF first.

### `cliniq-gemma4-e2b.gguf` (4.6 GB)
The Q8_0 "source of truth" — all `-Q*_K_*.gguf` variants above were
derived from this via `llama-quantize`. Regenerable from
`unsloth/gemma-4-E2B-it` + our compact LoRA through
`apps/mobile/convert/merge_and_convert.py`.

### `mlc-llm-build-output/2026-04-23/` (9.0 GB)
MLC-LLM weight dirs + compiled `.so` files accumulated during the
MLC port attempts (C11 mentioned these aren't reproducible from
commit 8da77e9 without a redo of weight conversion — see
`apps/mlc-llm/BASELINE.md`). MLC is no longer the primary demo path
per the mobile pivot.

### `demo-app-rs-target/` (3.5 GB)
`apps/demo-app-rs/target/` — Rust build artifacts only. Source code
stays in the repo. Regenerable via `cargo build`.

## Before / after

| Metric | Before | After |
|---|---|---|
| `/System/Volumes/Data` free | **32 GB** | **64 GB** |
| `/System/Volumes/Data` capacity | 97% | 93% |

## Recovering offloaded artifacts

```bash
# One-off
cp /Users/thinkstudio/mnt/models/cliniq/<thing> <local-path>

# Restore all gguf variants
rsync -av /Users/thinkstudio/mnt/models/cliniq/gguf-variants/ \
          /Users/thinkstudio/gemma4/models/

# Restore mlc-llm build output
rsync -av /Users/thinkstudio/mnt/models/cliniq/mlc-llm-build-output/2026-04-23/ \
          /Users/thinkstudio/gemma4/apps/mlc-llm/build-output/

# Restore demo-app-rs target (or just cargo build)
rsync -av /Users/thinkstudio/mnt/models/cliniq/demo-app-rs-target/ \
          /Users/thinkstudio/gemma4/apps/demo-app-rs/target/
```
