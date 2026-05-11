# MTP Track C — Memory Budgets

**Date:** 2026-05-05
**Source:** Explore agent computations from public Gemma 4 architecture specs
**Confidence:** methodology is sound; specific architecture numbers (layers, kv_heads, head_dim) reported by agent need source-check before any deployment decision.

## Methodology

For each (model, device, quantization) cell:
- Target weights = `params × dtype_bytes` (fp16=2, int8=1, int4=0.5)
- Drafter weights = same formula
- KV cache (target only — drafter shares per Google announcement) = `2 × num_layers × num_kv_heads × head_dim × context × dtype_bytes`
- Activation/runtime overhead ≈ 500 MB
- Verdict: fit within device LLM budget

## Device budgets

| Device | Total RAM | Usable LLM budget |
|---|---|---|
| Jetson Orin NX 8 GB | 8 GB | ~5–6 GB after OS/CUDA |
| iPhone 15 Pro / Pro Max | 8 GB | ~3–4 GB via LiteRT-LM |
| Apple Silicon Mac | 32 GB | ~28 GB unified |

## Architecture (agent-reported, NOT verified directly)

- E2B: 35 layers, 4 KV heads, 256 head_dim, 128K context (sliding)
- E4B: 42 layers, 4 KV heads, 256 head_dim, 128K context (sliding)
- 26B A4B MoE: 30 layers (25 sliding, 5 global), 8 KV heads (2 global), 256/512 head_dim, 256K context
- 31B Dense: 60 layers (50 sliding, 10 global), 16 KV heads (4 global), 256/512 head_dim, 256K context

## Fit matrix (GB; KV at fp16; +0.5 GB overhead included)

| Combo | Target int4 | Drafter int4 | KV @ 8K | KV @ 32K | Total @ 32K | Verdict |
|---|---|---|---|---|---|---|
| E2B + E2B-asst on Jetson | 2.6 | 0.04 | 0.14 | 0.56 | **3.7** | ✓ fits |
| E2B + E2B-asst on iPhone | 2.6 | 0.04 | 0.14 | 0.56 | **3.7** | ✗ tight at 32K, fits at 8K |
| E4B + E4B-asst on Jetson | 4.0 | 0.04 | 0.21 | 0.85 | **5.4** | ⚠ tight |
| E4B + E4B-asst on iPhone | 4.0 | 0.04 | 0.21 | 0.85 | **5.4** | ✗ no-fit |
| 26B-A4B + asst on Jetson | 12.6 | ~0.5 | 0.48 | 1.92 | **15.5** | ✗ no-fit |
| 26B-A4B + asst on Mac | 12.6 | ~0.5 | 0.48 | 1.92 | **15.5** | ✓ fits |
| 31B + asst on Mac | 15.4 | 0.25 | 0.96 | 3.84 | **20.0** | ✓ fits |

**Drafter overhead is ~1% of target — small enough that drafter doesn't change which device classes a target can fit on.** The constraint is the target weights + KV cache, exactly as before.

## Implications

- Our current edge-deployable target (E2B, our actual fine-tune target) is fine on Jetson and on iPhone at 8K context.
- **eICR documents are long** — 32K context is realistic, which pushes the iPhone budget. Consider sliding-window attention.
- E4B is the next step up but tight on iPhone and on Jetson with 8 GB. Anything bigger (26B / 31B) is desktop/server only — relevant only for the HF Spaces hosted-demo angle.

## Risks / unknowns

- Numbers above assume drafter actually shares KV cache. If it doesn't, KV memory ~doubles and iPhone E2B@32K is no longer safe.
- Drafter file size on disk not verified; estimated from param count × bytes.
- Sliding-window vs. full attention math may overstate KV at 32K for E2B/E4B.
- LiteRT-LM packaging overhead unknown.

## Next step before relying on this

Verify Gemma 4 E2B + E4B layer/head/dim numbers from the official HF model card; verify whether drafter weights are streamable or must be in RAM concurrently with target.
