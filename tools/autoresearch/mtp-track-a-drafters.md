# MTP Track A — Gemma 4 Drafters

**Date:** 2026-05-05
**Source:** Explore agent + spot-verification against HF
**Confidence:** mixed — see flags below.

## Verified facts

- `google/gemma-4-E2B-it-assistant` exists on HF, **78M params**, safetensors, updated ~2026-05-04. Verified by direct WebFetch of the model card.
- Google announcement (2026-05-04): https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/
- All four official drafters are Apache-2.0 and on both HF + Kaggle.
- The model card describes MTP as "extending the base model with a smaller, faster draft model" verified for E2B — language matches generic spec-decode framing.

## Released drafters (agent-reported, partially verified)

| Repo | Target | Drafter params | Format | License | Verified? |
|---|---|---|---|---|---|
| google/gemma-4-31B-it-assistant | gemma-4-31B-it | 0.5B | safetensors | Apache 2.0 | not directly |
| google/gemma-4-E4B-it-assistant | gemma-4-E4B-it | 78.8M | safetensors | Apache 2.0 | not directly |
| google/gemma-4-26B-A4B-it-assistant | gemma-4-26B-A4B-it | "3.8B active (25.2B total MoE)" — **suspect, looks like target's own active count** | safetensors | Apache 2.0 | not directly |
| google/gemma-4-E2B-it-assistant | gemma-4-E2B-it | 78M | safetensors | Apache 2.0 | **yes** |

MLX community ports reported (mlx-community/gemma-4-31B-it-assistant-bf16, mlx-community/gemma-4-26B-A4B-it-assistant-bf16) — not directly verified.

## LoRA compatibility — CRITICAL

**Result of direct check on E2B-assistant model card: no mention of LoRA fine-tuning compatibility.** The card does not state whether fine-tuning the target preserves drafter acceptance, nor whether the drafter consumes the target's last-layer activations. The Explore agent's earlier quote attributing layer-sharing details to "the documentation" is **not present on the model card I checked** — likely synthesized from generic EAGLE/Medusa/MTP literature, not this specific release.

What this means for our project:
- **The LoRA × MTP question is genuinely open and untested**, not just "undocumented for our tool". The empirical Mac/MLX bench (Phase 3 of the plan) is the only way to settle it.
- Plausible mechanism (from MTP literature, not this card): if the drafter shares any layers or hidden-state projections with the target, LoRA on those layers will likely degrade acceptance rate.

## Reported framework integrations (not independently verified)

- **HF Transformers**: day-0 via `target_model.generate(assistant_model=assistant_model)`.
- **vLLM v0.19.0**: spec-decode support announced; vLLM-Ascend issue #8392 reports `_v_up_proj` crash when batch token count exceeds `max_cudagraph_capture_size`.
- **MLX**: community bf16 ports exist; no first-party MTP integration PR reported.
- **LiteRT-LM**: announcement claims native support; agent reported v0.10.1 historically had issues (Jinja chat-template bug per our handoff). Re-verify on current version.
- **llama.cpp**: community issues #22337 and #21321 report E2B/E4B failing as draft models with "invalid vector subscript" — **directly hostile to our existing iPhone llama.cpp path**.
- **Ollama / SGLang**: mentioned as supported; no specific PRs found.

## Open questions to resolve before committing

1. Verify 26B-A4B and 31B drafter param counts directly from HF (the 3.8B figure is suspect).
2. Confirm whether the drafter shares the target's tokenizer (announcement implies yes).
3. Confirm the drafter file size on disk for memory-budgeting (Track C uses estimates).
4. Check whether llama.cpp issue #22337 has a fix in flight.

## Sources

- https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/
- https://huggingface.co/google/gemma-4-E2B-it-assistant (verified)
- HuggingFace pages for the other three drafters (agent-reported, not independently verified by me)
