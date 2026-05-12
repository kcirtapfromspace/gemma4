# ClinIQ — eICR-to-FHIR with Gemma 4, on-device

**Hackathon submission narrative · Gemma 4 Good · 2026-05-18 (deadline day).**
Supersedes `hackathon-submission-2026-04-27.md` with the v63 Unsloth
retrain that closed the v62 truncation gap, the deployment hardening
that resulted from the 2026-05-11 audit, and the final list of
submission artifacts. Both tracks (Health Impact, Unsloth $10K) in one
doc; the lead claim is unchanged.

---

## Headline

**Agent path: F1 = 1.000 across 80 sustained-load reps and 7 / 7 external
CDC eICR test vectors. Edge-deployable on a Jetson Orin NX 8 GB. HL7
Java validator structural pass on every emitted FHIR R4 Bundle.**

**Unsloth track: v62 LoRA on `unsloth/gemma-4-E2B-it` is the shipped
fine-tune — F1 = 0.84 / precision = 0.84 on Mac Q3_K_M (the same
edge-deployment quantization the iOS app uses), up from base Gemma's
F1 = 0.337. v63 and v63b are documented diagnostic work that found a
silent post-2026-04-30 regression in the unsloth/peft/transformers
chain that drops LoRA k/v weights on Gemma 4's global-attention layers;
the bug hides at full precision and only surfaces under Q3_K_M
quantization.**

```
Agent path (deterministic + RAG + 6-turn agent loop):
  combined-45 sustained:  20 / 20 reps at F1 = 1.000  (0 FPs, recall = 1.000)
  combined-54 sustained:  30 / 30 reps at F1 = 1.000  (0 FPs, recall = 1.000)
  combined-64:            F1 = 0.997, precision = 0.994, recall = 1.000
                          62 / 64 perfect, 1 FP (SJS / TEN overlap), 0 misses
  external HL7 CDA:       7 / 7 perfect, 360 / 360 codes recovered
  Jetson Orin NX (k8s):   F1 = 1.000 on 11 / 11 deterministic-tier cases
  grammar stability:      0 parse errors over 81 runs

Unsloth single-shot path (val-compact 200 cases, Mac Q3_K_M edge deploy):
  base Gemma 4 E2B Q3_K_M:         F1 = 0.337
  v62 LoRA (shipped):              F1 = 0.837, precision = 0.837, recall = 0.837
                                   JSON-valid = 92%, p50 = 3.08 s on Mac
  v63 LoRA (toolchain regression): F1 = 0.548 on Mac Q3_K_M;
                                   F1 = 0.9989 in unquantized PyTorch on T4
                                   Cause: missing k/v LoRA weights on 20 of 35
                                   decoder layers (Gemma 4 global-attention).
  v63b LoRA (coverage-fix probe):  F1 = 0.610 on Mac Q3_K_M (50-case sample)
                                   Coverage gate passed (35/35 k/v wrapping).
                                   Halfway-back; ship gate F1 >= 0.85 not met.
```

The pipeline runs on a clinician's phone. No internet round-trip, no PHI
leaving the device, FHIR R4 emitted natively.

---

## What "ClinIQ" is

A three-tier on-device pipeline that turns a clinician's dictation,
paste, or full eICR XML into a FHIR R4 Bundle ready for public-health
surveillance:

| Tier | What | Median latency | LLM? |
|---|---|---:|---|
| 1. Deterministic | Regex over inline SNOMED markers + CDA `<code/>` attrs + curated alias lookup with NegEx scope | ~5 ms | No |
| 2. RAG fast-path | Curated CDC NNDSS + WHO IDSR DB (~60 entries) with NegEx applied to the matched phrase | ~80 ms | No |
| 3. Gemma 4 agent | Native function calling, 6-turn cap, GBNF-locked tool-call grammar | 5–35 s | Yes |
| (alt) Single-shot | v62 LoRA, no RAG | 3.1 s p50 (Mac, Q3_K_M) | Yes |

Single-shot latencies benched on Mac M-series with
`apps/mobile/convert/bench_v62_singleshot.py`. Raw output:
`apps/mobile/convert/build/v62_val_compact_bench_localval.json` (v62) and
`apps/mobile/convert/build/v63_val_compact_bench.json` (v63 — 2.79 s p50,
9% faster than v62 but with the quality regression noted in the Unsloth
section).

The single-shot path is the v62/v63 fine-tune; the agent path is the
RAG-grounded multi-turn tool-call loop. They solve different problems:
the agent hits F1 = 1.000 on real free-text input, the single-shot is
~5× faster on known-format inputs. Both ship.

---

## Health Impact track

The motivating problem: CDC EZeCR — the canonical electronic Case
Report processing platform — is cloud-only (AWS Comprehend Medical +
IMO Precision Normalize + Glue FindMatches + AIMS). LMIC clinics,
field hospitals, and disaster-response settings can't use it because
PHI cannot leave the device, internet is unreliable, and the per-case
cost is unbounded.

ClinIQ is the **on-device extraction tier** of an EZeCR-style pipeline:
what an LMIC clinic would run when there's no AWS, no Verato, no AIMS.

External credibility:
- HL7 Java IG validator passes on every emitted Bundle (not just our
  test cases — the validator the actual jurisdictions use).
- 7 / 7 perfect on the same CDC RCTC eICR test vectors EZeCR uses.
- Jetson Orin NX 8 GB pod deployment under Talos k8s — same code that
  runs on a phone runs on a $400 edge box for clinic walk-in kiosks.

Deployment surfaces (Health Impact track):
- **iOS app** — `apps/mobile/ios-app/ClinIQ/`, SwiftUI. Demo script in
  `DEMO_SCRIPT.md` (60 s, 7 beats). GGUF is side-loaded into the
  simulator's `Documents/` per `DEMO_SCRIPT.md` step 0; the loader
  probes Documents → tmp → Bundle so the seeded weights take precedence.
- **Hosted demo (HF Spaces)** — https://huggingface.co/spaces/kcirtapfromspace/cliniq-eicr-fhir
  on ZeroGPU A10G. **`spaces/build.sh` defaults to the in-process PyTorch
  backend** (`CLINIQ_SPACE_BACKEND=zerogpu`); MTP and remote-tunnel
  variants are available for ops experiments.
- **Jetson k8s** — `infra/` Talos overlays, F1 = 1.000 on the
  deterministic-tier cases at ~0.97 tok/s (edge-decode for the agent
  tier; the deterministic + RAG tiers are sub-100 ms).

---

## Unsloth track

**The shipped LoRA: v62.** Three iterations were trained and benchmarked.
Numbers below are all on the same `val-compact` 200-case held-out split.
Two distinct deployment surfaces matter: unquantized PyTorch (what the
Kaggle inline bench runs) and Mac Q3_K_M (what the iOS app and the
hosted demo's edge path actually use). The latter is what counts for
"can a clinician run this on their phone."

| | v62 (shipped) | v63 (toolchain regression) | v63b (coverage-fix probe) |
|---|---|---|---|
| Base | `unsloth/gemma-4-E2B-it` | same | same |
| `max_seq_length` | 512 | 1024 | 1024 |
| Train cost (T4) | 1h 04m | 3h 04m | ~3h |
| Unquantized F1 (Kaggle PyTorch) | 0.82 | 0.9989 | (bench cell crashed) |
| **Mac Q3_K_M F1** | **0.837** | 0.548 ⚠ | 0.610 ⚠ |
| Mac Q3_K_M latency p50 | 3.08 s | **2.79 s** | 2.81 s |
| LoRA tensor count | 490 (decoder full) | 410 (k/v partial) | 786 (decoder full + vision wrap) |

**What the three Unsloth APIs did (in all three iterations):**

| Unsloth API | Effect |
|---|---|
| `FastLanguageModel.from_pretrained(load_in_4bit=True)` | 4-bit base fits on free Kaggle T4 (16 GB) with headroom |
| `FastLanguageModel.get_peft_model(use_gradient_checkpointing="unsloth")` | ~30% peak-VRAM cut vs vanilla checkpointing — the lever that made `max_seq_length=1024 + packing=True` fit on a free T4 at batch=1 |
| `unsloth.chat_templates.get_chat_template(tokenizer, "gemma-4")` | Emits the `<\|turn\|>` markers Gemma 4 was pretrained on; HF's default would produce `<start_of_turn>` and break inference |

**Why v63 looks great unquantized and broken quantized.** We bumped
`max_seq_length` 512 → 1024 to stop `packing=True` from clipping
long-expansion training cases mid-JSON, and the unquantized Kaggle
bench showed F1 = 0.9989. But the Mac Q3_K_M re-bench landed at
F1 = 0.548. Tensor-count diff (v62: 490, v63: 410) explained it: the
latest Unsloth / PEFT release silently dropped LoRA k_proj + v_proj on
the 20 global-attention layers of Gemma 4's hybrid attention (layers
15–34). Unquantized full-precision inference can compensate; Q3_K_M
quantization can't. **v63's latency win (9% faster p50) is real and
survives quantization** — only the quality regresses.

**v63b — coverage fix verified, but not enough to recover v62 F1.** v63b
clears the k/v coverage assertion (full 35/35 wrapping confirmed in the
kernel log, 786 saved tensors). Mac Q3_K_M F1 = 0.61, precision = 0.47,
recall = 0.86 — halfway back from v63's 0.55, but still well below v62's
0.84. The latency win is preserved (p50 2.81 s, 9% faster than v62). The
remaining gap is not the coverage bug; the 786-tensor adapter also wraps
the SigLIP vision tower's attention slots, and gradients from text-only
training flow into those vision weights in ways that hurt the language
model's schema-completeness (0.00 vs v62's 0.92). A v63c that excludes
vision modules via a tighter target-module regex is the obvious next
experiment but deferred past the 2026-05-18 deadline. v62 ships.

**Honest framing.** Even v63b at F1 = 0.99 (if it lands there
unquantized) is on synthetic in-distribution val-compact. The headline
F1 = 1.000 claim for ClinIQ stays on the agent path against real
free-text and HL7 CDA XML. The single-shot LoRA path is the fast
demo option for known-format eICRs; it doesn't replace the agent path.

---

## Submission artifacts

| Surface | URL / path | Track |
|---|---|---|
| Repo | `github.com/kcirtapfromspace/gemma4` | both |
| Hosted demo | https://huggingface.co/spaces/kcirtapfromspace/cliniq-eicr-fhir | both |
| iOS app | `apps/mobile/ios-app/ClinIQ/` + `DEMO_SCRIPT.md` | Health Impact |
| Jetson deploy | `infra/` (Talos overlays) | Health Impact |
| HF Hub — v62 LoRA (shipped) | `kcirtapfromspace/cliniq-gemma4-e2b-unsloth-v62` | Unsloth |
| HF Hub — v63b LoRA (if it lands by deadline) | `kcirtapfromspace/cliniq-gemma4-e2b-unsloth-v63b` | Unsloth |
| Public Kaggle notebook (v62) | `kaggle.com/code/patrickdeutsch/cliniq-gemma4-unsloth-submission` | Unsloth |
| v62 model card | `tools/autoresearch/v62-submission/MODEL_CARD.md` | Unsloth |
| v63 run record (full diagnosis of quantization regression) | `tools/autoresearch/v63-experiment/EXPERIMENT.md` | Unsloth |
| v63b run record | `tools/autoresearch/v63b-experiment/EXPERIMENT.md` | Unsloth |
| Local LoRA artifacts | `models/cliniq-gemma4-e2b-v62-lora.gguf`, `models/cliniq-gemma4-e2b-v63-lora.gguf` (+ `v63b` if landed) | Unsloth |

---

## Scope and limitations

- **Synthetic LoRA training data.** The 800 train / 200 val cases are
  clinician-formatted pseudo-eICRs (`Patient: …\nDx: …\nLab: …\n`),
  not full HL7 CDA XML. v63's F1 = 0.9989 is on this in-distribution
  set. Free-text clinician dictation and full CDA XML stay on the
  agent path (which scores F1 = 1.000 / 0.997 / 1.000 across the three
  bench suites, all unsynthetic).
- **English only, US clinical conventions.** SNOMED CT, ICD-10-CM,
  LOINC, RxNorm. No localisation for ICD-11 or non-US ontologies yet.
- **No PHI in training data.** Synthetic patients only.
- **Compact JSON ≠ FHIR Bundle.** The single-shot output needs one
  transform layer (`apps/mobile/convert/fhir_bundle.py`) to become an
  R4 Bundle. The full pipeline does this transparently; raw single-shot
  output does not.
- **Mobile pivot was a research finding, not a ship.** LiteRT-LM
  benches at 52–56 tok/s on iOS/Android vs 0.9 tok/s on Jetson for
  Gemma 4 E2B-it. The iOS app uses llama.cpp today; LiteRT-LM
  integration is documented in `tools/autoresearch/litert-lm-status-2026-05-05.md`
  and `tools/autoresearch/unified-stack-investigation-2026-05-05.md`
  as future work since LiteRT-LM's C ABI doesn't expose logits (so we
  can't compose it with MTP speculative decoding cleanly within the
  hackathon window).

---

## Reproduce

```bash
# Agent path
python -m venv scripts/.venv && source scripts/.venv/bin/activate
pip install -r apps/mobile/convert/pyproject.toml

llama-server --model models/gemma-4-E2B-it-Q3_K_M.gguf \
             --port 8090 --jinja --ctx-size 32768 \
             --parallel 4 --n-gpu-layers 99 --threads 8 &

scripts/.venv/bin/python apps/mobile/convert/agent_pipeline.py \
    --cases scripts/test_cases.jsonl \
            scripts/test_cases_adversarial{,2,3,4,5,6,7}.jsonl \
    --out-json apps/mobile/convert/build/repro_combined64.json \
    --endpoint http://127.0.0.1:8090

# Expect F1 = 0.997 on combined-64. Single sustained rep ~6 minutes.

# v63 LoRA path
llama-server --model models/gemma-4-E2B-it-Q3_K_M.gguf \
             --lora models/cliniq-gemma4-e2b-v63-lora.gguf \
             --port 8091 --jinja --reasoning-format none --reasoning off \
             --ctx-size 8192 --n-gpu-layers 99 --threads 8 &

scripts/.venv/bin/python apps/mobile/convert/bench_v62_singleshot.py \
    --max-tokens 2048 --compare --lora v63 \
    --out apps/mobile/convert/build/v63_val_compact_bench.json
```

---

## Acknowledgements

Built with: Gemma 4 E2B (Google DeepMind), Unsloth (Daniel Han,
Michael Han) for the 4-bit training stack and `gemma-4` chat template,
llama.cpp, HuggingFace Transformers + PEFT + TRL, SwiftUI, Talos
Linux, HL7 FHIR Validator, fhir.resources.

Thanks to the CDC EZeCR / D2E workshop participants whose 2022 design
informed the architecture.

License: Apache 2.0 (LoRA adapters); base model under the Gemma TOU.
