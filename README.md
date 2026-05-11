# ClinIQ — eICR → FHIR R4 on the edge with Gemma 4

A SwiftUI iPhone app that turns a clinician's dictation, paste, or eICR XML
into a FHIR R4 Bundle ready for public-health surveillance — all on-device,
no PHI off the phone.

**Submission for the Gemma 4 Good Hackathon** (Kaggle × Google DeepMind),
targeting the **Health Impact** and **Unsloth** ($10K) tracks. Deadline
2026-05-18.

---

## Headline numbers

```
F1 = 1.000 sustained across 80 deterministic reps
   (combined-45 ×20 + combined-54 ×30 + combined-64 ×30, 0 FPs ×50 reps)
F1 = 0.997 on the 64-case adversarial bench (recall = 1.000)
F1 = 1.000 on 7 / 7 external CDC eICR test vectors (360 / 360 codes recovered)
HL7 FHIR validator (Java IG): structural pass on every emitted Bundle
F1 = 1.000 on Jetson Orin NX 8 GB k8s pod (edge deployment)
0 parse errors over 81 runs of the agent path (grammar stability)
```

**Unsloth track delta** (val-compact, 200 cases held out from training,
Mac M-series Q3_K_M for like-for-like edge-deployment numbers):

| | base Gemma 4 E2B Q3_K_M | + v62 LoRA | + v63 LoRA |
|---|---:|---:|---:|
| Micro-F1 | 0.337 | **0.837** | 0.548 ⚠ |
| Micro-precision | 0.469 | **0.837** | 0.393 ⚠ |
| Micro-recall | 0.263 | 0.837 | **0.902** |
| JSON-validity | 100% | 92% | 92% |
| Latency p50 (Mac) | 6.6 s | 3.08 s | **2.79 s** |
| Latency p95 (Mac) | — | 4.24 s | **3.86 s** |
| Train wall-clock (T4) | — | 1h 04m | 3h 04m |

**v62 is the shipped Unsloth-track LoRA.** v63 retrained at
`max_seq_length=1024` benched at F1 = 0.9989 in unquantized PyTorch on
Kaggle T4 but regresses on Mac/Q3_K_M because the latest Unsloth/PEFT
release dropped k_proj + v_proj LoRA weights on Gemma 4's 20 global-
attention layers (layers 15–34). The partial-coverage adapter can't
survive Q3_K_M quantization. v63's latency win (9% faster than v62) is
real and survives quantization. A **v63b retrain with explicit
`layers_to_transform=range(35)` is queued** to recover Q3_K_M F1 ≥ v62
while keeping v63's latency. Full diagnosis in
`tools/autoresearch/v63-experiment/EXPERIMENT.md`; final submission
narrative in `tools/autoresearch/hackathon-submission-2026-05-18.md`.

---

## Try it

| Surface | URL | What you'll see |
|---|---|---|
| **Hosted demo (HF Spaces)** | https://huggingface.co/spaces/kcirtapfromspace/cliniq-eicr-fhir | Paste an eICR narrative, watch the 3-tier pipeline emit a FHIR Bundle on ZeroGPU H200 |
| **iOS app** | `apps/mobile/ios-app/ClinIQ/` | SwiftUI app, builds on `iPhone17ProDemo` simulator (UDID `CADA1806-F64D-4B02-B983-B75F197D1EF3`). Gemma 4 E2B Q3_K_M GGUF is **side-loaded into the simulator's Documents/ on first demo run** (see `DEMO_SCRIPT.md` step 0); the app's loader probes `Documents/ → tmp/ → Bundle.main` so seeded weights take precedence. |
| **60-second demo script** | `apps/mobile/ios-app/DEMO_SCRIPT.md` | Read-along narration timed to a 7-beat sim walkthrough |

---

## How the pipeline works

Three tiers, escalated by latency:

| Tier | What | Median latency | LLM? |
|---|---|---:|---|
| 1. Deterministic | Regex over inline `(SNOMED 12345)` markers + CDA `<code code="…" />` attrs + curated alias lookup with NegEx scope | ~5 ms | No |
| 2. RAG fast-path | Curated CDC NNDSS + WHO IDSR DB (~60 entries) with NegEx applied to the matched phrase | ~80 ms | No |
| 3. Gemma 4 agent | Native function calling, 6-turn cap, GBNF-locked tool-call grammar | 5–35 s | Yes |

Most cases never invoke the model. On the 54-case bench: 24 short-circuit
at Tier 1, 14 at Tier 2, 16 reach the agent.

The **Cand D gate** decides when the deterministic + lookup path is
trustworthy enough to bypass the agent entirely. See
`apps/mobile/convert/agent_pipeline.py` and the c20 ledger in
`tools/autoresearch/c20-llm-tuning-2026-04-25.md`.

---

## What's in the repo

```
apps/mobile/convert/      Python pipeline (regex / RAG / agent / bundle / score)
apps/mobile/ios-app/      SwiftUI iOS app
spaces/                   HF Spaces hosted demo (Gradio + ZeroGPU)
kaggle-training/          Kaggle T4 training kernel (compact LoRA)
tools/autoresearch/       Submission narrative + experiment ledgers
tools/autoresearch/v62-submission/   Unsloth-track artifacts (model card, notebook plan)
infra/                    Talos k8s overlays for Jetson cluster
data/eicr-samples/        CDC eICR test vectors
scripts/test_cases*.jsonl Bench cases (combined-27, adv4-7, longitudinal)
```

---

## Read in this order (judges, 5-min path)

1. **This README** — orientation
2. **[`tools/autoresearch/hackathon-submission-2026-05-18.md`](tools/autoresearch/hackathon-submission-2026-05-18.md)** — final judge-facing one-pager (Health Impact + Unsloth)
3. **[`tools/autoresearch/v62-submission/MODEL_CARD.md`](tools/autoresearch/v62-submission/MODEL_CARD.md)** — Unsloth-track model card (v62 shipped, v63 follow-up section appended)
4. **[`tools/autoresearch/v63-experiment/EXPERIMENT.md`](tools/autoresearch/v63-experiment/EXPERIMENT.md)** — v63 run record (max_seq=1024 retrain, F1=0.9989)
5. **[HF Space](https://huggingface.co/spaces/kcirtapfromspace/cliniq-eicr-fhir)** — see it run

15-min path: add `tools/autoresearch/c20-llm-tuning-2026-04-25.md` (full
ledger, 80-rep variance tables, every ablation) and `apps/mobile/convert/build/c45_sustained_*.json` (the 20 sustained-load reps).

---

## Reproduce

```bash
# Python venv with bench harness deps
python -m venv scripts/.venv && source scripts/.venv/bin/activate
pip install -r apps/mobile/convert/pyproject.toml

# Run the canonical bench against the local llama-server:
llama-server --model models/gemma-4-E2B-it-Q3_K_M.gguf \
             --port 8090 --jinja --ctx-size 32768 \
             --parallel 4 --n-gpu-layers 99 --threads 8 &

scripts/.venv/bin/python apps/mobile/convert/agent_pipeline.py \
    --cases scripts/test_cases.jsonl \
            scripts/test_cases_adversarial{,2,3,4,5,6,7}.jsonl \
    --out-json apps/mobile/convert/build/repro_combined64.json \
    --endpoint http://127.0.0.1:8090
```

Expect F1 = 0.997 on combined-64. Single sustained rep takes ~6 minutes.

To bench the v62 Unsloth fine-tune:

```bash
llama-server --model models/gemma-4-E2B-it-Q3_K_M.gguf \
             --lora models/cliniq-gemma4-e2b-v62-lora.gguf \
             --port 8091 --jinja --reasoning-format none --reasoning off \
             --ctx-size 8192 --n-gpu-layers 99 --threads 8 &

scripts/.venv/bin/python apps/mobile/convert/bench_v62_singleshot.py \
    --max-tokens 2048 --compare \
    --out apps/mobile/convert/build/v62_val_compact_bench.json
```

---

## What's novel

- **Three-tier extraction with a learned gate** — most cases land on regex or
  RAG. The agent only fires when both lower tiers are uncertain or
  multi-axis-incomplete. F1 = 1.000 with median ~5 ms, not seconds.
- **Autoresearch loop** — six `bench → bug-find → fix → re-bench` cycles
  grew the bench from 27 → 64 cases, surfacing 13 precision bugs, fixing
  8 in Python and Swift. The model never moved.
- **External credibility** — every Bundle structurally validates against
  HL7's official Java IG validator. 7/7 perfect on the same CDC eICR test
  vectors EZeCR uses.
- **Unsloth-distilled single-shot path** — `unsloth/gemma-4-E2B-it` v62
  LoRA trained on a free Kaggle T4 in 1h 4m turns the base model from
  F1 = 0.34 into F1 = 0.84 on Mac Q3_K_M with 0.84 precision, 9% faster
  per inference than the v63 follow-up (which has a known Mac-quantization
  regression being addressed in v63b). The fine-tune isn't replacing the
  agent (which already hits 1.000) — it's the speed tier for known-format
  inputs.

---

## Background

[CDC EZeCR](https://easyecr.org) (the CDC's electronic Case Report
processing platform) is the canonical reference architecture: NLP
extraction (Comprehend Medical) → ontology mapping (IMO Precision Normalize)
→ identity resolution (Glue FindMatches) → jurisdiction-specific rules
engine. ClinIQ is the **on-device extraction tier of an EZeCR-style
pipeline** — what an LMIC clinic would run when there's no AWS, no internet,
no Verato, and no AIMS.

The longitudinal "what's new" view (see `apps/mobile/ios-app/ClinIQ/ClinIQ/Views/Cases/PatientTimelineView.swift`) is the edge-side analogue
of EZeCR's flat-CSV diff between case versions for the same patient.

---

## License + acknowledgements

Apache 2.0 (model adapter follows base model's Gemma TOU).

Built with: Gemma 4 E2B (Google DeepMind), Unsloth, llama.cpp,
HuggingFace Transformers + PEFT + TRL, SwiftUI, Talos Linux,
HL7 FHIR Validator, fhir.resources (Python).

Thanks to the CDC EZeCR / D2E workshop participants whose 2022 design
informed the architecture, and to the Gemma team for the open weights.
