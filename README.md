# ClinIQ — eICR → FHIR R4 on the edge with Gemma 4

A SwiftUI iPhone app that turns a clinician's dictation, paste, or eICR XML
into a FHIR R4 Bundle ready for public-health surveillance — all on-device,
no PHI off the phone.

**Submission for the Gemma 4 Good Hackathon** (Kaggle × Google DeepMind),
targeting the **Health Impact** and **Unsloth** ($10K) tracks. Deadline
2026-05-18.

---

## Headline numbers

Canonical evidence and limitations live in
[`tools/autoresearch/evidence-ledger.md`](tools/autoresearch/evidence-ledger.md).
Use that ledger as the source of truth for submission copy.

```
Combined-64 default bench: F1 = 0.997, recall = 1.000, precision = 0.994
Combined-45 / combined-54 sustained loops: F1 = 1.000 in c20/c21 ledgers
External CDC/HL7 eICR vectors: 7 / 7, 360 / 360 authored codes recovered
FHIR R4: structural Bundle validation via fhir.resources.R4B; HL7 Java structure pass
Jetson Orin NX 8 GB: 11 / 11 edge smoke, ~0.97 tok/s agent decode
Agent stability: 0 parse errors over 81 combined-27 agent-path runs
iPhone Metal throughput: required final gate, not yet claimed as measured
```

**Unsloth track delta** (val-compact, 200 cases):

| | base Gemma 4 E2B Q3_K_M | + ClinIQ v62 LoRA | delta |
|---|---:|---:|---:|
| F1 | 0.337 | **0.823** | **+0.486** |
| Precision | 0.469 | **0.979** | +0.510 |
| Latency p50 | 6.6 s | **4.1 s** | 1.6× |

---

## Try it

| Surface | URL | What you'll see |
|---|---|---|
| **Hosted demo (HF Spaces)** | https://huggingface.co/spaces/kcirtapfromspace/cliniq-eicr-fhir | Paste an eICR narrative; deterministic/RAG tiers run without GPU, Gemma agent path reports availability explicitly |
| **iOS app** | `apps/mobile/ios-app/ClinIQ/` | SwiftUI app, builds on `iPhone17ProDemo`; Settings show model/fallback status, jurisdiction rules, and evidence gates |
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
2. **[`tools/autoresearch/evidence-ledger.md`](tools/autoresearch/evidence-ledger.md)** — canonical claims + final smoke checks
3. **[`tools/autoresearch/hackathon-submission-2026-04-27.md`](tools/autoresearch/hackathon-submission-2026-04-27.md)** — judge-facing one-pager
4. **[`tools/autoresearch/v62-submission/MODEL_CARD.md`](tools/autoresearch/v62-submission/MODEL_CARD.md)** — Unsloth-track model card
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
  vectors in the ClinIQ external bench.
- **Jurisdiction-aware review** — the iOS app tags accepted entities with
  demo public-health rules (`reportable`, `needs review`, `not included`)
  and copies a ClinIQ flat "what changed" diff JSON from the patient
  timeline banner.
- **Unsloth-distilled single-shot path** — `unsloth/gemma-4-E2B-it` LoRA
  trained for 1h 4m on a free Kaggle T4 ×2 turns the base model from
  F1 = 0.34 into F1 = 0.82 with 0.98 precision, 1.6× faster. The fine-tune
  isn't replacing the agent (which already hits 1.000) — it's the speed
  tier for known-format inputs.

---

## Background

ClinIQ is designed around public-health case-reporting workflows: NLP
extraction, ontology mapping, local identity grouping, jurisdiction-specific
rules, and FHIR Bundle payloads. It is the **on-device extraction tier of the
ClinIQ pipeline** — what an LMIC clinic would run when there's no AWS,
no internet, no Verato, and no AIMS.

The longitudinal "what's new" view (see `apps/mobile/ios-app/ClinIQ/ClinIQ/Views/Cases/PatientTimelineView.swift`) is the edge-side analogue
of ClinIQ's flat diff between case versions for the same patient. The
hackathon build intentionally stops at offline extraction/review, local
exact-match identity, demo jurisdiction rules, FHIR Bundle payloads, and
mock/optional POST sync; production mTLS, OAuth/FaceID, probabilistic
identity resolution, and a shared rules marketplace are out of scope.

---

## License + acknowledgements

Apache 2.0 (model adapter follows base model's Gemma TOU).

Built with: Gemma 4 E2B (Google DeepMind), Unsloth, llama.cpp,
HuggingFace Transformers + PEFT + TRL, SwiftUI, Talos Linux,
HL7 FHIR Validator, fhir.resources (Python).

Thanks to the CDC D2E workshop participants whose 2022 design informed the
architecture, and to the Gemma team for the open weights.
