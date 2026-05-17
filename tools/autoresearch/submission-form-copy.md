# Kaggle submission form copy — Gemma 4 Good

Ready-to-paste copy for the Kaggle submission form. Two tracks: Health
Impact and Unsloth ($10K). Paste each block into the corresponding field.

---

## Project title

**ClinIQ — On-device eICR → FHIR R4 with Gemma 4**

## One-line summary (≤140 chars)

> A clinician's phone turns dictation, paste, or eICR XML into a HL7-valid
> FHIR R4 Bundle — Gemma 4, fully offline, F1 = 1.000.

(135 chars)

## Tracks claimed

- [x] Health & Sciences (primary category)
- [x] Unsloth ($10K) (prize track)

## Short description (1–2 paragraphs)

ClinIQ is an offline-first, on-device extraction pipeline that turns a
clinician's dictation, paste, or full eICR XML into a FHIR R4 Bundle ready
for public-health surveillance. It is the on-device extraction tier of an
EZeCR-style architecture for LMIC clinics, field hospitals, and
disaster-response settings where AWS Comprehend Medical + IMO + Glue
FindMatches + AIMS are unreachable. Three tiers (deterministic regex over
SNOMED markers + curated RAG fast-path over CDC NNDSS / WHO IDSR + Gemma 4
agent with GBNF tool-call grammar) escalate by latency; the agent only
fires when the lower tiers are uncertain.

For the Unsloth track, we shipped **v62**, a `unsloth/gemma-4-E2B-it` LoRA
trained in 1 h 4 m on a free Kaggle T4. v62 lifts the base model on Mac
Q3_K_M from F1 = 0.337 to **F1 = 0.837** on a 200-case val-compact bench
(p50 = 3.08 s, JSON-valid 92%). The submission also documents v63 and v63b
diagnostic runs that found a silent post-2026-04-30 regression in the
unsloth/peft/transformers chain dropping LoRA k/v on Gemma 4's
global-attention layers under Q3_K_M quantization — a result that should
help downstream users.

## Headline numbers

| Metric | Result |
|---|---|
| Agent path F1 (combined-45 sustained, 20 reps) | **1.000** (0 FPs) |
| Agent path F1 (combined-54 sustained, 30 reps) | **1.000** (0 FPs) |
| Agent path F1 (combined-64 adversarial) | **0.997** (recall = 1.000) |
| External CDC eICR test vectors | **7 / 7 perfect**, 360 / 360 codes recovered |
| Jetson Orin NX 8 GB (k8s) | **F1 = 1.000** on 11 / 11 deterministic-tier cases |
| HL7 Java IG validator | structural pass on every Bundle |
| Grammar stability | 0 parse errors over 81 runs |
| Unsloth v62 LoRA (Mac Q3_K_M) | **F1 = 0.837** (from base 0.337) |
| Unsloth v62 latency p50 (Mac) | **3.08 s** (from base 6.6 s) |

## Submission artifacts

| Artifact | Link |
|---|---|
| **Hosted demo (HF Spaces)** | https://huggingface.co/spaces/kcirtapfromspace/cliniq-eicr-fhir |
| GitHub repo | https://github.com/kcirtapfromspace/gemma4 |
| Unsloth v62 LoRA (HF Hub) | https://huggingface.co/kcirtapfromspace/cliniq-gemma4-e2b-unsloth-v62 |
| Public Kaggle notebook (v62 training) | https://www.kaggle.com/code/patrickdeutsch/cliniq-gemma4-unsloth-submission |
| Training dataset (Kaggle) | https://www.kaggle.com/datasets/patrickdeutsch/eicr-fhir-training-data |
| Model card | [`tools/autoresearch/v62-submission/MODEL_CARD.md`](https://github.com/kcirtapfromspace/gemma4/blob/main/tools/autoresearch/v62-submission/MODEL_CARD.md) |
| Submission narrative | [`tools/autoresearch/hackathon-submission-2026-05-18.md`](https://github.com/kcirtapfromspace/gemma4/blob/main/tools/autoresearch/hackathon-submission-2026-05-18.md) |
| iOS demo script | [`apps/mobile/ios-app/DEMO_SCRIPT.md`](https://github.com/kcirtapfromspace/gemma4/blob/main/apps/mobile/ios-app/DEMO_SCRIPT.md) |
| v63 diagnostic record | [`tools/autoresearch/v63-experiment/EXPERIMENT.md`](https://github.com/kcirtapfromspace/gemma4/blob/main/tools/autoresearch/v63-experiment/EXPERIMENT.md) |
| v63b coverage-fix record | [`tools/autoresearch/v63b-experiment/EXPERIMENT.md`](https://github.com/kcirtapfromspace/gemma4/blob/main/tools/autoresearch/v63b-experiment/EXPERIMENT.md) |
| Demo video | `demo-video/cliniq-demo.mp4` (in repo) |

## What's novel (3 bullets if the form wants them)

- **Three-tier extraction with a learned gate.** Most cases short-circuit
  at regex (~5 ms) or curated RAG (~80 ms); the Gemma 4 agent only fires
  when both lower tiers are uncertain. F1 = 1.000 with median ~5 ms, not
  seconds.
- **Autoresearch loop.** Six `bench → bug-find → fix → re-bench` cycles
  grew the bench from 27 → 64 cases, found 13 precision bugs, fixed 8 in
  Python and Swift. The model never moved.
- **External credibility.** Every emitted Bundle structurally validates
  against HL7's official Java IG validator — the same one jurisdictions
  use — and the pipeline scores 7/7 on the CDC RCTC eICR test vectors
  that EZeCR uses.

## License

Apache 2.0 (LoRA adapters); base model under the Gemma TOU.

## Acknowledgements

Built with Gemma 4 E2B (Google DeepMind), Unsloth (Daniel Han / Michael
Han) for the 4-bit training stack and `gemma-4` chat template, llama.cpp,
HuggingFace Transformers + PEFT + TRL, SwiftUI, Talos Linux, HL7 FHIR
Validator, fhir.resources. Thanks to the CDC EZeCR / D2E workshop
participants whose 2022 design informed the architecture.
