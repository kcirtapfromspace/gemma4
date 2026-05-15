---
title: ClinIQ — eICR to FHIR (Gemma 4)
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.9.1"
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-a10g
suggested_hardware: zero-a10g
short_description: eICR → FHIR R4 via Gemma 4 with explicit demo status
---

# ClinIQ — eICR to FHIR (Gemma 4)

Hosted demo for the Gemma 4 Good hackathon submission. Paste an electronic
Initial Case Report (eICR) narrative, click **Run**, and watch a three-tier
pipeline emit a structurally valid FHIR R4 Bundle — same code as the iOS
app, exposed behind Gradio so judges can try it without Xcode.

## How it works

| Tier | What | Latency | LLM? |
|------|------|---------|------|
| 1. Deterministic | Regex over inline `(SNOMED 12345)` and CDA-XML `<code code="…">` attributes | ~5 ms | No |
| 2. RAG fast-path | Curated CDC NNDSS + WHO IDSR database (~60 entries) with NegEx filter on the matched phrase | ~80 ms | No |
| 3. Gemma 4 agent | Native function calling — agent invokes tiers (1) and (2) as tools, validates its own output, bounded at 6 turns | ~5–15 s | Yes |

Most cases land on tier 1 or 2 and never invoke a model. The agent tier
runs **Gemma 4 E2B-it** through the configured Space backend when the model
is available. `zerogpu_engine.py` detects ZeroGPU, ordinary CUDA, and CPU;
if the model cannot load, the UI shows an explicit agent-error state while
deterministic and RAG tiers continue to work. Each turn parses Gemma's native
`<|tool_call>...<tool_call|>` sentinels into OpenAI-format `tool_calls`
so `agent_pipeline.run_agent` works unchanged.

Set the model via `CLINIQ_GEMMA_MODEL_ID` (default
`unsloth/gemma-4-E2B-it`) — any HF-hosted Gemma 4 -it variant works.

Every Bundle is parsed through `fhir.resources.R4B`. The status row shows
a binary **✓ R4-valid** signal next to each extraction.

## Sample cases (provided)

| Sample | Tier hit | What it shows |
|--------|----------|---------------|
| COVID-19 (inline SNOMED + LOINC) | 1 | Tier-1 inline-code recall + RxNorm |
| Valley fever | 2 | RAG matches an alt-name (`valley fever` → coccidioidomycosis) |
| Marburg outbreak | 2 | RAG over a low-frequency disease |
| C. diff colitis | 2 | RAG over a colloquial abbreviation |
| Hard narrative | 3 / fallback | Deterministic + RAG miss; invokes Gemma agent when backend is available, otherwise shows a clear unavailable/error state |
| Negated lab | (no match) | Precision check — NegEx prevents `NOT detected` from emitting codes |

## Local run

```bash
python -m pip install -r spaces/requirements.txt
CLINIQ_DISABLE_AGENT_MODEL=1 python spaces/app.py
```

`app.py` finds `apps/mobile/convert/` automatically when run from the repo
root. The environment flag skips the multi-GB model download for local
screenshots; deterministic/RAG paths still run and the hard narrative sample
shows the explicit agent-unavailable state.

## Deploying to Hugging Face Spaces

The deploy bundle is flat (the `convert/` package sits next to `app.py`):

```bash
bash spaces/build.sh out/space   # copies convert/ next to app.py
cd out/space
huggingface-cli login
huggingface-cli repo create cliniq-eicr-fhir --type space --space-sdk gradio
git init && git remote add origin https://huggingface.co/spaces/<you>/cliniq-eicr-fhir
git add . && git commit -m "Initial commit" && git push origin main
```

The `app.py` import logic detects the flat layout and pulls modules from
`./convert/` — no code change needed between local and HF.

## Live agent loop on Spaces

Tier 3 is enabled by default, but it depends on the deployed Space hardware
and model download. If Gemma is unavailable, the status badge reports the
agent error instead of silently pretending the model ran. For the
deterministic + fast-path tiers, no model hardware is needed.

## Bench numbers

Same Python pipeline, no demo-specific tuning. Canonical source:
`tools/autoresearch/evidence-ledger.md`.

| Bench | F1 | Recall | Precision | Notes |
|-------|----|---------|-----------|-------|
| Combined-64 default | **0.997** | **1.000** | 0.994 | Current public headline for the adversarial suite |
| Combined-45 / combined-54 sustained loops | **1.000** | **1.000** | **1.000** | Reproducibility claim from the c20/c21 ledgers |
| Agent grammar stability | — | — | — | 0 parse errors over 81 combined-27 agent-path runs |
| FHIR R4 validity | — | — | — | Structural Bundle validation via `fhir.resources.R4B`; HL7 Java validator structure pass, terminology-snapshot warnings possible |
| External CDC/HL7 eICR vectors | **1.000** | **1.000** | **1.000** | Deterministic CDA path recovered 360/360 authored codes on 7/7 vectors |
| Jetson Orin NX 8GB | **1.000** | **1.000** | **1.000** | 11/11 edge smoke; agent decode about 0.97 tok/s, too slow for live demo |
| Unsloth v62 LoRA | **0.823** | 0.710 | **0.979** | JSON validity 86%; JSON-valid subset F1 0.895; GBNF grammar regressed to 0.780 |

### External validation

The submission passes two independent external credibility checks:

1. **HL7 reference R4 validator.** Every Bundle the pipeline emits is
   validated through both `fhir.resources.R4B` (pydantic structural) and
   the HL7-published `validator_cli.jar` 6.9.7 (canonical reference,
   `org.hl7.fhir.core`). Both backends agree on **structure** (cardinality,
   datatypes, invariants `bdl-7`/`bdl-8`, reference resolution); they
   diverge only on **terminology binding** for codes added to LOINC/SNOMED/
   RxNorm after the validator's bundled snapshots — a shared limitation
   with every FHIR extractor that targets newer codesets.

2. **HL7 CDA eICR sample test vectors.** The pipeline's deterministic CDA
   preparser is benched against 7 of HL7's official CDA eICR sample XMLs
   (across STU 1.1, 1.3.0, and 3.1.1 — see
   `https://github.com/HL7/CDA-phcaserpt`). Current public claim:
   **7/7 vectors, 360/360 authored codes recovered**. Use the evidence
   ledger as the canonical source before changing this number.

Reproducibility:

```bash
# HL7 Java validator (single bundle):
scripts/.venv/bin/python apps/mobile/convert/score_fhir.py --backend java

# Combined-54 bench (both backends):
scripts/.venv/bin/python apps/mobile/convert/score_fhir.py --bench --backend python \
  --cases scripts/test_cases.jsonl scripts/test_cases_adversarial{,2,3,4,5,6}.jsonl
scripts/.venv/bin/python apps/mobile/convert/score_fhir.py --bench --backend java \
  --cases scripts/test_cases.jsonl scripts/test_cases_adversarial{,2,3,4,5,6}.jsonl

# CDC / HL7 eICR external sample bench (deterministic):
scripts/.venv/bin/python scripts/run_det_external.py \
  --cases scripts/test_cases_external.jsonl \
  --out-json apps/mobile/convert/build/external_eicr_deterministic_only.json
```

The Java validator JAR (~177 MB) is distributed by HL7 at
`https://github.com/hapifhir/org.hl7.fhir.core/releases/latest/download/validator_cli.jar`.
Drop it at `/tmp/fhir-validator/validator_cli.jar` (or set
`CLINIQ_FHIR_VALIDATOR_JAR=/path/to/validator_cli.jar`) before running
`--backend java`.

## Screenshots

Captured against the local Gradio app (Python pipeline; agent path depends
on configured model hardware):

| State | Image |
|-------|-------|
| Homepage with sample dropdown open | ![homepage](screenshots/homepage.png) |
| Tier 1 deterministic hit (COVID-19 inline SNOMED + LOINC) — green badge, R4-valid, FHIR Bundle visible | ![deterministic](screenshots/deterministic_tier.png) |
| Tier 2 RAG fast-path hit (Valley fever → coccidioidomycosis) — blue badge, R4-valid, FHIR Bundle visible | ![fast-path](screenshots/fast_path_tier.png) |
| Negated lab precision check — grey "No match" (NegEx prevents false positives) | ![negated](screenshots/negated_lab.png) |

## Deploying

```bash
bash spaces/deploy.sh --space patrickdeutsch/cliniq-eicr-fhir
# Stages a commit in out/space/ and prints the push command.
# Add --push to actually run `git push origin main` (requires huggingface-cli login).
```

## License

Apache-2.0. The reportable-conditions database is sourced from public
CDC NNDSS and WHO IDSR pages — see `apps/mobile/convert/reportable_conditions.json`.
