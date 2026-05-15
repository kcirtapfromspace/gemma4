# ClinIQ Hackathon Evidence Ledger

**Canonical source of truth for public claims.** Last updated: 2026-04-30.

Use this file when updating the root README, HF Space README, mobile guide,
submission narrative, video script, or model card. If a claim is not listed
here, treat it as unverified and phrase it as planned or pending.

## Submission Positioning

ClinIQ is the offline **edge extraction and clinician-review tier** of a
ClinIQ public-health workflow. It is not a full surveillance backend. Real
production sync, mTLS, OAuth/FaceID, probabilistic identity resolution,
central retention, and a shared jurisdiction-rules marketplace remain out
of scope for this hackathon build.

## Canonical Claims

| Area | Public claim | Status | Evidence / artifact |
|---|---|---|---|
| Combined/adversarial extraction | Combined-64 default bench: F1 `0.997`, recall `1.000`, precision `0.994`; combined-45 and combined-54 sustained loops reached F1 `1.000` in the c20/c21 ledgers. | Verified in repo narrative; re-run before final submission if model/server artifacts are present. | `tools/autoresearch/hackathon-submission-2026-04-27.md`, `tools/autoresearch/c20-llm-tuning-2026-04-25.md` |
| Agent grammar stability | Agent path had `0` parse errors over `81` combined-27 runs. | Verified in repo narrative; artifact may be gitignored locally. | `tools/autoresearch/handoff-2026-04-27.md`, `toolcall_grammar_bench.json` when present |
| External CDC/HL7 eICR vectors | Deterministic CDA path recovered `360/360` authored codes on `7/7` external eICR vectors. | Verified in repo narrative; use this as the headline external-vector claim. | `scripts/test_cases_external.jsonl`, `scripts/run_det_external.py`, `external_eicr_deterministic_only.json` when present |
| FHIR validation | Bundles structurally validate with `fhir.resources.R4B`; HL7 Java validator passes structure, with terminology-snapshot warnings possible for newer codes. | Verified in repo narrative. Phrase as structural validation, not full production submission certification. | `apps/mobile/convert/score_fhir.py`, `external_fhir_validity_java_post.json` when present |
| HF Space | Public demo URL exists at `https://huggingface.co/spaces/kcirtapfromspace/cliniq-eicr-fhir`. Deterministic and RAG tiers do not require model hardware. Gemma agent path depends on Space hardware/model availability and must surface an explicit unavailable/error state. | Smoke before final submission. Do not say “ZeroGPU H200 verified” unless the live Space is checked. | `spaces/app.py`, `spaces/README.md` |
| iOS app | SwiftUI app shows offline intake, review, FHIR Bundle, outbox, longitudinal timeline, jurisdiction rule tags, and ClinIQ flat diff export. | Implemented locally. Physical-device inference still requires measurement. | `apps/mobile/ios-app/ClinIQ/` |
| Physical iPhone inference | Physical iPhone validation is required before claiming measured on-phone Gemma throughput. Until measured, say simulator CPU is measured and iPhone Metal is pending. | Pending. | Record cold load, warm extraction, tok/s, memory if available, and Metal success/failure here before final submission. |
| Jetson edge | Jetson Orin NX k8s pod ran deterministic/RAG-compatible cases at F1 `1.000` on `11/11`; agent decode was about `0.97 tok/s`, too slow for live demo. | Verified in repo narrative. Use as edge-deployability evidence, not live UX evidence. | `tools/autoresearch/jetson-bench-2026-04-26.md` |
| Unsloth v62 | v62 LoRA improves base Gemma 4 E2B Q3_K_M on val-compact: F1 `0.823`, precision `0.979`, recall `0.710`, JSON validity `86%`, p50 latency `4.05s`; JSON-valid subset F1 is `0.895`; GBNF grammar regressed to F1 `0.780`. | Verified from checked-in JSON. Ship without grammar. | `apps/mobile/convert/build/v62_val_compact_bench.json`, `apps/mobile/convert/build/v62_val_compact_grammar_bench.json`, `tools/autoresearch/v62-submission/MODEL_CARD.md` |

## Required Final Smoke Checks

```bash
python3 apps/mobile/convert/validate_fast_path.py
python3 -m json.tool apps/mobile/convert/build/v62_val_compact_bench.json >/dev/null
python3 -m json.tool apps/mobile/convert/build/v62_val_compact_grammar_bench.json >/dev/null
swiftc -O apps/mobile/ios-app/validate_rag.swift \
  apps/mobile/ios-app/ClinIQ/ClinIQ/Extraction/RagSearch.swift \
  apps/mobile/ios-app/ClinIQ/ClinIQ/Extraction/ReportableConditions.swift \
  -parse-as-library -o /tmp/validate_rag && /tmp/validate_rag
swiftc -O apps/mobile/ios-app/validate_preparser.swift \
  apps/mobile/ios-app/ClinIQ/ClinIQ/Extraction/EicrPreparser.swift \
  apps/mobile/ios-app/ClinIQ/ClinIQ/Extraction/LookupTable.swift \
  -parse-as-library -o /tmp/validate_preparser && \
  /tmp/validate_preparser scripts/test_cases.jsonl scripts/test_cases_adversarial.jsonl
xcodebuild -project apps/mobile/ios-app/ClinIQ/ClinIQ.xcodeproj \
  -scheme ClinIQ -destination 'platform=iOS Simulator,id=CADA1806-F64D-4B02-B983-B75F197D1EF3' build
```

For the physical iPhone gate, record:

| Device | Backend | Model artifact | Metal | Cold load | Warm extraction | tok/s | Notes |
|---|---|---|---|---:|---:|---:|---|
| Pending | llama.cpp GGUF | Pending | Pending | Pending | Pending | Pending | Do not claim measured iPhone throughput until this row is filled. |

## 2026-04-30 Local Validation Run

| Check | Result |
|---|---|
| Python fast-path validator | `8/8` pass |
| v62 JSON artifacts | `json.tool` pass for unconstrained + grammar bench files |
| Swift RAG validator | `19/19` pass |
| Swift preparser validator | `42/42` expected codes, `14/14` cases perfect |
| HF Space URL | `HTTP 200` for the public Space page |
| HF Space local import smoke | Blocked in base environment by missing `gradio`; use `CLINIQ_DISABLE_AGENT_MODEL=1` after installing `spaces/requirements.txt` |
| iOS simulator build | Pass on `iPhone17ProDemo` (`CADA1806-F64D-4B02-B983-B75F197D1EF3`) |
| Physical iPhone discovery | No attached physical iPhone found via `xcrun xctrace list devices`; only Mac + simulator listed |

## Final Narrative Guardrails

- Say “FHIR R4 Bundle payload” unless a real transaction/message integration
  is implemented.
- Say “mock/optional POST” for sync unless a real public-health endpoint is
  running.
- Say “exact local identity hash” for timeline grouping; do not imply Verato
  or probabilistic patient matching.
- Say “jurisdiction demo rules” for the local rule tags; do not imply a shared
  rules marketplace.
- Keep v62 as the Unsloth speed/distillation story, not the main quality path.
