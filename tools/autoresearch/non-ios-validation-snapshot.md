# Non-iOS Validation Snapshot

Scope: Python/Rust/training/research surfaces only. This intentionally excludes
the iOS app and TestFlight build work.

## Current Evidence

| Track | Result | Artifact |
|---|---:|---|
| Evaluation corpus summary | 90 curated code cases / 676 expected codes; 400 compact validation rows / 1,229 gold code-like items | `tools/autoresearch/eval-corpus-summary.md` |
| Broad90 agent/deterministic extraction | 676/676 matched, F1 1.000, 0 FP, 90/90 perfect | `apps/mobile/convert/build/broad90_agent_verify.json` |
| Broad90 FHIR R4, Python backend | 90/90 pass | `apps/mobile/convert/build/broad90_fhir_python_verify.json` |
| Broad90 FHIR R4, HL7 Java validator | 90/90 pass, 0 errors | `apps/mobile/convert/build/broad90_fhir_java_verify.json` |
| Official HL7 eICR coverage | 10/10 official XML samples covered | `scripts/run_det_external.py --audit-official-coverage` |
| External CDA deterministic extraction | 474/474 matched, F1 1.000, 0 FP, 10/10 perfect | `build/external/det_external_verify_after_agent_fix.json` |
| External CDA FHIR R4, Python backend | 10/10 pass | `build/external/fhir_python_external_verify.json` |
| External CDA FHIR R4, HL7 Java validator | 10/10 pass, 0 errors | `build/external/fhir_java_external_verify.json` |
| Adversarial8 deterministic extraction | 11/11 matched, F1 1.000, 0 FP, 10/10 perfect | CLI output from `regex_preparser.py` |
| Adversarial8 agent-bench path | 11/11 matched, F1 1.000, 0 FP, 10/10 perfect | `apps/mobile/convert/build/adv8_agent_verify.json` |
| Adversarial8 FHIR R4, Python backend | 10/10 pass | `build/external/fhir_python_adv8_verify.json` |
| Adversarial8 FHIR R4, HL7 Java validator | 10/10 pass, 0 errors | `build/external/fhir_java_adv8_verify.json` |
| Longitudinal expected diffs | 4/4 matched | `case_diff.py --bench-jsonl scripts/test_cases_longitudinal.jsonl` |
| Longitudinal agent-bench path | 19/19 matched, F1 1.000, 0 FP, 6/6 perfect | `apps/mobile/convert/build/longitudinal_verify.json` |
| Longitudinal EZeCR CSV | 23 diff rows plus header | `apps/mobile/convert/build/longitudinal_verify_diff.csv` |

## Fixes Landed In This Pass

- `apps/mobile/convert/case_diff.py`: longitudinal diffs now recover the
  original extracted vital component LOINC from FHIR BP/vital panel wrapper
  resources. This keeps user-visible diffs on `8480-6` instead of the FHIR
  profile parent `85354-9`.
- `apps/mobile/convert/regex_preparser.py`: the CLI perfect-case counter now
  counts zero-expected hard negatives as perfect when they emit no false
  positives.
- `apps/mobile/convert/agent_pipeline.py`: the agent-bench aggregate uses the
  same hard-negative perfect-case accounting.
- `scripts/summarize_eval_corpus.py`: reproducible sample-size summary for
  the curated code corpus, broad benchmark artifacts, and compact train/val
  datasets.

## Commands Re-run

```bash
python3 -m py_compile \
  apps/mobile/convert/case_diff.py \
  apps/mobile/convert/regex_preparser.py \
  scripts/run_det_external.py \
  scripts/distill_agent_traces.py

python3 apps/mobile/convert/case_diff.py \
  --bench-jsonl scripts/test_cases_longitudinal.jsonl

python3 apps/mobile/convert/regex_preparser.py \
  scripts/test_cases_adversarial8.jsonl

python3 scripts/run_det_external.py \
  --cases scripts/test_cases_external.jsonl \
  --out-json build/external/det_external_verify_after_agent_fix.json \
  --fail-on-imperfect

python3 apps/mobile/convert/agent_pipeline.py \
  --cases scripts/test_cases_adversarial8.jsonl \
  --out-json apps/mobile/convert/build/adv8_agent_verify.json \
  --chat-timeout 3 \
  --max-turns 2

python3 apps/mobile/convert/agent_pipeline.py \
  --cases scripts/test_cases_longitudinal.jsonl \
  --out-json apps/mobile/convert/build/longitudinal_verify.json \
  --prior-bundle apps/mobile/convert/build/longitudinal_verify_manifest.json \
  --chat-timeout 3 \
  --max-turns 2

python3 apps/mobile/convert/score_fhir.py \
  --diff-csv apps/mobile/convert/build/longitudinal_verify_diff.csv \
  --manifest apps/mobile/convert/build/longitudinal_verify_manifest.json

python3 apps/mobile/convert/agent_pipeline.py \
  --cases \
    scripts/test_cases.jsonl \
    scripts/test_cases_adversarial.jsonl \
    scripts/test_cases_adversarial2.jsonl \
    scripts/test_cases_adversarial3.jsonl \
    scripts/test_cases_adversarial4.jsonl \
    scripts/test_cases_adversarial5.jsonl \
    scripts/test_cases_adversarial6.jsonl \
    scripts/test_cases_adversarial7.jsonl \
    scripts/test_cases_adversarial8.jsonl \
    scripts/test_cases_external.jsonl \
    scripts/test_cases_longitudinal.jsonl \
  --out-json apps/mobile/convert/build/broad90_agent_verify.json \
  --chat-timeout 3 \
  --max-turns 2

scripts/.venv/bin/python apps/mobile/convert/score_fhir.py \
  --from-agent-bench apps/mobile/convert/build/broad90_agent_verify.json \
  --backend python \
  --out-json apps/mobile/convert/build/broad90_fhir_python_verify.json

CLINIQ_FHIR_TX_SERVER=n/a scripts/.venv/bin/python apps/mobile/convert/score_fhir.py \
  --from-agent-bench apps/mobile/convert/build/broad90_agent_verify.json \
  --backend java \
  --out-json apps/mobile/convert/build/broad90_fhir_java_verify.json

python3 scripts/summarize_eval_corpus.py \
  --agent-bench apps/mobile/convert/build/broad90_agent_verify.json \
  --out-md tools/autoresearch/eval-corpus-summary.md \
  --out-json build/eval_corpus_summary.json

scripts/.venv/bin/python apps/mobile/convert/score_fhir.py \
  --bench --backend python \
  --cases scripts/test_cases_external.jsonl \
  --out-json build/external/fhir_python_external_verify.json

scripts/.venv/bin/python apps/mobile/convert/score_fhir.py \
  --bench --backend python \
  --cases scripts/test_cases_adversarial8.jsonl \
  --out-json build/external/fhir_python_adv8_verify.json

CLINIQ_FHIR_TX_SERVER=n/a scripts/.venv/bin/python apps/mobile/convert/score_fhir.py \
  --bench --backend java \
  --cases scripts/test_cases_external.jsonl \
  --out-json build/external/fhir_java_external_verify.json

CLINIQ_FHIR_TX_SERVER=n/a scripts/.venv/bin/python apps/mobile/convert/score_fhir.py \
  --bench --backend java \
  --cases scripts/test_cases_adversarial8.jsonl \
  --out-json build/external/fhir_java_adv8_verify.json
```

## Next Non-iOS Moves

1. Add a larger held-out eICR/code corpus before making population-level
   accuracy claims. The current 90-case corpus is a curated regression suite;
   its 90/90 pass gives a 95% Wilson lower bound of 0.959 on case pass rate,
   while 676/676 code recall gives a 0.994 lower bound under a binomial model.
2. Run the v63 single-shot LoRA job from
   `tools/autoresearch/v63-singleshot-lora-runbook.md`.
3. Generate verified teacher traces and distill them with
   `scripts/distill_agent_traces.py`, keeping protected eval cases excluded.
4. Add the external hardening cases from
   `tools/autoresearch/external-cda-validation-runbook.md`: chunker dedup,
   messy CDA attributes, display-name bleed, and truncated CDA recovery.
5. Publish this non-iOS validation pack into the benchmark tracker once the
   next model candidate has measured latency/quality numbers.
