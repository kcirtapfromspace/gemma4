# LLM-Required Evaluation Track

This track covers the evidence gap left by the broad90 regression suite:
cases where deterministic extraction should not be enough. The fixture is:

- `scripts/test_cases_llm_required.jsonl`

It is intentionally small and high-signal. The clinical text avoids inline
SNOMED, LOINC, and RxNorm codes. Positive cases require disease-code inference
from descriptions, synonyms, multilingual prose, or lab wording. Hard negatives
require the agent to ignore disease names that appear only in quoted rumor,
family history, exposure, or negative-test context.

## What This Measures

The expected behavior is not "copy a visible code from text." A passing run
shows that the agent/model/RAG path can:

- bridge clinical descriptions to reportable condition codes;
- handle Spanish, French, Vietnamese, and Portuguese prose without English
  disease names;
- infer selected lab codes from prose lab names when no LOINC is inline;
- suppress false positives from quoted text, exposure-only context, family
  history, and explicitly negative diagnosis statements.

## How It Differs From Broad90

The broad90 suite is a broad regression pack with 90 curated cases and 676
expected codes. Current evidence shows 90/90 perfect with most cases resolved
through deterministic or deterministic-fallback paths. That is useful for
coverage and regression safety, but it does not isolate whether an LLM is
actually needed.

This fixture is the opposite: it is narrow, adversarial, and designed so the
correct answer often requires reasoning beyond literal code extraction. Treat
it as a model/agent capability check, not as a population-level accuracy claim.

## Local Validity Check

Validate JSONL shape without network access:

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("scripts/test_cases_llm_required.jsonl")
seen = set()
for lineno, line in enumerate(path.read_text().splitlines(), 1):
    row = json.loads(line)
    required = {"case_id", "description", "user", "expected_conditions", "expected_loincs", "expected_rxnorms"}
    missing = required - row.keys()
    if missing:
        raise SystemExit(f"{path}:{lineno}: missing {sorted(missing)}")
    if row["case_id"] in seen:
        raise SystemExit(f"{path}:{lineno}: duplicate case_id {row['case_id']}")
    seen.add(row["case_id"])
print(f"ok: {len(seen)} cases")
PY
```

Run the deterministic-only harness as a guardrail. It should not be expected
to pass perfectly; a perfect deterministic run would mean this fixture no
longer proves an LLM-required gap.

```bash
python3 scripts/run_det_external.py \
  --cases scripts/test_cases_llm_required.jsonl \
  --out-json /tmp/llm_required_deterministic.json
```

## Agent/LLM Run

Run this against the agent path used for broad90, with a reachable local model
or chat backend configured by the parent environment:

```bash
python3 apps/mobile/convert/agent_pipeline.py \
  --cases scripts/test_cases_llm_required.jsonl \
  --out-json apps/mobile/convert/build/llm_required_agent_verify.json \
  --force-agent \
  --chat-timeout 30 \
  --max-turns 4
```

`--force-agent` is required for this track. It bypasses the production
deterministic and fast-path short-circuits so the benchmark measures the
model/tool loop instead of the offline extractor.

If the run is intended to gate a candidate, score it through the same FHIR
verification path used for broad90 after the agent JSON is produced:

```bash
scripts/.venv/bin/python apps/mobile/convert/score_fhir.py \
  --from-agent-bench apps/mobile/convert/build/llm_required_agent_verify.json \
  --backend python \
  --out-json apps/mobile/convert/build/llm_required_fhir_python_verify.json
```

## Failure Interpretation

- Missing positive SNOMED codes means the model/RAG path is not bridging from
  clinical description, synonym, or multilingual prose into the reportable
  condition vocabulary.
- Missing expected LOINC codes means lab display-name lookup or model-side lab
  normalization is too brittle when no inline LOINC is present.
- Any false positive in hard negatives means the path is over-triggering on
  quoted text, exposure-only context, family history, or negated diagnoses.
- A perfect deterministic-only run is suspicious for this track: it means the
  fixture has drifted toward broad90-style literal matching and should be
  refreshed with harder no-inline-code cases.

## Pass Bar

For an evidence claim, require 8/8 perfect cases with zero false positives and
then preserve the agent-bench JSON plus FHIR verification JSON as artifacts.
For exploratory model tuning, inspect per-case misses first; aggregate F1 is
less useful here because each row represents a distinct reasoning failure mode.
