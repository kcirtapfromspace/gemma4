# Non-iOS Benchmark Evidence Aggregation Runbook

Use `scripts/aggregate_benchmark_evidence.py` to combine agent-bench JSON outputs
and optional FHIR score JSON outputs into a compact Markdown or JSON evidence
summary for competition reporting.

## Inputs

- Agent bench JSON: one or more files containing an array of per-case rows with
  `case_id`, `matched`, `expected`, `false_positives`, `path`, `n_tool_calls`,
  `extraction`, and optional `trace` timings.
- FHIR score JSON: optional files from the FHIR scorer with `backend`,
  `fhir_r4_pass_rate`, `n_pass`, `n_validated`, `source`, and `rows`.

The aggregator is read-only with respect to benchmark inputs. It does not call
the iOS app, conversion code, or build pipeline.

## Quick Commands

Markdown to stdout:

```bash
python3 scripts/aggregate_benchmark_evidence.py \
  build/llm_required/llm_required_agent_8091_v62_grounded.json \
  build/adversarial345_agent_after_rag.json \
  --fhir-score build/llm_required/llm_required_fhir_python_8091_v62_grounded.json \
  --fhir-score build/adversarial345_fhir_python_after_rag.json
```

Write both reporting formats:

```bash
python3 scripts/aggregate_benchmark_evidence.py \
  build/llm_required/llm_required_agent_8091_v62_grounded.json \
  build/adversarial345_agent_after_rag.json \
  --fhir-score build/llm_required/llm_required_fhir_python_8091_v62_grounded.json \
  --fhir-score build/adversarial345_fhir_python_after_rag.json \
  --markdown-out /tmp/non_ios_benchmark_evidence.md \
  --json-out /tmp/non_ios_benchmark_evidence.json \
  --out /tmp/non_ios_benchmark_evidence.md
```

JSON to stdout:

```bash
python3 scripts/aggregate_benchmark_evidence.py \
  build/llm_required/llm_required_agent_8091_v62_grounded.json \
  build/adversarial345_agent_after_rag.json \
  --format json
```

## Reported Metrics

- Micro precision, recall, and F1 from `matched`, `expected`, and
  `false_positives`.
- Exact-match case rate where `matched == expected` and `false_positives == 0`.
- Execution path mix, such as deterministic versus agent paths.
- Mean tool calls per case.
- Trace-derived p50 and p95 elapsed seconds when trace timings are present.
- Optional FHIR R4 structural pass rate and invalid case list.

## Competition Reporting Checklist

1. Confirm the agent-bench files are the intended non-iOS artifacts.
2. Include FHIR score files when available so the headline can state both code
   extraction quality and FHIR structural validity.
3. Review the `Review Notes` section for missed expected codes or false
   positives before copying headline claims.
4. Keep generated Markdown/JSON outputs outside the repo, such as under `/tmp`,
   unless you explicitly intend to create a tracked reporting artifact.
