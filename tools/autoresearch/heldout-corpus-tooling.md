# Held-Out Corpus Tooling

## Purpose

`scripts/build_heldout_corpus.py` builds benchmark-compatible candidate cases
from compact ClinIQ chat JSONL rows. It is intended to answer the sample-size
concern without touching extractor code or mixing generated cases into the
protected regression suite.

The helper:

- reads compact train/validation rows such as `kaggle-training/dataset/val-compact.jsonl`;
- converts assistant gold JSON into `expected_conditions`, `expected_loincs`,
  and `expected_rxnorms`;
- generates stable `heldout_synth_*` case IDs;
- scans protected `scripts/test_cases*.jsonl` files and excludes exact protected
  narrative reuse;
- writes a JSONL corpus plus a manifest that records source paths, counts,
  skipped rows, protected files scanned, and the claim boundary.

## Target Sizes

Use three tiers rather than a single headline number:

| Tier | Target | Use |
|---|---:|---|
| Smoke | 25-50 cases | CI or local benchmark wiring checks |
| Broad synthetic held-out | 400-1,000 cases | Larger sample for compact model behavior and deterministic throughput |
| Real external held-out | 100+ independently sourced eICR/CDA cases | Accuracy claims that generalize beyond project-generated data |

The current compact validation set has 400 rows. `train_v2` has 1,650 rows, but
rows derived from training data must be labeled synthetic and not used as
independent model generalization evidence for that trained model.

## Regression vs Held-Out Claims

Keep these claims separate:

- `scripts/test_cases*.jsonl`: protected regression corpus. Use for edge-case
  coverage, deterministic extractor regressions, FHIR validity, and known
  adversarial behaviors.
- compact validation rows: synthetic held-out candidates. Use for larger-sample
  model behavior only when the model was not trained on those rows.
- compact training rows: useful for scalable harness load tests and extractor
  smoke coverage, but not independent held-out evidence for a model trained on
  that data.
- real external eICR/CDA rows: required before making population-level clinical
  accuracy claims.

## Example

Dry run:

```bash
python3 scripts/build_heldout_corpus.py \
  --source kaggle-training/dataset/val-compact.jsonl \
  --limit 25 \
  --limit-mode round-robin-code \
  --dry-run
```

Write a 400-case synthetic held-out candidate set:

```bash
python3 scripts/build_heldout_corpus.py \
  --source kaggle-training/dataset/val-compact.jsonl \
  --out-jsonl tools/autoresearch/generated/heldout-val-compact.jsonl \
  --out-manifest tools/autoresearch/generated/heldout-val-compact.manifest.json
```

For load testing only, combine validation and training sources and keep the
manifest with the output:

```bash
python3 scripts/build_heldout_corpus.py \
  --source kaggle-training/dataset/val-compact.jsonl \
  --source kaggle-training/dataset/train_v2.jsonl \
  --limit 1000 \
  --limit-mode round-robin-code \
  --out-jsonl tools/autoresearch/generated/heldout-synth-1000.jsonl \
  --out-manifest tools/autoresearch/generated/heldout-synth-1000.manifest.json
```

When using `--limit` for a smoke or medium-size sample, prefer
`--limit-mode round-robin-code`. The default `first` mode preserves source
order, but source-order prefixes can overrepresent common codes. Round-robin
mode deterministically walks rare expected-code buckets first and records
`unique_expected_codes`, `top_expected_codes`, `limit_mode`, and
`candidate_cases_before_limit` in the manifest, making sample-size claims easier
to audit.
