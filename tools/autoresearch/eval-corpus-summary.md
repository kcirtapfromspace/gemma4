# ClinIQ Evaluation Corpus Summary

## Code Extraction Regression Corpus

- Cases: 90 (90 unique case IDs)
- Expected codes: 676 (SNOMED 231, LOINC 379, RxNorm 66)
- Hard-negative cases: 3
- Narrative chars p50/p95/max: 816 / 84855 / 209107

| File | Cases | Expected codes | Hard negatives |
|---|---:|---:|---:|
| `scripts/test_cases.jsonl` | 9 | 29 | 0 |
| `scripts/test_cases_adversarial.jsonl` | 5 | 13 | 0 |
| `scripts/test_cases_adversarial2.jsonl` | 8 | 21 | 0 |
| `scripts/test_cases_adversarial3.jsonl` | 5 | 7 | 0 |
| `scripts/test_cases_adversarial4.jsonl` | 8 | 22 | 0 |
| `scripts/test_cases_adversarial5.jsonl` | 10 | 31 | 0 |
| `scripts/test_cases_adversarial6.jsonl` | 9 | 23 | 0 |
| `scripts/test_cases_adversarial7.jsonl` | 10 | 26 | 1 |
| `scripts/test_cases_adversarial8.jsonl` | 10 | 11 | 2 |
| `scripts/test_cases_external.jsonl` | 10 | 474 | 0 |
| `scripts/test_cases_longitudinal.jsonl` | 6 | 19 | 0 |

## Broad Agent/Deterministic Result

- Cases: 90/90 perfect
- Codes: 676/676 matched
- False positives: 0
- Precision/recall/F1: 1.0000 / 1.0000 / 1.0000
- 95% Wilson lower bound, case pass rate: 0.9591
- 95% Wilson lower bound, code recall: 0.9943
- Path counts: {"deterministic": 78, "deterministic-fallback": 3, "fast": 9}

## Compact Model Dataset: train_v2

- Rows: 1650
- Schema-complete gold rows: 1650 (100.0%)
- Gold code/vital-like items: 5184
- Items per row p50/p95: 3 / 6
- Output chars p50/p95/max: 459 / 669 / 762
- Rows with sections: {"conditions": 1650, "labs": 1650, "meds": 1152, "vitals": 1569}

## Compact Model Dataset: val_compact

- Rows: 400
- Schema-complete gold rows: 400 (100.0%)
- Gold code/vital-like items: 1229
- Items per row p50/p95: 3 / 6
- Output chars p50/p95/max: 458 / 661 / 793
- Rows with sections: {"conditions": 400, "labs": 400, "meds": 274, "vitals": 382}

## Interpretation

The 90-case code corpus is a curated regression suite, not an iid clinical prevalence sample. Use it to claim coverage of known edge classes and FHIR validity. Use the 400-row compact validation set for model-training generalization claims, and add a larger held-out eICR/code corpus before making population-level accuracy claims.
