# Validation — model.litertlm

Harness: `validate_litertlm.py` on macOS CPU (arm64). This is a smoke test of
the `.litertlm` bundle, not a mobile benchmark — decoded tok/s on the phone
GPU is separate (~52-56 tok/s per the LiteRT-LM docs for this model).

## Setup

- Model: `build/litertlm/model.litertlm` (2556.2 MB)
- Backend: `cpu` (LiteRT-LM unknown)
- Max tokens: 768

## Prompt

System:
```
You are a clinical NLP assistant. Given an eICR summary, extract the primary conditions, labs, and medications as JSON with keys 'conditions' (SNOMED codes), 'loincs' (LOINC codes), and 'rxnorms' (RxNorm codes). Return only JSON.
```

User:
```
Patient: Wei Brown
Gender: M
DOB: 1958-08-07
Race: Asian
Ethnicity: Hispanic or Latino
Location: Seattle, WA 98101
Phone: +1-995-555-1144
Facility: Denver Health Medical Center (NPI: 1234567800)
Encounter: 2026-12-05
Reason: painless chancre for 39.0 days, regional lymphadenopathy, maculopapular rash on palms and soles
Dx: Syphilis (SNOMED 76272004)
Lab: Treponema pallidum Ab [Presence] in Serum by Immunoassay (LOINC 20507-0) - Positive [Serum, final]
Meds: penicillin G benzathine 2400000 UNT/injection (RxNorm 105220)
```

## Output

```
```json
{
  "conditions": [
  "SNOMES: 100000",
  "SNOMED: 2026-12-05",
  "LOINC: 20507-0"
]
}
```
```

## Stats

| metric | value |
|---|---|
| load time | 0.53 s |
| generate time | 5.33 s |
| approx tokens | 13 |
| approx tok/s | 2.4 |

## Expected extraction (ground truth from test_cases.jsonl)

- case_id:    `bench_minimal`
- description: Minimal: single condition, no vitals
- conditions: ['76272004']
- loincs:     ['20507-0']
- rxnorms:    ['105220']
