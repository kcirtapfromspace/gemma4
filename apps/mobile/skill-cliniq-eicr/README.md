# `skill-cliniq-eicr/` — ClinIQ eICR → FHIR Extractor skill

Drop-in text-only skill for the unmodified [AI Edge Gallery](https://github.com/google-ai-edge/gallery) app. Works with any model the gallery offers, but is tuned for **Gemma 4 E2B-it** (already in the Android allowlist at `1_0_12.json`).

## What this tests

Whether **base Gemma 4 E2B (no fine-tune)** can do our eICR → FHIR extraction with good-enough quality via prompt engineering alone. If yes, we don't need to ship our fine-tuned `.litertlm` at all → mobile demo is 0.5 days of work instead of 2-4.

## Install

1. Install **AI Edge Gallery** on your phone (Android or iOS 17+, from Play Store / App Store).
2. Download **Gemma 4 E2B-it** inside the app (Models tab → Gemma 4 → Download).
3. Zip this directory:
   ```bash
   cd apps/mobile/skill-cliniq-eicr
   zip -r ../cliniq-eicr.skill.zip SKILL.md examples/
   ```
4. Transfer `cliniq-eicr.skill.zip` to the phone (AirDrop / email / cloud).
5. In the gallery app → Skills tab → Add Skill → Import from Local File → pick the zip.
6. Open a chat with Gemma 4 E2B, select `ClinIQ eICR → FHIR Extractor` as the active skill, paste a test case from `scripts/test_cases.jsonl`.

## Measure extraction quality

After pasting a test case, copy the model's JSON output and compare against the `expected_conditions`, `expected_loincs`, `expected_rxnorms` in the same JSONL row. Score = fraction of expected codes present verbatim in the output.

**Pass bar:** ≥ 0.9 extraction_score across the 10 test cases → Tier 1 (skill-only) is viable. Ship it.

**Fail bar:** < 0.7 → base Gemma 4 can't handle it without fine-tune; move to Tier 2 (upload our `.litertlm` to HF + allowlist entry). See `apps/mobile/FORK_PLAN.md`.

## Why this might work without the fine-tune

- Gemma 4 E2B is instruction-tuned and knows SNOMED/LOINC/RxNorm from its pretraining.
- Our test cases **include the codes inline** in the input (e.g. `(SNOMED 840539006)`), so the extraction is largely copying rather than recall.
- Few-shot examples in `SKILL.md` anchor the output format.

## Why it might not work

- JSON structural consistency (no prose, no fences) is the hardest part for base Gemma 4. Fine-tune was explicitly trained on this.
- Longer documents may blow past the skill prompt's context budget before the model commits to the schema.
- The digit-mangling issue Team C7 flagged could affect base Gemma 4 too at int4 quant.

## Files

- `SKILL.md` — the skill definition (frontmatter + instructions + 3 few-shot examples).
- `examples/` — (optional) longer example eICR documents if we need to bundle them separately from SKILL.md.

## Provenance

Based on the skills system documented in `google-ai-edge/gallery/skills/README.md` (Apache 2.0).
