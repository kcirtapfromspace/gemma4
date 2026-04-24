# Mobile app fork plan — ClinIQ eICR/FHIR on AI Edge Gallery

**Author:** main orchestrator, 2026-04-23
**Status:** draft, awaiting Team C8 verification of the chat-template fix
**Base:** [`google-ai-edge/gallery`](https://github.com/google-ai-edge/gallery) — Apache 2.0, 22k stars, actively maintained, Kotlin (Android) + separate iOS build (available via App Store, iOS 17+).

## TL;DR

Three tiers of engineering effort, pick the cheapest one that demos well. Ranked best-to-worst on engineering cost:

| Tier | Effort | What you get |
|---|---|---|
| **1 — Skill-only** | 0.5 day | Install unmodified gallery + our custom eICR SKILL.md. Uses base Gemma 4 E2B (no fine-tune). Unknown demo quality. |
| **2 — Custom model entry** | 2 days | Upload our `.litertlm` to HF, fork `model_allowlist.json`, users get our fine-tune inside the stock chat UI. Quality should match C6/C8 validation. |
| **3 — Full UI fork** | 3-4 days | Fork the app, replace chat UI with eICR-paste → JSON-extract screen, bundle model in APK/IPA. Clinician-usable demo. |

Start with Tier 1 today (no engineering, just a SKILL.md we can write now), scale up based on quality measurements.

## Confirmed facts about the gallery

- **Gemma 4 E2B IT supported natively in Android allowlist 1_0_12** (committed 2026-04 per git log): `Gemma-4-E2B-it` → `gemma-4-E2B-it.litertlm`.
- **Gemma 4 not yet in `ios_1_0_0.json`** — iOS allowlist has Gemma 3n E2B/E4B and Gemma 3 1B IT only. Our model would have to be added there for iOS.
- **Skills system** in `skills/` with `SKILL.md` frontmatter + text (no code) OR JavaScript skills that run in a hidden webview. "Import from URL" is built-in.
- **All models are `.litertlm`** in current allowlists — matches C6's pipeline output. (Older `model_allowlist.json` at the repo root still uses MediaPipe `.task` — ignore; it's legacy.)
- **HuggingFace OAuth** required for download. Model must live on HF; gallery downloads on first run.
- **iOS source is not in this repo.** The App Store iOS app consumes the same allowlists (see `ios_1_0_0.json`) but its Swift code is presumably in a sibling repo or internal. For Tier 2 iOS, we rely on the App Store app + our entry in the allowlist.

## Tier 1 — Skill-only (0.5 day, no model upload, no fork)

### What it produces

A `.skill.zip` file the user can import into the unmodified gallery app. Inside: a `SKILL.md` that instructs the LLM to act as a clinical entity extractor, with schema, 2-3 few-shot examples from `scripts/test_cases.jsonl`, and output format (strict JSON).

### Files to create

```
apps/mobile/skill-cliniq-eicr/
├── SKILL.md
├── examples/
│   ├── covid.xml
│   ├── meningitis.xml
│   └── syphilis.xml
└── README.md  (how to import into gallery)
```

### `SKILL.md` sketch

```markdown
---
name: ClinIQ eICR Extractor
description: Extract SNOMED, LOINC, RxNorm codes and FHIR-ready JSON from eICR CDA/XML clinical documents. Invoke when the user pastes a clinical summary, eICR, or CCD document.
version: 0.1.0
---

# ClinIQ Clinical Entity Extractor

You are a clinical NLP assistant. When the user provides any clinical
document (eICR, CCD, HL7 v2, or free-text clinical notes), extract:

- **conditions**: array of `{code, system, display}` using SNOMED CT
- **labs**: array of `{code, system, display, value, unit}` using LOINC
- **medications**: array of `{code, system, display}` using RxNorm
- **patient**: `{gender, birth_date}`
- **vitals**: `{temp_c, hr, rr, spo2}` if present

Return **JSON only**, no prose, no markdown fences.

## Example 1 — COVID-19

Input:
...

Output:
{"conditions":[{"code":"840539006","system":"SNOMED","display":"COVID-19"}],"labs":[...]}
```

### Risk

Unknown whether base Gemma 4 E2B (no fine-tune) can hit extraction_score ≥ 0.9 on our 5 test cases with prompt engineering alone. Easy to find out: install gallery, add the skill, paste a test case, compare output. ~1 hour of human time.

## Tier 2 — Custom model entry (2 days)

### What it produces

Our fine-tuned `cliniq-gemma4-e2b.litertlm` (2.4 GB) uploaded to HuggingFace as a public (or gated) repo, added to the Android allowlist `1_0_13.json` via PR, downloaded by users in the unmodified gallery app. Quality matches whatever C8 verification lands at.

### Work breakdown

1. **Upload `.litertlm` to HF** (~30 min including `huggingface-cli login` + upload). Org: probably `cliniq-demo` or personal. Name: `cliniq-demo/cliniq-gemma4-e2b-litert-lm`.
2. **Author `model_allowlist.json` delta.** New entry matching the existing Gemma-4-E2B-it template:
   ```json
   {
     "name": "ClinIQ-Gemma-4-E2B",
     "modelId": "cliniq-demo/cliniq-gemma4-e2b-litert-lm",
     "modelFile": "cliniq-gemma4-e2b.litertlm",
     "description": "Gemma 4 E2B fine-tuned on eICR → FHIR clinical extraction. Returns JSON with SNOMED/LOINC/RxNorm codes.",
     "sizeInBytes": 2556000000,
     "estimatedPeakMemoryInBytes": 5900000000,
     "version": "20260423",
     "defaultConfig": {"topK": 1, "topP": 0.95, "temperature": 0.1, "maxTokens": 1024, "accelerators": "gpu,cpu"},
     "taskTypes": ["llm_chat", "llm_prompt_lab"]
   }
   ```
3. **PR to `google-ai-edge/gallery`** adding the entry — OR, faster, host our own allowlist URL and swap it in the app's config. Gallery has a "remote allowlist URL" mechanism (need to verify in the Kotlin source).
4. **Test on Android device** — install gallery from Play Store, download our model, run extraction.
5. **iOS path** — add our entry to `ios_1_0_0.json` via same PR. Risk: iOS app hasn't shipped Gemma 4 support yet (none in `ios_1_0_0.json`). May have to wait for Google to ship Gemma 4 iOS support, or fork iOS app (not in repo — need to find source).

### Risks

- Gemma 4 iOS support not shipped yet → iOS demo delayed until Google releases it.
- HF OAuth: gated repo may require each user to auth. Non-gated public repo solves it but exposes our fine-tune.

## Tier 3 — Full UI fork (3-4 days)

### What it produces

A dedicated ClinIQ mobile app (both platforms): single screen, paste eICR → tap Extract → see JSON, SNOMED/LOINC/RxNorm highlighted, optional "Send to FHIR server" button. Clinician-usable demo.

### Work breakdown

1. **Android fork** (~1 day). Clone `google-ai-edge/gallery`, strip `llm_prompt_lab`/`llm_ask_image`/`llm_chat` UIs, add a single `EicrExtractScreen` in Jetpack Compose, wire to `Engine.createConversation().sendMessage(...)` with our system prompt hardcoded.
2. **iOS**: without the iOS source repo, need to build from scratch against the LiteRT-LM Swift package (see `apps/mobile/SKETCH.md`). ~2 days.
3. **Model distribution**: bundle the 2.4 GB `.litertlm` via Play Asset Delivery (Android) or a first-run download (iOS).
4. **Demo polish**: loading states, test-case picker, FHIR export mock, error handling.

### Risks

- LiteRT-LM Swift binding is "in dev" per their docs — may need to vendor the C++ core via a bridging header (see SKETCH.md fallback).
- 2.4 GB model doesn't fit in Play Store 200 MB base APK — Play Asset Delivery has a 1.5 GB per-pack limit, requires multi-pack split.
- Apple IPA hard limit is 4 GB, fits our model at 2.4 GB, but App Store review may push back on model-in-IPA vs download.

## Recommended starting sequence

**Day 0 (today, after C8 verifies):**
- Write the Tier 1 skill (0.5 day, mostly SKILL.md authoring).
- Test on the unmodified Play Store gallery app with base Gemma 4 E2B already in the allowlist. Measure extraction_score on the same 5 test cases.

**Day 1-2:**
- If Tier 1 quality ≥ 0.9 extraction_score → ship Tier 1 as "v0 demo." Start Tier 2 (upload model) for "v1 demo" next week.
- If Tier 1 < 0.9 → jump to Tier 2 directly. Upload, test, adjust.

**Day 3-4 (optional polish):**
- Tier 3 UI fork only if the chat-style UI is too clunky for the demo narrative.

## Open questions

- Does the gallery iOS app support adding custom HF models, or is it allowlist-locked? (Probably allowlist-locked — check `ios_1_0_0.json` update cadence.)
- Can Tier 1 skill invoke structured JSON output reliably from base (non-fine-tuned) Gemma 4? Unknown.
- Is `huggingface-cli login` on every user's phone acceptable UX? Or do we need a public-no-auth repo?
