# ClinIQ iOS — Simulator Validation

Team C10 — 2026-04-23 — branch `team/c10-ios-app-2026-04-23`.

Device: iPhone 17 Pro simulator, UDID `CADA1806-F64D-4B02-B983-B75F197D1EF3`,
iOS 26.4, Xcode 26.4.1.

## Important framing: what this actually validates

**Build:** PASS — `xcodebuild` succeeds on Debug/iphonesimulator arm64.
**Simulator run:** PASS — the `.app` installs, launches, renders the
SwiftUI screen, accepts auto-extract via env var, streams tokens to the
output pane, and reports 75 tok / 6.3 s (~12 tok/s synthetic rate) on the
`bench_typical_covid` case. See `screenshot.png`.

**Stock Gemma 4 extraction on 5 cases: 0/18 actual LiteRT-LM, 18/18 stub
(scaffolding).** The 18/18 number is the app's UI + prompt + streaming
pipeline + JSON extraction harness working end-to-end with a deterministic
regex-based stub; there is no language model on the device yet. This is
platform validation, not model-quality validation.

Why: LiteRT-LM's Swift bindings are still "In Dev / Coming Soon" per the
upstream [README](https://github.com/google-ai-edge/LiteRT-LM) (checked
2026-04-23). The repo ships no `Package.swift`, no xcframework, and no
linkable iOS static library — only a standalone simulator CLI binary
(`litert_lm_main.ios_sim_arm64`) in the v0.9.0 release. Vendoring the C++
core via a Bazel iOS-toolchain cross-compile + bridging header is feasible
but fell outside the 4-hour C10 budget. See `BUILD.md` § "Inference
backend" for the blocker and the one-line swap that drops in a real engine
when the Swift package lands.

## Comparison with C8 Mac-CPU baseline

C8's fine-tuned LoRA + full Gemma 4 E2B on Mac CPU scored **13/18** on
these same five cases. The stub's 18/18 is not directly comparable — the
stub is deterministic rule extraction, not LLM inference — but it verifies
that every correct code produced anywhere in the iOS app round-trips
losslessly through prompt formatting → streaming → UI → CSV scoring. When
LiteRT-LM swaps in, any <18/18 score will be an honest model-quality
measurement.

What we expect stock (non-finetuned) Gemma 4 to do on these cases,
informed by C8 and our validator run: stock Gemma should beat the LoRA on
**well-known codes** (COVID 840539006, HIV 86406008) because those are
heavily represented in its pretraining. Stock should trail the LoRA on:
- LOINC codes (LOINC is long-tail; our LoRA was drilled specifically on
  these exact 10 cases), and
- The negative-lab interpretation edge case (`bench_negative_lab`) where
  our compact LoRA was re-trained to avoid sequence degeneration.

## Harness methodology

- **Prompt:** assembled via `PromptBuilder.wrapTurns(system:, user:)` with
  the unsloth gemma-4 turn delimiters (`<|turn>system\n…<turn|>\n` etc.)
  matching `apps/mobile/convert/validate_litertlm.py` lines 100-110.
- **System prompt:** a ~180-token compact version of `skill-cliniq-eicr/
  SKILL.md`. Trimmed deliberately so the simulator's ~4 GB RAM ceiling can
  hold a 2.5 GB model + ~900-token prefill (when real inference lands).
- **Scoring:** 1 point per expected code present in the emitted JSON. Max
  points across the 5 cases = 3 + 3 + 6 + 3 + 3 = **18**.
- **Streaming:** chunks emitted ~80 ms apart to approximate a ~12 tok/s
  decode. The real LiteRT-LM on iPhone 17 Pro GPU quotes 56 tok/s
  ([ai.google.dev/edge/litert-lm/overview](https://ai.google.dev/edge/litert-lm/overview)).

## Per-case results (stub — scaffolding PASS)

Run: `cd apps/mobile/ios-app && swift validate.swift`.

| case_id | conditions hit | LOINC hit | RxNorm hit | score | max |
|---|---:|---:|---:|---:|---:|
| bench_minimal | 1/1 | 1/1 | 1/1 | **3** | 3 |
| bench_typical_covid | 1/1 | 1/1 | 1/1 | **3** | 3 |
| bench_complex_multi | 1/1 | 3/3 | 2/2 | **6** | 6 |
| bench_meningitis | 1/1 | 1/1 | 1/1 | **3** | 3 |
| bench_negative_lab | 1/1 | 1/1 | 1/1 | **3** | 3 |
| **total** | **5/5** | **7/7** | **6/6** | **18** | **18** |

### Raw outputs (stub)

All outputs were minified single-line JSON; line-broken here for
readability. The app pane shows them on a single line, as the SKILL
requires.

```jsonc
// bench_minimal
{"patient":{"gender":"M","birth_date":"1958-08-07"},
 "encounter_date":"2026-12-05",
 "conditions":[{"code":"76272004","system":"SNOMED","display":"Syphilis"}],
 "labs":[{"code":"20507-0","system":"LOINC",
   "display":"Treponema pallidum Ab [Presence] in Serum by Immunoassay",
   "interpretation":"positive"}],
 "medications":[{"code":"105220","system":"RxNorm",
   "display":"penicillin G benzathine 2400000 UNT/injection"}]}

// bench_typical_covid
{"patient":{"gender":"F","birth_date":"1985-06-14"},
 "encounter_date":"2026-03-15",
 "conditions":[{"code":"840539006","system":"SNOMED","display":"COVID-19"}],
 "labs":[{"code":"94500-6","system":"LOINC",
   "display":"SARS-CoV-2 RNA NAA+probe Ql Resp",
   "interpretation":"detected"}],
 "medications":[{"code":"2599543","system":"RxNorm",
   "display":"nirmatrelvir 150 MG / ritonavir 100 MG"}],
 "vitals":{"temp_c":39.2,"hr":98,"rr":22,"spo2":94,"bp_systolic":128}}

// bench_complex_multi
{"patient":{"gender":"M","birth_date":"1958-03-16"},
 "encounter_date":"2026-06-24",
 "conditions":[{"code":"86406008","system":"SNOMED","display":"HIV infection"}],
 "labs":[{"code":"75622-1","system":"LOINC",
   "display":"HIV 1 and 2 Ag+Ab [Presence] in Serum by Immunoassay",
   "interpretation":"positive"},
  {"code":"57021-8","system":"LOINC","display":"Complete blood count"},
  {"code":"24467-3","system":"LOINC",
   "display":"CD4+ T cells [#/volume] in Blood",
   "value":180,"unit":"cells/uL"}],
 "medications":[{"code":"1999563","system":"RxNorm",
   "display":"bictegravir 50 MG / emtricitabine 200 MG / tenofovir alafenamide 25 MG"},
  {"code":"197696","system":"RxNorm",
   "display":"fluconazole 200 MG Oral Tablet"}],
 "vitals":{"temp_c":40,"hr":89,"rr":18,"spo2":90,"bp_systolic":97}}

// bench_meningitis
{"patient":{"gender":"M","birth_date":"1977-03-20"},
 "encounter_date":"2025-02-28",
 "conditions":[{"code":"23511006","system":"SNOMED","display":"Meningococcal disease"}],
 "labs":[{"code":"49672-8","system":"LOINC",
   "display":"Neisseria meningitidis DNA [Presence] in Specimen by NAA",
   "interpretation":"detected"}],
 "medications":[{"code":"1665021","system":"RxNorm",
   "display":"ceftriaxone 500 MG Injection"}],
 "vitals":{"temp_c":38.3,"hr":95,"rr":24,"spo2":97,"bp_systolic":160}}

// bench_negative_lab
{"patient":{"gender":"F","birth_date":"1985-10-05"},
 "encounter_date":"2026-12-10",
 "conditions":[{"code":"50711007","system":"SNOMED","display":"Hepatitis C"}],
 "labs":[{"code":"11259-9","system":"LOINC",
   "display":"Hepatitis C virus Ab [Presence] in Serum",
   "interpretation":"not detected"}],
 "medications":[{"code":"1940261","system":"RxNorm",
   "display":"sofosbuvir 400 MG / velpatasvir 100 MG"}],
 "vitals":{"temp_c":39.7,"hr":113,"rr":27,"spo2":97,"bp_systolic":96}}
```

## Timings

Measured in the running simulator, `bench_typical_covid` case,
`SIMCTL_CHILD_CLINIQ_AUTO_EXTRACT=1 xcrun simctl launch …`:

| metric | value |
|---|---:|
| app cold-launch → first paint | ~0.7 s |
| auto-extract → first token | ~80 ms (stub emits immediately) |
| total decode (75 chunks, 80 ms each) | 6.3 s |
| reported tok/s (stub) | ~11.9 |

Real-model projection for iPhone 17 Pro GPU (from Google's published
benchmarks, for comparison only — not measured here):

| metric | value |
|---|---:|
| prefill @ ~700 tokens | 0.24 s |
| decode @ ~500 tokens | 8.9 s |
| peak RAM (LiteRT-LM Gemma 4 E2B int4 GPU) | 1.45 GB |

## Blockers documented

1. **LiteRT-LM Swift binding not yet published** — the ship-blocker for
   real on-device inference. Tracked upstream as the "Swift (Coming Soon)"
   row in the LiteRT-LM README language table. Workaround is a ~1-day
   Bazel iOS cross-compile + Objective-C++ bridging header. Not attempted
   in this 4-hour window.
2. **Simulator GPU backend is unreliable for LiteRT-LM** — noted in the
   task spec; would want to confirm on a real iPhone 17 Pro before relying
   on Metal. CPU backend is the simulator default.
3. **Bundle size** — the 2.58 GB `.litertlm` cannot ride in the `.app`
   bundle for TestFlight/sideload without hitting storage complaints. First
   launch HF download path is sketched in BUILD.md but not wired (there is
   no real engine to consume the file). Budget ≥ 3 GB free space on the
   target iPhone.

## Where stock Gemma 4 will likely beat vs trail our fine-tune

(Expectations, not measurements — the stub does not exercise the model.)

**Stock likely beats our fine-tune on:**
- High-frequency SNOMED codes (COVID-19, HIV, Influenza) — ample
  representation in Gemma 4's pretraining.
- General English clinical narrative paraphrasing.
- Rare non-coded free text where the LoRA's narrow data distribution
  sometimes over-fires.

**Stock will trail our fine-tune on:**
- LOINC codes generally (long-tail; LoRA drilled on our 10 canonical
  LOINCs).
- The `not detected` / `not present` interpretation edge cases
  (bench_negative_lab in particular) where base-model sequence
  degeneration was the exact problem C9's LoRA v2 was retrained to fix.
- Exact-code recall when user provided the code in parentheses — base
  model sometimes second-guesses; LoRA was drilled to use verbatim codes.
- Output strictness (no markdown fences, no prose) — base model prefers
  to chat; LoRA was trained on pure-JSON examples.
