# Autoresearch proposals — 2026-04-24

**Author:** autoresearch agent (orchestrator pass)
**Deadline:** 2026-05-18 (24 days remaining)
**Pivot reference:** 2026-04-23 — mobile (LiteRT-LM) replaces Jetson as primary target.
**Current pole-position fact set:**

- C16 retrain `cliniq-gemma4-e2b.litertlm` (int4, 2.55 GB) scores **18/29 = 0.621** on the 9-case bench (`scripts/test_cases.jsonl`, macOS CPU, `validate_all_cases.py`).
- 5/9 cases are perfect; 2 cases (`bench_negative_lab`, `bench_complex_multi`) hit int4 sequence degeneration ("repeat-zero" tail).
- C8 Mac CPU reference for the same fine-tune via llama.cpp/GGUF: **13/18 = 0.722** (6-case slice). Mobile int4 is ~10pp lower than CPU GGUF on the comparable slice.
- C14 Swift decode on iPhone 17 Pro **simulator CPU**: 13.84–15.55 tok/s; on-device GPU expected 52–56 tok/s per upstream Google numbers.
- C12 `llama.cpp` iOS sim: 4.0 tok/s, 12/18 = 0.667 — strictly inferior to LiteRT-LM on speed, comparable on quality.
- KV-sharing-aware Unsloth retrain (C9 r=32, 1650 rows, response-only loss, 205 trainable modules) is **already queued** on Kaggle T4 — do **not** re-propose it.

The hackathon-judge axes I optimized for: (a) demo robustness on the 9-case bench, (b) latency on a "3-yo phone" floor (iPhone 14 / Pixel 7), (c) a credible offline FHIR story, (d) at least one "wow" datapoint a non-engineer judge will remember.

Each proposal block uses the format requested in the brief.
Ranking is by `(impact × p(works)) / engineer-hours`; rank-1 is the highest ratio.

---

## Rank 1 — Anti-degeneration constrained decoding via grammar-locked JSON schema

**Hypothesis (one sentence):** Forcing the int4 model to emit a strict JSON schema via a tokenizer-level grammar (LiteRT-LM's `formatter`/sampler-side constraint or a llama.cpp GBNF on the GGUF fallback path) eliminates the "repeat-zero" tail on `bench_negative_lab` and `bench_complex_multi` *without retraining*, lifting aggregate score from 0.62 → ~0.85+.

**Mechanism (what to actually change):**

1. Author a GBNF / EBNF for our exact extraction schema: `conditions[]` (string of 6–9 digits), `loincs[]` (`\d+-\d`), `rxnorms[]` (digits), terminated by `}`. Keep it generous on order/whitespace, strict on the *tail* — the grammar must require either a comma-separated next element or `]`/`}`, never a digit-after-zero state machine that allows degenerate repetition.
2. For the LiteRT-LM path: integrate via `LlmInferenceParams.formatter` (proto field — newly exposed in 0.10.2) or, if unavailable, run a thin **token-level repetition rejector** at the C-shim layer (`apps/mobile/litertlm-swift/Sources/LiteRtLmCShim/litertlm_c_shim.cc`) that vetoes any sampled token whose decoded substring matches `(0)\1{6,}`. The shim already owns the decode loop.
3. For llama.cpp/GGUF fallback (C12 + Mac validator): use `--grammar-file cliniq.gbnf` directly. Note the prior `--json-schema {}` crash (results.tsv row 26) was an empty-schema sampler init bug, not a grammar feature failure — we author a *real* grammar.
4. Validate via `apps/mobile/convert/validate_all_cases.py --grammar cliniq.gbnf` against the same 9 cases.

**Cost estimate:** **3 engineer-hours.**
- 1 h grammar authoring (existing schema is small).
- 1 h LiteRT-LM `formatter` plumbing + iOS rebuild.
- 1 h shim-side repetition rejector as a safety net.
- No Kaggle, no Jetson reflash, no Mac-Studio compute.

**Measurable outcome:** `experiments.avg_extraction_score` on the C16 row jumps from 0.694 → ≥0.85 (i.e., 25/29 across the 9 cases). Specifically: `bench_negative_lab` 0/3 → 2/3, `bench_complex_multi` 1/6 → 5/6. The 5 currently-perfect cases must not regress (kill criterion). Decode tok/s should fall by ≤10% — the grammar prunes branches but the int4 weight read is unchanged.

**Risk / killer assumption:** LiteRT-LM 0.10.2's exposed sampler API may not accept arbitrary grammars (Google ships an `xnnpack` formatter primarily for tool-use). If `formatter` is restricted, we still have the shim-side repetition rejector — slower path but proven (we own that code). Fallback degrades the proposal to "anti-repetition only" and recovers ~half the gain.

**Pivot fit:** **High on both axes.** Helps mobile demo immediately; on Jetson llama.cpp side it also patches the `bench_negative_lab` failure that has plagued every earlier sweep.

---

## Rank 2 — Two-pass extraction: tiny prefill + regex-first short-circuit

**Hypothesis:** Because the SNOMED/LOINC/RxNorm codes are already verbatim in the eICR prompt, a regex pre-scan can correctly emit codes for ~80% of cases with **zero LLM tokens**, and the LLM only runs on the residual (free-text deciding which condition is "primary," disambiguating "negative" labs, summary). Result: median end-to-end latency on iPhone 14 drops from ~46 s to ~6 s, and `extraction_score` on code fields hits **1.00** by construction on every prompt that contains the codes literally — independent of int4 quality.

**Mechanism:**

1. Add a Swift-side `EicrPreparser` (in `apps/mobile/ios-app/ClinIQ/`) that scans the input for `SNOMED \d{6,9}`, `LOINC \d+-\d`, `RxNorm \d+`, plus negation context (`Not detected`, `Negative`).
2. Build a deterministic JSON skeleton from the regex hits.
3. Only invoke `LiteRtLmEngine` if (a) the input lacks parenthesized codes (free-text-only eICR — currently 0/9 of our bench but realistic for messy eCR clones), or (b) for the optional `summary` / `primary_dx` fields the regex can't compute. When LLM runs, it gets a **constrained** prompt of <100 tokens, not the full 700.
4. Display in the iOS UI: "Codes: deterministic. Summary: AI-generated."

**Cost estimate:** **5 engineer-hours.**
- 2 h regex + Swift implementation (mirrors the existing `apps/mobile/ios-app/ClinIQ/.../tolerant parser` from commit `2f4511b`).
- 1 h JSON-skeleton builder.
- 1 h benchmark wiring through `validate_all_cases.py`.
- 1 h judge-narrative copy ("offline-first hybrid, deterministic on codes").

**Measurable outcome:** On the 9-case bench, code-level extraction goes to **29/29 = 1.00** (regex hits all printed codes). Aggregate `extraction_score` published to DuckDB clears 0.95. Wall-clock per case (iPhone 14 GPU projection): from 46 s to ~6 s for cases with codes, ~25 s for code-less cases. New `experiment_name`: `c17-hybrid-regex-llm-ios`.

**Risk / killer assumption:** Real-world eICRs often *don't* have inline codes — they're in CDA `<code code="…">` XML elements. The 9 bench cases all have parenthesized codes, which makes this look better than it is on naturalistic input. **Mitigation:** add 3 "code-stripped" variants of bench cases to the test suite as a stress test before claiming the win publicly. Even with stripped cases, the proposal still wins because the regex path is additive — it never makes things worse.

**Pivot fit:** Native to mobile, but identical logic ports to the Jetson llama.cpp path (it's a Python regex pre-pass before the prompt is sent). Doubles as a story for "LLM as the *last* line of defense, not the first."

---

## Rank 3 — Quality-preserving int8 weights via mixed-precision LiteRT-LM bundle

**Hypothesis:** Switching the LiteRT-LM quantization recipe from `dynamic_wi4_afp32` (current, int4 weights) to `dynamic_wi8_afp32` (int8 weights) on attention/MLP while keeping `lm_head` and embeddings at fp16 will eliminate the digit-token logit drift identified in C7 DIAGNOSIS.md (S2), recovering ~80% of the 10pp mobile-vs-CPU quality gap, at the cost of ~1.6× model size (2.55 GB → ~4.0 GB) and ~30% decode slowdown (52 tok/s → ~37 tok/s) on iPhone 17 Pro GPU.

**Mechanism:**

1. In `apps/mobile/convert/merge_and_convert.py`, change the `--quant` flag to `dynamic_wi8_afp32` (the `litert_torch` exporter supports this — the recipe key is in `ai_edge_torch.generative.quantize.quant_recipes`).
2. Pin **`lm_head`** and `embed_tokens` at fp16 with `quant_recipes.full_fp16_recipe()` selectively masked to those modules — exact addressing pattern follows `ai-edge-torch/.../gemma4` example. The lm_head is the suspect for digit-token errors per C7.
3. Build `cliniq-gemma4-e2b-int8.litertlm`, target file size ≤4.2 GB to stay inside the Apple IPA 4 GB limit (drop tokenizer JSON if needed).
4. Re-run `validate_all_cases.py` on macOS CPU first (fast feedback), then seed into the iOS sim and the on-device build.
5. Bench peak RAM — must stay under 4 GB on iPhone 14's 6 GB total to keep that device in the demo.

**Cost estimate:** **6 engineer-hours.**
- 1 h recipe edit + selective quant addressing.
- 1 h re-run of the conversion pipeline (~6 min wall time per the C16 REGEN.md log).
- 2 h iOS rebuild + simulator + on-device validation.
- 2 h decode tok/s + RAM benchmarking on at least one physical phone.

**Measurable outcome:** New row `c17-litertlm-int8-macos-cpu` and `c17-litertlm-int8-ios`. `avg_extraction_score` lifts from 0.694 → ≥0.83 on the 9-case bench. Decode tok/s on iPhone 17 Pro GPU: stays ≥35 (still 35× the Jetson 7W number, still inside the 60 s/case demo budget). Peak RAM: report measured value; kill criterion is >4 GB on a 6 GB device.

**Risk / killer assumption:** `litert_torch`'s int8 path may not be supported for Gemma 4's KV-shared layers — the same architectural friction C6 fought through to ship int4. If int8 conversion fails, we're stuck in a 1-day debug spiral. **Mitigation:** time-box to 6 h; if the converter errors out within 90 minutes, drop to **`dynamic_wi4_afp32` with selective fp16 lm_head only** (a quarter-step that should still recover the digit-token problem).

**Pivot fit:** Mobile-only. On Jetson the equivalent is "use Q4_K_M instead of Q3_K_M" (already in our results.tsv at score 1.00). This proposal exists *because* the mobile pivot regressed a quality dimension that GGUF didn't.

---

## Rank 4 — Speculative decoding via MedGemma-1B distilled draft (LiteRT-LM CPU/GPU)

**Hypothesis:** The known-bad llama.cpp speculative decoding result (results.tsv, "spec-decode SKIPPED — Gemma family doesn't speculate well") was about an **un-distilled** small model. A purpose-distilled 1B draft (or even a 270M Gemma 4 nano if it lands in time) running on CPU while the int4 E2B target runs on GPU should give 1.5–2× decode tok/s on iPhone 17 Pro because mobile is bandwidth-bound exactly like Jetson was. The recent llama.cpp Gemma 4 day-0 shows reasonable distill-then-speculate gains in community benchmarks.

**Mechanism:**

1. Pull `litert-community/gemma-4-270m-it-litert-lm` (if shipped — verify on HF) or distill our own 270M from the merged compact LoRA via `ai_edge_torch` distillation recipe (4 h on Mac Studio CPU, no T4 needed for a 270M target).
2. Extend `apps/mobile/litertlm-swift/Sources/LiteRtLmCShim/litertlm_c_shim.cc` with a `litertlm_session_set_draft_model(path)` symbol, calling LiteRT-LM's `SpeculativeDecodingSettings` (proto field, 0.10.2).
3. Bench on iPhone 17 Pro physical device; target 1.5×.

**Cost estimate:** **8 engineer-hours.**
- 4 h distill / verify draft model availability.
- 2 h shim plumbing.
- 2 h iPhone bench.

**Measurable outcome:** `c17-litertlm-spec-decode-iphone-17-pro-gpu` row at ≥80 tok/s decode (vs 56 tok/s baseline). Quality unchanged (verifier rejects bad drafts).

**Risk / killer assumption:** LiteRT-LM 0.10.2 may not actually expose a speculative-decoding entry point (Google's docs show it for the C++ engine but the Kotlin/Swift surface may lag). If absent, this is dead — drop to Rank 5.

**Pivot fit:** Mobile-first; the Jetson MLC-LLM port could later inherit the same draft model (5–8 → 12+ tok/s plausible). Useful for both branches.

---

## Rank 5 — Adversarial bench expansion: 5 messy real-world eICR cases

**Hypothesis:** The current 9-case bench is too clean (all have parenthesized SNOMED/LOINC/RxNorm codes inline). Judges from public-health backgrounds will probe with realistic CDA XML where codes live in `<code code="…">` attributes mixed with displayName attributes, plus prose-only narratives. We need a *believable* failure-mode story before the demo, not during. Adding 5 adversarial cases now turns one weakness (only 9 cases) into one strength (battle-tested on harder inputs). It also gates Rank 1/2/3 from over-fitting their changes to the easy bench.

**Mechanism:**

1. Take 5 publicly-available CDA eICR samples from CDC EZeCR / HL7 sample bundles (the user has prior CDC EZeCR experience per memory).
2. Hand-author ground truth in the same `expected_conditions/loincs/rxnorms` schema.
3. Add 3 "code-attribute-only" variants (codes in XML attrs, not parenthesized) and 2 prose-only variants (clinician narrative, no codes — only display strings the LLM must look up).
4. Update `scripts/test_cases.jsonl` from 9 → 14 cases.
5. Re-baseline C16 on the new bench; expect a drop to ~0.50 aggregate, which is **the honest number** we should be optimizing.

**Cost estimate:** **4 engineer-hours.**
- 2 h case curation + ground truthing.
- 1 h validator regex tweaks for XML-attribute extraction.
- 1 h re-baseline.

**Measurable outcome:** A new `experiments` row `c17-bench-expanded-baseline` at the new (lower) aggregate score, plus *every subsequent proposal scored on the harder bench*. This is a quality-of-research-program change as much as a numerical change.

**Risk / killer assumption:** None really — worst case the new cases are too hard and everything looks bad in the report. The benefit is that the demo team can rehearse against a realistic input distribution. **Killer:** if licensing on a particular CDA sample is unclear, drop it; CDC EZeCR samples are public-domain US-government works.

**Pivot fit:** Universal. Bench improvements help every runtime equally.

---

## Rank 6 — Submission narrative: "From 0.88 to 56 tok/s" data-rich landing page + 90-second demo video

**Hypothesis:** The single biggest delta in hackathon outcomes is whether judges can **understand the technical journey in 90 seconds**. We already have `apps/dashboard/journey.html` (Field Report №01) and `demo-video/cliniq-demo.mp4`. Investing 6 hours to (a) add a live in-browser demo of the 5-perfect cases running on a real phone via `litert-lm-cli` over WebSerial / video screencap, and (b) re-cut the demo video to lead with the Jetson 0.9 → mobile 56 tok/s pivot, is worth more judge-points than another 3pp on extraction score.

**Mechanism:**

1. Embed an autoplay 30 s video in `journey.html` of the iOS app extracting `bench_typical_covid` end-to-end on an iPhone 17 Pro, with a stopwatch overlay (use `simctl io` recording → `ffmpeg`).
2. Split-screen the same case running on Jetson at 0.9 tok/s — visceral comparison.
3. Add a "Try it" QR code linking to a TestFlight build (gated invite — judges-only).
4. Edit `demo-video/cliniq-demo.mp4` to a 90 s cut: 0–15 s problem, 15–45 s the pivot insight, 45–75 s live extraction, 75–90 s offline + FHIR validity.

**Cost estimate:** **6 engineer-hours** (mostly video editing).

**Measurable outcome:** Not a benchmark row. Soft metric: "judges can understand the project without reading a README." Pre-registered: get 3 non-engineer test viewers to summarize the project after watching the video. Pass = all 3 mention "offline" and "phone."

**Risk / killer assumption:** Time spent here is time not spent on Rank 1/2/3 quality wins. Defer until at least Rank 1 lands.

**Pivot fit:** Mobile-pivot-defining. Don't run this until the pivot is irreversibly the demo path.

---

## Rank 7 — Output schema redesign: short-form FHIR-lite reduces output token count by 60%

**Hypothesis:** Our current model emits ~170 tokens of JSON per `bench_complex_multi` case. A short-form schema (`{"c":["86406008"],"l":["75622-1","57021-8","24467-3"],"r":["1999563","197696"]}` — 70 tokens) cuts decode time by 60% and reduces the surface area on which int4 sequence degeneration can fire. Client-side post-processor expands `c/l/r` → `conditions/loincs/rxnorms` and adds the FHIR scaffolding.

**Mechanism:**

1. Update training prompt and validator to accept both schemas (so we don't have to retrain to test).
2. Add a Swift post-processor that maps the short keys back to canonical FHIR R4 resource arrays.
3. Re-bench with the short schema on the same 9 cases.

**Cost estimate:** **3 engineer-hours**, no retraining.

**Measurable outcome:** Decode tok/s × tokens-per-case → wall-clock per case. Should drop from 8.5 s to ~4 s on macOS CPU C16. Aggregate score should rise modestly because shorter output = less room for degeneration.

**Risk / killer assumption:** The model was fine-tuned on the long schema. Without retraining, it may not reliably emit the short keys — only a few-shot example in the prompt to test. If it doesn't take, drop or stack with the queued C9 retrain (add a short-schema variant to the training mix; ~1 h add).

**Pivot fit:** Universal. Same gain on Jetson llama.cpp.

---

## Rank 8 — A15-class device floor benchmark: actual iPhone 14 measurement

**Hypothesis:** The iPhone 14 number (~12 tok/s) is *projected* from `modelfit.io` and AI Edge Gallery community reports — we don't have a measured number ourselves. The remote-clinic narrative collapses if a 3-yo phone actually OOMs or thermal-throttles to single-digit tok/s on sustained workload. We need to *measure* this before the demo, not at it.

**Mechanism:**

1. Borrow / source one iPhone 14 (A15, 6 GB RAM) from team member or used-iPhone marketplace.
2. Provision the C15 ClinIQ build, run all 9 bench cases back-to-back, measure tok/s + peak RAM + thermal throttling (3-case sustained run).
3. Publish row `c17-litertlm-iphone-14-gpu-measured`.

**Cost estimate:** **4 engineer-hours** (includes provisioning).

**Measurable outcome:** Replace the projected 12 tok/s row with a measured number. If measured ≥10 tok/s on 9-case sustained, ship that as a key demo claim ("works on a 3-yo phone"). If <5 tok/s or thermal-trips, narrow the demo claim to "iPhone 15+" honestly.

**Risk / killer assumption:** iPhone 14 not borrowable inside the 24-day window. Fall back to iPhone SE 3rd gen (also A15) if the team has one, or a Pixel 7 (Tensor G2).

**Pivot fit:** Mobile-only. Doesn't help Jetson.

---

## Rank 9 — FHIR R4 validity scoring on the 9-case bench

**Hypothesis:** Judges who care about clinical interop will want to see "the output passes a FHIR R4 validator." Today we measure code-presence but never check that the model's JSON, when wrapped in a FHIR `Bundle`, actually validates against the R4 spec. Shipping a row that says "9/9 cases produce R4-valid Bundles" is a categorical credibility signal that score-deltas don't deliver.

**Mechanism:**

1. Add `apps/mobile/convert/score_fhir.py` that takes the model output + a Bundle template, runs it through `fhir.resources` (Python lib) or `fhirpath` for structural validation.
2. New scoring axis `fhir_r4_pass_rate` in the DuckDB schema.
3. Re-bench C16 — expect ~6/9 pass given the JSON-shape variability we already see; that becomes an explicit KPI.

**Cost estimate:** **5 engineer-hours.**

**Measurable outcome:** New schema column populated; the C16 row shows `fhir_r4_pass_rate`. Subsequent proposals must hold or improve this number.

**Risk / killer assumption:** `fhir.resources` strict validation may reject every output for trivial reasons (missing meta.profile, etc.). Mitigation: validate against a "permissive profile" (just `Bundle.entry[*].resource.resourceType` legal + required fields present) on first pass.

**Pivot fit:** Universal. Demo-only narrative impact.

---

## Rank 10 — AWQ baseline comparison (concrete: convert merged HF → AWQ → re-quantize via `ai-edge-torch`'s int4 import)

**Hypothesis:** Naive int4 (`dynamic_wi4_afp32`, our current C16 path) loses more accuracy on long-tail digit tokens than AWQ-derived int4 because AWQ preserves activation-aware salient channels at higher precision. Convert the merged HF model to AWQ via `autoawq`, then *re-import* the AWQ-int4 weights into the LiteRT-LM bundle (via `ai-edge-torch`'s pre-quantized weight import path, which exists for symmetric int4 sources). Target: +0.10 extraction score on `bench_negative_lab` and `bench_complex_multi` without regressing the 5 perfect cases.

**Mechanism:**

1. `pip install autoawq && autoawq quantize build/cliniq-gemma4-e2b-merged --w_bit 4 --group_size 128 --out build/cliniq-gemma4-e2b-awq`.
2. Use `ai_edge_torch.generative.quantize.import_pretrained_int4(...)` (this API exists for `gemma2`; verify it lands for `gemma4` — if not, this proposal is dead).
3. Bench on macOS CPU; if win, push to iOS sim.

**Cost estimate:** **4 engineer-hours.**

**Measurable outcome:** `c17-litertlm-awq-int4-macos-cpu` row. Aggregate score ≥0.78 (vs 0.694 baseline). 5 currently-perfect cases must remain perfect.

**Risk / killer assumption:** **`ai-edge-torch`'s pre-quantized-int4 import path may not be supported for Gemma 4 yet** — the C16 retry only ran the dynamic post-training quantization recipe. If unsupported, AWQ output isn't reachable through the LiteRT-LM bundler, the proposal is dead, and we fall back to Rank 3 (mixed-precision int8).

**Pivot fit:** Mobile-only.

---

## Ranking summary table

| Rank | Proposal | Hours | p(works) | Demo impact | Score = impact·p / hours |
|---:|---|---:|---:|---:|---:|
| 1 | Grammar-locked anti-degeneration | 3 | 0.85 | High | **2.55** |
| 2 | Regex pre-pass / hybrid extraction | 5 | 0.95 | High | **1.90** |
| 3 | Mixed-precision int8 LiteRT-LM bundle | 6 | 0.55 | Med-High | 0.92 |
| 4 | Speculative decoding via 270M draft | 8 | 0.40 | Med | 0.45 |
| 5 | Adversarial bench (CDA XML, prose-only) | 4 | 0.95 | Med | 0.95 |
| 6 | Demo video + judge-facing landing | 6 | 1.00 | Med-High | 1.00 |
| 7 | Short-form output schema | 3 | 0.55 | Low-Med | 0.92 |
| 8 | iPhone 14 measured floor | 4 | 0.80 | Med | 0.80 |
| 9 | FHIR R4 structural validity scoring | 5 | 0.85 | Med | 0.68 |
| 10 | AWQ-int4 import path | 4 | 0.30 | Med | 0.45 |

**Recommended sprint sequencing for the 24-day runway:**

- **Days 1–3** (this week): Rank 1 + Rank 2 in parallel. These two together should clear extraction_score 0.90+.
- **Days 4–7**: Rank 5 (bench expansion) — re-baseline everyone honestly. Rank 9 (FHIR validity) folded in.
- **Days 8–14**: Pick *one* of Rank 3 or Rank 10 based on `ai-edge-torch` API availability check (1 h spike on day 8).
- **Days 15–20**: Rank 8 (real iPhone 14 measurement) + Rank 7 if extraction is still wobbly.
- **Days 21–24**: Rank 6 (demo polish). Lock the model bundle on day 22; days 23–24 are video-only.

Don't run Rank 4 unless Ranks 1–3 plateau — speculative decoding is the kind of "engineer-attractive but judge-invisible" bet that historically eats schedule.
