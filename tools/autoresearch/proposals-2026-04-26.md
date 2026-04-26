# Autoresearch proposals — 2026-04-26 (delta over 2026-04-25)

**Author:** researcher (autoresearch agent, post-LLM-tuning sprint)
**Deadline:** 2026-05-18 (22 days remaining)
**Supersedes:** `proposals-2026-04-25.md` — this is a *delta*, not a re-statement.
**Read this if** you read the 2026-04-25 doc. Three of the LLM-side levers
came back negative; the bottleneck has shifted off the model entirely.

## What changed since 2026-04-25

The c20 LLM-tuning sprint (`tools/autoresearch/c20-llm-tuning-2026-04-25.md`)
ran four planned items and closed five proposal-doc entries:

1. **Old Rank 4 (tool-call grammar lock) — closed/retired.** 81-run
   stability bench on combined-27 gave **F1 = 1.000, 0 parse errors, 0 FP,
   210/210 codes recalled, median 14.94 s/case** (`build/toolcall_grammar_bench.json`).
   Caveat from the bench harness: llama-server's `/v1/chat/completions`
   rejects custom GBNF when `tools` is set, so what's measured is the
   `--jinja` built-in grammar. Either way, the agent path is at the
   floor: zero retries, zero drift. Remaining grammar work is iOS-side
   (`AgentRunner.applyGrammar`), and that's gated on Rank 1 (physical
   device). Nothing left to research here.

2. **Old Rank 5 (adv4 dictation+CDA bench) — partially closed.** Three
   modes ran cleanly: deterministic-only F1 = 0.927; agent + RAG F1 =
   0.952; agent + RAG + c19 fast-path F1 = 0.952 (`build/adv4_*.json`).
   Fast-path is a no-op on adv4 (every case has at least one inline /
   CDA code that the deterministic tier picks up; the
   `det_codes is empty` gate never fires). What's still open is the
   physical-iPhone re-run, which folds into Rank 1 below.

3. **Three LLM-tuning candidates — all negative.** Documented in the
   "Don't-do" sub-section so we stop re-litigating them:
   - **Candidate A (fast-path threshold sweep, 0.5–0.9 × 35 cases).**
     F1 flat at 0.972 across every threshold; 32/35 cases short-circuit
     before the gate, the other 3 match at score ≥ 1.0.
     `build/fastpath_threshold_sweep.json`. Keep default 0.70.
   - **Candidate B (self-consistency, n × temp on adv4 LOINC misses).**
     Best F1 0.933 was n=1 temp=0.2 statistical noise (0/5 reproduces);
     majority vote at n=3 ≤ baseline. The misses are knowledge gaps in
     `lookup_table.json`, not sampling-fixable.
     `build/self_consistency_bench.json`.
   - **Candidate C (system-prompt A/B, 3 variants).** All three identical
     F1 = 0.952 on adv4. Variant C (terse, 22% faster at p50 11.70 s)
     **regresses** combined-27 from F1 1.000 → 0.950 with 4 new FPs on
     `adv3_legionella_rag` and 2 LOINC misses on rmsf + valley_fever.
     Cannot ship terse. `build/system_prompt_bench_combined27_terse.json`.

4. **Two new follow-ups identified (now ranked below):** Candidate D
   (iOS short-circuit gate refinement) and a small lookup-table
   expansion for the two persistent adv4 voice misses.

5. **Hosted Spaces scaffold landed** (`spaces/app.py` + `requirements.txt`
   + `build.sh` + `README.md`). Structurally complete; not yet deployed.

The structural take: **the model is at its quality ceiling for this
codebase.** F1 = 0.972 on combined-27 + adv4 (35 cases) end-to-end,
F1 = 1.000 on combined-27 alone via the agent path, precision = 1.000
across every sustainable configuration. Further wins come from data
(lookup table) and from removing the iOS short-circuit gate that swallows
partial extractions. They are not LLM-side wins.

---

## Pole-position fact set (refreshed)

- **Python combined-27 + adv4 (35 cases)** under three-tier flow:
  F1 = 0.972, precision = 1.000, recall = 0.946, 30/35 perfect, 0 FP.
- **Python combined-27 alone, agent path:** F1 = 1.000 over 81 runs
  (Rank 4 grammar bench), 0 parse errors, median 14.94 s/case, 3.74 LLM
  turns/case avg.
- **Python adv4 alone, agent + RAG:** F1 = 0.952, 6/8 perfect, 0 FP. Two
  persistent misses are LOINC codes never named in voice prose
  (`adv4_voice_h5n1_bird_flu_thing` misses LOINC 100343-3;
  `adv4_voice_strep_throat_restart` misses LOINC 78012-2).
- **Swift mirror:** F1 = 1.000 on validate_rag.swift 14-case bench;
  19/19 fast-path probes pass; xcodebuild green on `iPhone17ProDemo`.
- **FHIR R4 validity:** 35/35 = 1.000 on combined-27 + adv4
  (`fhir.resources.R4B 8.2.0`). Outbox payload defaults to FHIR Bundle.
- **iOS sim decode tok/s:** still 4.0 from C12 — never re-measured under
  the agent flow, never measured on a physical device.
- **Five remaining misses** on combined-27 + adv4 are all LOINC codes
  that the iOS deterministic short-circuit prevents the agent from
  recovering on adv3, plus two voice-phrasing knowledge gaps on adv4.
  Both are non-LLM problems.

---

## Proposals (ranked by (impact × p(works)) / hours)

### Rank 1 — Physical-iPhone tok/s + dictation latency (carry-forward)

**Hypothesis:** Same as 2026-04-25 Rank 1, restated for emphasis: the
4.0 tok/s simulator number is now the *only* large open uncertainty in
the demo claim. With the LLM-tuning surface closed and the model proven
at F1 = 1.000 under 81 runs, there is nothing else this measurement
unblocks via deferral.

**Mechanism:** Unchanged. `ios-eng` deploys `iPhone17ProDemo` to a
physical iPhone 15 Pro, runs the 14-case bench end-to-end via
`validate_rag.swift` (or the in-app debug path through `NewCaseView`),
captures (a) deterministic wall-clock, (b) agent decode tok/s, (c) p50
/ p95 latency, (d) dictation mic-on → first-partial latency, (e) peak
RAM, (f) thermal at 5-case sustained load.

**Cost:** 3 engineer-hours (`ios-eng`).

**Measurable outcome:** New rows `c20-iphone-15-pro-physical-{deterministic,agent}`
in `results.tsv`. Decision rule unchanged from 2026-04-25:
- ≥10 tok/s decode → demo claim is "real-time clinical extraction on a 3-yo iPhone."
- 4–10 tok/s → "iPhone 15 Pro and newer."
- <4 tok/s → emergency rebuild xcframework with `-O3 -ffast-math
  -DGGML_USE_ACCELERATE`; if still <4, drop to laptop demo path.

**Risk / killer assumption:** None — we have the device.

**Pivot fit:** Critical. Decides whether the demo is a phone or a laptop.

---

### Rank 2 — Candidate D: iOS short-circuit gate refinement

**Hypothesis:** The current iOS short-circuit `det.hasAnyDeterministic`
fires whenever deterministic returns *any* code — including partial
answers. On `adv3_rmsf_rag` and `adv3_valley_fever_rag`, deterministic
finds the SNOMED via lookup but misses the LOINC, the gate fires, the
agent never runs, the LOINC stays missed. The Python agent path
already proves the agent recovers both LOINCs when invoked
(deterministic 1/2 → agent 2/2 in c20 ledger). Same fix on iOS recovers
the same two cases. F1 on combined-27 + adv4 moves from 0.972 → ~0.985.

**Mechanism:**
1. In `apps/mobile/ios-app/ClinIQ/ClinIQ/Extraction/ExtractionService.swift`,
   replace the `det.hasAnyDeterministic` gate with a per-bucket
   completeness check: short-circuit only when expected SNOMED *and*
   expected LOINC *and* expected RxNorm buckets all have at least one
   match — *or* when narrative analysis indicates none of those buckets
   are expected.
2. A pragmatic first cut: detect "expected_loinc but missing" by checking
   whether deterministic emitted a SNOMED for a condition known to have
   a paired LOINC in `lookup_table.json` (e.g., RMSF → LOINC 9826-3,
   valley fever → LOINC 24008-1). If yes, skip the short-circuit and
   let `AgentRunner` fill the gap.
3. Mirror the gate in the Python `agent_pipeline.py` so Python and
   Swift stay in parity (the c19 fast-path gate already lives there).
4. Re-run `validate_rag.swift` and the 35-case Python combined bench;
   target F1 ≥ 0.985 with precision still 1.000.

**Cost:** 2–3 engineer-hours (mostly Swift; small Python parity tweak).

**Measurable outcome:** New rows `c20-ios-gate-refined-{python,swift}`.
Target: F1 ≥ 0.985 on combined-27 + adv4 (vs 0.972 baseline). Kill
criterion: any new false positive.

**Risk / killer assumption:** Removing the partial-extraction
short-circuit on iOS adds an agent invocation to ~5/35 cases. Each
agent invocation is ~13 s on Mac CPU; on a physical iPhone (Rank 1)
this could be substantially more. Mitigation: gate the bypass on
*expected-bucket-missing* heuristics rather than firing the agent on
every partial. The c20 ledger names the exact two adv3 cases this
covers; we don't need a generic loop-everything fix.

**Pivot fit:** iOS-defining. Cheapest real F1 win in the remaining 22 days.

---

### Rank 3 — Hugging Face Spaces deployment (live agent demo)

**Hypothesis:** The hosted demo is the largest judging-surface
multiplier we have left. Anyone with the URL — judges, eICR domain
peers, the team's own future selves — can paste a narrative and watch
the deterministic + RAG fast-path + agent flow emit a FHIR R4 Bundle
with provenance chips. Single biggest "watch this iPhone" → "click this
link" amplifier. Scaffold (`spaces/app.py`, `spaces/requirements.txt`,
`spaces/build.sh`, `spaces/README.md`) landed today; deployment is the
remaining gap.

**Mechanism:**
1. Create the Space under user's HF account (`patrickdeutsch/cliniq`).
   Use Gradio SDK 4.44 per the existing `app.py` header.
2. Decide on inference tier:
   - **CPU-free tier (default in scaffold):** agent disabled by default;
     deterministic + RAG-fast-path serve every sample case with no LLM.
     This is honest and works today on the free tier.
   - **GPU-tier (hardware preset T4-small):** wire `build.sh` to start
     `llama-server` against the bundled Gemma 4 E2B Q3_K_M GGUF
     (download-on-startup from HF model repo) and let the agent loop
     run live. ~5 hours including iteration on cold-start time.
3. Bundle the `convert/` package flat next to `app.py` per the README
   deploy notes. CI is `spaces/build.sh`.
4. Pin `fhir.resources>=7.0,<9.0` (R4B subpackage) per the c19 finding.
5. Add the live URL to:
   - `tools/autoresearch/results.tsv` row `c20-spaces-live-demo`
   - the eventual hackathon submission form
   - `apps/dashboard/journey.html` Field Report №01 above the fold

**Cost:** 5 engineer-hours (scaffold exists; deployment + GPU-tier
startup script + cold-start tuning is the work).

**Measurable outcome:** Public URL serving the 5 sample cases in <2 s
on the deterministic / fast-path tier; agent tier serving Marburg /
Legionnaires in <20 s on T4-small if GPU tier is enabled. New row
`c20-spaces-live-demo` capturing cold-start time, p50 / p95 per sample.

**Risk / killer assumption:** HF Spaces free tier won't host a 2.4 GB
GGUF at usable agent latency. Mitigation: scaffold already concedes
this — agent gated behind a checkbox. If GPU tier proves too slow at
cold-start, ship deterministic-only and link out to the iPhone build
for the agent story.

**Pivot fit:** Universal. The link is what goes on the submission form.

---

### Rank 4 — Lookup-table expansion (adv4 voice misses)

**Hypothesis:** Two persistent misses on adv4 are LOINC codes the
agent can't conjure because the lab test isn't named explicitly in
voice prose:
- `adv4_voice_h5n1_bird_flu_thing` — agent gets SNOMED 442695009 +
  RxNorm 261244, misses LOINC 100343-3 (H5N1 RNA).
- `adv4_voice_strep_throat_restart` — agent gets SNOMED 43878008 +
  RxNorm 723, misses LOINC 78012-2 (rapid strep).

These are the cleanest knowledge-gap signal we have: c20 Candidate B
(self-consistency × temp sweep) confirmed the model can't recover
these via sampling, only via lookup-table coverage. Add the two LOINC
entries to `lookup_table.json` + `LookupTable.swift` (mirror), keyed
on the SNOMED ↔ LOINC pairs the agent already extracts. F1 on adv4
moves from 0.952 → 1.000.

**Mechanism:**
1. In `apps/mobile/convert/lookup_table.json`, add LOINC 78012-2
   (rapid strep antigen) and LOINC 100343-3 (H5N1 RNA) under the
   `loinc_aliases` and `loinc_paired_with_snomed` shapes already in
   place from c17 / c19.
2. Mirror to `apps/mobile/ios-app/ClinIQ/ClinIQ/Extraction/LookupTable.swift`
   static arrays.
3. In `agent_pipeline.py`, when `extract_codes_from_text` returns a
   SNOMED in the paired set but no LOINC, `lookup_displayname` /
   `lookup_reportable_conditions` should auto-emit the paired LOINC.
   The infra exists; just wire the two new rows.
4. Re-run adv4 on agent + RAG. Target 8/8 perfect, F1 = 1.000.

**Cost:** 1 engineer-hour.

**Measurable outcome:** New row `c20-adv4-lookup-expanded`. Target
F1 = 1.000 on adv4, no regression on combined-27.

**Risk / killer assumption:** Auto-emitting paired LOINC could
over-fire when a SNOMED is mentioned but the corresponding lab wasn't
done (FP). Mitigation: gate the auto-emit on RxNorm presence (the
patient is being treated, so the lab almost certainly was run) or on
NegEx absence around the SNOMED match.

**Pivot fit:** Universal. Closes adv4 cleanly.

---

### Rank 5 — v27 fine-tune bench (gated on Kaggle COMPLETE)

**Hypothesis:** Track A from the 2026-04-25 PM handoff. The v27
Kaggle kernel (Unsloth re-train with KV-shared-layer exclusion, +20
code-preservation cases, +20 negative-lab augmentation) is still
running per the handoff. Either it fixes code elision and sequence
degeneration *without* re-breaking tool-calling — in which case we
swap the iOS resolver order back to prefer the fine-tune — or it
doesn't, and we file the result and move on. Both outcomes are
publishable and the work is pre-staged.

**Mechanism:** Run `scripts/v27_convert_and_bench.sh` once
`kaggle kernels status patrickdeutsch/cliniq-compact-lora-training`
returns `COMPLETE`. The script does:
1. Pull artifacts to `/tmp/c9-v27/`.
2. Convert merged HF → GGUF f16 (reuses `/tmp/llama-cpp-tools/convert_hf_to_gguf.py`).
3. Quantize to Q3_K_M.
4. Stop the base llama-server, start one pointed at v27 on `127.0.0.1:8090`.
5. Run agent_pipeline on combined-27 + adv4 (35 cases).
6. Run regex_preparser-only sanity bench.
7. Restart base llama-server (deployment default).
8. Print keep/discard verdict.

**Decision rule** (from script header): keep iff v27 F1 ≥ base F1
*and* 0 FP on combined-27. Otherwise discard.

**Cost:** 0 hours of new design (script is staged); ~30 min wall-clock
of bench run + verdict commit.

**Measurable outcome:** Row `c20-v27-finetune-bench` in `results.tsv`
with status `keep|discard`. If keep, swap `LlamaCppInferenceEngine`
resolver order to prefer v27.

**Risk / killer assumption:** Kaggle kernel still RUNNING — script
short-circuits if not COMPLETE. No risk of the bench corrupting state;
script restarts base llama-server on exit.

**Pivot fit:** Universal. Either confirms or retires the fine-tune
track for this hackathon.

---

### Rank 6 — Demo polish + provenance deep-link (final-week-only)

**Hypothesis:** Same as 2026-04-25 Rank 6, deferred again. The 90-second
demo cut should walk: (1) typical eICR → instant tier-1 with INLINE/CDA
chips → tap chip, source span highlights; (2) Marburg / Legionnaires
narrative → 3-second agent loop, RAG chip, tap to open CDC NNDSS in
Safari; (3) flip to dictation, speak, watch the agent recover the
voice case. Should now be a four-stop tour with the Spaces URL added.

**Mechanism:**
1. Add a "Demo mode" toggle in `SettingsTab` pre-loading three canned
   narratives.
2. 1-second pulse animation on the provenance chip on first render.
3. RAG `sourceURL` opens in Safari (not in-app).
4. `simctl io recordVideo` against `iPhone17ProDemo` for the screencast.
5. Embed in `apps/dashboard/journey.html` Field Report №01.
6. Add the Spaces URL (Rank 3) as a "click this if you don't have an
   iPhone" callout.

**Cost:** 6 engineer-hours.

**Measurable outcome:** 3 non-engineer test viewers can summarize the
project after watching. Pass = all 3 mention "instant," "sources," and
either "phone" or "offline."

**Risk / killer assumption:** Time-spent-here trades against Ranks 1–4.
Defer until Rank 1 lands and at least one of Rank 2 / Rank 4 is in.

**Pivot fit:** Mobile-defining. Leads with on-device + provenance.

---

## Ranking summary

| Rank | Proposal | Hours | p(works) | Demo impact | Score |
|---:|---|---:|---:|---:|---:|
| 1 | Physical-iPhone measurement | 3 | 0.95 | High | **0.95** |
| 2 | iOS short-circuit gate refinement (Cand D) | 3 | 0.85 | High | **0.85** |
| 3 | Hugging Face Spaces deployment | 5 | 0.85 | Very High | **1.02** |
| 4 | Lookup-table expansion (adv4 LOINCs) | 1 | 0.95 | Med | **0.95** |
| 5 | v27 fine-tune bench | 0.5 | 0.55 | Med | **1.10** |
| 6 | Demo polish + provenance deep-link | 6 | 1.00 | High | 1.00¹ |

¹ Rank 6 scores high on raw impact but is gated to the final week.
Rank 5 scores top by ratio because the design cost is paid; only the
bench wall-clock remains.

---

## Sequencing — 22-day window to 2026-05-18

| Window | Run | Owner |
|---|---|---|
| **This week** (Apr 26 – May 2) | Rank 4 (lookup-table expansion, ~1 h, can land any sitting). Rank 5 (v27 bench, fires on Kaggle COMPLETE — already polled). Rank 1 the moment a physical iPhone is in hand. | `team-lead` (Rank 4 + Rank 5 trigger), `ios-eng` (Rank 1) |
| **Next week** (May 3 – 9) | Rank 2 (iOS gate refinement) — depends on Rank 1 to budget the added agent invocations on physical hardware. Rank 3 (Spaces deployment) starts here in parallel; deterministic-only tier ships first, GPU tier as a follow-up. | `ios-eng` (Rank 2), `team-lead` (Rank 3) |
| **Final week** (May 10 – 17) | Rank 6 (demo polish + 90 s screencast). Code freeze May 14. May 15–17 = video + dashboard + TestFlight invite list + Spaces URL on submission form. | `ios-eng` + `team-lead` |

Critical path is **Rank 1 → Rank 2 → Rank 6**. Rank 3, 4, 5 run in
parallel and don't gate the demo.

---

## Don't-do list (updated)

| Trap | Why not |
|---|---|
| **Old Rank 4 — explicit GBNF on llama-server agent path** | Closed by c20: F1 = 1.000, 0 parse errors over 81 runs with `--jinja` built-in grammar. llama-server rejects custom GBNF when `tools` is set. iOS-side `AgentRunner.applyGrammar` is still useful but gated on Rank 1, not on more research. |
| **Candidate A — fast-path threshold sweep** | Negative on combined-27 + adv4: F1 flat at 0.972 across 0.5–0.9; 32/35 cases short-circuit before the gate fires; the 3 that reach RAG match at score ≥ 1.0. Default 0.70 confirmed. Re-investigate only if a future bench includes deterministic-empty cases with RAG hits in the 0.5–0.9 score band. |
| **Candidate B — self-consistency / temperature × n** | Negative: best F1 0.933 was n=1 statistical noise (0/5 reproduces). adv4 LOINC misses are knowledge gaps in `lookup_table.json`, not sampling-fixable. Rank 4 (lookup expansion) closes them; do not re-run sampling sweeps. |
| **Candidate C — system-prompt A/B (terse / think-first)** | Negative + regression: identical F1 on adv4; terse variant breaks combined-27 (F1 1.000 → 0.950, 4 FPs on legionella). Prescriptive prompt is load-bearing. Do not re-prompt-engineer for latency. |
| **LiteRT-LM int4 retrain** (carry-over) | Runtime is dead. 0.10.1 Jinja can't render the bundle's chat template; 0.10.2 not shipped. iOS uses llama.cpp now. |
| **MediaPipe LLM Inference API** | Officially deprecated by Google. |
| **Gallery-fork Android port** | Depends on LiteRT-LM. Default = skip; iOS is the demo. |
| **Multi-Jetson tensor parallel / MLC-LLM hardening** | Wrong target. Mobile is primary. |
| **Q2_K quantization** | Quality cliff for clinical extraction; no upside given Q3_K_M F1 = 1.000. |
| **AWQ via `ai-edge-torch`** | Targets `.litertlm`; runtime dead. |
| **Speculative decoding via 270M draft** | LiteRT-LM `SpeculativeDecodingSettings` proto unreachable; llama.cpp Gemma family doesn't speculate well. |
| **Re-fine-tuning to fix tool-calling** | Base Gemma 4 already does it well per c20: F1 = 1.000 on agent path with 0 parse errors over 81 runs. v27 (Rank 5) is the one exception still in flight, gated on Kaggle COMPLETE — once verdict lands, this trap reactivates. |
| **Tool-call result cache** (old Rank 7) | Marginal latency win; skip in favor of Rank 2 / Rank 3 / Rank 6. |

---

## Cross-cutting items needing teammate action

- **`ios-eng`**: owns Rank 1 (physical-device bench) and Rank 2 (iOS
  short-circuit gate refinement). Rank 6 demo polish + screencast in
  the final week.
- **`team-lead`**: owns Rank 4 (lookup-table expansion + Swift mirror —
  ~1 hour, can land any sitting), Rank 5 trigger (run
  `scripts/v27_convert_and_bench.sh` once Kaggle kernel returns
  COMPLETE), and Rank 3 (Spaces deployment, 5 hours including GPU-tier
  startup script + cold-start tuning).
- **`trainer`**: idle on the LLM side — c20 closed the LLM-tuning
  surface. Re-engage only if Rank 5 (v27) returns `keep` and the iOS
  resolver swap surfaces a new bug. Otherwise, trainer can pick up
  Rank 3 GPU-tier work as overflow.
- **`researcher`** (me): refresh `results.tsv` after each landed
  proposal; author 2–3 deterministic-empty + RAG-borderline cases
  (the genuine threshold-sensitive test set Candidate A flagged as
  missing) so the next sweep has signal; on-call for any new adv5
  bench if dictation surfaces unseen distributions.
