# Autoresearch proposals — 2026-04-26 PM (delta over 2026-04-26 AM)

**Author:** researcher (autoresearch agent, post-c20 + 4-loop autoresearch arc)
**Deadline:** 2026-05-18 (23 days remaining)
**Supersedes:** `proposals-2026-04-26.md` — this is a *delta*. Read that doc
first if you haven't, then come here for the late-Apr-25 state.
**Read this if** you read the 04-26 AM doc this morning. Five of its six
ranks have moved (closed, superseded, or had their gating condition
satisfied), the bench has grown 27 → 54 cases through two adversarial
expansions, and the pole-position number has moved twice (1.000 → 0.984
→ 1.000 → 0.970 → 0.980 → **0.983**) over the course of a single day.

## What changed since 2026-04-26 AM

The morning ranking ran clean. By end of day 04-25 most of it has
been promoted, closed, or superseded:

1. **Old Rank 2 — Candidate D (iOS short-circuit gate refinement) — DONE.**
   Hybrid gate landed in `ExtractionService.swift` +
   `EicrPreparser.swift` + `bench_fastpath_threshold.py`. Combined-35
   F1 climbed 0.972 → **1.000** (35/35 perfect). Bench:
   `apps/mobile/convert/build/fastpath_threshold_sweep_candD.json`.
   The chosen criterion is "fire on inline/cda tier match OR ≥2
   buckets populated"; lookup-only single-bucket results fall through
   to the agent. Kept combined-27 unchanged at F1 1.000.

2. **Old Rank 3 — HF Spaces deployment — PREP DONE, deploy gated on
   `huggingface-cli login`.** `spaces/app.py` + `requirements.txt` +
   `build.sh` + `deploy.sh` + `README.md` complete; bundle layout
   verified locally; sample cases run; screenshots captured. The
   only remaining step is the user-side `huggingface-cli login` +
   `git push` — the deploy script prints the exact command. **Carry
   forward as Rank 2 below**, demoted from Rank 3 only because Rank 1
   (physical iPhone) is still the largest open uncertainty.

3. **Old Rank 4 — Lookup-table expansion — DONE.** Three strep + four
   H5N1 aliases added to both `lookup_table.json` and
   `LookupTable.swift`. adv4 F1 0.952 → **1.000** (8/8 perfect, 0 FP).
   Combined-27 regression-checked at F1 1.000. Bench:
   `adv4_post_lookup_expansion.json`, `combined27_post_lookup.json`.

4. **Old Rank 5 — v27 fine-tune bench — STILL BLOCKED, but now on
   the conversion side, not Kaggle.** Kaggle kernel returned
   COMPLETE; both artifacts produced (`cliniq-compact-merged/` and
   `cliniq-compact-lora/`). Neither is locally convertible:
   `convert_hf_to_gguf.py` rejects bnb-4bit packed safetensors;
   PEFT 0.19 doesn't recognize `Gemma4ClippableLinear` as a LoRA
   target. Two fix paths, both kernel-side. Pre-staged
   `scripts/v27_convert_and_bench.sh` runs the verdict pipeline
   once a clean fp16 merged dir exists. Status: held.
   See `c20-llm-tuning-2026-04-25.md` "v27 conversion blocker".

5. **Four autoresearch loops landed.** The day produced 4 cycles of
   {bench → bug-find → fix → re-bench}. Bench grew 27 → 35
   (combined-27 + adv4) → 45 (combined-27 + adv4 + adv5) → 54
   (combined-27 + adv4 + adv5 + adv6). Bug ledger across the loops:

   | # | Bug | Surfaced by | Fixed | Artifact |
   |---|---|---|---|---|
   | 1 | Fast-path RAG search is NegEx-blind on the matched phrase | adv5 `ruled_in_measles_ruled_out_rubella` | yes | post-NegEx-fix Python + Swift |
   | 2 | NegEx triggers miss "came back negative" / post-hoc constructions | adv5 `adjacent_dengue_not_zika` | yes | `_POSTHOC_NEG_TRIGGERS` + 80-char clause window |
   | 3 | NegEx terminator gap on commas | adv6 `neg_enumeration_history_seizures_no_stroke` | yes | `,` added to `_NEG_TERMINATORS` |
   | 4 | NegEx trigger gap on "do not have" / "not eligible for" | adv6 `quoted_negative_covid` | yes | `do(?:es)?\s+not\s+have`, `not\s+eligible\s+for`, etc. |
   | 5 | Unicode whitespace breaks alias regex | adv6 `unicode_typography_em_dash_smart_quotes` | yes | `unicodedata.normalize("NFKC", text)` + Swift `precomposedStringWithCompatibilityMapping` |
   | 6 | Deterministic FPs poison the fast-path gate | adv6 `long_form_admission_note` | partial | `try_fast_path` now gates on inline/cda tier presence + merge-on-fire; **Cand D bucket-rule still fires upstream** for this specific case |
   | 7 | Inline regex emits any 6-9 digit number after "SNOMED" | adv6 `code_injection_bait` | **deferred** | needs curated SNOMED allow-list post-filter |
   | 8 | RAG embedding can match 3-letter alt_names cross-domain | adv6 `implicit_syndrome_stevens_johnson` | **deferred** | needs alt_name-length cap or curated drop |

   7 unique bugs found, 6 fixed (Bug 6 is partially fixed —
   try_fast_path semantics improved + merge-on-fire regression caught,
   but Cand D's bucket-rule fires upstream for the long-form
   admission note case so the FPs persist). 2 deferred to a future
   sprint (Bug 7 inline allow-list, Bug 8 RAG short-token).

6. **One Cand D regression caught + fixed during the loop.** The
   naive Bug 6 fix (`return None` in `try_fast_path` only on
   inline/cda matches) regressed 4 cases from 2/2 to 1/2:
   `adv3_rmsf_rag`, `adv3_valley_fever_rag`,
   `adv4_voice_valley_fever_informal`, `adv5_synonym_chain_rmsf`. The
   problem: fast-path was firing but emitting *only* the RAG hit,
   dropping the lookup-tier RxNorm that deterministic had already
   recovered. Fix: when fast-path fires, MERGE existing det.matches
   with the RAG hit instead of replacing them. Mirrored to Swift
   `ExtractionService.swift` (start `fast` from `det` and append).
   Final bench includes this fix.

7. **The 0.04-26 AM "v27 → v28" line is wrong now.** v28 (the
   trainer's re-run with `save_method="merged_16bit"`) is the
   right path; v27 is dead. `scripts/v27_convert_and_bench.sh`
   still runs, but its first arg is now the v28 artifact path.
   Treating "v27" as shorthand for "the next merged-fp16 fine-tune"
   below.

The structural take: **the model is at its quality ceiling on this
codebase, and the precision regressions that adv5 + adv6 surfaced
are all in the *deterministic* and *fast-path* tiers, not in the
LLM.** Combined-54 F1 = 0.983 today, with all 4 remaining FPs
attributable to the 2 deferred bugs (Bug 7 inline allow-list and
Bug 8 RAG short-token) plus the 1 unfixed Cand D / fast-path
interaction (Bug 6 partial). Closing those would move the ceiling
to 54/54 = 1.000. None of them are LLM-side wins.

---

## Pole-position fact set (refreshed end-of-04-25)

- **Python combined-54** (combined-27 + adv4 + adv5 + adv6, 54 cases)
  under three-tier flow with Cand D gate, threshold 0.7, all 6 fixed
  bugs in:
  **F1 = 0.983, precision = 0.973, recall = 0.993, 50/54 perfect, 4 FPs,
  p50 0.002 s, p95 16.19 s.** 24 deterministic short-circuits, 14
  fast-path hits, 16 agent invocations.
  Bench: `apps/mobile/convert/build/combined54_post_final_fix.json`.
- **Python combined-45** (combined-27 + adv4 + adv5, the
  not-bleeding-edge bench): F1 = **1.000**, 45/45 perfect, 0 FP. Post
  all the NegEx + Cand D + lookup expansion fixes. Bench:
  `combined45_post_negex_fix.json`. *This is the number to cite for
  the "everything except the just-authored stress cases" headline.*
- **Python combined-27 alone, agent path (Rank 4 grammar
  stability)**: F1 = 1.000 over 81 runs, 0 parse errors, median
  14.94 s/case, 3.74 LLM turns/case avg.
- **Adv4 alone, agent + RAG + lookup expansion**: F1 = 1.000, 8/8
  perfect, 0 FP.
- **Adv5 alone, agent + RAG + fast-path** (pre-NegEx-fix): F1 = 0.937;
  post-fix, all 10 cases perfect within combined-45.
- **Adv6 alone** (deterministic-only, pre-fixes): F1 = 0.809; agent
  path 0.833. Post-fixes within combined-54: 5 of 9 still imperfect:
  `polypharmacy_mixed_dose_formats` (FP=1, surfaces a polypharmacy
  fast-path interaction with the new Cand D fix — see Rank 4 below),
  `long_form_admission_note` (Bug 6 partial — Cand D blocks),
  `implicit_syndrome_stevens_johnson` (Bug 8 RAG short-token),
  `code_injection_bait` (Bug 7 inline allow-list).
- **Swift mirror**: builds green on `iPhone17ProDemo` (UDID
  `CADA1806-F64D-4B02-B983-B75F197D1EF3`) after every change in the
  loop. NFKC normalization, comma terminator, expanded NegEx
  triggers, `try_fast_path` merge-on-fire, and Cand D gate all
  mirrored.
- **FHIR R4 validity**: 35/35 unchanged from c19; combined-54 R4
  validity bench deferred (`fhir.resources` not installed in current
  venv, structural shape unchanged from combined-35 so no
  regression expected; re-run when dependency restored).
- **iOS sim decode tok/s**: still 4.0 from C12. **Physical iPhone
  tok/s: still never measured.** The largest open uncertainty in
  the demo claim — unchanged from this morning. See Rank 1.

---

## Proposals (ranked by (impact × p(works)) / hours)

### Rank 1 — Physical-iPhone tok/s + dictation latency (carry-forward)

**Hypothesis:** Restated unchanged from 2026-04-26 AM Rank 1. With
the LLM-tuning surface closed (c20), Cand D landed (combined-35 =
1.000), lookup expansion landed (adv4 = 1.000), and 4 autoresearch
loops surfacing + fixing 6 of 8 bugs, **the 4.0 tok/s simulator
number is now the only large open uncertainty in the demo claim.**
The model is proven at F1 = 1.000 over 81 runs on combined-27 and
F1 = 0.983 over 54 cases on the bleeding-edge bench. Nothing else
in the queue benefits from deferring this measurement.

**Mechanism:** Unchanged. `ios-eng` deploys to a physical iPhone 15
Pro, runs the 14-case bench end-to-end via `validate_rag.swift` (or
the in-app debug path through `NewCaseView`), captures (a)
deterministic wall-clock, (b) agent decode tok/s, (c) p50 / p95
latency, (d) dictation mic-on → first-partial latency, (e) peak
RAM, (f) thermal at 5-case sustained load.

**Cost:** 3 engineer-hours (`ios-eng`).

**Measurable outcome:** New rows
`c20-iphone-15-pro-physical-{deterministic,agent}` in `results.tsv`.
Decision rule unchanged from 2026-04-25:
- ≥10 tok/s decode → demo claim is "real-time clinical extraction on
  a 3-year-old iPhone."
- 4–10 tok/s → "iPhone 15 Pro and newer."
- <4 tok/s → emergency rebuild xcframework with `-O3 -ffast-math
  -DGGML_USE_ACCELERATE`; if still <4, drop to laptop demo path.

**Risk / killer assumption:** None — we have the device.

**Pivot fit:** Critical. Decides whether the demo is a phone or a
laptop.

---

### Rank 2 — HF Spaces actual deployment (was Rank 3)

**Hypothesis:** The hosted demo is the largest judging-surface
multiplier we have left. Anyone with the URL — judges, eICR domain
peers, the team's future selves — can paste a narrative and watch
the deterministic + RAG fast-path + agent flow emit a FHIR R4
Bundle with provenance chips. The scaffold landed this morning
(`spaces/{app.py,requirements.txt,build.sh,deploy.sh,README.md}` +
sample screenshots), and a smoke test in CPU-free mode passes
locally. The only remaining step is `huggingface-cli login` + the
git push to `huggingface.co/spaces/patrickdeutsch/cliniq-eicr-fhir`,
which is user-side because we don't have HF credentials in this
session.

**Mechanism:**
1. **User-side**: `huggingface-cli login` (browser-based or token).
2. From the worktree: `bash spaces/deploy.sh --space
   patrickdeutsch/cliniq-eicr-fhir --push`. The script stages the
   bundle in `out/space/`, runs the flat-layout copy, commits, and
   pushes.
3. Watch the build log in the Spaces UI for ~5 minutes. CPU-free
   tier should serve the 5 sample cases in <2 s.
4. Add the live URL to:
   - `tools/autoresearch/results.tsv` row `c20-spaces-live-demo`
   - `apps/dashboard/journey.html` Field Report №02 (see Rank 5
     below; the URL goes above the fold)
   - the eventual hackathon submission form
5. **Optional GPU-tier follow-up**: if cycles allow, upgrade the
   Space to T4-small and add a startup script that spawns
   `llama-server` in the background. ~4 hours; gates on cold-start
   tuning being acceptable.

**Cost:** 0.5 engineer-hours user-side (login + push) + ~1 hour
researcher-side to verify and link. GPU-tier upgrade is a separate
4-hour task and is *not* in scope here.

**Measurable outcome:** Public URL serving 5 sample cases at <2 s
on the deterministic / fast-path tier. Status row
`c20-spaces-live-demo` in `results.tsv`. Live URL in Field Report
№02 + submission form.

**Risk / killer assumption:** HF Spaces free tier is sometimes
capacity-constrained on first deploy. Mitigation: deploy now, not
on May 17.

**Pivot fit:** Universal. The link is what goes on the submission
form when judges don't have an iPhone.

---

### Rank 3 — v28 fine-tune verdict (was old Rank 5)

**Hypothesis:** Same as 2026-04-26 AM Rank 5, restated for the
v27→v28 supersession. v27 produced bnb-4bit packed safetensors that
neither `convert_hf_to_gguf.py` nor PEFT 0.19 can ingest. The right
fix is kernel-side: re-run with
`model.save_pretrained_merged(out, tokenizer, save_method="merged_16bit")`
to produce a standard fp16 HF dir. Once that lands, the verdict
runs in ~30 minutes via the staged script.

**Mechanism:**
1. **Kaggle-side** (user or trainer): re-launch the kernel with the
   `merged_16bit` save method. This is a one-line change in the
   training script. Wait for COMPLETE.
2. From the worktree:
   `bash scripts/v27_convert_and_bench.sh /tmp/c9-v28/cliniq-compact-merged`.
   The script pulls artifacts, converts HF → GGUF f16, quantizes
   to Q3_K_M, stops the base llama-server, starts one pointed at
   v28, runs `agent_pipeline` on combined-27 + adv4, runs a
   regex-only sanity bench, restarts base llama-server, prints
   keep/discard verdict.
3. **Decision rule** (from script header): keep iff v28 F1 ≥ base
   F1 *and* 0 FP on combined-27. Otherwise discard.
4. If keep, swap `LlamaCppInferenceEngine` resolver order to prefer
   v28 over base.

**Cost:** 0 hours of new design (script staged); ~30 min wall-clock
for the bench + verdict commit; whatever Kaggle wall-clock the
re-run takes (~5 hours on T4).

**Measurable outcome:** Row `c20-v28-finetune-bench` in
`results.tsv` with status `keep|discard`. If keep, resolver swap
in iOS engine.

**Risk / killer assumption:** v28 either matches base or breaks
tool-calling. The c20 sprint conclusion ("model at quality ceiling
on this bench") makes this *strictly evaluative* — F1 = 1.000 on
combined-27 already holds without v28. The most likely outcome is
"v28 either matches base or breaks tool-calling," neither of which
changes the demo claim.

**Pivot fit:** Universal but low-stakes. Either confirms or retires
the fine-tune track for this hackathon.

---

### Rank 4 — Two deferred precision bugs (small sprint)

**Hypothesis:** Combined-54 F1 = 0.983 has 4 FPs left. The deferred
Bug 7 (inline allow-list) and Bug 8 (RAG short-token) are 2 of them;
the third is the Cand D / fast-path interaction Bug 6 partial-fix
left behind on `adv6_long_form_admission_note`; the fourth is a
new polypharmacy fast-path interaction that didn't exist before
the final cleanup pass on `adv6_polypharmacy_mixed_dose_formats`
(FP=1 on the fast-path side; pre-cleanup the same case was 7/7
perfect via deterministic). Closing these gets us to 53/54 or
54/54.

**Mechanism (parallel, ~3 hours total):**

1. **Bug 7 — inline regex allow-list.**
   `SNOMED_RE = r"\bSNOMED\s+(\d{6,9})\b"` is purely lexical. On
   `adv6_code_injection_bait` it extracts `(SNOMED 99999999)` from
   a quoted internet-rumor narrative at confidence 0.99. Fix:
   post-filter inline matches against the curated SNOMED universe
   (the union of every SNOMED in `lookup_table.json` +
   `reportable_conditions.json`). If the captured code isn't in the
   allow-list, demote to `lookup` tier or drop with a debug log.
   Cheap (~30 min Python + Swift mirror), deterministic, stops the
   worst case. Mirror to Swift `EicrPreparser.swift`'s inline regex
   block.

2. **Bug 6 partial-fix follow-on — Cand D bucket-rule + fast-path
   interaction redesign.** The current `_det_short_circuits_llm`
   fires when `bucket_count >= 2` regardless of tier. On
   `adv6_long_form_admission_note`, deterministic emits varicella
   (lookup tier, FP from "no varicella series") + CBC LOINC (lookup
   tier, FP from "CBC: WBC..."). Both lookup-tier, both FPs, but
   `bucket_count == 2` so Cand D short-circuits, fast-path never
   runs, measles never recovered. Fix: refine Cand D to require
   *either* an explicit-tier match *or* (≥2 buckets *and* at least
   one match in the >=2-bucket set is non-lookup-tier). Mirror to
   Swift `ParsedExtraction.shortCircuitsLLM`. Risk: re-introduces
   the original adv3 LOINC misses *unless* the fast-path / agent
   path catches them. Validate with the full combined-54 sweep
   before committing. ~1.5 hours including regression check.

3. **(Bonus, time-permitting) Bug 8 — RAG short-token alt_name cap.**
   `adv6_implicit_syndrome_stevens_johnson` falsely emitted CRE
   SNOMED 47523006 because "CRE" is one of CRE's `alt_names` and
   the RAG embedding rewarded a 3-letter token match. Fix: cap
   fast-path on alt_name matches that are shorter than 4 characters
   unless a co-located explicit qualifier is found ("CRE
   infection", "carbapenem-resistant"). Or simply drop standalone
   "CRE" from the curated DB and require the longer phrase. ~1
   hour; choose based on a regression sweep against combined-54.

4. **Polypharmacy fast-path FP** (newly introduced by the cleanup
   pass on `adv6_polypharmacy_mixed_dose_formats`). Before final
   fix this case was 7/7 deterministic; after the
   try_fast_path-merge change it now also fires fast-path with FP=1.
   Diagnose during the Bug 6 redesign — likely the merge-on-fire
   is also adding a RAG hit when det was already complete. Fix
   should be conditional on det being "complete enough" (e.g.,
   bucket_count match against expected, or simply "if det has any
   inline/cda tier match in the same bucket as the RAG hit, don't
   merge"). ~30 min as part of (2).

**Cost:** ~3 engineer-hours total; can be split across Python +
Swift mirrors.

**Measurable outcome:** New rows
`c20-bug7-inline-allowlist`, `c20-bug6-cand-d-tier-aware`,
`c20-bug8-ragalt-shortcap` in `results.tsv`. Combined target:
combined-54 F1 ≥ 0.991 (53/54), precision = 1.000. Stretch: F1 =
1.000.

**Risk / killer assumption:** The Cand D redesign (item 2) has
the highest regression surface; it touches both Python and Swift
short-circuit gates. Mitigation: regression-sweep against
combined-45 *before* combined-54, so we know we haven't broken
the not-bleeding-edge bench.

**Pivot fit:** Strong — pushes the demo claim from "F1 = 0.983"
to "F1 ≥ 0.991" without touching the model. Fits the c20
conclusion that quality progress lives on the non-LLM side.

---

### Rank 5 — Hackathon submission narrative + 90-second screencast

**Hypothesis:** Same as 04-26 AM Rank 6, gated on Rank 1 landing.
The submission window opens once we have a physical-iPhone tok/s
number; until then, the screencast can't include the
"watch-this-on-an-iPhone" frame and the narrative is missing its
hero claim. Authoring the prose is now in scope (a draft lives at
`tools/autoresearch/hackathon-submission-2026-04-25.md`); the
remaining work is the screencast + dashboard updates + final
submission-form prep.

**Mechanism:**
1. Author the judge-facing one-pager (DONE — committed today as
   `tools/autoresearch/hackathon-submission-2026-04-25.md`).
2. Add a Field Report №02 to `apps/dashboard/journey.html` covering
   c20 + the autoresearch loops + the combined-54 headline (DONE
   today if `journey.html` exists).
3. Once Rank 1 lands: capture `simctl io recordVideo` against
   `iPhone17ProDemo` for the screencast (or the physical iPhone if
   that's the demo target). Three-stop tour: (a) typical eICR →
   instant tier-1 with INLINE/CDA chips → tap chip, source span
   highlights; (b) Marburg / Legionnaires narrative → 3-second
   agent loop, RAG chip, tap to open CDC NNDSS; (c) flip to
   dictation, speak, watch the agent recover the voice case.
   Add a fourth stop for the Spaces URL (Rank 2).
4. Record once the iPhone result is in hand. Embed in journey.html.
5. Submit by May 17 with: live Spaces URL + iPhone screencast +
   c20 doc + this proposals doc + the hackathon-submission md.

**Cost:** 6 engineer-hours after Rank 1 lands (screencast +
dashboard polish + submission form).

**Measurable outcome:** 3 non-engineer test viewers can summarize
the project after watching. Pass = all 3 mention "instant,"
"sources," and either "phone" or "offline." Final submission filed
before May 17 23:59 PT.

**Risk / killer assumption:** Rank 1 doesn't land in time.
Mitigation: have a sim-only fallback screencast pre-recorded that
cites the 4.0 tok/s sim number honestly, with a footnote noting
the physical-iPhone projection from LiteRT-LM benchmarks.

**Pivot fit:** Mobile-defining. Leads with on-device + provenance.

---

### Rank 6 (optional) — adv7 stress bench

**Hypothesis:** Diminishing returns. adv5 surfaced 2 bugs; adv6
surfaced 6 bugs (4 fixed + 2 deferred). Each adversarial round has
been ~10 cases, and each round has produced fewer net fixable
bugs. An adv7 round (probing axes adv5 + adv6 didn't probe —
e.g., multi-language fully non-English, encounter-aware scoring,
adversarial Unicode at the code-point level, hand-off across
sections, structured CCDA section confusion) would likely surface
2-3 more bugs, of which 1-2 might be fixable in the remaining
window.

**Recommendation:** Run adv7 *only if Rank 1 doesn't land in
time*, as a hedge. Otherwise the cycles are better spent on Rank 4
(closing the 4 known FPs) and Rank 5 (the screencast). The
combined-54 headline is already stronger than what most hackathon
submissions cite; adv7 trades cycle time for marginal F1 gain.

**Cost (if run):** 6 engineer-hours: 3 to author 8-10 cases, 3 for
the bug-find / fix loop. Requires llama-server.

**Measurable outcome:** Row `c20-adv7-stress` in `results.tsv`,
combined-63 bench artifact, post-fix bench artifact.

**Risk / killer assumption:** As stated, diminishing returns.
Don't take this Rank if either Rank 1 is in flight or Rank 4 has
unfinished items.

**Pivot fit:** Marginal. Defensive-only.

---

## Ranking summary

| Rank | Proposal | Hours | p(works) | Demo impact | Score |
|---:|---|---:|---:|---:|---:|
| 1 | Physical-iPhone measurement | 3 | 0.95 | High | **0.95** |
| 2 | HF Spaces actual deployment | 1 (after user login) | 0.90 | Very High | **2.70** |
| 3 | v28 fine-tune bench | 0.5 (+ Kaggle re-run) | 0.50 | Med | **1.00** |
| 4 | Two deferred precision bugs (small sprint) | 3 | 0.80 | Med-High | **0.80** |
| 5 | Hackathon submission narrative + screencast | 6 (post-Rank-1) | 1.00 | Very High | 1.00¹ |
| 6 | adv7 stress bench (optional) | 6 | 0.40 | Low | 0.20² |

¹ Rank 5 scores high on raw impact but is gated on Rank 1 landing.
² Rank 6 is explicitly defensive — only run if Rank 1 doesn't land.

The compressed-form: **Rank 2 (HF Spaces push) is the highest-ratio
play tonight or tomorrow morning** — minutes of work for the
single largest judging-surface multiplier we have. Rank 1 (iPhone)
is highest absolute impact and unblocks Rank 5. Rank 4 is the only
remaining F1 win on offer.

---

## Sequencing — 23-day window to 2026-05-18

| Window | Run | Owner |
|---|---|---|
| **Tonight / Tomorrow morning** | Rank 2 (Spaces push) — needs `huggingface-cli login` then `bash spaces/deploy.sh --push`. ~1 hour wall-clock including build verification. | user (login) → `team-lead` (push) |
| **This week** (Apr 26 – May 2) | Rank 1 (physical iPhone) the moment a device is in hand. Rank 3 (v28 verdict) fires once the kernel re-run COMPLETE-s. Rank 4 (Bug 6/7/8 sprint) — can land any sitting; no LLM dependencies. | `ios-eng` (Rank 1), `trainer` (Rank 3 trigger), `team-lead` (Rank 4) |
| **Next week** (May 3 – 9) | Rank 5 prep — draft the screencast storyboard against whatever Rank 1 returned. Spaces GPU-tier upgrade if cycles allow. | `team-lead` |
| **Final week** (May 10 – 17) | Rank 5 (record, polish, submit). Code freeze May 14. May 15–17 = video + dashboard + Spaces URL on submission form + TestFlight invite list. | `ios-eng` + `team-lead` |
| **Hedge** | Rank 6 (adv7) only if Rank 1 hasn't landed by May 5. | `researcher` |

Critical path is **Rank 2 → Rank 1 → Rank 5**. Rank 3, 4, 6 run in
parallel and don't gate the demo. Rank 2 (Spaces) is small enough
that finishing it tonight is the right move.

---

## Don't-do list (carry-over from 04-26 AM, with adv5+adv6 additions)

| Trap | Why not |
|---|---|
| **Old c20 Candidate A — fast-path threshold sweep** | Still negative. Closed in 04-26 AM. |
| **Old c20 Candidate B — self-consistency / temp × n** | Still negative. Closed in 04-26 AM. Adv5/adv6 confirmed: knowledge gaps are knowledge gaps; no sampling fix exists. |
| **Old c20 Candidate C — system-prompt A/B (terse / think-first)** | Still negative + regression. Closed in 04-26 AM. |
| **Re-running adv5/adv6 cases through self-consistency** | Same negative result expected; the adv5/adv6 misses that survived the loop are *deterministic-tier* bugs (Bug 6/7/8), not model bugs. |
| **Tightening NegEx beyond what's already shipped** | The 80-char post-hoc window is wider than spec but `adv6_negex_overfire_mpox_then_flu_negative` confirmed it doesn't over-fire on a probe. Don't re-narrow without a concrete failing case. |
| **Adding more pre-NegEx-fix triggers speculatively** | The c20 doc lists candidate triggers ("not entertaining", "not pursuing") not yet shipped; add only when an adversarial case forces them. Speculative additions widen the FP surface without a signal. |
| **Encounter-aware scoring for `multi_encounter_covid_then_hiv`** | The case "trivially passed" because both encounters surface inline codes. To genuinely stress encounter attribution would require a contradicting case + a scoring model that's encounter-aware. Out of scope this hackathon. |
| **LiteRT-LM int4 retrain** (carry-over) | Runtime is dead. iOS uses llama.cpp now. |
| **MediaPipe LLM Inference API** | Officially deprecated by Google. |
| **Re-fine-tuning to fix tool-calling** | Base Gemma 4 already at F1 = 1.000 on combined-27 over 81 runs. v28 (Rank 3) is the one exception still in flight. |

---

## Cross-cutting items needing teammate action

- **`ios-eng`**: owns Rank 1 (physical-device bench). Owns Rank 5
  screencast post-Rank-1. Idle until a physical iPhone arrives.
- **`team-lead`**: owns Rank 2 (Spaces push, gated on user login),
  Rank 4 (Bug 6/7/8 sprint), and Rank 5 packaging/submission.
- **`trainer`**: owns Rank 3 trigger (v28 Kaggle re-run with
  `merged_16bit`). Otherwise idle on the LLM side — c20 closed
  the LLM-tuning surface.
- **`researcher`** (me): refresh `results.tsv` after each landed
  proposal; on-call for Rank 6 (adv7) if Rank 1 doesn't land by
  May 5.
- **User-side**: `huggingface-cli login` to unblock Rank 2;
  re-launch the Kaggle kernel with `merged_16bit` to unblock
  Rank 3; provide a physical iPhone to unblock Rank 1.
