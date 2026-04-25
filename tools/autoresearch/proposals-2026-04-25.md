# Autoresearch proposals — 2026-04-25 (delta over 2026-04-24)

**Author:** researcher (autoresearch agent, delta pass)
**Deadline:** 2026-05-18 (23 days remaining)
**Supersedes:** `proposals-2026-04-24.md` — this is a *delta*, not a re-statement.
**Read this if** you read the 2026-04-24 doc. The bottleneck has moved.

## What changed since 2026-04-24

Three landed deliverables collapsed the 2026-04-24 problem-set:

1. **Compact pipeline (fbb6228)** — 2.5× faster inference via shorter output tokens. Subsumes old Rank 7 (short-form schema). Drop.
2. **C17 deterministic preparser + Gemma 4 agent + RAG** (results.tsv rows 30–47). On the combined 27-case bench (originals 9 + adv1 5 + adv2 8 + adv3 5) the Python pipeline hits **F1 = 0.986** (68/70 recall, 1.000 precision, 25/27 perfect, 0 FP). The Swift mirror hits **F1 = 1.000** on the 14-case bench (`EicrPreparser.swift` + `LookupTable.swift` + `ReportableConditions.swift` + `RagSearch.swift` + `AgentRunner.swift` + `GemmaToolTemplate.swift` + `ToolCallParser.swift`). On-device provenance chips and offline dictation also shipped.
3. **LiteRT-LM is dead** (results.tsv row 46). LiteRT-LM 0.10.1's Jinja runtime can't render the embedded chat template (`map has no method named get`). 0.10.1 is the latest published. iOS now ships **llama.cpp + base Gemma 4 E2B Q3_K_M** in `Frameworks/`, not the fine-tune. This kills every old proposal whose mechanism touched `.litertlm`: Rank 3 (mixed-int8), Rank 4 (spec-decode via LiteRT-LM), Rank 10 (AWQ via `ai-edge-torch`). All three are now Don't-Do (see end).

The residual risk surface is small and narrow. Six proposals, sharply ranked.

---

## Pole-position fact set

- **Python (Mac CPU, llama-server v2 GGUF Q3_K_M)** agent + RAG, 27-case combined bench: **F1 = 0.986**, precision = 1.000, 25/27 perfect, 0 FP. Avg **13.0 s/case**, 2.64 tool calls/case, 3.64 LLM turns/case.
- **Python deterministic-only**, 22-case bench (originals + adv1 + adv2): **F1 = 1.000**, 22/22 perfect.
- **Swift on iPhone17ProDemo simulator** (llama.cpp + base Gemma 4 E2B Q3_K_M), 14-case bench: **1.000**, 11/11 deterministic. App installs and launches; tier-1 short-circuit confirmed via `last-extraction-raw.txt` debug dump.
- **iOS Sim decode tok/s (llama.cpp)**: 4.0 tok/s last measured in C12. Not re-measured under the new agent flow.
- **Provenance UI shipped** — INLINE/CDA/LOOKUP/RAG chips, confidence %, tap-to-expand source span + sourceURL deep-link.
- **Offline dictation shipped** — Apple Speech `requiresOnDeviceRecognition=true`, contextual medical priors (50+ disease/drug terms), live waveform meter.
- **Two residual Python-bench fails (2/27)** are knowledge-coverage, not precision: model can't conjure correct codes for inputs it has never seen and that aren't in the lookup table or NNDSS RAG.
- **Bottleneck is no longer model quality.** Bottleneck is (in order): demo coherence, on-device latency uncertainty, FHIR-validity credibility signal, agent loop robustness on adversarial inputs.

---

## Proposals (ranked by (impact × p(works)) / hours)

### Rank 1 — Physical-iPhone tok/s + dictation latency measurement

**Hypothesis:** The 4.0 tok/s simulator number is the largest open uncertainty in the demo claim. We have no measured on-device number for llama.cpp + base Gemma 4 E2B Q3_K_M post-c17. Either it's good (≥15 tok/s on flagship, ≥6 on iPhone 14) and we lead the demo with it, or it's bad and we narrow the device claim now, not at the demo.

**Mechanism:** `ios-eng` deploys the current `iPhone17ProDemo` scheme to a physical phone (iPhone 15 Pro is the team-available baseline). Run the 14-case bench end-to-end via `validate_rag.swift` harness or via a debug button in `NewCaseView`. Capture (a) deterministic-tier wall-clock per case (target <100 ms), (b) agent-tier decode tok/s, (c) agent-tier total latency p50/p95, (d) dictation latency from mic-on to first partial transcript, (e) peak RAM during agent loop, (f) thermal behavior on 5-case sustained run.

**Cost:** **3 engineer-hours** (`ios-eng`, includes USB-cable provisioning).

**Measurable outcome:** New rows in results.tsv: `c18-iphone-15-pro-physical-{deterministic,agent}`. Decision rule:
- If agent decode ≥10 tok/s → demo claim is "real-time clinical extraction on a 3-yo-class iPhone."
- If 4–10 tok/s → demo claim is "iPhone 15 Pro and newer."
- If <4 tok/s → emergency: rebuild llama.cpp xcframework with `-O3 -ffast-math -DGGML_USE_ACCELERATE`; if still <4, drop to "Mac CPU live" demo path on a 14" MBP via Lightning-out HDMI.

**Risk / killer assumption:** None. We have the device.

**Pivot fit:** Critical. Decides whether the demo is a phone or a laptop.

---

### Rank 2 — Single-turn agent fast path (deterministic-empty + RAG-confident)

**Hypothesis:** When `EicrPreparser.extractWithProvenance` returns empty AND `RagSearch.search(narrative_keywords)` returns a top hit with score ≥ 0.7, we can skip the agent loop entirely — directly emit the RAG result as a synthetic "lookup tier" extraction, no LLM. This collapses the 13.0 s agent-tier cases to <500 ms on the cases that currently dominate that tail. F1 should hold because RAG already covers the long-tail (Legionnaires, C diff, Marburg, RMSF, valley fever) per row 47.

**Mechanism:**
1. In `apps/mobile/convert/agent_pipeline.py`, add `--fast-path-rag-threshold 0.7` flag. Before invoking the agent loop, check `rag_search(narrative)`; if top hit ≥ threshold AND deterministic empty, skip the loop, fabricate the extraction from the RAG hit, mark provenance tier = `RAG_FAST`.
2. In `apps/mobile/ios-app/ClinIQ/ClinIQ/Extraction/ExtractionService.swift` `run(narrative:)`, between tier-1 and tier-2, insert the same shortcut against `RagSearch`.
3. Re-bench Python on the 27-case combined bench AND a held-out subset where deterministic was previously empty (the cases that hit the agent in the first place). Confirm F1 holds and latency drops.

**Cost:** **4 engineer-hours** (2 Python, 2 Swift mirror).

**Measurable outcome:** New rows `c18-agent-fast-path-{python,swift}`. Target: F1 ≥ 0.97 on combined-27 (vs 0.986 baseline; small drop tolerable if latency win is large), median latency on agent-tier-only cases drops from ~13 s to <1 s. Kill criterion: any new false positive (precision drops below 1.000).

**Risk / killer assumption:** RAG might over-fire on narratives that mention a reportable condition by name but where the *real* extraction is something else (e.g., "ruled out Legionnaires"). NegEx already runs in tier 1, so a "ruled out" phrase is suppressed before RAG ever sees it. Mitigation: require RAG fast-path to also check NegEx didn't suppress that exact disease in tier 1.

**Pivot fit:** Universal. Latency win matters most on low-end iPhones.

---

### Rank 3 — FHIR R4 Bundle wrapper + structural validator on the 27-case bench

**Hypothesis:** No proposal so far ships a "the output passes a real FHIR R4 validator" credibility signal. Clinical-interop judges (the `eICR-to-FHIR` track is one of them) will probe this directly. Wrapping our extraction in an R4 `Bundle` of `Condition`/`Observation`/`MedicationStatement` resources and validating via `fhir.resources` is a 1-day task that buys a categorical claim score-deltas can't.

**Mechanism:**
1. Add `apps/mobile/convert/fhir_bundle.py` — pure-function `to_bundle(extraction: dict, patient_ref: str) -> dict` that assembles a minimal R4 Bundle: `Patient` stub, `Condition` per SNOMED, `Observation` per LOINC, `MedicationStatement` per RxNorm. Source URLs from provenance go into `Resource.meta.source`.
2. Add `apps/mobile/convert/score_fhir.py` — pip-install `fhir.resources==7.x`, run `Bundle(**bundle_dict)`; count structural validation pass/fail. Emit `fhir_r4_pass_rate` per row.
3. Mirror the bundle assembler in Swift: `apps/mobile/ios-app/ClinIQ/ClinIQ/FHIR/BundleBuilder.swift`. Validation is Python-only for now (no Swift FHIR validator we trust).
4. Add `--write-bundle` flag to `validate_all_cases.py` and `agent_pipeline.py`. Write to `build/bundles/<case_id>.json`.
5. Re-run on the 27-case bench. Publish row `c18-fhir-r4-validity`.

**Cost:** **5 engineer-hours.**

**Measurable outcome:** `fhir_r4_pass_rate` column populated. Target ≥25/27 = 0.926 on first run; iterate the bundle template until 27/27 = 1.000. Demo claim: "100% R4-valid Bundles, on-device, offline."

**Risk / killer assumption:** `fhir.resources` strict R4 may reject bundles for missing required fields (e.g., `Condition.subject`). Mitigation: build bundles to satisfy required cardinality on first authoring; iterate until clean. This is a known finite engineering problem, not a research question.

**Pivot fit:** Universal. Useful for any judging axis that mentions interoperability.

---

### Rank 4 — Tool-call grammar lock + malformed-call recovery in `AgentRunner`

**Hypothesis:** results.tsv row 37 (Mode B, lookup OFF, LLM grammar fallback) showed the agent gets *invoked* but fails to recover the long-tail cases — partly because the fine-tune emits malformed tool calls (per `ExtractionService.swift` comment, "fine-tune emits malformed tool calls"). On adversarial inputs that exceed context or contain unusual prose, base Gemma 4 also occasionally drifts into JSON-quote-mismatched tool calls (`ToolCallParser.swift` already has a JSON-drift fallback). A small GBNF that locks the *tool-call payload syntax* (not the model output schema) would prevent the loop from ever needing the drift fallback.

**Mechanism:**
1. Author `apps/mobile/convert/cliniq_toolcall.gbnf` — restricts the tokens between `<|tool_call_start|>` and `<|tool_call_end|>` to legal JSON with one of three tool-name strings (`extract_codes_from_text`, `lookup_reportable_conditions`, `validate_fhir_extraction`, `lookup_displayname`) and arg shapes matching the registered tools.
2. Pass via `--grammar-file` on the llama-server invocation that backs `LlamaCppInferenceEngine` on iOS. Plumb through `AgentRunner` when the engine is in agent mode (`engine.beginAgentTurn(grammar:)` — small surface change).
3. In `ToolCallParser.swift`, add a hard error path when grammar is on and parse still fails — currently silent fallback hides bugs.
4. Re-run the 27-case bench × 3 to measure non-determinism. Target: 0 unrecoverable parse failures.

**Cost:** **3 engineer-hours.**

**Measurable outcome:** Row `c18-agent-toolcall-grammar`. Stability metric: 0/(27×3) parse failures (vs. prior unmeasured baseline; we see ~1–2 retries per run). Quality unchanged or up. Decode tok/s drops by ≤5%.

**Risk / killer assumption:** Grammar may force the model into a tool call when the right answer is to emit final JSON and stop. Mitigation: grammar is conditional — applied only when the previous turn ended with `<|tool_response|>`; final answer turns use no grammar. `GemmaToolTemplate.swift` already tracks turn role.

**Pivot fit:** iOS + Mac both. Pure llama.cpp work; LiteRT-LM independent.

---

### Rank 5 — Adversarial-4: dictation-disfluency + CDA-XML mixed bench (8 cases)

**Hypothesis:** The dictation feature ships a *new* input distribution we have never benched. Voice-dictated narratives include disfluencies ("uh", "um", trailing thoughts, restart phrases, mis-segmented punctuation), informal disease names ("the H5N1 thing", "that valley fever case"), and frequent code-omission. Simultaneously, real CDA eICR clones often nest the same diagnosis in both prose and `<code code="…">` XML — testing the *interaction* of the two parsers. Adv4 = 4 voice-dictated + 4 CDA-XML-heavy cases.

**Mechanism:**
1. Author `scripts/test_cases_adversarial4.jsonl` — 4 voice cases hand-written to mimic the team's own dictation samples (record a few `DictationButton` outputs first; mine `last-extraction-raw.txt` for shape), 4 CDA cases adapted from CDC EZeCR sample bundles.
2. Run through deterministic-only, agent-only, agent+RAG, and the new fast-path (Rank 2) variants.
3. Publish row `c18-adv4-{deterministic,agent,fast-path}`.

**Cost:** **5 engineer-hours.**

**Measurable outcome:** Honest F1 on a realistic-input bench. Expected drop to ~0.85 on agent+RAG; drop to ~0.95 with Rank 2 fast-path because RAG already covers most disease-name patterns.

**Risk / killer assumption:** Hand-authored dictation cases may not capture real disfluency distribution. Mitigation: generate them by transcribing 4 real reads of a written case via the shipped `SpeechDictationService`. Cheaper and more authentic than synthesizing.

**Pivot fit:** Mobile-first. Gates demo claims about "describe in your own language."

---

### Rank 6 — Demo polish (final-week-only): provenance deep-link sequence + sub-second story

**Hypothesis:** The single most memorable judge artifact is **provenance chips that deep-link to CDC NNDSS / WHO IDSR sources**, paired with a sub-second deterministic-tier short-circuit on the majority of cases. No one else in the hackathon has this. The 90-second demo cut should be: (1) paste a typical eICR → instant extraction with INLINE/CDA chips → tap a chip, see the source span highlighted; (2) paste a long-tail Marburg/Legionnaires narrative → 3-second agent loop, RAG chip, tap it, deep-link to CDC NNDSS page; (3) flip to dictation, speak a case, watch the same flow run on transcribed text.

**Mechanism:** Coordinate with `ios-eng`. Concrete asks:
1. Add a "Demo mode" toggle in `SettingsTab` that pre-loads three canned narratives (typical, long-tail, voice).
2. Add a 1-second pulse animation on the provenance chip to draw the eye on first render.
3. Confirm RAG chip's `sourceURL` deep-link opens in Safari (not in-app webview — judges want to leave and come back).
4. Pre-record a 90 s screencast with `simctl io recordVideo` against `iPhone17ProDemo` once the physical-device measurement (Rank 1) lands. Voice-over from `team-lead` or `ios-eng`.
5. Embed in `apps/dashboard/journey.html` Field Report №01 above the fold.

**Cost:** **6 engineer-hours** (mostly `ios-eng` + video editing, light `team-lead` review).

**Measurable outcome:** 3 non-engineer test viewers can summarize the project after watching. Pass = all 3 mention "instant," "sources," and either "phone" or "offline."

**Risk / killer assumption:** Time-spent-here trades against Ranks 1–5 quality wins. Defer until Rank 1 (physical device) lands and either Rank 2 (fast path) or Rank 4 (grammar) is in.

**Pivot fit:** Mobile-defining. Leads with the on-device + provenance story.

---

### Rank 7 (optional, only if Ranks 1–4 finish early) — Tool-call result cache + p95 trace

**Hypothesis:** ~80% of the agent's 13.0 s wall-clock is base-model decode tokens, not tool execution. But `extract_codes_from_text` is sometimes called twice per loop (model retries after `validate_fhir_extraction` complains). Hashing the input narrative and caching the deterministic extraction would shave a turn off ~30% of cases.

**Mechanism:** Add an LRU cache around `tool_extract_codes_from_text` in `apps/mobile/convert/agent_pipeline.py` keyed on `hash(args["text"])`. Mirror in `AgentRunner.swift` with a `@MainActor` dictionary scoped to the current `run(narrative:)` invocation.

**Cost:** **2 engineer-hours.**

**Measurable outcome:** Median agent latency 13.0 s → ~10 s. p95 tighter. F1 unchanged.

**Risk / killer assumption:** Marginal. Skip if Ranks 1–6 fill the calendar.

**Pivot fit:** Universal.

---

## Ranking summary

| Rank | Proposal | Hours | p(works) | Demo impact | Score |
|---:|---|---:|---:|---:|---:|
| 1 | Physical-iPhone measurement | 3 | 0.95 | High | **0.95** |
| 2 | Single-turn agent fast path | 4 | 0.75 | High | **0.94** |
| 3 | FHIR R4 Bundle validity | 5 | 0.85 | Med-High | 0.85 |
| 4 | Tool-call grammar lock | 3 | 0.85 | Med | 0.85 |
| 5 | Adv4 dictation+CDA bench | 5 | 0.95 | Med | 0.57 |
| 6 | Demo polish + provenance deep-link | 6 | 1.00 | High | 1.00¹ |
| 7 | Tool-call result cache | 2 | 0.70 | Low | 0.35 |

¹ Rank 6 scores highest on raw impact but is gated to the final week.

---

## Sequencing

| Window | Run |
|---|---|
| **This week** (Apr 25 – May 2) | Rank 1 (`ios-eng`), Rank 2 (`trainer` Python + `ios-eng` Swift mirror), Rank 4 (`trainer`). Rank 3 starts here if Rank 2 finishes early. |
| **Next week** (May 3 – 9) | Rank 3 finishes. Rank 5 (adv4 bench, `researcher` to author cases, `trainer` to score). Rank 7 only if calendar allows. |
| **Final week** (May 10 – 17) | Rank 6 (demo polish + 90 s video). Code freeze on May 14. May 15–17 is video + dashboard + TestFlight invite list. |

---

## Don't-do list (traps, after fbb6228 + c17)

| Trap | Why not |
|---|---|
| **LiteRT-LM int4 retrain** (old Rank 3 echo) | Runtime is dead. 0.10.1 Jinja can't render the bundle's chat template; 0.10.2 not shipped. iOS uses llama.cpp now. Re-quantizing produces an unloadable artifact. |
| **MediaPipe LLM Inference API** | Officially deprecated by Google (per `apps/mobile/RECOMMENDATION.md`). |
| **Gallery-fork Android port** | Depends on LiteRT-LM. Use llama.cpp NDK on Android only if `ios-eng` or `trainer` has truly idle cycles in the final week — and even then it's a 16-h port at best. Default = skip; iOS is the demo. |
| **Multi-Jetson tensor parallel / MLC-LLM port hardening** | Wrong target. Mobile is primary. Jetson cluster is a secondary "we built it" footnote, not a demo path. |
| **Q2_K quantization** | Quality cliff for clinical extraction; no upside given Q3_K_M already gives F1=1.000 on Swift bench. |
| **AWQ via `ai-edge-torch`** (old Rank 10) | Import path targets `.litertlm`; runtime dead. |
| **Speculative decoding via 270M draft** (old Rank 4) | LiteRT-LM `SpeculativeDecodingSettings` proto unreachable in the shipping version; llama.cpp Gemma family doesn't speculate well per results.tsv row 25. |
| **Mixed-precision int8 LiteRT-LM bundle** (old Rank 3) | Same — `.litertlm` runtime is dead. |
| **Re-running the LoRA fine-tune for tool-call format** | The base Gemma 4 E2B already does tool-calling well per `agent_pipeline.py` 22/22 result. The fine-tune *broke* tool-calling (see `ExtractionService.swift` resolver-order comment). Don't retrain to fix something that works. |

---

## Cross-cutting items needing teammate action

- **`ios-eng`**: owns Rank 1 (physical-device bench), Rank 2 Swift mirror, Rank 6 demo polish.
- **`trainer`**: owns Rank 2 Python, Rank 3 (FHIR R4 wrapper + validator), Rank 4 (grammar GBNF + `AgentRunner` plumbing), Rank 7.
- **`researcher`** (me): on-call to author Rank 5 adv4 bench cases when triggered, refresh `results.tsv` after each landed proposal.
- **`team-lead`**: arbitration on Rank 6 (final-week scope) and any Don't-Do exception requests.
