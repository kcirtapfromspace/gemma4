# Handoff — 2026-04-25 PM (post-c19 sprint)

**Author:** team-lead (solo orchestrator after team shutdown)
**Branch:** `autoresearch/apr15`  •  **HEAD:** `97fea00`
**Hackathon deadline:** 2026-05-18 (23 days remaining)
**Read this if** you're picking up the project after a `/clear`. Cross-reference `tools/autoresearch/proposals-2026-04-25.md` for the strategic ranking.

## Where we left off

The team (researcher, ios-eng, trainer) was spawned this morning and shut down by the user around 16:26 UTC after the c18/c19 work landed. From that point, team-lead has been operating solo. All work lives on `autoresearch/apr15` in the main repo at `/Users/thinkstudio/gemma4`. The worktree under `.claude/worktrees/calm-ray-2no2/` is fast-forwarded to the same HEAD.

**Worktree branch invariant:** if a fresh worktree is created from `main`, it lags ~95 commits behind `autoresearch/apr15`. Always `git merge --ff-only autoresearch/apr15` from the worktree before tasking agents. Documented in memory `project_branch_strategy.md`.

**Stale stash:** `stash@{0}` ("pre-merge snapshot of worktree uncommitted state — already preserved in main repo at 4469831") is redundant. Safe to drop manually with `git stash drop stash@{0}`. Safety-net blocks automated drop.

## What shipped this sprint (commits since `fbb6228`)

| commit | summary |
|---|---|
| `5869937` | ios c19 single-turn fast-path (Rank 2 Swift mirror) |
| `926c8ef` | ios c19 NegEx tighten + smoke bench (19/19) |
| `c98990b` | ios c19 demo seed (Sofia Reyes / valley fever) + screenshot + conf clamp |
| `042f2da` | researcher adv4 bench (4 voice + 4 CDA) |
| `d5b5726` | team-lead c19 Python fast-path mirror — agent_pipeline.py + 8/8 parity probes |
| `4469831` | team-lead integrated trainer's c19 tool-call grammar (Rank 4) |
| `a61f1f5` | team-lead c19 FHIR R4 Bundle wrapper + validator (Rank 3) — 35/35 R4-valid |
| `037f6a2` | results.tsv commit-hash backfill |
| `1c6b607` | team-lead c19 BundleBuilder.swift mirror + "View FHIR Bundle" sheet |
| `2dc4b2b` | ios c19 Outbox payload = FHIR R4 Bundle (SyncService) |
| `97fea00` | publish c17/c18/c19 sprint experiments + fix sync hook venv |

**Pre-sprint (already in tree at fbb6228 + autoresearch/apr15):** the c1-c18 stack including 5 c18 polish commits, the deterministic preparser + agent + RAG pipeline, dictation, provenance UI.

## Pole-position facts

- **Python pipeline:** F1 = 0.986 across 27-case combined bench (originals 9 + adv1 5 + adv2 8 + adv3 5). 25/27 perfect, 0 false positives, recall 0.971, precision 1.000. Avg 13.0 s/case agent-tier. Source: `tools/autoresearch/results.tsv` rows tagged `c17`.
- **Swift mirror:** F1 = 1.000 on `validate_rag.swift` 14-case bench. 19/19 fast-path probes pass. xcodebuild green on `iPhone17ProDemo` (UDID `CADA1806-F64D-4B02-B983-B75F197D1EF3`).
- **FHIR R4 validity:** 35/35 = 1.000 on combined-27 + adv4 via `apps/mobile/convert/score_fhir.py` (uses `fhir.resources.R4B 8.2.0` in `scripts/.venv`). Outbox payload now defaults to FHIR Bundle wire format.
- **Fast-path parity:** Python and Swift agree on all 8 probes (Marburg 1.387 / valley fever 1.218 / C diff 1.225 / Plasmodium 1.514 fire; Legionnaires-token-only / negated / bare decline). Locks the matched-phrase NegEx contract across runtimes.
- **Tool-call grammar:** `cliniq_toolcall.gbnf` authored, AgentRunner plumbed. Key finding: llama-server's `/v1/chat/completions` rejects custom grammar when `tools` is set — `--jinja` already grammar-locks. So the explicit GBNF only matters on the iOS AgentRunner path. Stability bench (target: 0 unrecoverable parse failures × 27 × 3) gated on llama-server.
- **iOS sim decode tok/s:** still 4.0 from C12. **Physical iPhone tok/s: never measured.** Largest open uncertainty.

## Tasks (where everything is)

### In flight

| ID | Subject | Status | Blocker |
|---|---|---|---|
| #2 | Kaggle Unsloth fine-tune iteration (C9 v2 retrain) | in_progress | Kernel v27 RUNNING on T4 last check (pushed ~21:25 UTC); poll with `kaggle kernels status patrickdeutsch/cliniq-compact-lora-training` |
| #11 | Bench v2 fine-tune Q3_K_M GGUF vs base Gemma 4 | in_progress | Gated on #2 finishing |

### Open / blocked

| ID | Subject | Blocker |
|---|---|---|
| #4 | Rank 1: Physical iPhone tok/s + dictation latency | Needs user to provide a physical iPhone (15 Pro target) |
| #8 | Rank 5: adv4 (8 cases) full LLM scoring + Rank 4 grammar stability bench | Needs `llama-server` running locally on `127.0.0.1:8090` with `--jinja` |

### Completed this sprint

`#1` proposals-2026-04-25.md  •  `#3` c18 iOS polish  •  `#5` c19 fast-path Swift  •  `#6` FHIR R4 wrapper + validator  •  `#7` tool-call grammar lock  •  `#9` Python fast-path mirror  •  `#10` BundleBuilder.swift + sheet  •  `#12` Outbox = FHIR Bundle. Plus `#2` is technically marked completed *and* in_progress (the v26 Kaggle artifact is in hand at `/tmp/c9-existing/`; v27 is a fresh re-run of the same code/data the user requested).

## What to work on next (priority order)

### 1. Finish the v2 fine-tune bench (Track A) — gated on kernel v27

When `kaggle kernels status patrickdeutsch/cliniq-compact-lora-training` returns `COMPLETE`:

```bash
export KAGGLE_API_TOKEN=KGAT_815bad4b042568001ff75ed86e46852b
mkdir -p /tmp/c9-v27
kaggle kernels output patrickdeutsch/cliniq-compact-lora-training -p /tmp/c9-v27

# llama.cpp source already cloned at /tmp/llama-cpp-tools (full shallow,
# has convert_hf_to_gguf.py + gguf-py).
# Convert merged HF → GGUF:
scripts/.venv/bin/python /tmp/llama-cpp-tools/convert_hf_to_gguf.py \
    /tmp/c9-v27/cliniq-compact-merged \
    --outfile /tmp/c9-v27/cliniq-gemma4-e2b-v2.f16.gguf \
    --outtype f16

# Quantize to Q3_K_M (matches our deployment):
llama-quantize /tmp/c9-v27/cliniq-gemma4-e2b-v2.f16.gguf \
               /tmp/c9-v27/cliniq-gemma4-e2b-v2-Q3_K_M.gguf Q3_K_M

# Bench: needs llama-server running with the v2 GGUF on 127.0.0.1:8090.
# scripts/benchmark.py is the fixed eval harness — DO NOT MODIFY.
# Compare F1 / gen tok/s / prompt tok/s vs base Gemma 4 E2B Q3_K_M.

# Log row:
# Append to tools/autoresearch/results.tsv with status keep|discard.
# If keep AND mountable size: copy to apps/mobile/ios-app/ClinIQ/Frameworks/
# (note: existing Frameworks/ has cliniq-compact-lora.gguf and the base Q3_K_M;
# the resolver order in LlamaCppInferenceEngine.swift prefers BASE over the
# fine-tune intentionally because the c17 fine-tune broke tool-calling. If
# v2 doesn't break tool-calling, change resolver order accordingly).
```

The honest question: does C9 v2's KV-shared-layer exclusion + +20 code-preservation + +20 negative-lab augmentation actually fix code elision and sequence degeneration *without* re-breaking tool-calling? Either answer (keep or discard) is publishable.

### 2. Hosted web demo on Hugging Face Spaces (#3 from earlier menu)

`agent_pipeline.py` + RAG behind a Gradio frontend. Anyone with the URL can paste an eICR and watch the deterministic+agent flow. Massively wider judging surface than "watch this iPhone screen." Needs:
- Spaces app under user's HF account
- Pin `fhir.resources>=7.0,<9.0` in requirements.txt
- Bundle the base Gemma 4 E2B Q3_K_M GGUF (2.4 GB; use HF model repo + download-on-startup)
- llama-server in a subprocess inside the Space; `agent_pipeline.run_agent` against it
- Gradio output: render the extraction + the FHIR Bundle JSON + clickable provenance source URLs
- Estimated: 5 hours

The hosted URL is the obvious link to put on the hackathon submission form.

### 3. Validate adv4 + grammar bench when llama-server is up (#8)

These two were never executable solo because they need a live LLM endpoint. When you have llama-server running on `127.0.0.1:8090`:

```bash
# adv4 scoring (all 4 modes — deterministic-only / agent-only / agent+RAG / fast-path):
scripts/.venv/bin/python apps/mobile/convert/agent_pipeline.py \
    --cases scripts/test_cases_adversarial4.jsonl \
    --out-json apps/mobile/convert/build/adv4_agent_bench.json
# scripts/.venv/bin/python apps/mobile/convert/agent_pipeline.py \
#     --cases scripts/test_cases_adversarial4.jsonl --no-fast-path \
#     --out-json apps/mobile/convert/build/adv4_no_fast_path.json

# Grammar stability bench (27×3, target: 0 parse failures):
scripts/.venv/bin/python apps/mobile/convert/bench_toolcall_grammar.py \
    --grammar-file apps/mobile/convert/cliniq_toolcall.gbnf --repeats 3
```

Append rows to `results.tsv`. The hook will auto-sync if you also re-run `scripts/publish_benchmarks.py` after.

### 4. Physical iPhone measurement (#4) — when device available

`ios-eng`'s old brief covers it. Run `validate_rag.swift` harness via Xcode's command-line target on a physical iPhone 15 Pro. Capture: deterministic-tier wall-clock, agent decode tok/s, p50/p95 latency, dictation latency, peak RAM, thermal under 5-case sustained load. Decision rule per `proposals-2026-04-25.md` Rank 1.

### Don't-do (still trapped)

- LiteRT-LM int4 retrain (runtime is dead — Jinja chat-template bug in 0.10.1)
- MediaPipe LLM Inference API (deprecated)
- Multi-Jetson tensor parallel (wrong target; mobile is primary)
- Q2_K quantization (quality cliff)
- Re-fine-tuning to fix tool-calling (base Gemma 4 already does it well via agent_pipeline)
- Rank 6 final-week demo polish (gated on physical-device numbers landing)
- Rank 7 tool-call result cache (low value)

## Environment / setup

- **Kaggle auth:** KGAT tokens go in `KAGGLE_API_TOKEN` env var, NOT `kaggle.json` `key` field (returns 401). Token: `KGAT_815bad4b042568001ff75ed86e46852b`. Username: `patrickdeutsch`. Memory: `reference_kaggle_auth.md`.
- **Obsidian Local REST API:** `https://127.0.0.1:27124`, key in `.claude/settings.local.json`. Vault path for blog notes: `the_archives/archives/`. Single article auto-updated: `gemma4-experiments.md` (lists every experiment as a section).
- **kcirtap.io blog:** Zola, pulls from `the_archives/archives/`. Should auto-publish on next sync after Obsidian update. Memory: `reference_obsidian_blog.md`.
- **PostToolUse sync hook:** `.claude/settings.json` runs `scripts/.venv/bin/python scripts/sync_experiment_notes.py` after any Bash that matches `scripts/(benchmark|publish_benchmarks)\.py`. Was previously broken (used system `python3` without deps); fixed at `97fea00`.
- **Python venv:** `scripts/.venv/` has duckdb + requests + fhir.resources + pandas etc. Use it for any DB-touching work.
- **llama.cpp source:** `/tmp/llama-cpp-tools/` has full shallow clone. `convert_hf_to_gguf.py` + `gguf-py/` available. Reuse this for the v27 conversion to avoid re-cloning.
- **Kernel v26 artifacts:** `/tmp/c9-existing/cliniq-compact-lora/` (LoRA adapter) + `cliniq-compact-merged/` (merged HF model). Same code/data as v27, just timestamped 2026-04-24.

## Coordination notes for next session

- The team (researcher / ios-eng / trainer) was useful for parallel work but spawned and shut down within one session. Solo orchestration has been working fine for the residual queue. Re-spawn only when there's truly parallel work (multiple independent tracks needing distinct skills).
- All work lands on `autoresearch/apr15` in the main repo. Worktrees under `.claude/worktrees/` should ff-merge before agents touch files.
- The user calls out when something doesn't show up on the blog. Always re-run `scripts/publish_benchmarks.py` after a sprint of new experiments to keep the duckdb store + Obsidian + kcirtap.io blog current.
- The user's term "Upstash" means Unsloth (LoRA fine-tuning lib). Memory: `feedback_term_aliases.md`.
