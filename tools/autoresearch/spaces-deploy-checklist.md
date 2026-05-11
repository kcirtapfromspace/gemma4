# HF Spaces deploy checklist — ClinIQ eICR → FHIR (safety-net)

**Status:** Staged 2026-05-04 by the spaces-stage-deploy agent. Bundle is at
`out/space/`, snapshot at `out/space.safety-net-2026-05-05/`. Local smoke
test passed (HTTP 200 on `127.0.0.1:7871`, all five sample cases route
through tier 1 + tier 2 correctly with the agent disabled).

**Decision needed from user:** approve and run the push commands in §5.

---

## 1. Authoritative facts captured during staging

| Fact | Value |
|---|---|
| HF account | `kcirtapfromspace` (member of org `humanitys-last-hackathon`) |
| Recommended Space name | `cliniq-eicr-fhir` (matches `spaces/deploy.sh` default) |
| Recommended namespace | `kcirtapfromspace/cliniq-eicr-fhir` (personal — simplest) — OR `humanitys-last-hackathon/cliniq-eicr-fhir` (hackathon org, if judges check the org listing) |
| SDK | `gradio` 5.9.1 (per `out/space/README.md` frontmatter) |
| Hardware tier | `zero-a10g` (free ZeroGPU H200 quota for the agent tier) |
| Python | 3.12 (per `python_version` in README frontmatter) |
| App entrypoint | `app.py` |
| License | apache-2.0 |
| Expected URL | `https://huggingface.co/spaces/<owner>/cliniq-eicr-fhir` |

## 2. Bundle contents (verified)

`out/space/` after `bash spaces/build.sh out/space` + the case_diff fixup
described in §6:

```
out/space/
├── app.py                    18.4 KB   — Gradio UI + run_pipeline driver
├── zerogpu_engine.py          9.8 KB   — Gemma 4 in-process backend (ZeroGPU)
├── requirements.txt           ~120 B   — gradio + fhir.resources + spaces + transformers + torch
├── README.md                  9.4 KB   — Spaces YAML frontmatter + deploy + bench
└── convert/
    ├── agent_pipeline.py     55.7 KB   — orchestrator (tier 1+2+3)
    ├── case_diff.py          17.5 KB   — *added by this agent* — agent_pipeline imports it
    ├── fhir_bundle.py        14.1 KB   — FHIR R4 Bundle assembler
    ├── lookup_table.json      6.2 KB   — deterministic-tier code aliases
    ├── rag_search.py         14.7 KB   — fast-path RAG + NegEx
    ├── regex_preparser.py    35.8 KB   — tier 1 regex + CDA-XML preparser
    └── reportable_conditions.json  19.5 KB   — curated CDC NNDSS / WHO IDSR DB
```

NOT in the bundle (intentional): `screenshots/`, `case_diff.py` references
elsewhere, model artifacts, GGUF files, the bench scripts. Spaces will pull
the Gemma 4 weights from `unsloth/gemma-4-E2B-it` on first boot.

## 3. Spaces metadata (from `out/space/README.md` frontmatter)

```yaml
---
title: ClinIQ — eICR to FHIR (Gemma 4)
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.9.1"
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-a10g
suggested_hardware: zero-a10g
short_description: eICR → FHIR R4 via Gemma 4 on ZeroGPU
---
```

This drives the Spaces dashboard (title/emoji/colors), pins the Python
runtime, picks the Gradio version, and most importantly requests
`zero-a10g` (free ZeroGPU H200) so the in-process `transformers` Gemma 4
backend gets GPU under `@spaces.GPU(duration=120)`. No edits needed.

## 4. Local smoke-test result (2026-05-04 wall clock)

Smoke venv: `out/space/.venv-smoke/` (Python 3.11.x, lightweight deps only
— gradio, fhir.resources, huggingface_hub; we deliberately skipped torch +
transformers + spaces because the deploy will install them on Spaces
hardware and `CLINIQ_DISABLE_AGENT=1` lets us boot without them).

```
$ cd out/space
$ CLINIQ_DISABLE_AGENT=1 GRADIO_SERVER_PORT=7871 GRADIO_SERVER_NAME=127.0.0.1 \
    .venv-smoke/bin/python app.py &
[boot in <2 s, no model download]

$ curl -sS -o /dev/null -w "HTTP %{http_code} bytes=%{size_download}\n" \
    http://127.0.0.1:7871/
HTTP 200 bytes=22807

# programmatic check across all 5 SAMPLES:
[deterministic     ] COVID-19 (inline SNOMED + LOINC)        bundle.entries=5
[fast_path         ] Valley fever (RAG fast-path)            bundle.entries=2
[fast_path         ] Marburg outbreak (RAG fast-path)        bundle.entries=2
[fast_path         ] C. diff colitis (RAG fast-path)         bundle.entries=2
[no_match          ] Negated lab (precision check)           bundle.entries=0
```

All five samples route correctly. R4 validation passes for every
non-empty bundle. This matches the prior PoC result in
`tools/autoresearch/mtp-spaces-poc.md` § 4.

What the smoke test did NOT cover (deploy-only):
- `zerogpu_engine.py` model loading (no torch installed locally)
- ZeroGPU `@spaces.GPU` decorator (no-op outside Spaces)
- Live agent path against an actual eICR-to-FHIR case requiring tier 3

These will execute on Spaces ZeroGPU hardware on first boot and on the
first agent-tier case the user runs.

## 5. Push commands (run these once, in this order)

**Pre-flight (already done by this agent — no action needed):**
- `huggingface-cli whoami` → `kcirtapfromspace` (logged in)
- `out/space/` bundle is built and validated
- Snapshot at `out/space.safety-net-2026-05-05/` exists

**Step 1 — Create the Space repo on Hugging Face.** Pick the namespace
(personal vs hackathon org). Personal is simpler for first-time deploy;
the org is more visible to judges if they browse the org page.

```bash
# Option A (recommended for first deploy): personal namespace
huggingface-cli repo create cliniq-eicr-fhir \
    --type space --space-sdk gradio

# Option B: hackathon org namespace (requires push permission to the org)
huggingface-cli repo create cliniq-eicr-fhir \
    --type space --space-sdk gradio \
    --organization humanitys-last-hackathon
```

This creates the Space at `https://huggingface.co/spaces/<owner>/cliniq-eicr-fhir`
with `gradio` SDK selected. Hardware will default to **CPU basic** —
upgrade to ZeroGPU after the first push (Step 4) so you can verify the
push succeeded before consuming ZeroGPU quota.

**Step 2 — Initialize the bundle as a git repo and add the HF remote.**

```bash
cd /Users/thinkstudio/gemma4/out/space
git init -b main
git remote add origin https://huggingface.co/spaces/<owner>/cliniq-eicr-fhir
```

Replace `<owner>` with `kcirtapfromspace` or `humanitys-last-hackathon`
depending on Step 1.

**Step 3 — Stage, commit, and push.**

```bash
cd /Users/thinkstudio/gemma4/out/space
git add .
git commit -m "Initial deploy — safety-net 2026-05-05"
git push origin main
```

The first push will prompt for credentials. If `huggingface-cli login`
configured the git credential helper, the push should succeed without
prompting. If it prompts:
- Username: `kcirtapfromspace`
- Password: a HF write token (NOT your account password). Get one from
  https://huggingface.co/settings/tokens — must be **write** scope.

**Alternative — use the included deploy.sh (does Steps 2 + 3 in one shot).**
Note: this re-runs `build.sh` first, which would overwrite the case_diff.py
fixup unless you've already updated `spaces/build.sh` per §6. So **do
NOT run deploy.sh until §6 is applied.** Once §6 is applied:

```bash
bash /Users/thinkstudio/gemma4/spaces/deploy.sh \
    --space kcirtapfromspace/cliniq-eicr-fhir --push
```

**Step 4 — Switch hardware to ZeroGPU.** Visit
`https://huggingface.co/spaces/<owner>/cliniq-eicr-fhir/settings`,
scroll to "Space hardware", change from CPU basic to **ZeroGPU**
(it should already be auto-suggested because the README frontmatter
says `suggested_hardware: zero-a10g`). The Space will rebuild and
restart. First model load will take ~60-90 s (Gemma 4 E2B = ~5 GB
download from `unsloth/gemma-4-E2B-it`).

**Step 5 — Verify.** Open the Space URL. Expected behavior:
- Page loads with the ClinIQ Gradio UI
- "Backend" line in the Advanced accordion shows `Gemma 4 (gemma-4-E2B-it,
  ~2.0 B params, bfloat16) running in-process on ZeroGPU H200`
- Click "Run extraction" with the COVID-19 sample → green "Deterministic"
  badge in <100 ms (no GPU consumed — tier 1 hits)
- Click "Run extraction" with Valley fever → blue "RAG fast-path" badge
- Click "Run extraction" with the negated-lab sample → grey "No match"
  badge (precision check)
- Toggle off "Enable Gemma 4 agent loop" + run a free-text case that
  isn't in the curated DB → grey "No match" (skips tier 3)
- Toggle agent back on + same case → purple "Gemma 4 agent" badge after
  ~5–15 s, with tool calls visible in the trace

## 6. **MANDATORY pre-deploy fixup — `spaces/build.sh` is missing case_diff.py**

`apps/mobile/convert/agent_pipeline.py` line 44 imports `case_diff`. The
current `spaces/build.sh` PIPELINE list does NOT include `case_diff.py`,
so the bundle that build.sh produces is broken: `app.py` import crashes
with `ModuleNotFoundError: No module named 'case_diff'` before the UI
ever renders.

This agent worked around it by manually copying `case_diff.py` into
`out/space/convert/case_diff.py` after build.sh ran. The snapshot at
`out/space.safety-net-2026-05-05/` contains the fix.

**Apply this one-line fix to `spaces/build.sh` before the next deploy
that goes through `deploy.sh` (so any re-build keeps `case_diff.py`):**

```diff
 PIPELINE=(
   agent_pipeline.py
+  case_diff.py
   fhir_bundle.py
   rag_search.py
   regex_preparser.py
   lookup_table.json
   reportable_conditions.json
 )
```

This staging agent did NOT touch `spaces/build.sh` per the user's
constraint ("DO NOT modify any file under `spaces/` source"). The user
should make this edit before the second deploy or before letting the
parallel `spaces-mtp-integration` agent's outputs flow into the bundle.

## 7. Pre-flight gotchas

1. **Repo creation requires HF write token, not just login.** If
   `huggingface-cli repo create` errors with `403`, run
   `huggingface-cli login` and paste a token with **write** scope (visit
   https://huggingface.co/settings/tokens and create one if your current
   token is read-only).
2. **The default git credential helper on macOS won't store the HF
   token.** First push usually works because `huggingface-cli login`
   configures the git helper, but if it prompts and you have to paste,
   use the HF write token (NOT your password) as the git password.
3. **ZeroGPU quota is per-account, not per-Space.** If `kcirtapfromspace`
   has used ZeroGPU recently for other Spaces, the agent tier may queue.
   Tier 1 + 2 are unaffected.
4. **First boot will redownload Gemma 4 weights (~5 GB) into the Space's
   ephemeral storage.** Expect the first build to take 5–10 minutes.
   Subsequent restarts are cached.
5. **Spaces rebuild on every push.** Even a one-line README change
   triggers a full rebuild. Batch edits.
6. **The `spaces` Python package is not on PyPI's mirror everywhere.** If
   the build fails with `Could not find a version that satisfies the
   requirement spaces`, the install runs against the wrong index. The
   default Spaces builder uses the right index — only an issue if the
   user customizes the build.

## 8. Rollback plan if the deploy breaks

The snapshot at `out/space.safety-net-2026-05-05/` is the known-good
bundle. To roll back from a broken push:

```bash
# Option 1 — re-push from the snapshot, overwriting the broken commit:
cd /tmp && rm -rf cliniq-rollback && \
    cp -R /Users/thinkstudio/gemma4/out/space.safety-net-2026-05-05/ \
          /tmp/cliniq-rollback
cd /tmp/cliniq-rollback
git init -b main
git remote add origin https://huggingface.co/spaces/<owner>/cliniq-eicr-fhir
git add .
git commit -m "Rollback to safety-net 2026-05-05"
git push -f origin main      # force-push to overwrite the broken HEAD

# Option 2 — delete the Space and re-create:
huggingface-cli repo delete <owner>/cliniq-eicr-fhir --type space
# then re-run §5 Steps 1–5
```

If the model fails to load on Spaces (unlikely — `unsloth/gemma-4-E2B-it`
is public and well-tested), set `CLINIQ_DISABLE_AGENT=1` as a Space
secret to keep tiers 1 + 2 working while you debug. Visit
`https://huggingface.co/spaces/<owner>/cliniq-eicr-fhir/settings`,
scroll to "Variables and secrets", add `CLINIQ_DISABLE_AGENT=1` as a
public variable, restart the Space. The agent tier will surface a clear
"backend unavailable" status; the deterministic + RAG tiers stay live.

## 9. What this agent did NOT do (per the user's constraints)

- ❌ Did NOT push to Hugging Face.
- ❌ Did NOT create the Space repo on Hugging Face.
- ❌ Did NOT modify any file under `spaces/` (build.sh fix is documented
  in §6 for the user to apply).
- ❌ Did NOT commit anything to git locally.
- ✅ DID build `out/space/`, snapshot to `out/space.safety-net-2026-05-05/`,
  install a smoke venv at `out/space/.venv-smoke/`, boot the app on
  `127.0.0.1:7871` with `CLINIQ_DISABLE_AGENT=1`, hit `/` with curl
  (HTTP 200), tear down, and write this checklist.

## 10. One-line answer

**Is this ready for the user to push?** Yes — bundle is staged and
verified at `out/space/`. The user must (a) optionally apply the
build.sh fix in §6, (b) run §5 Steps 1–5. ETA from approval to live
Space: ~10 minutes (5–10 min for the first ZeroGPU build + 1–2 min for
manual hardware switch + verification).
