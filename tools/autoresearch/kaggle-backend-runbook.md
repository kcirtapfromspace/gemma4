# Kaggle T4 Inference Backend — Operator Runbook

**Date:** 2026-05-06 • **Hackathon deadline:** 2026-05-18

## What this is

A free external LLM backend for the Gemma 4 hackathon Space, running on Kaggle's
free-tier T4 GPU and bridged to the Space via a [cloudflared](https://github.com/cloudflare/cloudflared)
quick tunnel. Lets the live demo at
<https://huggingface.co/spaces/kcirtapfromspace/cliniq-eicr-fhir> serve the
Tier-3 agent loop without paying for ZeroGPU Pro.

```
HF Spaces (cpu-basic, free)            Kaggle T4 (free, ~30h/wk)
─────────────────────────              ──────────────────────────
app.py + agent_pipeline                cliniq_inference_server.py
  Tier 1 deterministic                   transformers main HEAD
  Tier 2 RAG fast-path                   + Gemma 4 E2B-it (target)
  Tier 3 → POST → ─tunnel→               + Gemma 4 E2B-it-assistant (drafter)
                                         → 1.92× decode via assistant_model
```

## Files

| Path | Role |
|---|---|
| `kaggle/cliniq_inference_server.py` | Kaggle kernel script (FastAPI + cloudflared) |
| `kaggle/cliniq_inference_kernel.json` | Kaggle CLI metadata for `kaggle kernels push` |
| `spaces/zerogpu_engine_remote.py` | Spaces-side HTTP client (mirrors the in-process engine API) |
| `spaces/requirements-remote.txt` | Slim Spaces requirements (no torch / transformers) |

The remote engine is a parallel-track copy alongside `zerogpu_engine.py`
(safety-net) and `zerogpu_engine_mtp.py` (in-process MTP). All three expose
the same `chat_completion` / `chat_http_shim` / `model_banner` surface so the
deploy chooses one by renaming files in the build bundle — `app.py` itself
stays unchanged.

## Prerequisites

- **Kaggle CLI installed and authenticated.** The local check is
  `which kaggle` — verified present at `/opt/homebrew/bin/kaggle` (v2.0.1).
  If absent, `pip install kaggle`. The CLI also needs `~/.kaggle/kaggle.json`
  with `username` set to your Kaggle handle. **Important per
  `~/.claude/projects/-Users-thinkstudio-gemma4/memory/reference_kaggle_auth.md`:**
  KGAT-prefixed tokens (the kind in the project handoff) go in the
  `KAGGLE_API_TOKEN` env var, *not* in the `key` field of `kaggle.json`. A
  `KGAT_*` token in the JSON file returns 401.
  ```bash
  export KAGGLE_USERNAME=patrickdeutsch
  # KGAT_... value lives in tools/autoresearch/handoff-2026-04-27.md.
  # Not duplicated here; export it before running the snippets below.
  export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:?set from handoff first}"
  ```
- **HF Spaces secret edit access** for `kcirtapfromspace/cliniq-eicr-fhir`
  (need to set `CLINIQ_REMOTE_URL`).

## One-time setup

1. Verify the Kaggle metadata file matches your username — `cliniq_inference_kernel.json`
   ships with `"id": "patrickdeutsch/cliniq-inference-server"`. Edit if you're
   pushing under a different handle.
2. From the repo root:
   ```bash
   kaggle kernels push -p kaggle/
   ```
   This creates the kernel `patrickdeutsch/cliniq-inference-server` and runs
   it once. First-run cold-start is dominated by:
   - `pip install` of transformers main HEAD (~90 s on Kaggle T4 nodes),
   - HF download of Gemma 4 E2B-it weights (~5 GB → 60–90 s on Kaggle's
     uplink),
   - drafter download (~150 MB → seconds),
   - cloudflared download + first model load (~30 s).
   Plan on **~3–5 minutes** before the tunnel URL is printed.

## Each session (after the kernel auto-stops)

Kaggle GPU kernels auto-terminate after ~9 hours of GPU runtime, and the
weekly free quota is **30 GPU hours**. Each fresh push gets a *new*
`https://*.trycloudflare.com` URL, so the Space secret has to be updated
each time you start a new session.

```bash
# 1. Start a fresh kernel run (re-runs the script from scratch).
kaggle kernels push -p kaggle/

# 2. Wait for it to reach RUNNING (script kernels go COMPLETE only when
# the script returns; ours blocks in uvicorn, so RUNNING is the steady
# state). Poll until status flips off "queued".
kaggle kernels status patrickdeutsch/cliniq-inference-server

# 3. Once RUNNING and the model has loaded (~3–5 min), grab the tunnel
# URL the script wrote to /kaggle/working/tunnel_url.txt:
mkdir -p /tmp/cliniq-tunnel
kaggle kernels output patrickdeutsch/cliniq-inference-server -p /tmp/cliniq-tunnel/
cat /tmp/cliniq-tunnel/tunnel_url.txt
```

Alternative if `kernels output` is slow / unavailable mid-run: open the
kernel page in the Kaggle web UI and copy the `https://*.trycloudflare.com`
URL from the live console output (the script prints it in a banner with
`====` separators).

## Wiring it to the Space

1. In the HF Spaces UI for `kcirtapfromspace/cliniq-eicr-fhir`, open
   **Settings → Variables and secrets**.
2. **Add a secret** named `CLINIQ_REMOTE_URL` with the trycloudflare.com URL
   from `tunnel_url.txt` (no trailing slash; the engine strips one anyway).
3. **Remove** the `CLINIQ_DISABLE_AGENT` env var if it's set — the agent path
   is the whole point of wiring this up.
4. Restart the Space worker so the new secret is picked up.

The Space can stay on cpu-basic; no GPU upgrade needed.

## Swap procedure: in-process → remote

Edit `spaces/build.sh` so it ships the remote engine and slim requirements:

```diff
-cp "${REPO_ROOT}/spaces/zerogpu_engine_mtp.py"  "${OUT_DIR}/zerogpu_engine.py"
-cp "${REPO_ROOT}/spaces/requirements-mtp.txt"   "${OUT_DIR}/requirements.txt"
+cp "${REPO_ROOT}/spaces/zerogpu_engine_remote.py"  "${OUT_DIR}/zerogpu_engine.py"
+cp "${REPO_ROOT}/spaces/requirements-remote.txt"   "${OUT_DIR}/requirements.txt"
```

The destination filenames stay `zerogpu_engine.py` and `requirements.txt`
so `app.py` (which does `from zerogpu_engine import …`) doesn't change.
Re-run `bash spaces/build.sh`, commit, push to the Space remote.

## Quick verification

After the Space rebuilds:

1. Open the Space URL.
2. Expand the **Advanced — agent loop** accordion. The "Backend:" line should
   read something like:
   `Gemma 4 (gemma-4-E2B-it, 5.0 B params, bfloat16) on CUDA (Tesla T4) · MTP on (drafter gemma-4-E2B-it-assistant, 78 M) · via remote tunnel https://...trycloudflare.com`
3. Pick the "Valley fever (RAG fast-path)" sample to confirm Tier 2 still
   works (no remote call).
4. Pick a case that misses both Tier 1 and Tier 2 — or write your own — to
   exercise Tier 3 end-to-end. The Pipeline Trace tab should show a tool
   loop and the badge should read "Gemma 4 agent · N tool call(s) · ✓ R4-valid".

If `model_banner()` reports
`Remote backend unhealthy: ConnectionError: ...`, the tunnel is dead — push
a fresh Kaggle run and update the secret.

## Limits and gotchas

- **Kaggle T4 quota:** 30 GPU-h/week, free tier. Each kernel auto-terminates
  after ~9 GPU-h. Plan demos to land inside a single session.
- **Cold start:** ~3–5 min from `kaggle kernels push` to the tunnel URL
  appearing. Don't hot-reload before a judging window.
- **Ephemeral URLs:** quick tunnels mint a new subdomain on every cloudflared
  start. Pin the URL in the Space secret right before the demo, not the day
  before.
- **No persistent Kaggle "always-on":** Kaggle is interactive notebooks, not
  a hosted service. There is no "deploy and forget"; somebody has to push
  the kernel before the demo.
- **Cloudflared is rate-limited at the edge.** Quick tunnels are explicitly
  not for production traffic. For a single judge clicking through the demo
  this is fine; for a public viral link it would melt.
- **Internet required on the kernel.** `enable_internet: true` in
  `cliniq_inference_kernel.json` is set; verify it didn't get reset by a
  Kaggle UI edit (the toggle lives on the notebook settings panel).
- **Single-tenant.** The FastAPI server has no concurrency control beyond
  uvicorn's default loop. If two judges click "Run" simultaneously they'll
  serialize on the GPU. For a higher-fanout demo, queue requests on the
  Space side.

## Rollback

If the Kaggle backend isn't ready for a judging slot, or the cloudflared
tunnel keeps dying, revert the deploy to the disabled-agent fallback:

```diff
-cp "${REPO_ROOT}/spaces/zerogpu_engine_remote.py"  "${OUT_DIR}/zerogpu_engine.py"
-cp "${REPO_ROOT}/spaces/requirements-remote.txt"   "${OUT_DIR}/requirements.txt"
+cp "${REPO_ROOT}/spaces/zerogpu_engine.py"     "${OUT_DIR}/zerogpu_engine.py"
+cp "${REPO_ROOT}/spaces/requirements.txt"      "${OUT_DIR}/requirements.txt"
```

Plus set `CLINIQ_DISABLE_AGENT=1` in the Space secrets. The deterministic +
RAG tiers (which cover ~70% of the bench cases) keep working; Tier 3 will
just surface a clean "Agent backend unavailable" badge instead of crashing.

## Test plan (when you can run end-to-end)

- `kaggle kernels push -p kaggle/` returns a kernel URL.
- `kaggle kernels status` flips RUNNING within ~30 s of push.
- After ~3–5 min, `kaggle kernels output` produces `tunnel_url.txt` with a
  `https://*.trycloudflare.com` line.
- `curl ${TUNNEL}/healthz` returns `{"ok": true, "banner": "Gemma 4 ... CUDA (Tesla T4) · MTP on..."}`.
- `curl -X POST ${TUNNEL}/v1/chat/completions -H 'Content-Type: application/json' \
   -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}'`
  returns a `choices[0].message.content` string.
- After setting `CLINIQ_REMOTE_URL` and rebuilding the Space, the
  Advanced row banner shows the remote backend, the Tier-3 sample produces
  a tool-call trace, and the R4 validity badge shows ✓.
