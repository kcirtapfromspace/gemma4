"""Gemma 4 ZeroGPU engine — REMOTE backend (Kaggle T4 via cloudflared).

Third parallel-track variant of `zerogpu_engine.py`. Public API surface is
identical (`chat_completion`, `chat_http_shim`, `model_banner`) so `app.py`
can import it under the same name as the in-process engines via
`spaces/build.sh`.

Why this exists:
  HF Spaces "kcirtapfromspace" is a free account, so ZeroGPU isn't available.
  In-process Gemma 4 on cpu-basic is unusably slow (CPU bf16 = minutes per
  case). The remote engine moves Tier 3 (the agent loop) to a Kaggle T4 +
  MTP backend (`kaggle/cliniq_inference_server.py`) reachable through a
  cloudflared tunnel. Tiers 1+2 stay in-process on the Space (deterministic
  preparser + RAG fast-path are pure-Python and fast on CPU).

Failure modes:
  - `CLINIQ_REMOTE_URL` env var unset → `model_banner()` says so;
    `chat_completion(...)` returns a 503-ish error response instead of
    crashing. `app.py` already handles that path through its
    `_AGENT_BACKEND_ERROR` badge surface.
  - Remote unreachable / 5xx → same: error response, no crash.

See `tools/autoresearch/kaggle-backend-runbook.md` for the operator runbook
and `kaggle/cliniq_inference_server.py` for the server side.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import requests

REMOTE_URL = os.environ.get("CLINIQ_REMOTE_URL", "").rstrip("/")
MODEL_ID = os.environ.get("CLINIQ_GEMMA_MODEL_ID", "google/gemma-4-E2B-it")

_TIMEOUT_S = float(os.environ.get("CLINIQ_REMOTE_TIMEOUT", "60"))
_HEALTH_TIMEOUT_S = float(os.environ.get("CLINIQ_REMOTE_HEALTH_TIMEOUT", "5"))

# Cached health probe so repeated banner / chat calls don't re-poll.
_HEALTH_CACHE: dict[str, Any] | None = None
_HEALTH_FETCHED_AT: float = 0.0
_HEALTH_TTL_S: float = 30.0


def _health_probe(force: bool = False) -> dict[str, Any]:
    """GET ${REMOTE_URL}/healthz once, cache for `_HEALTH_TTL_S`."""
    global _HEALTH_CACHE, _HEALTH_FETCHED_AT
    if not REMOTE_URL:
        return {"ok": False, "error": "CLINIQ_REMOTE_URL unset"}
    if (
        not force
        and _HEALTH_CACHE is not None
        and (time.time() - _HEALTH_FETCHED_AT) < _HEALTH_TTL_S
    ):
        return _HEALTH_CACHE
    try:
        resp = requests.get(f"{REMOTE_URL}/healthz", timeout=_HEALTH_TIMEOUT_S)
        resp.raise_for_status()
        _HEALTH_CACHE = resp.json()
    except Exception as exc:  # noqa: BLE001 — surface, don't crash
        _HEALTH_CACHE = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}"[:300],
        }
    _HEALTH_FETCHED_AT = time.time()
    return _HEALTH_CACHE


def _error_response(message: str) -> dict[str, Any]:
    """Shape-compatible error envelope so `app.py` can render a clean badge."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": f"[remote-backend-error] {message}",
                "tool_calls": None,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "timings": {
            "predicted_per_second": 0.0,
            "predicted_n": 0,
            "predicted_ms": 0.0,
            "mtp": {
                "enabled": False,
                "active": False,
                "drafter_id": None,
                "drafter_load_error": None,
            },
            "remote": {"ok": False, "error": message},
        },
    }


def chat_completion(
    *,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 3072,
    **_ignored: Any,
) -> dict:
    """POST OpenAI-format request to the remote Kaggle backend.

    Mirrors the public signature of `zerogpu_engine.chat_completion` so the
    `agent_pipeline.chat = chat_http_shim` monkey-patch in `app.py` keeps
    working unchanged.
    """
    if not REMOTE_URL:
        return _error_response(
            "CLINIQ_REMOTE_URL is not set in the Space's secrets. "
            "Start the Kaggle backend, copy the trycloudflare.com URL, and "
            "set it as a Space secret.",
        )

    payload: dict[str, Any] = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if tools is not None:
        payload["tools"] = tools

    try:
        resp = requests.post(
            f"{REMOTE_URL}/v1/chat/completions",
            json=payload,
            timeout=_TIMEOUT_S,
            headers={"Content-Type": "application/json"},
        )
    except requests.exceptions.Timeout:
        return _error_response(
            f"Remote backend timed out after {_TIMEOUT_S:.0f}s. "
            f"The Kaggle kernel may have stopped or the tunnel is stale.",
        )
    except requests.exceptions.RequestException as exc:
        return _error_response(
            f"Remote backend unreachable: {type(exc).__name__}: {exc}"[:300],
        )

    if resp.status_code != 200:
        body_preview = resp.text[:200] if resp.text else "<empty body>"
        return _error_response(
            f"Remote backend returned HTTP {resp.status_code}: {body_preview}",
        )

    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        return _error_response(
            f"Remote backend returned non-JSON ({exc}): {resp.text[:200]}",
        )

    # Server already returns the same shape as the in-process engines (see
    # kaggle/cliniq_inference_server.py — it intentionally mirrors
    # zerogpu_engine_mtp.chat_completion's response). Pass through unchanged.
    return data


def chat_http_shim(endpoint: str, payload: dict, timeout: float = 900.0) -> dict:
    """Drop-in replacement for `agent_pipeline.chat()` — same as in the
    in-process engines so the monkey-patch in `app.py` works identically.

    `endpoint` and `timeout` from the caller are ignored; we use
    `CLINIQ_REMOTE_URL` and `CLINIQ_REMOTE_TIMEOUT` instead so the wire
    target is configured at the Space level, not per call.
    """
    _ = endpoint, timeout
    return chat_completion(**payload)


def model_banner() -> str:
    """Status string for the Gradio UI's Advanced row.

    Calls `/healthz` once (cached) so the banner reflects the actual remote
    state, not just env var presence. Same return type as the in-process
    engines.
    """
    if not REMOTE_URL:
        return "Remote backend unavailable (CLINIQ_REMOTE_URL unset)"

    health = _health_probe()
    if not health.get("ok"):
        err = health.get("error", "unknown error")
        return f"Remote backend unhealthy: {err} (target {REMOTE_URL})"

    banner = health.get("banner") or "remote Gemma 4"
    return f"{banner} · via remote tunnel {REMOTE_URL}"
