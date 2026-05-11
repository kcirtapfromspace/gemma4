"""ClinIQ Gemma 4 inference server — Kaggle T4 + MTP.

Standalone Kaggle "kernel script" that:
  1. Pip-installs transformers main HEAD (the SHA proven by the mtp-mlx-bench
     to ship the `gemma4_assistant` model_type), accelerate, fastapi, uvicorn,
     sentencepiece.
  2. Downloads the cloudflared binary.
  3. Loads `google/gemma-4-E2B-it` (target) and
     `google/gemma-4-E2B-it-assistant` (78 M drafter) on cuda.
  4. Serves an OpenAI-compatible `/v1/chat/completions` endpoint that mirrors
     the request/response shape produced by `spaces/zerogpu_engine_mtp.py`,
     including the Gemma 4 `<|tool_call>...<tool_call|>` parser → OpenAI
     `tool_calls` translation.
  5. Adds `assistant_model=drafter` to `model.generate(...)` for ~1.92×
     decode speedup (Multi-Token Prediction / assisted decoding) — this is
     the whole reason for pinning transformers main HEAD.
  6. Exposes `/healthz` for the Spaces client to probe.
  7. Boots uvicorn on :8000 then subprocesses cloudflared to mint a public
     `https://*.trycloudflare.com` tunnel URL, prints it, and writes it to
     `/kaggle/working/tunnel_url.txt` so `kaggle kernels output` can fetch it.

Companion: `spaces/zerogpu_engine_remote.py` is the HTTP client wrapper that
the Spaces app uses to talk to this server. They share the same response
schema so the Spaces UI works without code changes.

Run via Kaggle CLI:
    kaggle kernels push -p kaggle/

See `tools/autoresearch/kaggle-backend-runbook.md` for the full operator
runbook.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. Dependency install (Kaggle GPU notebooks ship with torch + transformers
# tagged, but we need the main-HEAD SHA for `gemma4_assistant` + assisted
# decoding, plus FastAPI / uvicorn for the OpenAI shim.)

_TRANSFORMERS_SHA = "41c3a5ac425e81aa1c9b3e6288eebaccf0c89835"
_PIP_PACKAGES = [
    f"git+https://github.com/huggingface/transformers.git@{_TRANSFORMERS_SHA}",
    "accelerate>=1.0",
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "sentencepiece>=0.2",
]


def _pip_install(packages: list[str]) -> None:
    print(f"[bootstrap] pip install: {packages}", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", *packages],
    )


_pip_install(_PIP_PACKAGES)


# -----------------------------------------------------------------------------
# 2. cloudflared binary download (no apt; Kaggle internet is on)

_CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64"
)
_CLOUDFLARED_BIN = Path("/kaggle/working/cloudflared")


def _ensure_cloudflared() -> Path:
    if _CLOUDFLARED_BIN.exists():
        print(f"[bootstrap] cloudflared already at {_CLOUDFLARED_BIN}", flush=True)
        return _CLOUDFLARED_BIN
    print(f"[bootstrap] downloading cloudflared from {_CLOUDFLARED_URL}", flush=True)
    subprocess.check_call(
        ["curl", "-L", "--silent", "--show-error", "--fail",
         "-o", str(_CLOUDFLARED_BIN), _CLOUDFLARED_URL],
    )
    _CLOUDFLARED_BIN.chmod(0o755)
    return _CLOUDFLARED_BIN


_ensure_cloudflared()


# -----------------------------------------------------------------------------
# 3. Heavy imports — only after pip install above so we get the new SHA.

import json  # noqa: E402
import re  # noqa: E402
import threading  # noqa: E402
import uuid  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_ID = os.environ.get("CLINIQ_GEMMA_MODEL_ID", "google/gemma-4-E2B-it")
DRAFTER_ID = os.environ.get(
    "CLINIQ_GEMMA_DRAFTER_ID", "google/gemma-4-E2B-it-assistant",
)
_MTP_ENABLED = os.environ.get("MTP_ENABLED", "true").lower() not in (
    "0", "false", "no", "off",
)

if torch.cuda.is_available():
    _DEVICE = "cuda"
    _DTYPE = torch.bfloat16
    _BACKEND = f"CUDA ({torch.cuda.get_device_name(0)})"
else:
    _DEVICE = "cpu"
    _DTYPE = torch.float32
    _BACKEND = "CPU (no GPU detected — Kaggle kernel must enable T4)"

print(
    f"[engine] loading {MODEL_ID} on {_DEVICE} ({_BACKEND}) "
    f"+ drafter={DRAFTER_ID if _MTP_ENABLED else '<disabled>'}",
    flush=True,
)

_LOAD_ERROR: str | None = None
_DRAFTER_LOAD_ERROR: str | None = None
_tokenizer = None
_model = None
_drafter = None
_N_PARAMS_B = 0.0
_DRAFTER_N_PARAMS_M = 0.0

try:
    _t0 = time.time()
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=_DTYPE,
        device_map=_DEVICE,
    )
    _N_PARAMS_B = sum(p.numel() for p in _model.parameters()) / 1e9
    print(
        f"[engine] loaded target {MODEL_ID} in {time.time() - _t0:.1f}s "
        f"({_N_PARAMS_B:.2f} B params, {_DTYPE}, device={_model.device})",
        flush=True,
    )
except Exception as exc:  # noqa: BLE001
    _LOAD_ERROR = f"{type(exc).__name__}: {exc}"[:400]
    print(f"[engine] ERROR loading {MODEL_ID}: {_LOAD_ERROR}", flush=True)

if _LOAD_ERROR is None and _MTP_ENABLED:
    try:
        _t0 = time.time()
        _drafter = AutoModelForCausalLM.from_pretrained(
            DRAFTER_ID,
            torch_dtype=_DTYPE,
            device_map=_DEVICE,
        )
        _DRAFTER_N_PARAMS_M = sum(p.numel() for p in _drafter.parameters()) / 1e6
        print(
            f"[engine] loaded drafter {DRAFTER_ID} in {time.time() - _t0:.1f}s "
            f"({_DRAFTER_N_PARAMS_M:.0f} M params, {_DTYPE}, "
            f"device={_drafter.device})",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        _DRAFTER_LOAD_ERROR = f"{type(exc).__name__}: {exc}"[:400]
        _drafter = None
        print(
            f"[engine] WARN drafter load failed ({_DRAFTER_LOAD_ERROR}); "
            f"falling back to non-MTP decode.",
            flush=True,
        )


# -----------------------------------------------------------------------------
# 4. Gemma 4 tool-call parser — bit-identical to the in-process engines so the
# response shape matches exactly. Keep these three impls in sync (this file,
# spaces/zerogpu_engine.py, spaces/zerogpu_engine_mtp.py).

_TOOL_CALL_RE = re.compile(
    r"<\|tool_call\>\s*call:\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?P<body>\{.*?\})\s*<tool_call\|>",
    re.DOTALL,
)
_SENTINEL_QUOTE = "<|\"|>"


def _normalize_gemma_body(body: str) -> str:
    s = body.replace(_SENTINEL_QUOTE, '"')
    s = re.sub(
        r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)',
        r'\1"\2"\3',
        s,
    )
    return s


def parse_gemma_tool_calls(text: str) -> tuple[str | None, list[dict] | None]:
    calls: list[dict] = []
    cleaned_chunks: list[str] = []
    cursor = 0
    for m in _TOOL_CALL_RE.finditer(text):
        cleaned_chunks.append(text[cursor:m.start()])
        cursor = m.end()
        name = m.group("name")
        body = m.group("body")
        try:
            args_dict = json.loads(_normalize_gemma_body(body))
        except json.JSONDecodeError:
            args_dict = {"_raw": body}
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(args_dict),
            },
        })
    cleaned_chunks.append(text[cursor:])
    cleaned = "".join(cleaned_chunks).strip() or None
    if not calls:
        return text.strip() or None, None
    return cleaned, calls


# -----------------------------------------------------------------------------
# 5. FastAPI app

app = FastAPI(title="ClinIQ Gemma 4 inference (Kaggle T4 + MTP)")


class _ChatRequest(BaseModel):
    messages: list[dict]
    tools: list[dict] | None = None
    tool_choice: Any = None  # accepted for OpenAI compat; not consumed
    temperature: float = 0.0
    max_tokens: int = Field(default=3072)
    # Optional alias OpenAI-style clients send.
    max_completion_tokens: int | None = None
    model: str | None = None  # accepted but ignored (we serve a single model)
    # Catch-all for fields we want to ignore rather than 422 on.
    stream: bool | None = False

    class Config:
        extra = "allow"


# Force pydantic to finalize the schema at import time. Avoids a
# `class-not-fully-defined` error when the module is imported dynamically
# (e.g. via importlib.exec_module in tests, or by uvicorn under some
# loader configurations).
_ChatRequest.model_rebuild()


def model_banner() -> str:
    if _LOAD_ERROR is not None:
        return f"Gemma 4 unavailable — {_LOAD_ERROR}"
    base = (
        f"Gemma 4 ({MODEL_ID.split('/')[-1]}, {_N_PARAMS_B:.1f} B params, "
        f"{str(_DTYPE).split('.')[-1]}) on {_BACKEND}"
    )
    if _drafter is not None:
        return (
            f"{base} · MTP on (drafter "
            f"{DRAFTER_ID.split('/')[-1]}, {_DRAFTER_N_PARAMS_M:.0f} M)"
        )
    if _MTP_ENABLED and _DRAFTER_LOAD_ERROR is not None:
        return f"{base} · MTP requested but drafter failed: {_DRAFTER_LOAD_ERROR}"
    return f"{base} · MTP disabled"


def _chat_completion(
    messages: list[dict],
    tools: list[dict] | None,
    temperature: float,
    max_tokens: int,
) -> dict:
    if _LOAD_ERROR is not None or _model is None:
        raise RuntimeError(
            f"Gemma 4 model unavailable on this Kaggle kernel: "
            f"{_LOAD_ERROR or 'model not loaded'}",
        )

    chat_out = _tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # transformers main HEAD returns BatchEncoding here; tagged returns Tensor.
    if isinstance(chat_out, torch.Tensor):
        prompt_ids = chat_out.to(_model.device)
        gen_inputs: dict[str, Any] = {"input_ids": prompt_ids}
    else:
        gen_inputs = {
            k: (v.to(_model.device) if isinstance(v, torch.Tensor) else v)
            for k, v in chat_out.items()
        }
        prompt_ids = gen_inputs["input_ids"]

    do_sample = temperature > 0.0
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_tokens),
        "do_sample": do_sample,
        "pad_token_id": _tokenizer.pad_token_id or _tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)

    mtp_active = _drafter is not None
    if mtp_active:
        gen_kwargs["assistant_model"] = _drafter

    t0 = time.time()
    with torch.inference_mode():
        output_ids = _model.generate(**gen_inputs, **gen_kwargs)
    elapsed = max(time.time() - t0, 1e-6)

    prompt_len = prompt_ids.shape[1]
    response_ids = output_ids[0][prompt_len:]
    n_completion = int(response_ids.shape[0])
    response_text = _tokenizer.decode(response_ids, skip_special_tokens=False)

    content, tool_calls = parse_gemma_tool_calls(response_text)
    finish_reason = "tool_calls" if tool_calls else "stop"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            },
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": int(prompt_len),
            "completion_tokens": n_completion,
            "total_tokens": int(prompt_len) + n_completion,
        },
        "timings": {
            "predicted_per_second": n_completion / elapsed,
            "predicted_n": n_completion,
            "predicted_ms": elapsed * 1000.0,
            "mtp": {
                "enabled": _MTP_ENABLED,
                "active": mtp_active,
                "drafter_id": DRAFTER_ID if mtp_active else None,
                "drafter_load_error": _DRAFTER_LOAD_ERROR,
            },
        },
    }


@app.get("/healthz")
def healthz() -> dict:
    ok = _LOAD_ERROR is None and _model is not None
    return {
        "ok": ok,
        "banner": model_banner(),
        "model_id": MODEL_ID,
        "drafter_id": DRAFTER_ID if _drafter is not None else None,
        "mtp_active": _drafter is not None,
        "load_error": _LOAD_ERROR,
        "drafter_load_error": _DRAFTER_LOAD_ERROR,
    }


@app.post("/v1/chat/completions")
def chat_completions(req: _ChatRequest) -> Any:
    max_tok = req.max_completion_tokens or req.max_tokens
    try:
        out = _chat_completion(
            messages=req.messages,
            tools=req.tools,
            temperature=req.temperature,
            max_tokens=max_tok,
        )
    except Exception as exc:  # noqa: BLE001 — surface JSON, don't 500-html
        return JSONResponse(
            status_code=503,
            content={"error": {"message": f"{type(exc).__name__}: {exc}",
                               "type": "engine_error"}},
        )
    return out


# -----------------------------------------------------------------------------
# 6. cloudflared tunnel + uvicorn

_TUNNEL_URL_FILE = Path("/kaggle/working/tunnel_url.txt")
_TUNNEL_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
_TUNNEL_DATASET_SLUG = os.environ.get("CLINIQ_TUNNEL_DATASET", "cliniq-tunnel-url")
_TUNNEL_DATASET_DIR = Path("/kaggle/working/cliniq-tunnel-meta")


_NTFY_TOPIC = os.environ.get(
    "CLINIQ_NTFY_TOPIC", "cliniq-gemma4-tunnel-kcirtapfromspace-2026-05-06"
)


def _publish_url_to_ntfy(url: str) -> None:
    """POST the tunnel URL to a public ntfy.sh topic — most reliable exfil.

    No auth needed; the host polls https://ntfy.sh/<topic>/raw?poll=1.
    """
    import urllib.request
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{_NTFY_TOPIC}",
            data=url.encode("utf-8"),
            method="POST",
            headers={"Title": "ClinIQ tunnel up", "Priority": "high"},
        )
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode("utf-8", errors="replace")[:200]
            print(f"[tunnel-ntfy] POST ntfy.sh/{_NTFY_TOPIC} -> "
                  f"{r.status} {body!r}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[tunnel-ntfy] POST FAILED: {type(exc).__name__}: {exc}",
              flush=True)


def _publish_url_to_dataset(url: str) -> None:
    """Push the tunnel URL to a Kaggle Dataset so the host can poll for it.

    Inside a Kaggle kernel, the kaggle CLI is auto-authenticated as the
    kernel owner via the runtime's KAGGLE_USERNAME / KAGGLE_KEY env vars.
    This lets us "export" the live tunnel URL out-of-band — `kernels output`
    and `kernels files` only return data after a kernel COMPLETES, but
    datasets can be created/versioned mid-run.
    """
    try:
        owner = os.environ.get("KAGGLE_USERNAME", "kaggle")
        slug = _TUNNEL_DATASET_SLUG
        full_id = f"{owner}/{slug}"
        _TUNNEL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        (_TUNNEL_DATASET_DIR / "tunnel_url.txt").write_text(url + "\n")
        # Include the timestamp so the host can sanity-check freshness.
        (_TUNNEL_DATASET_DIR / "minted_at.txt").write_text(
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n"
        )
        meta = {
            "title": "ClinIQ Gemma 4 tunnel URL",
            "id": full_id,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (_TUNNEL_DATASET_DIR / "dataset-metadata.json").write_text(
            __import__("json").dumps(meta, indent=2)
        )
        # Try `version` first (covers the steady-state case where the
        # dataset already exists). If the dataset doesn't exist yet,
        # `version` 404s and we fall through to `create --public`.
        for cmd in (
            ["kaggle", "datasets", "version", "-p", str(_TUNNEL_DATASET_DIR),
             "-m", f"tunnel up @ {time.strftime('%H:%M:%SZ', time.gmtime())}",
             "--dir-mode", "zip"],
            ["kaggle", "datasets", "create", "-p", str(_TUNNEL_DATASET_DIR),
             "--public", "--dir-mode", "zip"],
        ):
            r = subprocess.run(cmd, capture_output=True, text=True)
            print(f"[tunnel-export] {' '.join(cmd[:3])} rc={r.returncode}",
                  flush=True)
            if r.stdout:
                print(f"[tunnel-export] stdout: {r.stdout.strip()[:400]}",
                      flush=True)
            if r.stderr:
                print(f"[tunnel-export] stderr: {r.stderr.strip()[:400]}",
                      flush=True)
            if r.returncode == 0:
                print(f"[tunnel-export] published to "
                      f"https://www.kaggle.com/datasets/{full_id}",
                      flush=True)
                return
        print("[tunnel-export] both version + create failed; URL only "
              "available via /kaggle/working/tunnel_url.txt", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[tunnel-export] FAILED: {type(exc).__name__}: {exc}",
              flush=True)


def _start_tunnel(port: int) -> subprocess.Popen:
    cmd = [
        str(_CLOUDFLARED_BIN), "tunnel",
        "--url", f"http://127.0.0.1:{port}",
        "--no-autoupdate",
    ]
    print(f"[tunnel] launching: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _watch_tunnel(proc: subprocess.Popen) -> None:
    """Stream cloudflared output, capture the trycloudflare.com URL, persist."""
    captured = False
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(f"[cloudflared] {line}")
        sys.stdout.flush()
        if not captured:
            m = _TUNNEL_URL_RE.search(line)
            if m:
                url = m.group(0)
                _TUNNEL_URL_FILE.write_text(url + "\n")
                # Banner print for the user, with whitespace so it's spottable
                # in the noisy Kaggle console.
                banner = (
                    "\n" + "=" * 72 +
                    f"\nCLINIQ_REMOTE_URL={url}\n" +
                    f"(also written to {_TUNNEL_URL_FILE})\n" +
                    "=" * 72 + "\n"
                )
                print(banner, flush=True)
                captured = True
                _publish_url_to_ntfy(url)
                _publish_url_to_dataset(url)


def main(port: int = 8000) -> None:
    if _LOAD_ERROR is not None:
        print(f"[main] WARNING: model load failed ({_LOAD_ERROR}); "
              f"server will start but /v1/chat/completions will 503.",
              flush=True)

    tunnel_proc = _start_tunnel(port)
    watcher = threading.Thread(target=_watch_tunnel, args=(tunnel_proc,), daemon=True)
    watcher.start()

    print(f"[main] starting uvicorn on 0.0.0.0:{port}", flush=True)
    # uvicorn.run blocks until SIGTERM / kernel timeout.
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main(port=int(os.environ.get("PORT", "8000")))
