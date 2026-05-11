"""Gemma 4 ZeroGPU inference engine — MTP / assisted-decoding variant.

This is a parallel-track copy of `zerogpu_engine.py` that adds Hugging Face
**assisted decoding** (a.k.a. speculative decoding / Multi-Token Prediction
in the Gemma 4 release announcement) using the official 78 M drafter
`google/gemma-4-E2B-it-assistant`.

API surface is identical to `zerogpu_engine.py` — `chat_completion(...)`,
`chat_http_shim(endpoint, payload, timeout=...)`, and `model_banner()` —
so `app.py` only needs the import line swapped.

Why a separate file:
  The safety-net deploy (handled by `spaces-deploy-prep`) ships the original
  `zerogpu_engine.py` against tagged transformers. MTP requires a main-HEAD
  transformers pin (see `requirements-mtp.txt`); we don't want to risk the
  safety-net's reproducibility on that pin. Both engines coexist in the
  repo; deployment picks one.

Bench evidence (mlx-bench, 2026-05-04, MPS fp16, 9 eICR prompts):
  - base E2B-it (no MTP):    14.24 tok/s
  - base E2B-it + drafter:   23.80 tok/s  (1.67× speedup, 0.72 acceptance)
  - cliniq-fp16-merged + d:  29.13 tok/s  (1.92× speedup, 0.74 acceptance)

See `tools/autoresearch/mtp-mlx-bench-results.md` for the full report and
`tools/autoresearch/spaces-mtp-swap.md` for the deploy-swap procedure.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any

import spaces  # HF Spaces ZeroGPU SDK; @spaces.GPU is a no-op outside ZeroGPU
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("CLINIQ_GEMMA_MODEL_ID", "unsloth/gemma-4-E2B-it")
DRAFTER_ID = os.environ.get(
    "CLINIQ_GEMMA_DRAFTER_ID", "google/gemma-4-E2B-it-assistant"
)
# MTP_ENABLED kill-switch — flip to "0"/"false"/"no" to bypass the drafter
# entirely at runtime without redeploying. When disabled this module behaves
# identically to `zerogpu_engine.py` (modulo the extra import time).
_MTP_ENABLED = os.environ.get("MTP_ENABLED", "true").lower() not in (
    "0",
    "false",
    "no",
    "off",
)

# Hardware detection — matches zerogpu_engine.py exactly.
_IS_ZEROGPU = os.environ.get("SPACES_ZERO_GPU", "").lower() == "true"
_HAS_CUDA = torch.cuda.is_available()
if _IS_ZEROGPU or _HAS_CUDA:
    _DEVICE = "cuda"
    _DTYPE = torch.bfloat16
    _BACKEND = "ZeroGPU H200" if _IS_ZEROGPU else "CUDA GPU"
else:
    _DEVICE = "cpu"
    _DTYPE = torch.float32
    _BACKEND = "CPU (slow — switch the Space hardware to ZeroGPU for usable latency)"

print(
    f"[zerogpu_engine_mtp] loading {MODEL_ID} on {_DEVICE} ({_BACKEND}) "
    f"+ drafter={DRAFTER_ID if _MTP_ENABLED else '<disabled>'} ...",
    flush=True,
)
_LOAD_ERROR: str | None = None
_DRAFTER_LOAD_ERROR: str | None = None
_tokenizer = None  # type: ignore[assignment]
_model = None  # type: ignore[assignment]
_drafter = None  # type: ignore[assignment]
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
        f"[zerogpu_engine_mtp] loaded target {MODEL_ID} in "
        f"{time.time() - _t0:.1f}s ({_N_PARAMS_B:.2f} B params, "
        f"{_DTYPE}, device={_model.device})",
        flush=True,
    )
except Exception as exc:  # noqa: BLE001 — surface to the UI, don't crash boot
    _LOAD_ERROR = f"{type(exc).__name__}: {exc}"[:400]
    print(f"[zerogpu_engine_mtp] ERROR loading {MODEL_ID}: {_LOAD_ERROR}", flush=True)

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
            f"[zerogpu_engine_mtp] loaded drafter {DRAFTER_ID} in "
            f"{time.time() - _t0:.1f}s ({_DRAFTER_N_PARAMS_M:.0f} M params, "
            f"{_DTYPE}, device={_drafter.device})",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 — drafter failure is non-fatal
        _DRAFTER_LOAD_ERROR = f"{type(exc).__name__}: {exc}"[:400]
        _drafter = None
        print(
            f"[zerogpu_engine_mtp] WARN drafter load failed ({_DRAFTER_LOAD_ERROR}); "
            f"falling back to non-MTP decode for the target model.",
            flush=True,
        )


def _ensure_loaded() -> None:
    if _LOAD_ERROR is not None or _model is None:
        raise RuntimeError(
            f"Gemma 4 model unavailable on this Space: "
            f"{_LOAD_ERROR or 'model not loaded'}. "
            "Check the Space's Hardware setting (ZeroGPU recommended) and "
            "the build log for the model-download trace."
        )


# -----------------------------------------------------------------------------
# Gemma 4 tool-call parser — bit-identical to zerogpu_engine.py.
# Keep the two implementations in sync if either changes.

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
# OpenAI-compatible chat completion
#
# Identical to zerogpu_engine.chat_completion EXCEPT:
#   - `assistant_model=_drafter` is passed to model.generate when the drafter
#     is loaded (i.e. MTP_ENABLED=true and the load succeeded).
#   - Adds `mtp` block to the response under `timings` for observability.

@spaces.GPU(duration=120)
def chat_completion(
    *,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 3072,
    **_ignored: Any,
) -> dict:
    _ensure_loaded()
    chat_out = _tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # transformers main HEAD (5.8.0.dev0) returns a BatchEncoding here, but
    # tagged versions return a bare Tensor. Normalize so both work.
    if isinstance(chat_out, torch.Tensor):
        prompt_ids = chat_out.to(_model.device)
        gen_inputs: dict[str, Any] = {"input_ids": prompt_ids}
    else:
        # BatchEncoding / dict — move every tensor to the model device.
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


def chat_http_shim(endpoint: str, payload: dict, timeout: float = 900.0) -> dict:
    """Drop-in replacement for `agent_pipeline.chat()`.

    Signature-compat with the HTTP transport so monkey-patching is one line:
        agent_pipeline.chat = zerogpu_engine_mtp.chat_http_shim
    `endpoint` and `timeout` are ignored — inference is in-process.
    """
    _ = endpoint, timeout
    return chat_completion(**payload)


def model_banner() -> str:
    """Short status string for the UI."""
    if _LOAD_ERROR is not None:
        return f"⚠️ Gemma 4 unavailable — {_LOAD_ERROR}"
    base = (
        f"Gemma 4 ({MODEL_ID.split('/')[-1]}, {_N_PARAMS_B:.1f} B params, "
        f"{str(_DTYPE).split('.')[-1]}) running in-process on {_BACKEND}"
    )
    if _drafter is not None:
        return (
            f"{base} · MTP on (drafter "
            f"{DRAFTER_ID.split('/')[-1]}, {_DRAFTER_N_PARAMS_M:.0f} M)"
        )
    if _MTP_ENABLED and _DRAFTER_LOAD_ERROR is not None:
        return f"{base} · MTP requested but drafter failed: {_DRAFTER_LOAD_ERROR}"
    return f"{base} · MTP disabled (MTP_ENABLED={os.environ.get('MTP_ENABLED', 'true')!r})"
