"""Gemma 4 ZeroGPU inference engine for HF Spaces.

Replaces the HTTP llama-server backend with in-process transformers inference
on ZeroGPU's H200, exposing an OpenAI-compatible `chat_completion` so the
existing `agent_pipeline.run_agent` works unchanged via a small monkey-patch
in `app.py`.

Why transformers and not llama-cpp-python:
  ZeroGPU relies on PyTorch's CUDA emulation outside `@spaces.GPU` so models
  can be loaded on `cuda` at module level (HF docs: "Lazy-loading or moving
  models to CUDA inside @spaces.GPU is discouraged"). llama-cpp-python is
  a C++ library with its own CUDA bindings — it doesn't benefit from the
  PyTorch emulation, so it would have to cold-load on every call (~5-10s
  overhead). transformers + ZeroGPU is the canonical pattern.

Cost: we lose llama-server's GBNF grammar locking on the tool-call wire
format. To compensate we (a) decode at temperature=0.0, (b) parse Gemma 4's
native `<|tool_call>...<tool_call|>` sentinels into OpenAI-format tool_calls
ourselves (the same parser the iOS Swift `ToolCallParser` runs), (c) keep
the existing agent_pipeline's malformed-call recovery as a safety net.
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

# Hardware detection. SPACES_ZERO_GPU=true is set inside ZeroGPU runtimes;
# torch.cuda.is_available() is True on standard GPU hardware. Outside both,
# we fall back to CPU so the Space at least boots — the agent path will be
# slow (~minutes per case) but the deterministic + fast-path tiers stay
# instant, and judges still see the model fire eventually.
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

print(f"[zerogpu_engine] loading {MODEL_ID} on {_DEVICE} ({_BACKEND}) ...", flush=True)
_LOAD_ERROR: str | None = None
_tokenizer = None  # type: ignore[assignment]
_model = None  # type: ignore[assignment]
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
        f"[zerogpu_engine] loaded {MODEL_ID} in {time.time() - _t0:.1f}s "
        f"({_N_PARAMS_B:.2f} B params, {_DTYPE}, device={_model.device})",
        flush=True,
    )
except Exception as exc:  # noqa: BLE001 — surface to the UI, don't crash boot
    _LOAD_ERROR = f"{type(exc).__name__}: {exc}"[:400]
    _N_PARAMS_B = 0.0
    print(f"[zerogpu_engine] ERROR loading {MODEL_ID}: {_LOAD_ERROR}", flush=True)


def _ensure_loaded() -> None:
    if _LOAD_ERROR is not None or _model is None:
        raise RuntimeError(
            f"Gemma 4 model unavailable on this Space: "
            f"{_LOAD_ERROR or 'model not loaded'}. "
            "Check the Space's Hardware setting (ZeroGPU recommended) and "
            "the build log for the model-download trace."
        )


# -----------------------------------------------------------------------------
# Gemma 4 tool-call parser
#
# Gemma 4's chat template emits tool calls in this wire format (see
# apps/mobile/convert/cliniq_toolcall.gbnf for the GBNF spec):
#
#     <|tool_call>call:NAME{key:"value",key2:42,...}<tool_call|>
#
# Keys are bareword identifiers OR sentinel-quoted strings; values can be
# Gemma's `<|"|>...<|"|>` sentinel-quoted strings, JSON-quoted strings,
# numbers, bools, null, or nested arrays/objects. To convert to OpenAI's
# `tool_calls` shape we extract NAME + the brace-delimited body, normalize
# Gemma's sentinel quotes to JSON quotes, then json.loads the result.

_TOOL_CALL_RE = re.compile(
    r"<\|tool_call\>\s*call:\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?P<body>\{.*?\})\s*<tool_call\|>",
    re.DOTALL,
)
_SENTINEL_QUOTE = "<|\"|>"


def _normalize_gemma_body(body: str) -> str:
    """Best-effort conversion from Gemma's bareword/sentinel object syntax to
    strict JSON, suitable for json.loads.

    Two observed forms in Gemma 4 outputs (per cliniq_toolcall.gbnf comments):

      A. JSON-strict already:        {"text": "..."}
      B. Bareword keys + JSON values: {text: "...", n: 3}
      C. Sentinel-quoted strings:    {text: <|"|>...<|"|>}

    Strategy: replace sentinel quotes with JSON quotes, then quote any
    bareword keys (the regex matches `key:` only when preceded by `{` or `,`).
    """
    s = body.replace(_SENTINEL_QUOTE, '"')
    s = re.sub(
        r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)',
        r'\1"\2"\3',
        s,
    )
    return s


def parse_gemma_tool_calls(text: str) -> tuple[str | None, list[dict] | None]:
    """Extract `<|tool_call>...<tool_call|>` blocks from Gemma 4 output.

    Returns `(content_residual, tool_calls)`:
      - `content_residual` is the text with all tool-call blocks removed
        (or the original text if there were none). Empty string → `None`.
      - `tool_calls` is a list of OpenAI-format tool_call dicts, or `None`
        if the model emitted plain text only.
    """
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
# Drop-in replacement for `apps/mobile/convert/agent_pipeline.chat()`. Same
# request/response shape, just no HTTP — the model is in-process on the
# ZeroGPU H200 allocated for the duration of this call.

@spaces.GPU(duration=120)
def chat_completion(
    *,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 3072,
    **_ignored: Any,
) -> dict:
    """Single chat-completion turn on Gemma 4 + ZeroGPU.

    `messages` and `tools` follow the OpenAI shape. Returns an OpenAI-format
    response with `choices[0].message.{content, tool_calls}` and a synthetic
    `timings.predicted_per_second` so downstream bench harnesses keep their
    tok/s columns populated.
    """
    _ensure_loaded()
    prompt_ids = _tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(_model.device)

    do_sample = temperature > 0.0
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_tokens),
        "do_sample": do_sample,
        "pad_token_id": _tokenizer.pad_token_id or _tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)

    t0 = time.time()
    with torch.inference_mode():
        output_ids = _model.generate(prompt_ids, **gen_kwargs)
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
        },
    }


def chat_http_shim(endpoint: str, payload: dict, timeout: float = 900.0) -> dict:
    """Drop-in replacement for `agent_pipeline.chat()`.

    Signature-compat with the HTTP transport so monkey-patching is one line:
        agent_pipeline.chat = zerogpu_engine.chat_http_shim
    `endpoint` and `timeout` are ignored — inference is in-process.
    """
    _ = endpoint, timeout
    return chat_completion(**payload)


def model_banner() -> str:
    """Short status string for the UI."""
    if _LOAD_ERROR is not None:
        return f"⚠️ Gemma 4 unavailable — {_LOAD_ERROR}"
    return (
        f"Gemma 4 ({MODEL_ID.split('/')[-1]}, {_N_PARAMS_B:.1f} B params, "
        f"{str(_DTYPE).split('.')[-1]}) running in-process on {_BACKEND}"
    )
