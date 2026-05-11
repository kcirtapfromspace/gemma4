"""LiteRT-LM v0.11.0 Mac preflight bench.

Measures end-to-end decode tok/s on real eICR prompts for the
{cpu,gpu} x {no-MTP,MTP} matrix, using the litert_lm Python API.

Also explicitly tests issue #2181 (TypeError when
enable_speculative_decoding=True on Mac Python Engine).
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

from litert_lm.engine import Engine
from litert_lm.interfaces import Backend, SamplerConfig

MODEL = "/Volumes/models/hf/hub/models--litert-community--gemma-4-E2B-it-litert-lm/snapshots/b4f4f4df93418ddb4aa7da8bf33b584602a5b9f8/gemma-4-E2B-it.litertlm"

PROMPT = """You are a clinical decision support assistant. Read the following eICR snippet and produce a one-paragraph patient summary.

Patient: 6-year-old female, presents to ED with 4 days of fever (Tmax 39.5C), non-productive cough, fatigue, and 2 days of bilateral conjunctivitis. Maculopapular rash noted on day 4 starting on the face and spreading to trunk. Koplik spots seen on buccal mucosa. Immunization records show no MMR vaccine given (parental refusal). Lives in a household with two unvaccinated younger siblings. Recent travel: family vacation to Romania 10 days ago.

Vitals: T 39.2C, HR 128, RR 24, SpO2 96% RA. Lab: WBC 4.1 (low), lymphocyte-predominant; CRP 3.2 mg/dL.

Provide:
1. Most likely diagnosis with reasoning.
2. Reportable disease status.
3. Public health action items."""


def run_one(backend: Backend, mtp: bool) -> dict:
    """Run a single bench config and return metrics."""
    label = f"{backend.name}_mtp={mtp}"
    print(f"\n=== {label} ===", flush=True)
    out = {"label": label, "backend": backend.name, "mtp": mtp}

    t0 = time.perf_counter()
    try:
        eng = Engine(
            model_path=MODEL,
            backend=backend,
            enable_speculative_decoding=mtp,
            max_num_tokens=4096,
        )
    except Exception as e:
        out["status"] = "engine_init_failed"
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
        print(f"  ENGINE INIT FAILED: {out['error']}", flush=True)
        return out
    out["engine_init_s"] = time.perf_counter() - t0

    try:
        # top_k=1 for greedy; deterministic comparison.
        sampler = SamplerConfig(top_k=1)
        conv = eng.create_conversation(sampler_config=sampler)
    except Exception as e:
        out["status"] = "conversation_create_failed"
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
        print(f"  CONV CREATE FAILED: {out['error']}", flush=True)
        eng.close()
        return out

    # Use streaming so we can time tok-by-tok and count tokens.
    chunks = []
    n_chunks = 0
    first_chunk_t = None
    t_start = time.perf_counter()
    try:
        for chunk in conv.send_message_async(PROMPT):
            now = time.perf_counter()
            if first_chunk_t is None:
                first_chunk_t = now
            n_chunks += 1
            # Collect text content.
            content = chunk.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        chunks.append(c.get("text", ""))
            elif isinstance(content, str):
                chunks.append(content)
    except Exception as e:
        out["status"] = "decode_failed"
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
        print(f"  DECODE FAILED: {out['error']}", flush=True)
        try:
            conv.close()
        except Exception:
            pass
        eng.close()
        return out
    t_end = time.perf_counter()

    body = "".join(chunks)
    out["status"] = "ok"
    out["n_stream_chunks"] = n_chunks
    out["wall_total_s"] = t_end - t_start
    out["ttft_s"] = (first_chunk_t - t_start) if first_chunk_t else None
    out["body_chars"] = len(body)

    # Tokenize the produced text to get a fair token count.
    try:
        toks = eng.tokenize(body)
        out["body_tokens"] = len(toks)
        if out["ttft_s"] is not None and (t_end - first_chunk_t) > 0:
            out["decode_tok_s"] = (len(toks) - 1) / (t_end - first_chunk_t)
        else:
            out["decode_tok_s"] = None
    except Exception as e:
        out["body_tokens"] = None
        out["decode_tok_s"] = None
        out["tokenize_error"] = str(e)

    print(
        f"  status=ok wall={out['wall_total_s']:.2f}s "
        f"ttft={out['ttft_s']:.2f}s tokens={out['body_tokens']} "
        f"tok/s={out['decode_tok_s']}",
        flush=True,
    )
    out["body_preview"] = body[:300]

    try:
        conv.close()
    except Exception:
        pass
    eng.close()
    return out


def main():
    results = []
    # Run only one engine per process to avoid interference.
    matrix = [
        (Backend.CPU, False),
        (Backend.CPU, True),
        (Backend.GPU, False),
        (Backend.GPU, True),
    ]
    only = os.environ.get("ONLY", "").strip()
    if only:
        idx = int(only)
        matrix = [matrix[idx]]
    for backend, mtp in matrix:
        results.append(run_one(backend, mtp))

    out_path = os.environ.get("OUT", "/tmp/litert-preflight-bench.json")
    with open(out_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\n[wrote results to {out_path}]")


if __name__ == "__main__":
    main()
