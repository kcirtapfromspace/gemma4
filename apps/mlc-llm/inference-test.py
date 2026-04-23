#!/usr/bin/env python3
"""Baseline inference test against v2 compiled lib.so on Jetson.

Measures tok/s, emits per-20-token markers on the output so we can see
exactly where degeneration begins.
"""
import os, sys, time, json
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu/tegra:/usr/local/lib/python3.10/dist-packages/tvm:/usr/local/cuda/lib64"

from mlc_llm import MLCEngine
from transformers import AutoTokenizer

MODEL = "/models/mlc-models/gemma4-e2b-q4f16_1-v2"
LIB   = "/models/mlc-models/gemma4-e2b-q4f16_1-v2/lib.so"

t0 = time.time()
engine = MLCEngine(model=MODEL, model_lib=LIB, device="cuda", mode="interactive")
print(f"[load] engine ready in {time.time()-t0:.1f}s", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

PROMPTS = [
    ("SHORT_STORY", "Write a short story about a robot learning to paint."),
    ("LONG_STORY",  "Write a detailed 800-word story about a time traveler who visits ancient Rome. Describe what they see, hear, and feel. Be descriptive and use many senses."),
    ("CLINICAL",    "Extract the key clinical facts from this note as JSON:\n\nPatient: John Doe, 65yo male. Chief complaint: chest pain x 3 hours. Vitals: BP 158/92, HR 104, SpO2 97%. ECG: ST elevation in leads II, III, aVF consistent with inferior MI. Started ASA 325mg, IV heparin, nitroglycerin drip. Cath lab activated.\n\nJSON:"),
    ("FACTS",       "List 10 facts about the human cardiovascular system:"),
]
MAX_TOK = 400

for label, prompt in PROMPTS:
    print(f"\n================ {label} ================", flush=True)
    print(f"[prompt] {prompt[:200]}", flush=True)
    t_start = time.time()
    full = ""
    first_tok_time = None
    chunk_ts = []
    try:
        stream = engine.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOK,
            temperature=0.0,
            top_p=1.0,
            stream=True,
        )
        for chunk in stream:
            now = time.time() - t_start
            if first_tok_time is None:
                first_tok_time = now
            for choice in chunk.choices:
                d = choice.delta.content
                if d:
                    full += d
                    chunk_ts.append((len(full), now))
    except Exception as e:
        print(f"[error] {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        continue
    elapsed = time.time() - t_start
    n_tokens = len(tok.encode(full, add_special_tokens=False))
    tps = n_tokens / max(elapsed, 0.001)
    print(f"[stats] {n_tokens} tokens in {elapsed:.2f}s = {tps:.2f} tok/s  (TTFT={first_tok_time:.2f}s)", flush=True)
    ids = tok.encode(full, add_special_tokens=False)
    marked = []
    for i in range(0, len(ids), 20):
        piece = tok.decode(ids[i:i+20])
        marked.append(f"[TOK{i}]{piece}")
    print(f"[output-marked-by-20]", flush=True)
    print("".join(marked), flush=True)
    print(f"--- END {label} ---", flush=True)

print("\n[done]", flush=True)
