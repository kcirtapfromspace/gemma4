#!/usr/bin/env python3
"""
Phase 1 diagnosis for Team C7:
Load the 490 LoRA tensors from models/cliniq-compact-lora.gguf, compute
||B @ A||_F for every (layer, projection) pair, then compare shared-KV
layers (15-34) against non-shared layers (0-14) and against always-trained
Q/O/gate/up/down projections.

The model has 35 layers; num_kv_shared_layers=20, so layers [15..34] reuse
the K/V caches of layers [0..14] at INFERENCE time. During training with
past_key_values=None (the normal SFT setup), the HF gemma3n forward pass
still calls self.k_proj(x) / self.v_proj(x) on those layers, so LoRA
deltas *will* accumulate gradients there.

Outcomes:
 (a) ||BA|| on shared-K/V ~ near-zero    -> drops benign, real regression elsewhere
 (b) ||BA|| on shared-K/V ~ trained norms -> drops harmful, retrain with masked target_modules
 (c) mixed                                -> document and recommend
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gguf
import numpy as np

GGUF_PATH = Path("/Users/thinkstudio/gemma4/models/cliniq-compact-lora.gguf")
NUM_LAYERS = 35
NUM_KV_SHARED = 20
FIRST_SHARED = NUM_LAYERS - NUM_KV_SHARED  # 15

GGUF_PROJ_TO_HF = {
    "attn_q": "q_proj",
    "attn_k": "k_proj",
    "attn_v": "v_proj",
    "attn_output": "o_proj",
    "ffn_gate": "gate_proj",
    "ffn_up": "up_proj",
    "ffn_down": "down_proj",
}


def _to_numpy(tensor: "gguf.ReaderTensor") -> np.ndarray:
    """Dequantise a GGUF reader tensor to float32 numpy.

    LoRA adapters produced by llama.cpp's convert_lora_to_gguf.py are stored
    in F16 or F32; we only need to handle those.
    """
    data = tensor.data
    if data.dtype == np.float16:
        arr = np.asarray(data, dtype=np.float32)
    elif data.dtype == np.float32:
        arr = np.asarray(data, dtype=np.float32)
    else:
        # Try to view raw bytes; this will raise if quantised.
        arr = np.asarray(data).astype(np.float32)
    # GGUF stores matrices row-major with the *trailing* dim being fastest.
    # Tensor shape from gguf is reversed compared to a NumPy memory order,
    # so reinterpret:
    shape = tuple(int(x) for x in tensor.shape[::-1])
    try:
        arr = arr.reshape(shape)
    except ValueError as exc:
        raise RuntimeError(
            f"Cannot reshape {tensor.name} data (size {arr.size}) to {shape}: {exc}"
        )
    return arr


def load_lora_pairs(path: Path) -> Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]]:
    """Return dict keyed by (layer_idx, hf_proj) with (A, B) matrices."""
    reader = gguf.GGUFReader(str(path))
    ab: Dict[Tuple[int, str], Dict[str, np.ndarray]] = defaultdict(dict)
    for tensor in reader.tensors:
        name = tensor.name
        # Expected form: blk.<layer>.<gguf_proj>.weight.lora_<a|b>
        if not name.startswith("blk."):
            continue
        parts = name.split(".")
        if len(parts) < 5:
            continue
        layer = int(parts[1])
        gguf_proj = parts[2]
        suffix = parts[-1]
        if gguf_proj not in GGUF_PROJ_TO_HF:
            continue
        if suffix not in ("lora_a", "lora_b"):
            continue
        hf_proj = GGUF_PROJ_TO_HF[gguf_proj]
        side = "A" if suffix == "lora_a" else "B"
        ab[(layer, hf_proj)][side] = _to_numpy(tensor)

    pairs: Dict[Tuple[int, str], Tuple[np.ndarray, np.ndarray]] = {}
    for key, parts in ab.items():
        if "A" not in parts or "B" not in parts:
            raise RuntimeError(f"Missing A/B for {key}: keys={list(parts)}")
        pairs[key] = (parts["A"], parts["B"])
    return pairs


def frobenius(a: np.ndarray, b: np.ndarray) -> float:
    """Compute ||B @ A||_F without fully materialising the product when large."""
    # A typical LoRA: A is (r, in), B is (out, r). ``B @ A`` is (out, in).
    # For small r and modest out/in we just compute directly; it is a single
    # matmul of shape (out, r) x (r, in) -> (out, in). For Gemma-E2B hidden
    # dims at most ~16384, this is <16M entries so memory is fine.
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"Expected 2D A/B, got A.shape={a.shape} B.shape={b.shape}")
    # llama.cpp's convert_lora_to_gguf.py stores lora_a with shape (r, in)
    # and lora_b with shape (out, r). If the reversal changed this, one of
    # the two orderings will fail the inner-dim check; handle both.
    if a.shape[0] == b.shape[1]:
        # a = (r, in), b = (out, r), matmul B @ A
        prod = b @ a
    elif a.shape[1] == b.shape[0]:
        # a = (in, r), b = (r, out), matmul A @ B
        prod = a @ b
    else:
        raise ValueError(f"Cannot align A.shape={a.shape} B.shape={b.shape}")
    return float(np.linalg.norm(prod))


def initial_lora_std(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Return (std_A, std_B, expected_||BA||_F_if_B_were_zero).

    PEFT initialises A ~ kaiming_uniform and B = 0.
    If B is *still* exactly zero, ||BA||_F = 0. If B has only numerical
    roundoff, the norm should be <=  sqrt(A.size * B.size) * eps, which is
    tiny.
    """
    std_a = float(np.std(a))
    std_b = float(np.std(b))
    expected = std_b * math.sqrt(b.size) * std_a * math.sqrt(a.shape[-1])
    return std_a, std_b, expected


def main() -> int:
    if not GGUF_PATH.exists():
        print(f"GGUF not found: {GGUF_PATH}", file=sys.stderr)
        return 1

    pairs = load_lora_pairs(GGUF_PATH)
    print(f"Loaded {len(pairs)} (layer, proj) LoRA pairs from {GGUF_PATH}\n")

    # Organise
    results: Dict[Tuple[int, str], Dict[str, float]] = {}
    for (layer, proj), (a, b) in pairs.items():
        std_a, std_b, _ = initial_lora_std(a, b)
        norm_ba = frobenius(a, b)
        results[(layer, proj)] = {
            "norm_ba": norm_ba,
            "std_a": std_a,
            "std_b": std_b,
            "a_shape": tuple(a.shape),
            "b_shape": tuple(b.shape),
        }

    # Summary buckets
    def bucket(layers, projs):
        vals = []
        for L in layers:
            for P in projs:
                r = results.get((L, P))
                if r is not None:
                    vals.append(r["norm_ba"])
        return vals

    non_shared = list(range(0, FIRST_SHARED))      # 0..14
    shared = list(range(FIRST_SHARED, NUM_LAYERS))  # 15..34

    kv = ["k_proj", "v_proj"]
    qo = ["q_proj", "o_proj"]
    mlp = ["gate_proj", "up_proj", "down_proj"]

    ns_kv = bucket(non_shared, kv)
    sh_kv = bucket(shared, kv)
    sh_qo = bucket(shared, qo)
    sh_mlp = bucket(shared, mlp)
    ns_qo = bucket(non_shared, qo)
    ns_mlp = bucket(non_shared, mlp)

    def stats(label, vals):
        if not vals:
            print(f"{label}: (empty)")
            return
        print(
            f"{label}: n={len(vals)} mean={statistics.mean(vals):.4g} "
            f"median={statistics.median(vals):.4g} "
            f"min={min(vals):.4g} max={max(vals):.4g}"
        )

    print("=== L2 norms of ||B @ A||_F grouped by layer range and projection type ===\n")
    stats("layers  0-14 k/v_proj (trained, own KV)       ", ns_kv)
    stats("layers 15-34 k/v_proj (SHARED — allegedly dropped)", sh_kv)
    stats("layers  0-14 q/o_proj (trained, own attn)     ", ns_qo)
    stats("layers 15-34 q/o_proj (trained, own Q/O)      ", sh_qo)
    stats("layers  0-14 gate/up/down (trained, MLP)      ", ns_mlp)
    stats("layers 15-34 gate/up/down (trained, MLP)      ", sh_mlp)

    # Per-layer per-proj table for K/V only (the interesting case)
    print("\n=== Per-layer ||BA||_F table: K/V projections ===\n")
    print(f"{'layer':>5} | {'k_proj':>10} | {'v_proj':>10} | {'shared-KV?':>10}")
    print("-" * 48)
    for L in range(NUM_LAYERS):
        k = results.get((L, "k_proj"), {}).get("norm_ba", float("nan"))
        v = results.get((L, "v_proj"), {}).get("norm_ba", float("nan"))
        marker = "YES" if L >= FIRST_SHARED else ""
        print(f"{L:>5d} | {k:>10.4g} | {v:>10.4g} | {marker:>10}")

    # Ratios for outcome classification
    if ns_kv and sh_kv:
        ratio = statistics.mean(sh_kv) / statistics.mean(ns_kv)
        print(
            f"\nshared-KV mean-||BA|| / non-shared-KV mean-||BA|| = {ratio:.3f}"
        )
        if ratio < 0.1:
            outcome = "(a) near-zero  — training effectively ignored the shared-KV projections"
        elif ratio > 0.5:
            outcome = "(b) comparable — training DID update shared-KV projections (bug!)"
        else:
            outcome = "(c) mixed      — investigate per-layer"
        print(f"OUTCOME (by ratio heuristic): {outcome}")

    # JSON dump for follow-up
    per_pair = {}
    for (layer, proj), r in results.items():
        per_pair[f"L{layer}.{proj}"] = {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()
        }
    out_json = Path("/Users/thinkstudio/gemma4/.claude/worktrees/agent-aa94a1cb/apps/training/diagnose_lora_output.json")
    out_json.write_text(json.dumps({
        "num_layers": NUM_LAYERS,
        "num_kv_shared_layers": NUM_KV_SHARED,
        "first_shared_layer_idx": FIRST_SHARED,
        "per_pair": per_pair,
        "summary": {
            "ns_kv": ns_kv,
            "sh_kv": sh_kv,
            "ns_qo": ns_qo,
            "sh_qo": sh_qo,
            "ns_mlp": ns_mlp,
            "sh_mlp": sh_mlp,
        },
    }, indent=2))
    print(f"\nWrote details to {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
