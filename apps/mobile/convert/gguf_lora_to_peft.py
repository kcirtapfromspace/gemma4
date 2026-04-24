#!/usr/bin/env python3
"""Convert a Gemma 4 GGUF LoRA adapter back to a PEFT-compatible safetensors directory.

Why: our compact LoRA was fine-tuned with Unsloth + PEFT, saved as PEFT adapter,
then llama.cpp's ``convert_lora_to_gguf.py`` serialized it to GGUF. The PEFT
directory was NOT synced back from the training box, so we round-trip from
the GGUF.

GGUF LoRA tensor layout (llama.cpp):
    blk.N.attn_q.weight.lora_a   shape = (in_features,  r)     F16
    blk.N.attn_q.weight.lora_b   shape = (r,           out_features)  F16

PEFT safetensors layout (peft 0.19):
    base_model.model.model.layers.N.self_attn.q_proj.lora_A.default.weight
        shape = (r, in_features)
    base_model.model.model.layers.N.self_attn.q_proj.lora_B.default.weight
        shape = (out_features, r)

So each GGUF tensor is just the transpose of the PEFT one. That's the entire
trick.

We also emit a minimal ``adapter_config.json`` — PEFT needs it and we can
reconstruct every required field from the GGUF KV header.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors.torch import save_file


# GGUF projection name -> HF/PEFT attr path suffix (under each decoder layer)
_PROJ_MAP = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}

# Regex: blk.N.<proj>.weight.lora_{a,b}
_GGUF_NAME_RE = re.compile(r"^blk\.(\d+)\.(\w+)\.weight\.lora_([ab])$")


def _kv_scalar(reader: gguf.GGUFReader, key: str):
    """Extract a scalar KV value regardless of dtype."""
    f = reader.fields[key]
    val = f.parts[f.data[0]]
    if hasattr(val, "tolist"):
        v = val.tolist()
        if isinstance(v, list) and len(v) == 1:
            return v[0]
        return v
    return val


def _kv_str(reader: gguf.GGUFReader, key: str) -> str:
    f = reader.fields[key]
    val = f.parts[f.data[0]]
    return val.tobytes().decode("utf-8", errors="replace")


def convert(
    gguf_path: Path, out_dir: Path, base_model_override: str | None = None
) -> None:
    reader = gguf.GGUFReader(str(gguf_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- inspect header ---
    kv = {}
    for fname in reader.fields:
        try:
            kv[fname] = reader.fields[fname]
        except Exception:  # noqa
            pass

    lora_alpha = int(_kv_scalar(reader, "adapter.lora.alpha"))
    base_model = base_model_override or _kv_str(
        reader, "general.base_model.0.repo_url"
    ).rsplit("/huggingface.co/", 1)[-1].replace("https://huggingface.co/", "")
    print(f"base_model: {base_model}")
    print(f"adapter.lora.alpha: {lora_alpha}")

    # --- collect tensors into PEFT naming ---
    target_modules: set[str] = set()
    rank: int | None = None
    tensors: dict[str, torch.Tensor] = {}

    for t in reader.tensors:
        m = _GGUF_NAME_RE.match(t.name)
        if not m:
            print(f"[skip] {t.name} (not a lora tensor)")
            continue
        layer_idx, gguf_proj, ab = int(m.group(1)), m.group(2), m.group(3)
        if gguf_proj not in _PROJ_MAP:
            print(f"[skip] {t.name} (unknown proj {gguf_proj})")
            continue
        peft_suffix = _PROJ_MAP[gguf_proj]
        target_modules.add(peft_suffix.split(".")[-1])  # e.g. "q_proj"

        arr = np.array(t.data, dtype=np.float16).reshape(tuple(t.shape))
        # GGUF stores row-major (in_features, r) for lora_a / (r, out_features) for lora_b
        # PEFT lora_A shape = (r, in_features)  -> transpose of lora_a
        # PEFT lora_B shape = (out_features, r) -> transpose of lora_b
        torch_tensor = torch.from_numpy(arr).transpose(0, 1).contiguous()
        # identify r
        if ab == "a":
            if rank is None:
                rank = torch_tensor.shape[0]  # (r, in)
            peft_key = (
                f"base_model.model.model.layers.{layer_idx}.{peft_suffix}."
                f"lora_A.default.weight"
            )
        else:  # "b"
            peft_key = (
                f"base_model.model.model.layers.{layer_idx}.{peft_suffix}."
                f"lora_B.default.weight"
            )
        tensors[peft_key] = torch_tensor

    print(f"rank: {rank}")
    print(f"target_modules: {sorted(target_modules)}")
    print(f"tensor count: {len(tensors)}")

    # --- write adapter_model.safetensors ---
    safetensors_path = out_dir / "adapter_model.safetensors"
    # save_file requires contiguous float tensors
    save_file({k: v.to(torch.float16) for k, v in tensors.items()}, str(safetensors_path))
    print(f"wrote {safetensors_path} ({safetensors_path.stat().st_size/1e6:.1f} MB)")

    # --- write adapter_config.json ---
    cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "r": rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": sorted(target_modules),
        "fan_in_fan_out": False,
        "inference_mode": True,
        "revision": None,
        "modules_to_save": None,
        "init_lora_weights": True,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "rank_pattern": {},
        "alpha_pattern": {},
        "use_dora": False,
        "use_rslora": False,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "loftq_config": {},
        "exclude_modules": None,
        "eva_config": None,
        "corda_config": None,
        "trainable_token_indices": None,
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out_dir / 'adapter_config.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="Path to LoRA GGUF")
    ap.add_argument("--out", required=True, help="Output PEFT dir")
    ap.add_argument("--base-model", default=None, help="Override base model HF id")
    args = ap.parse_args()
    convert(Path(args.gguf), Path(args.out), args.base_model)


if __name__ == "__main__":
    main()
