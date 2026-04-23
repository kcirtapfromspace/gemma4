#!/usr/bin/env python3
"""Inspect the compact GGUF LoRA adapter to see if we can reconstruct
a PEFT-compatible safetensors adapter from it.

Usage: python inspect_lora_gguf.py /Users/thinkstudio/gemma4/models/cliniq-compact-lora.gguf
"""

import sys
from pathlib import Path
import gguf


def main(path: str) -> None:
    reader = gguf.GGUFReader(path)
    print(f"=== {path} ===")
    print(f"file size (MB): {Path(path).stat().st_size / 1e6:.1f}")
    print()
    print("KV fields:")
    for f in reader.fields.values():
        try:
            val = f.parts[f.data[0]]
            # handle scalar / array
            if hasattr(val, "tobytes"):
                try:
                    s = val.tobytes().decode("utf-8", errors="replace")
                    if len(s) < 120 and all(c.isprintable() or c.isspace() for c in s):
                        display = repr(s)
                    else:
                        display = f"<bytes len={len(val)}>"
                except Exception:
                    display = f"<array dtype={val.dtype} shape={val.shape}>"
            else:
                display = repr(val)
        except Exception as e:  # noqa
            display = f"<unreadable: {e}>"
        print(f"  {f.name}: {display}")
    print()
    print(f"Tensor count: {len(reader.tensors)}")
    # sample a few tensor names + shapes
    for i, t in enumerate(reader.tensors[:30]):
        print(f"  [{i:3d}] {t.name:70s} shape={tuple(t.shape)} dtype={t.tensor_type.name}")
    if len(reader.tensors) > 30:
        print(f"  ... ({len(reader.tensors)-30} more)")
        # final few
        for i, t in enumerate(reader.tensors[-5:]):
            idx = len(reader.tensors) - 5 + i
            print(f"  [{idx:3d}] {t.name:70s} shape={tuple(t.shape)} dtype={t.tensor_type.name}")
    # unique name patterns
    import re
    patterns = set()
    for t in reader.tensors:
        p = re.sub(r"\.(\d+)\.", ".<N>.", t.name)
        patterns.add(p)
    print(f"\nUnique tensor-name patterns ({len(patterns)}):")
    for p in sorted(patterns):
        print(f"  {p}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/Users/thinkstudio/gemma4/models/cliniq-compact-lora.gguf")
