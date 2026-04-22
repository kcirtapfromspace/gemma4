#!/usr/bin/env python3
"""Verify that all patch targets exist in the container's source code."""
import sys

def check(path, needle, label):
    with open(path) as f:
        content = f.read()
    if needle not in content:
        print(f"  FAIL: {label}")
        # Show a snippet around the expected area
        key = needle[:60]
        for i, line in enumerate(content.split('\n')):
            if key[:30] in line:
                print(f"    Near line {i}: {line[:100]}")
        return False
    print(f"  OK: {label}")
    return True

print("=== C++ patches (TVM) ===")
ok = True

# attn_utils.h
ok &= check(
    "/opt/mlc-llm/3rdparty/tvm/src/runtime/relax_vm/attn_utils.h",
    "enum class AttnKind : int {\n  kMHA = 0,\n  kMLA = 1,\n  kLinearAttn = 2,\n};",
    "attn_utils.h: AttnKind enum"
)
ok &= check(
    "/opt/mlc-llm/3rdparty/tvm/src/runtime/relax_vm/attn_utils.h",
    "if (attn_kind == AttnKind::kMHA) {\n    // Ignore v_head_dim",
    "attn_utils.h: GetKVCacheShape"
)

# paged_kv_cache.cc
ok &= check(
    "/opt/mlc-llm/3rdparty/tvm/src/runtime/relax_vm/paged_kv_cache.cc",
    "if (append_before_attn_) {\n      // Right now we use different kernels when depth is 1 or not 1.\n      // For the case where maximum depth is 1, we create the auxiliary\n      // data structure with regard to the page table after appending.\n      for (int i = 0; i < cur_batch_size_; ++i) {\n        ReserveAppendLengthInSeq(sequences[i], append_lengths[i]);\n      }\n    }",
    "paged_kv_cache.cc: first reserve block"
)
ok &= check(
    "/opt/mlc-llm/3rdparty/tvm/src/runtime/relax_vm/paged_kv_cache.cc",
    "if (!append_before_attn_) {\n      // Right now we use different kernels when depth is 1 or not 1.\n      // For the case where maximum depth is not 1, we create the auxiliary\n      // data structure with regard to the page table before appending.\n      for (int i = 0; i < cur_batch_size_; ++i) {\n        ReserveAppendLengthInSeq(sequences[i], append_lengths[i]);\n      }\n    }",
    "paged_kv_cache.cc: second reserve block"
)
ok &= check(
    "/opt/mlc-llm/3rdparty/tvm/src/runtime/relax_vm/paged_kv_cache.cc",
    "if (attn_kind == AttnKind::kMHA) {\n      MHASelfAttnInternal(q_data, k_data, v_data, o_data, lse_data, sm_scale);\n    } else {\n      MLASelfAttnInternal(q_data, k_data, v_data, o_data, lse_data, sm_scale);\n    }",
    "paged_kv_cache.cc: self-attn dispatch"
)
ok &= check(
    "/opt/mlc-llm/3rdparty/tvm/src/runtime/relax_vm/paged_kv_cache.cc",
    "if (attn_kind == AttnKind::kMHA) {\n      MHACrossAttnInternal(local_layer_id, q_data, o_data, lse_data, sm_scale,\n                           /*is_first_kernel=*/true);\n    } else {\n      MLACrossAttnInternal(local_layer_id, q_data, o_data, lse_data, sm_scale);\n    }",
    "paged_kv_cache.cc: cross-attn dispatch"
)

print("\n=== Python patches (TVM) ===")

# kv_cache.py
ok &= check(
    "/opt/mlc-llm/3rdparty/tvm/python/tvm/relax/frontend/nn/llm/kv_cache.py",
    "MHA = 0\n    MLA = 1",
    "kv_cache.py: AttnKind enum"
)

with open("/opt/mlc-llm/3rdparty/tvm/python/tvm/relax/frontend/nn/llm/kv_cache.py") as f:
    content = f.read()
count = content.count("[int(getattr(AttnKind, attn_kind.upper())) for _ in range(num_hidden_layers)]")
if count == 2:
    print(f"  OK: kv_cache.py: {count} uniform attn_kinds patterns")
else:
    print(f"  FAIL: kv_cache.py: expected 2 patterns, found {count}")
    ok = False

# position_embedding.py
ok &= check(
    "/opt/mlc-llm/3rdparty/tvm/python/tvm/relax/frontend/nn/llm/position_embedding.py",
    'def rope_freq_gptj(s: tir.Var, d: tir.Var, d_range: int, theta: float, dtype: str):\n    """Compute the inverse frequency of RoPE for gptj RoPE scaling."""\n    freq = s / tir.power(theta, 2 * (d // 2) % d_range / tir.const(d_range, "float32"))',
    "position_embedding.py: rope_freq_gptj"
)

print("\n=== MLC-LLM patches ===")

ok &= check(
    "/opt/mlc-llm/python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py",
    '    assert args[0].value in ["mha", "mla"]',
    "dispatch_kv_cache_creation.py: assertion"
)
ok &= check(
    "/opt/mlc-llm/python/mlc_llm/compiler_pass/dispatch_kv_cache_creation.py",
    '        "attn_kind": args[0].value,',
    "dispatch_kv_cache_creation.py: extraction"
)
ok &= check(
    "/opt/mlc-llm/python/mlc_llm/nn/kv_cache.py",
    'attn_kind: Literal["mha", "mla"]',
    "mlc_llm/nn/kv_cache.py: type hint"
)
ok &= check(
    "/opt/mlc-llm/python/mlc_llm/nn/kv_cache.py",
    "rx.StringImm(attn_kind),",
    "mlc_llm/nn/kv_cache.py: StringImm"
)

print()
if ok:
    print("=== ALL CHECKS PASSED ===")
else:
    print("=== SOME CHECKS FAILED ===")
    sys.exit(1)
