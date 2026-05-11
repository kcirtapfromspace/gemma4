#!/usr/bin/env python3
"""Bench `spaces/zerogpu_engine_mtp.chat_completion` with MTP on vs off.

Mirrors `tools/autoresearch/mtp_bench.py` but goes through the engine
wrapper (the same code path the Spaces UI hits) instead of bare
`model.generate(...)`. Goal: confirm the 1.67–1.92× MTP speedup measured
in mtp_bench.py survives the wrapper overhead.

Strategy: process is one-shot per scenario because `zerogpu_engine_mtp` is
a module-level singleton — flipping `MTP_ENABLED` requires re-import. So we
shell out per scenario and merge JSON results.

Usage:
  # MTP enabled (default)
  MTP_ENABLED=true \
    tools/autoresearch/spaces-mtp-venv/bin/python \
    tools/autoresearch/spaces_mtp_bench.py --scenario mtp \
    --out tools/autoresearch/spaces-mtp-bench.mtp.json

  # MTP disabled
  MTP_ENABLED=false \
    tools/autoresearch/spaces-mtp-venv/bin/python \
    tools/autoresearch/spaces_mtp_bench.py --scenario nomtp \
    --out tools/autoresearch/spaces-mtp-bench.nomtp.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/Users/thinkstudio/gemma4")
TEST_CASES = REPO_ROOT / "scripts" / "test_cases.jsonl"

SYS_PROMPT = (
    "You are an eICR-extraction agent. Given a clinical encounter summary, "
    "list SNOMED conditions, LOINC labs, and RxNorm meds you find. Be concise."
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True, choices=["mtp", "nomtp"])
    p.add_argument("--n-prompts", type=int, default=9)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # Make spaces/ importable.
    sys.path.insert(0, str(REPO_ROOT / "spaces"))

    # Sanity: env must agree with the scenario flag (the engine reads
    # MTP_ENABLED at import time).
    env_flag = os.environ.get("MTP_ENABLED", "true").lower()
    enabled = env_flag not in ("0", "false", "no", "off")
    expect_enabled = args.scenario == "mtp"
    if enabled != expect_enabled:
        print(
            f"WARN scenario={args.scenario} but MTP_ENABLED={env_flag!r} "
            f"resolves to enabled={enabled}; respecting the env",
            flush=True,
        )

    # Force CPU + spaces shim for off-GPU smoke. The engine module already
    # does its own device detection, but we want to fail loudly if torch
    # decides to grab MPS (which it shouldn't, the engine only uses cuda).
    print(f"[bench] importing zerogpu_engine_mtp (scenario={args.scenario})", flush=True)
    t_import_0 = time.time()
    import zerogpu_engine_mtp as eng  # type: ignore[import-not-found]
    print(
        f"[bench] import done in {time.time() - t_import_0:.1f}s; "
        f"banner: {eng.model_banner()}",
        flush=True,
    )

    # Build prompts from the same 9 cases mtp_bench used.
    cases = []
    with open(TEST_CASES) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    cases = cases[: args.n_prompts]
    prompts = [
        (
            c["case_id"],
            [{"role": "user", "content": SYS_PROMPT + "\n\n" + c["user"]}],
        )
        for c in cases
    ]

    # Warmup with a tiny budget — first call compiles graphs, downloads any
    # missing weights, etc. Don't include in the timing summary.
    print(f"[bench] warmup ({prompts[0][0]}, max_tokens=8)", flush=True)
    _ = eng.chat_completion(messages=prompts[0][1], max_tokens=8)

    results = []
    for cid, msgs in prompts:
        t0 = time.time()
        try:
            resp = eng.chat_completion(
                messages=msgs,
                max_tokens=args.max_new_tokens,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            wall = time.time() - t0
            results.append({
                "case_id": cid,
                "error": f"{type(exc).__name__}: {exc}",
                "wall_secs": wall,
            })
            print(f"  [{cid}] ERROR: {exc}", flush=True)
            continue
        wall = time.time() - t0
        timings = resp.get("timings", {})
        usage = resp.get("usage", {})
        rec = {
            "case_id": cid,
            "wall_secs": wall,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "engine_tok_per_s": timings.get("predicted_per_second"),
            "engine_predicted_n": timings.get("predicted_n"),
            "engine_predicted_ms": timings.get("predicted_ms"),
            "mtp": timings.get("mtp"),
            "wall_tok_per_s": (
                (usage.get("completion_tokens") or 0) / wall if wall > 0 else 0.0
            ),
            "finish_reason": (resp.get("choices") or [{}])[0].get("finish_reason"),
        }
        results.append(rec)
        print(
            f"  [{cid}] {rec['completion_tokens']} tok in {wall:.2f}s "
            f"= {rec['wall_tok_per_s']:.2f} tok/s "
            f"(engine reports {rec['engine_tok_per_s']:.2f})",
            flush=True,
        )

    summary = {
        "scenario": args.scenario,
        "mtp_env": env_flag,
        "banner": eng.model_banner(),
        "n_prompts": len(results),
        "max_new_tokens": args.max_new_tokens,
        "totals": {
            "tokens": sum((r.get("completion_tokens") or 0) for r in results),
            "wall_secs": sum(r.get("wall_secs", 0.0) for r in results),
        },
        "results": results,
    }
    summary["totals"]["tok_per_s"] = (
        summary["totals"]["tokens"] / summary["totals"]["wall_secs"]
        if summary["totals"]["wall_secs"] > 0
        else 0.0
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[bench] wrote {out_path}", flush=True)
    print(
        f"[bench] SUMMARY scenario={args.scenario} "
        f"total {summary['totals']['tokens']} tok in "
        f"{summary['totals']['wall_secs']:.2f}s "
        f"= {summary['totals']['tok_per_s']:.2f} tok/s avg",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
