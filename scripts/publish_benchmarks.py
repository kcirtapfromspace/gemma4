#!/usr/bin/env python3
"""Publish known benchmark results into scripts/benchmarks.duckdb.

Extends the existing Jetson-oriented schema with multi-backend/device
dimensions and ingests every hackathon experiment we have concrete
numbers for (Jetson llama.cpp, Jetson MLC-LLM, iOS llama.cpp, iOS
LiteRT-LM, macOS LiteRT-LM validation, upstream Google mobile
benchmarks, projections).

Idempotent: each experiment is keyed by `experiment_name` and replaced
on re-run. Safe to run repeatedly from CI or after new sprint work.
"""

import json
import statistics
import uuid
from datetime import datetime
from pathlib import Path

import duckdb

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
DB = SCRIPT_DIR / "benchmarks.duckdb"


def new_id() -> str:
    return str(uuid.uuid4())[:12]


def ts(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


EXPERIMENT_EXTRA_COLS: list[tuple[str, str]] = [
    ("backend", "VARCHAR"),
    ("device", "VARCHAR"),
    ("runtime", "VARCHAR"),
    ("model_variant", "VARCHAR"),
    ("model_format", "VARCHAR"),
    ("team_tag", "VARCHAR"),
    ("extraction_pass_rate", "DOUBLE"),
    ("data_source", "VARCHAR"),
]

RUN_EXTRA_COLS: list[tuple[str, str]] = [
    ("backend", "VARCHAR"),
    ("device", "VARCHAR"),
    ("runtime", "VARCHAR"),
    ("model_variant", "VARCHAR"),
    ("model_format", "VARCHAR"),
    ("team_tag", "VARCHAR"),
]


def extend_schema(conn: duckdb.DuckDBPyConnection) -> None:
    for col, typ in EXPERIMENT_EXTRA_COLS:
        try:
            conn.execute(f"ALTER TABLE experiments ADD COLUMN {col} {typ}")
        except duckdb.CatalogException:
            pass
    for col, typ in RUN_EXTRA_COLS:
        try:
            conn.execute(f"ALTER TABLE benchmark_runs ADD COLUMN {col} {typ}")
        except duckdb.CatalogException:
            pass

    conn.execute(
        """
        CREATE OR REPLACE VIEW experiment_timeline AS
        SELECT
            e.team_tag,
            e.experiment_name,
            e.created_at,
            e.backend,
            e.device,
            e.runtime,
            e.model_variant,
            e.model_format,
            e.data_source,
            e.avg_gen_tok_s,
            e.p50_gen_tok_s,
            e.avg_ttft_ms,
            e.p95_total_ms,
            e.avg_prompt_tok_s,
            e.success_rate,
            e.avg_extraction_score,
            e.extraction_pass_rate,
            e.total_runs,
            e.speedup_pct,
            e.notes
        FROM experiments e
        ORDER BY e.created_at
        """
    )


def upsert_experiment(conn: duckdb.DuckDBPyConnection, row: dict) -> None:
    conn.execute(
        "DELETE FROM experiments WHERE experiment_name = ?",
        [row["experiment_name"]],
    )
    cols = list(row.keys())
    placeholders = ",".join("?" for _ in cols)
    conn.execute(
        f"INSERT INTO experiments ({','.join(cols)}) VALUES ({placeholders})",
        [row[c] for c in cols],
    )


def upsert_runs(conn: duckdb.DuckDBPyConnection, experiment_id: str, runs: list[dict]) -> None:
    conn.execute(
        "DELETE FROM benchmark_runs WHERE experiment_id = ?",
        [experiment_id],
    )
    for r in runs:
        cols = list(r.keys())
        placeholders = ",".join("?" for _ in cols)
        conn.execute(
            f"INSERT INTO benchmark_runs ({','.join(cols)}) VALUES ({placeholders})",
            [r[c] for c in cols],
        )


def publish_c5_upstream(conn: duckdb.DuckDBPyConnection) -> None:
    # Google-published LiteRT-LM benchmarks for Gemma 4 E2B.
    # Source: apps/mobile/FEASIBILITY.md (citations: ai.google.dev/edge/litert-lm).
    upstream = [
        ("c5-upstream-iphone-17-pro-gpu",  "iphone-17-pro",     "gpu", 2878.0, 56.0,  300.0, 1450),
        ("c5-upstream-iphone-17-pro-cpu",  "iphone-17-pro",     "cpu",  532.0, 25.0, 1900.0,  607),
        ("c5-upstream-s26-ultra-gpu",      "samsung-s26-ultra", "gpu", 3808.0, 52.0,  300.0,  676),
        ("c5-upstream-s26-ultra-cpu",      "samsung-s26-ultra", "cpu",  557.0, 47.0, 1800.0, 1733),
        ("c5-upstream-macbook-m4-gpu",     "macbook-pro-m4",    "gpu", 7835.0, 160.0, 100.0, 1623),
        ("c5-upstream-rpi5-cpu",           "rpi5",              "cpu",  133.0,  8.0, 7800.0, 1546),
    ]
    for name, device, runtime, prefill_tok_s, decode_tok_s, ttft_ms, mem_mb in upstream:
        upsert_experiment(conn, {
            "experiment_id": new_id(),
            "experiment_name": name,
            "created_at": ts("2026-04-23T12:00:00"),
            "backend": "litert-lm",
            "device": device,
            "runtime": runtime,
            "model_variant": "gemma-4-E2B-base",
            "model_format": "litertlm-int4",
            "team_tag": "c5",
            "data_source": "upstream-bench",
            "avg_gen_tok_s": decode_tok_s,
            "p50_gen_tok_s": decode_tok_s,
            "avg_prompt_tok_s": prefill_tok_s,
            "avg_ttft_ms": ttft_ms,
            "notes": f"Upstream Google LiteRT-LM benchmark. Peak mem {mem_mb} MB.",
        })

    upsert_experiment(conn, {
        "experiment_id": new_id(),
        "experiment_name": "c5-projected-iphone-14",
        "created_at": ts("2026-04-23T12:00:00"),
        "backend": "litert-lm",
        "device": "iphone-14",
        "runtime": "gpu",
        "model_variant": "gemma-4-E2B-base",
        "model_format": "litertlm-int4",
        "team_tag": "c5",
        "data_source": "projected",
        "avg_gen_tok_s": 12.0,
        "notes": (
            "Projected from modelfit.io A15 + AI Edge Gallery reports. "
            "~46 s e2e on 700-prefill + 500-decode workload — inside the 60 s demo budget."
        ),
    })


def publish_c8_mac_baseline(conn: duckdb.DuckDBPyConnection) -> None:
    upsert_experiment(conn, {
        "experiment_id": new_id(),
        "experiment_name": "c8-mac-cpu-baseline",
        "created_at": ts("2026-04-22T12:00:00"),
        "backend": "llama-cpp-server",
        "device": "macbook-pro-m4",
        "runtime": "cpu",
        "model_variant": "cliniq-compact-lora-v1",
        "model_format": "gguf-q3km",
        "team_tag": "c8",
        "data_source": "measured",
        "avg_extraction_score": 13 / 18,
        "extraction_pass_rate": 13 / 18,
        "total_runs": 5,
        "notes": (
            "C8 Mac-CPU baseline — fine-tuned LoRA + Gemma 4 E2B on Mac CPU. "
            "13/18 on the 5 canonical bench_* cases. Referenced as apples-to-apples target by C12/C16."
        ),
    })


def publish_c12_llama_cpp_ios(conn: duckdb.DuckDBPyConnection) -> None:
    # Source: apps/mobile/ios-app/VALIDATION.md § "Per-case results (real inference) — C12 measured"
    per_case = [
        ("bench_minimal",        140, 37.5, 3.74, 0, 3),
        ("bench_typical_covid",  172, 42.9, 4.01, 3, 3),
        ("bench_complex_multi",  317, 59.1, 5.37, 4, 6),
        ("bench_meningitis",     165, 41.2, 4.00, 2, 3),
        ("bench_negative_lab",   182, 41.7, 4.37, 3, 3),
    ]
    experiment_id = new_id()
    runs: list[dict] = []
    for case_id, tokens, elapsed_s, tok_s, matched, expected in per_case:
        runs.append({
            "run_id": new_id(),
            "experiment_id": experiment_id,
            "experiment_name": "c12-llama-cpp-ios-sim",
            "case_id": case_id,
            "run_number": 1,
            "timestamp": ts("2026-04-23T20:00:00"),
            "endpoint": "ios-sim://iphone-17-pro",
            "total_ms": elapsed_s * 1000.0,
            "completion_tokens": tokens,
            "gen_tok_per_sec": tok_s,
            "valid_json": True,
            "extraction_score": matched / expected if expected else None,
            "backend": "llama-cpp-ios",
            "device": "iphone-17-pro-sim",
            "runtime": "cpu",
            "model_variant": "cliniq-compact-lora-v1",
            "model_format": "gguf-q3km",
            "team_tag": "c12",
        })

    tok_values = [r["gen_tok_per_sec"] for r in runs]
    score_values = [r["extraction_score"] for r in runs]
    upsert_experiment(conn, {
        "experiment_id": experiment_id,
        "experiment_name": "c12-llama-cpp-ios-sim",
        "created_at": ts("2026-04-23T20:00:00"),
        "backend": "llama-cpp-ios",
        "device": "iphone-17-pro-sim",
        "runtime": "cpu",
        "model_variant": "cliniq-compact-lora-v1",
        "model_format": "gguf-q3km",
        "team_tag": "c12",
        "data_source": "measured",
        "model_file": "Documents/cliniq-gemma4-e2b-Q3_K_M.gguf",
        "avg_gen_tok_s": statistics.mean(tok_values),
        "p50_gen_tok_s": statistics.median(tok_values),
        "total_runs": len(runs),
        "success_rate": 1.0,
        "avg_extraction_score": statistics.mean(score_values),
        "extraction_pass_rate": 12 / 18,
        "notes": (
            "C12 on-device llama.cpp on iPhone 17 Pro simulator (CPU, 4 threads). "
            "12/18 extraction (93% of C8 Mac baseline), median 4.0 tok/s. Projected physical "
            "iPhone 17 Pro Metal throughput: 10-20 tok/s per upstream benchmarks."
        ),
    })
    upsert_runs(conn, experiment_id, runs)


def publish_c14_litertlm_ios(conn: duckdb.DuckDBPyConnection) -> None:
    # Source: apps/mobile/litertlm-swift/STATUS.md § "C14 decode results"
    runs_input = [
        ("smoke_colors",       6, 433.0, 13.84),
        ("smoke_france_paris", 2, 128.6, 15.55),
    ]
    experiment_id = new_id()
    runs: list[dict] = []
    for case_id, tokens, elapsed_ms, tok_s in runs_input:
        runs.append({
            "run_id": new_id(),
            "experiment_id": experiment_id,
            "experiment_name": "c14-litertlm-ios-sim",
            "case_id": case_id,
            "run_number": 1,
            "timestamp": ts("2026-04-23T22:00:00"),
            "endpoint": "ios-sim://iphone-17-pro",
            "total_ms": elapsed_ms,
            "completion_tokens": tokens,
            "gen_tok_per_sec": tok_s,
            "valid_json": False,
            "backend": "litert-lm",
            "device": "iphone-17-pro-sim",
            "runtime": "cpu",
            "model_variant": "gemma-4-E2B-base",
            "model_format": "litertlm-int4",
            "team_tag": "c14",
        })

    tok_values = [r["gen_tok_per_sec"] for r in runs]
    upsert_experiment(conn, {
        "experiment_id": experiment_id,
        "experiment_name": "c14-litertlm-ios-sim",
        "created_at": ts("2026-04-23T22:00:00"),
        "backend": "litert-lm",
        "device": "iphone-17-pro-sim",
        "runtime": "cpu",
        "model_variant": "gemma-4-E2B-base",
        "model_format": "litertlm-int4",
        "team_tag": "c14",
        "data_source": "measured",
        "avg_gen_tok_s": statistics.mean(tok_values),
        "p50_gen_tok_s": statistics.median(tok_values),
        "total_runs": len(runs),
        "success_rate": 1.0,
        "notes": (
            "C14 LiteRT-LM Swift decode — iPhone 17 Pro simulator CPU. 13.84-15.55 tok/s on "
            "smoke prompts (XCTest + CLI). Metal/GPU path is iPhone-hardware-only; expect ~56 "
            "tok/s per C5 upstream benchmark."
        ),
    })
    upsert_runs(conn, experiment_id, runs)


def publish_c16_litertlm_macos(conn: duckdb.DuckDBPyConnection) -> None:
    # Source: apps/mobile/convert/build/validation/SUMMARY.json
    summary_path = REPO_ROOT / "apps/mobile/convert/build/validation/SUMMARY.json"
    summary = json.loads(summary_path.read_text())

    experiment_id = new_id()
    runs: list[dict] = []
    for case in summary["cases"]:
        runs.append({
            "run_id": new_id(),
            "experiment_id": experiment_id,
            "experiment_name": "c16-litertlm-macos-cpu",
            "case_id": case["case_id"],
            "run_number": 1,
            "timestamp": ts("2026-04-24T14:00:00"),
            "endpoint": "local://macos-cpu",
            "total_ms": case["gen_s"] * 1000.0,
            "completion_tokens": case["token_len"],
            "gen_tok_per_sec": case["tok_s"],
            "valid_json": case["total_matched"] > 0,
            "extraction_score": case["extraction_score"],
            "raw_response": case["output"][:2000],
            "conditions_found": json.dumps(case["matched_conditions"]),
            "loincs_found": json.dumps(case["matched_loincs"]),
            "rxnorms_found": json.dumps(case["matched_rxnorms"]),
            "backend": "litert-lm",
            "device": "macbook-pro-m4",
            "runtime": "cpu",
            "model_variant": "c16-retrain",
            "model_format": "litertlm-int4",
            "team_tag": "c16",
        })

    tok_values = [r["gen_tok_per_sec"] for r in runs]
    score_values = [r["extraction_score"] for r in runs]
    upsert_experiment(conn, {
        "experiment_id": experiment_id,
        "experiment_name": "c16-litertlm-macos-cpu",
        "created_at": ts("2026-04-24T14:00:00"),
        "backend": "litert-lm",
        "device": "macbook-pro-m4",
        "runtime": "cpu",
        "model_variant": "c16-retrain",
        "model_format": "litertlm-int4",
        "team_tag": "c16",
        "data_source": "measured",
        "model_file": summary["model"],
        "avg_gen_tok_s": statistics.mean(tok_values),
        "p50_gen_tok_s": statistics.median(tok_values),
        "total_runs": len(runs),
        "success_rate": sum(1 for r in runs if r["valid_json"]) / len(runs),
        "avg_extraction_score": statistics.mean(score_values),
        "extraction_pass_rate": summary["totals"]["aggregate_extraction_score"],
        "notes": (
            "C16 retry — freshly-quantized cliniq-gemma4-e2b.litertlm on macOS CPU. "
            "18/29 = 0.621 across 9 cases. 5/9 perfect. Two cases hit known int4 degeneration "
            "(negative_lab, complex_multi) pending KV-sharing-aware Unsloth retrain."
        ),
    })
    upsert_runs(conn, experiment_id, runs)


def publish_jetson_milestones(conn: duckdb.DuckDBPyConnection) -> None:
    upsert_experiment(conn, {
        "experiment_id": new_id(),
        "experiment_name": "c1-jetson-orin-nx-7w",
        "created_at": ts("2026-04-23T10:00:00"),
        "backend": "llama-cpp-server",
        "device": "jetson-orin-nx",
        "runtime": "gpu",
        "model_variant": "cliniq-compact-lora-v1",
        "model_format": "gguf-q3km",
        "team_tag": "c1",
        "data_source": "measured",
        "avg_gen_tok_s": 0.9,
        "notes": (
            "Jetson Orin NX 8GB on Talos @ 7W (EMC 2133 MHz). 15W mode blocked on Talos "
            "without image rebuild per team/c1 POWER_MODE.md. 0.9 tok/s — ~60x below mobile GPU."
        ),
    })

    upsert_experiment(conn, {
        "experiment_id": new_id(),
        "experiment_name": "mlc-llm-jetson-orin-nx",
        "created_at": ts("2026-04-19T12:00:00"),
        "backend": "mlc-llm",
        "device": "jetson-orin-nx",
        "runtime": "gpu",
        "model_variant": "gemma-4-E2B-base",
        "model_format": "mlc-q4f16",
        "team_tag": "mlc",
        "data_source": "measured",
        "avg_gen_tok_s": 6.5,
        "notes": (
            "MLC-LLM port (commit 8da77e9) on Jetson Orin NX: 5-8 tok/s — 6x over llama.cpp. "
            "Not production-ready (no LoRA, sequence degeneration)."
        ),
    })


def print_summary(conn: duckdb.DuckDBPyConnection) -> None:
    print("\n=== experiments_by_backend_device ===")
    rows = conn.execute(
        """
        SELECT backend, device, runtime, data_source, COUNT(*) AS n,
               ROUND(AVG(avg_gen_tok_s), 2) AS avg_tok_s,
               ROUND(AVG(extraction_pass_rate), 3) AS avg_pass
        FROM experiments
        WHERE backend IS NOT NULL
        GROUP BY backend, device, runtime, data_source
        ORDER BY avg_tok_s DESC NULLS LAST
        """
    ).fetchall()
    header = f"  {'backend':<18} {'device':<22} {'runtime':<7} {'source':<14} {'n':>3} {'tok/s':>8} {'pass':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for backend, device, runtime, source, n, tok, pass_ in rows:
        tok_s = f"{tok:.1f}" if tok is not None else "—"
        pass_s = f"{pass_:.3f}" if pass_ is not None else "—"
        print(f"  {backend:<18} {device:<22} {runtime:<7} {source:<14} {n:>3} {tok_s:>8} {pass_s:>6}")

    n_exp = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    n_runs = conn.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0]
    n_tagged = conn.execute(
        "SELECT COUNT(*) FROM experiments WHERE team_tag IS NOT NULL"
    ).fetchone()[0]
    print(f"\nTotal experiments:      {n_exp} ({n_tagged} with team_tag)")
    print(f"Total per-case runs:    {n_runs}")


def main() -> None:
    conn = duckdb.connect(str(DB))
    try:
        extend_schema(conn)
        publish_c5_upstream(conn)
        publish_c8_mac_baseline(conn)
        publish_c12_llama_cpp_ios(conn)
        publish_c14_litertlm_ios(conn)
        publish_c16_litertlm_macos(conn)
        publish_jetson_milestones(conn)
        conn.commit()
        print_summary(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
