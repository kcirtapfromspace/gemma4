#!/usr/bin/env python3
"""Benchmark harness for ClinIQ llama-server inference.

Sends standardized test cases to the inference endpoint, measures detailed
performance metrics and extraction quality, stores results in DuckDB.

Usage:
    python benchmark.py --experiment-name "baseline-cpu" --endpoint http://localhost:8080
    python benchmark.py --experiment-name "cuda-on" --runs 10 --config-json '{"cuda": true}'
    python benchmark.py --experiment-name "prompt-compact" --system-prompt "Extract entities..."
    python benchmark.py --experiment-name "maxtok-384" --max-tokens 384
"""

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import requests

SCRIPT_DIR = Path(__file__).parent
DEFAULT_DB = SCRIPT_DIR / "benchmarks.duckdb"
DEFAULT_CASES = SCRIPT_DIR / "test_cases.jsonl"

DEFAULT_SYSTEM_PROMPT = (
    "Extract clinical entities from this eICR summary. Output JSON with: "
    "patient demographics, conditions (SNOMED/ICD-10), labs (LOINC), "
    "medications (RxNorm), vitals, and a case summary. "
    "Include confidence scores. Output valid JSON only."
)


def init_db(db_path: str) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id   VARCHAR PRIMARY KEY,
            experiment_name VARCHAR NOT NULL,
            created_at      TIMESTAMP NOT NULL DEFAULT current_timestamp,
            -- Config snapshot
            model_file      VARCHAR,
            quantization    VARCHAR,
            cuda_enabled    BOOLEAN,
            n_gpu_layers    INTEGER,
            ctx_size        INTEGER,
            threads         INTEGER,
            batch_size      INTEGER,
            ubatch_size     INTEGER,
            flash_attn      BOOLEAN,
            mlock           BOOLEAN,
            cache_type_k    VARCHAR,
            cache_type_v    VARCHAR,
            extra_args      TEXT,
            -- Aggregate results
            avg_gen_tok_s   DOUBLE,
            p50_gen_tok_s   DOUBLE,
            p95_total_ms    DOUBLE,
            avg_ttft_ms     DOUBLE,
            avg_prompt_tok_s DOUBLE,
            success_rate    DOUBLE,
            total_runs      INTEGER,
            notes           TEXT,
            -- Baseline comparison
            baseline_id     VARCHAR,
            speedup_pct     DOUBLE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            run_id              VARCHAR PRIMARY KEY,
            experiment_id       VARCHAR NOT NULL,
            experiment_name     VARCHAR NOT NULL,
            case_id             VARCHAR NOT NULL,
            run_number          INTEGER NOT NULL,
            timestamp           TIMESTAMP NOT NULL,
            endpoint            VARCHAR,
            -- Metrics
            total_ms            DOUBLE,
            ttft_ms             DOUBLE,
            prompt_tokens       INTEGER,
            completion_tokens   INTEGER,
            gen_tok_per_sec     DOUBLE,
            prompt_tok_per_sec  DOUBLE,
            valid_json          BOOLEAN,
            conditions_found    TEXT,
            -- Raw
            raw_response        TEXT
        )
    """)
    # Add new columns for quality scoring (idempotent)
    for col, typ in [
        ("extraction_score", "DOUBLE"),
        ("loincs_found", "TEXT"),
        ("rxnorms_found", "TEXT"),
        ("schema_keys_found", "TEXT"),
    ]:
        try:
            conn.execute(f"ALTER TABLE benchmark_runs ADD COLUMN {col} {typ}")
        except duckdb.CatalogException:
            pass  # Column already exists
    # Add avg extraction score to experiments
    try:
        conn.execute("ALTER TABLE experiments ADD COLUMN avg_extraction_score DOUBLE")
    except duckdb.CatalogException:
        pass
    try:
        conn.execute("ALTER TABLE experiments ADD COLUMN system_prompt TEXT")
    except duckdb.CatalogException:
        pass
    try:
        conn.execute("ALTER TABLE experiments ADD COLUMN max_tokens INTEGER")
    except duckdb.CatalogException:
        pass
    conn.execute("""
        CREATE OR REPLACE VIEW experiment_timeline AS
        SELECT
            e.experiment_name,
            e.created_at,
            e.quantization,
            e.cuda_enabled,
            e.flash_attn,
            e.ctx_size,
            e.avg_gen_tok_s,
            e.p50_gen_tok_s,
            e.p95_total_ms,
            e.success_rate,
            e.avg_extraction_score,
            e.speedup_pct,
            e.notes
        FROM experiments e
        ORDER BY e.created_at
    """)
    return conn


def load_test_cases(path: str) -> list[dict]:
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_inference_streaming(endpoint: str, user_prompt: str, system_prompt: str, max_tokens: int) -> dict:
    """Call llama-server with streaming to measure TTFT, then collect full response."""
    payload = {
        "model": "gemma4-eicr-fhir",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": True,
    }

    t_start = time.perf_counter()
    ttft = None
    chunks = []

    resp = requests.post(
        f"{endpoint}/v1/chat/completions",
        json=payload,
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if ttft is None:
            ttft = (time.perf_counter() - t_start) * 1000
        chunks.append(chunk)

    t_total = (time.perf_counter() - t_start) * 1000

    content_parts = []
    for c in chunks:
        delta = c.get("choices", [{}])[0].get("delta", {})
        text = delta.get("content")
        if text:
            content_parts.append(text)
    content = "".join(content_parts)

    last = chunks[-1] if chunks else {}
    usage = last.get("usage", {})
    timings = last.get("timings", {})

    return {
        "content": content,
        "total_ms": t_total,
        "ttft_ms": ttft or t_total,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "gen_tok_per_sec": timings.get("predicted_per_second"),
        "prompt_tok_per_sec": timings.get("prompt_per_second"),
    }


def run_inference_sync(endpoint: str, user_prompt: str, system_prompt: str, max_tokens: int) -> dict:
    """Non-streaming call."""
    payload = {
        "model": "gemma4-eicr-fhir",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
        "stream": False,
    }

    t_start = time.perf_counter()
    resp = requests.post(
        f"{endpoint}/v1/chat/completions",
        json=payload,
        timeout=600,
    )
    t_total = (time.perf_counter() - t_start) * 1000
    resp.raise_for_status()
    body = resp.json()

    content = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    timings = body.get("timings", {})

    return {
        "content": content,
        "total_ms": t_total,
        "ttft_ms": None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "gen_tok_per_sec": timings.get("predicted_per_second"),
        "prompt_tok_per_sec": timings.get("prompt_per_second"),
    }


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code block wrappers if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def _extract_codes(parsed: dict, code_key: str, section_key: str, alt_keys: list[str] | None = None) -> list[str]:
    """Extract ontology codes from a parsed JSON extraction result."""
    found = []
    for item in parsed.get(section_key, []) or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get(code_key, ""))
        if not code and alt_keys:
            for alt in alt_keys:
                val = str(item.get(alt, ""))
                if val:
                    # Handle "SNOMED 76272004" or "LOINC 20507-0" format
                    parts = val.split()
                    code = parts[-1] if parts else ""
                    break
        if code:
            found.append(code)
    return found


def validate_output(content: str, case: dict) -> dict:
    """Validate output quality: JSON validity, schema, SNOMED/LOINC/RxNorm accuracy.

    Returns dict with: valid_json, extraction_score, conditions_found, loincs_found,
    rxnorms_found, schema_keys_found.
    """
    text = _strip_markdown_fences(content)

    result = {
        "valid_json": False,
        "extraction_score": 0.0,
        "conditions_found": [],
        "loincs_found": [],
        "rxnorms_found": [],
        "schema_keys_found": [],
    }

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return result

    if not isinstance(parsed, dict):
        return result

    result["valid_json"] = True

    # Schema completeness: check for expected top-level keys
    # Handle both fine-tuned format (patient, conditions, labs, meds, vitals)
    # and base model format (patient_demographics, medications, etc.)
    expected_keys = {"patient", "conditions", "labs", "meds", "vitals"}
    alt_key_map = {
        "patient": ["patient_demographics", "demographics"],
        "conditions": ["diagnoses", "diagnosis"],
        "labs": ["laboratory", "lab_results", "results"],
        "meds": ["medications", "medication"],
        "vitals": ["vital_signs"],
    }
    keys_found = []
    for key in expected_keys:
        if key in parsed:
            keys_found.append(key)
        else:
            for alt in alt_key_map.get(key, []):
                if alt in parsed:
                    keys_found.append(key)
                    break
    result["schema_keys_found"] = keys_found
    schema_score = len(keys_found) / len(expected_keys) if expected_keys else 1.0

    # Condition codes (SNOMED)
    conditions_found = _extract_codes(parsed, "snomed", "conditions", ["code"])
    result["conditions_found"] = conditions_found
    expected_conditions = case.get("expected_conditions", [])
    cond_score = (
        len(set(conditions_found) & set(expected_conditions)) / len(expected_conditions)
        if expected_conditions else 1.0
    )

    # Lab codes (LOINC)
    loincs_found = _extract_codes(parsed, "loinc", "labs", ["code", "test_code"])
    result["loincs_found"] = loincs_found
    expected_loincs = case.get("expected_loincs", [])
    loinc_score = (
        len(set(loincs_found) & set(expected_loincs)) / len(expected_loincs)
        if expected_loincs else 1.0
    )

    # Medication codes (RxNorm)
    rxnorms_found = _extract_codes(parsed, "rxnorm", "meds", ["code", "medication_code"])
    # Also check "medications" key for base model format
    if not rxnorms_found:
        rxnorms_found = _extract_codes(parsed, "rxnorm", "medications", ["code"])
    result["rxnorms_found"] = rxnorms_found
    expected_rxnorms = case.get("expected_rxnorms", [])
    rxnorm_score = (
        len(set(rxnorms_found) & set(expected_rxnorms)) / len(expected_rxnorms)
        if expected_rxnorms else 1.0
    )

    # Composite score: JSON 20%, Schema 20%, Conditions 30%, Labs 15%, Meds 15%
    result["extraction_score"] = (
        0.20 * 1.0  # JSON valid
        + 0.20 * schema_score
        + 0.30 * cond_score
        + 0.15 * loinc_score
        + 0.15 * rxnorm_score
    )

    return result


def run_benchmark(
    endpoint: str,
    cases: list[dict],
    runs: int,
    warmup: int,
    experiment_id: str,
    experiment_name: str,
    conn: duckdb.DuckDBPyConnection,
    system_prompt: str,
    max_tokens: int,
    use_streaming: bool = True,
) -> list[dict]:
    """Run all test cases and store results."""
    all_results = []

    def infer(user_prompt):
        if use_streaming:
            return run_inference_streaming(endpoint, user_prompt, system_prompt, max_tokens)
        return run_inference_sync(endpoint, user_prompt, system_prompt, max_tokens)

    for case in cases:
        case_id = case["case_id"]
        user_prompt = case["user"]
        desc = case.get("description", case_id)

        print(f"\n  [{case_id}] {desc}")

        # Warmup runs (discarded)
        for w in range(warmup):
            print(f"    warmup {w + 1}/{warmup}...", end="", flush=True)
            try:
                infer(user_prompt)
                print(" ok")
            except Exception as e:
                print(f" error: {e}")

        # Actual runs
        for r in range(runs):
            run_id = str(uuid.uuid4())[:12]
            print(f"    run {r + 1}/{runs}...", end="", flush=True)

            try:
                result = infer(user_prompt)
                qual = validate_output(result["content"], case)

                conn.execute(
                    """INSERT INTO benchmark_runs (
                        run_id, experiment_id, experiment_name, case_id, run_number,
                        timestamp, endpoint, total_ms, ttft_ms, prompt_tokens,
                        completion_tokens, gen_tok_per_sec, prompt_tok_per_sec,
                        valid_json, conditions_found, raw_response,
                        extraction_score, loincs_found, rxnorms_found, schema_keys_found
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        run_id, experiment_id, experiment_name, case_id, r + 1,
                        datetime.now(timezone.utc).isoformat(), endpoint,
                        result["total_ms"], result["ttft_ms"],
                        result["prompt_tokens"], result["completion_tokens"],
                        result["gen_tok_per_sec"], result["prompt_tok_per_sec"],
                        qual["valid_json"], json.dumps(qual["conditions_found"]),
                        result["content"][:2000],
                        qual["extraction_score"],
                        json.dumps(qual["loincs_found"]),
                        json.dumps(qual["rxnorms_found"]),
                        json.dumps(qual["schema_keys_found"]),
                    ],
                )
                all_results.append({**result, **qual, "case_id": case_id, "run_number": r + 1})

                tok_s = result["gen_tok_per_sec"]
                tok_str = f"{tok_s:.1f} tok/s" if tok_s else "N/A"
                score_str = f"{qual['extraction_score']:.2f}"
                print(f" {result['total_ms']:.0f}ms | {tok_str} | "
                      f"json={'ok' if qual['valid_json'] else 'FAIL'} | score={score_str}")

            except Exception as e:
                print(f" ERROR: {e}")
                all_results.append({
                    "case_id": case_id, "run_number": r + 1,
                    "total_ms": None, "gen_tok_per_sec": None, "valid_json": False,
                })

    return all_results


def compute_aggregates(
    conn: duckdb.DuckDBPyConnection,
    experiment_id: str,
    baseline_id: str | None,
) -> dict:
    """Compute aggregate metrics for an experiment."""
    stats = conn.execute("""
        SELECT
            AVG(gen_tok_per_sec)   AS avg_gen,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gen_tok_per_sec) AS p50_gen,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_ms)       AS p95_ms,
            AVG(ttft_ms)          AS avg_ttft,
            AVG(prompt_tok_per_sec) AS avg_prompt,
            AVG(CASE WHEN valid_json THEN 1.0 ELSE 0.0 END) AS success,
            COUNT(*)              AS total,
            AVG(extraction_score) AS avg_score
        FROM benchmark_runs
        WHERE experiment_id = ?
          AND total_ms IS NOT NULL
    """, [experiment_id]).fetchone()

    agg = {
        "avg_gen_tok_s": stats[0],
        "p50_gen_tok_s": stats[1],
        "p95_total_ms": stats[2],
        "avg_ttft_ms": stats[3],
        "avg_prompt_tok_s": stats[4],
        "success_rate": stats[5],
        "total_runs": stats[6],
        "avg_extraction_score": stats[7],
    }

    speedup = None
    if baseline_id:
        baseline = conn.execute(
            "SELECT avg_gen_tok_s FROM experiments WHERE experiment_id = ?",
            [baseline_id],
        ).fetchone()
        if baseline and baseline[0] and agg["avg_gen_tok_s"]:
            speedup = ((agg["avg_gen_tok_s"] - baseline[0]) / baseline[0]) * 100

    agg["speedup_pct"] = speedup
    agg["baseline_id"] = baseline_id
    return agg


def print_summary(conn: duckdb.DuckDBPyConnection, experiment_id: str, experiment_name: str):
    """Print per-case summary for this experiment."""
    rows = conn.execute("""
        SELECT
            case_id,
            COUNT(*) AS runs,
            AVG(total_ms) AS avg_ms,
            AVG(gen_tok_per_sec) AS avg_tok_s,
            AVG(extraction_score) AS avg_score,
            AVG(CASE WHEN valid_json THEN 1.0 ELSE 0.0 END) * 100 AS pct_valid
        FROM benchmark_runs
        WHERE experiment_id = ?
          AND total_ms IS NOT NULL
        GROUP BY case_id
        ORDER BY case_id
    """, [experiment_id]).fetchall()

    print(f"\n{'='*78}")
    print(f"  Experiment: {experiment_name} ({experiment_id})")
    print(f"{'='*78}")
    print(f"  {'Case':<25} {'Runs':>5} {'Avg ms':>9} {'Tok/s':>8} {'Score':>7} {'Valid%':>7}")
    print(f"  {'-'*25} {'-'*5} {'-'*9} {'-'*8} {'-'*7} {'-'*7}")
    for r in rows:
        case, runs, avg_ms, avg_tok, avg_score, pct = r
        tok_str = f"{avg_tok:.1f}" if avg_tok else "N/A"
        score_str = f"{avg_score:.2f}" if avg_score is not None else "N/A"
        print(f"  {case:<25} {runs:>5} {avg_ms:>9.0f} {tok_str:>8} {score_str:>7} {pct:>6.0f}%")

    overall = conn.execute("""
        SELECT
            AVG(gen_tok_per_sec),
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gen_tok_per_sec),
            AVG(total_ms),
            AVG(extraction_score)
        FROM benchmark_runs
        WHERE experiment_id = ? AND total_ms IS NOT NULL
    """, [experiment_id]).fetchone()

    avg_tok = f"{overall[0]:.1f}" if overall[0] else "N/A"
    p50_tok = f"{overall[1]:.1f}" if overall[1] else "N/A"
    avg_lat = f"{overall[2]:.0f}" if overall[2] else "N/A"
    avg_scr = f"{overall[3]:.2f}" if overall[3] is not None else "N/A"
    print(f"\n  Overall: avg={avg_tok} tok/s | p50={p50_tok} tok/s | "
          f"avg_latency={avg_lat}ms | extraction_score={avg_scr}")
    print()


def main():
    parser = argparse.ArgumentParser(description="ClinIQ inference benchmark harness")
    parser.add_argument("--endpoint", default="http://localhost:8080",
                        help="llama-server endpoint URL")
    parser.add_argument("--experiment-name", required=True,
                        help="Human-readable experiment label")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of runs per test case")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup runs (discarded)")
    parser.add_argument("--config-json", default="{}",
                        help="JSON blob of server config for this experiment")
    parser.add_argument("--baseline", default=None,
                        help="Baseline experiment_id for speedup comparison")
    parser.add_argument("--output-db", default=str(DEFAULT_DB),
                        help="DuckDB file path")
    parser.add_argument("--test-cases", default=str(DEFAULT_CASES),
                        help="JSONL test cases file")
    parser.add_argument("--no-stream", action="store_true",
                        help="Use sync mode instead of streaming (no TTFT)")
    parser.add_argument("--notes", default="",
                        help="Free-text notes for this experiment")
    parser.add_argument("--system-prompt", default=None,
                        help="Override system prompt (default: standard ClinIQ prompt)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max tokens for generation (default: 1024)")
    args = parser.parse_args()

    system_prompt = args.system_prompt or DEFAULT_SYSTEM_PROMPT

    # Init
    conn = init_db(args.output_db)
    cases = load_test_cases(args.test_cases)
    experiment_id = str(uuid.uuid4())[:12]
    config = json.loads(args.config_json) if args.config_json else {}

    print(f"Benchmark: {args.experiment_name}")
    print(f"  Endpoint:    {args.endpoint}")
    print(f"  Experiment:  {experiment_id}")
    print(f"  Test cases:  {len(cases)}")
    print(f"  Runs/case:   {args.runs}")
    print(f"  Warmup:      {args.warmup}")
    print(f"  Max tokens:  {args.max_tokens}")
    print(f"  Streaming:   {not args.no_stream}")
    if args.system_prompt:
        print(f"  Prompt:      {system_prompt[:60]}...")

    # Create experiment record
    conn.execute(
        """INSERT INTO experiments (
            experiment_id, experiment_name, created_at,
            model_file, quantization, cuda_enabled, n_gpu_layers,
            ctx_size, threads, batch_size, ubatch_size,
            flash_attn, mlock, cache_type_k, cache_type_v, extra_args,
            notes, system_prompt, max_tokens
        ) VALUES (?, ?, current_timestamp, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            experiment_id, args.experiment_name,
            config.get("model_file"), config.get("quantization"),
            config.get("cuda_enabled"), config.get("n_gpu_layers"),
            config.get("ctx_size"), config.get("threads"),
            config.get("batch_size"), config.get("ubatch_size"),
            config.get("flash_attn"), config.get("mlock"),
            config.get("cache_type_k"), config.get("cache_type_v"),
            json.dumps({k: v for k, v in config.items()
                       if k not in ("model_file", "quantization", "cuda_enabled",
                                    "n_gpu_layers", "ctx_size", "threads", "batch_size",
                                    "ubatch_size", "flash_attn", "mlock",
                                    "cache_type_k", "cache_type_v")}),
            args.notes,
            system_prompt if args.system_prompt else None,
            args.max_tokens,
        ],
    )

    # Run benchmarks
    run_benchmark(
        endpoint=args.endpoint,
        cases=cases,
        runs=args.runs,
        warmup=args.warmup,
        experiment_id=experiment_id,
        experiment_name=args.experiment_name,
        conn=conn,
        system_prompt=system_prompt,
        max_tokens=args.max_tokens,
        use_streaming=not args.no_stream,
    )

    # Compute and store aggregates
    agg = compute_aggregates(conn, experiment_id, args.baseline)
    conn.execute(
        """UPDATE experiments SET
            avg_gen_tok_s = ?, p50_gen_tok_s = ?, p95_total_ms = ?,
            avg_ttft_ms = ?, avg_prompt_tok_s = ?, success_rate = ?,
            total_runs = ?, baseline_id = ?, speedup_pct = ?,
            avg_extraction_score = ?
        WHERE experiment_id = ?""",
        [
            agg["avg_gen_tok_s"], agg["p50_gen_tok_s"], agg["p95_total_ms"],
            agg["avg_ttft_ms"], agg["avg_prompt_tok_s"], agg["success_rate"],
            agg["total_runs"], agg["baseline_id"], agg["speedup_pct"],
            agg["avg_extraction_score"],
            experiment_id,
        ],
    )

    # Print summary
    print_summary(conn, experiment_id, args.experiment_name)

    if agg["speedup_pct"] is not None:
        sign = "+" if agg["speedup_pct"] >= 0 else ""
        print(f"  vs baseline: {sign}{agg['speedup_pct']:.1f}%")

    if agg["avg_extraction_score"] is not None:
        print(f"  extraction quality: {agg['avg_extraction_score']:.2f}")

    print(f"\n  Results saved to: {args.output_db}")
    print(f"  Experiment ID: {experiment_id}")

    conn.close()


if __name__ == "__main__":
    main()
