#!/usr/bin/env python3
"""
ClinIQ Inference Optimization — Experiment Runner.

This is the file the autoresearch agent modifies. It wraps the benchmark
harness with experiment-specific configuration.

Usage:
    python tools/autoresearch/run_experiment.py \
        --endpoint http://192.168.150.41:30083 \
        --name "ctx1024-threads1"

The agent modifies the EXPERIMENT CONFIG section below, then runs this script.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "benchmark.py"
BENCHMARK_DB = REPO_ROOT / "scripts" / "benchmarks.duckdb"
TEST_CASES = REPO_ROOT / "scripts" / "test_cases.jsonl"

# ============================================================
# EXPERIMENT CONFIG — The agent modifies THIS section
# ============================================================

# Server-side args (require pod restart to take effect).
# To apply these, the agent must kubectl delete/apply the deployment
# with updated args. Set to None to skip server restart.
SERVER_ARGS = {
    "ctx_size": 1536,
    "n_gpu_layers": 99,
    "reasoning_budget": 0,
    "lora": "/models/cliniq-compact-lora.gguf",
}

# Model file on Jetson (under /var/lib/ollama/models/)
MODEL_FILE = "gemma-4-E2B-it-Q3_K_M.gguf"

# Client-side params (no server restart needed)
SYSTEM_PROMPT = (
    "Extract clinical entities from this eICR summary. Output JSON with: "
    "patient demographics, conditions (SNOMED/ICD-10), labs (LOINC), "
    "medications (RxNorm), vitals, and a case summary. "
    "Output valid JSON only."
)
MAX_TOKENS = 1024
USE_STREAMING = False       # non-streaming avoids chunk loss
RUNS_PER_CASE = 3          # validation run: 3 reps for stable numbers
WARMUP = 1                 # 1 warmup for stable results

# Quantization label for tracking
QUANTIZATION = "Q3_K_M"

# ============================================================
# END EXPERIMENT CONFIG
# ============================================================


def build_server_command_args() -> list[str]:
    """Build the llama-server args list from SERVER_ARGS."""
    args = [
        "-m", f"/models/{MODEL_FILE}",
        "--port", "8080",
        "--host", "0.0.0.0",
    ]
    if SERVER_ARGS.get("ctx_size"):
        args += ["--ctx-size", str(SERVER_ARGS["ctx_size"])]
    if SERVER_ARGS.get("n_gpu_layers") is not None:
        args += ["--n-gpu-layers", str(SERVER_ARGS["n_gpu_layers"])]
    if SERVER_ARGS.get("threads"):
        args += ["--threads", str(SERVER_ARGS["threads"])]
    if SERVER_ARGS.get("batch_size"):
        args += ["--batch-size", str(SERVER_ARGS["batch_size"])]
    if SERVER_ARGS.get("ubatch_size"):
        args += ["--ubatch-size", str(SERVER_ARGS["ubatch_size"])]
    if SERVER_ARGS.get("flash_attn"):
        args += ["--flash-attn"]
    if SERVER_ARGS.get("reasoning_budget") is not None:
        args += ["--reasoning-budget", str(SERVER_ARGS["reasoning_budget"])]
    if SERVER_ARGS.get("mlock"):
        args += ["--mlock"]
    if SERVER_ARGS.get("cache_type_k"):
        args += ["--cache-type-k", SERVER_ARGS["cache_type_k"]]
    if SERVER_ARGS.get("cache_type_v"):
        args += ["--cache-type-v", SERVER_ARGS["cache_type_v"]]
    if SERVER_ARGS.get("lora"):
        args += ["--lora", SERVER_ARGS["lora"]]
    if SERVER_ARGS.get("model_draft"):
        args += ["--model-draft", SERVER_ARGS["model_draft"]]
    return args


def check_endpoint(endpoint: str) -> bool:
    """Verify llama-server is reachable."""
    import requests
    try:
        resp = requests.get(f"{endpoint}/health", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def restart_server_if_needed(endpoint: str, args: argparse.Namespace) -> bool:
    """Restart the llama-server pod with updated config if server args changed.

    Returns True if server is ready, False if restart failed.
    """
    if args.skip_restart:
        print("Skipping server restart (--skip-restart)")
        return check_endpoint(endpoint)

    # Build the deployment patch with new args
    server_cmd_args = build_server_command_args()
    print(f"Server args: {' '.join(server_cmd_args)}")

    # For now, just verify the server is reachable.
    # To actually restart with new args, the agent should:
    #   1. Edit apps/llama-server/deployment.yaml
    #   2. kubectl apply -f apps/llama-server/deployment.yaml
    #   3. Wait for rollout
    #
    # Automated restart via kubectl patch:
    if not args.no_kubectl:
        patch = {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "llama-server",
                            "args": server_cmd_args
                        }]
                    }
                }
            }
        }
        patch_json = json.dumps(patch)
        print(f"Applying kubectl patch...")
        result = subprocess.run(
            ["kubectl", "patch", "deployment", "llama-server",
             "-n", "gemma4", "--type=strategic",
             "-p", patch_json],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"kubectl patch failed: {result.stderr}")
            return False
        print(f"Patch applied: {result.stdout.strip()}")

        # Wait for rollout
        print("Waiting for rollout...")
        result = subprocess.run(
            ["kubectl", "rollout", "status", "deployment/llama-server",
             "-n", "gemma4", "--timeout=300s"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Rollout failed: {result.stderr}")
            return False
        print("Rollout complete.")

        # Wait for health check
        print("Waiting for health check...", end="", flush=True)
        for i in range(60):
            if check_endpoint(endpoint):
                print(" ready!")
                return True
            time.sleep(5)
            print(".", end="", flush=True)
        print(" TIMEOUT")
        return False

    return check_endpoint(endpoint)


def run_benchmark(endpoint: str, name: str) -> dict:
    """Run the benchmark harness and parse results."""
    config_blob = {
        "model_file": MODEL_FILE,
        "quantization": QUANTIZATION,
        "n_gpu_layers": SERVER_ARGS.get("n_gpu_layers"),
        "ctx_size": SERVER_ARGS.get("ctx_size"),
        "threads": SERVER_ARGS.get("threads"),
        "batch_size": SERVER_ARGS.get("batch_size"),
        "ubatch_size": SERVER_ARGS.get("ubatch_size"),
        "flash_attn": SERVER_ARGS.get("flash_attn"),
    }
    config_blob = {k: v for k, v in config_blob.items() if v is not None}

    cmd = [
        sys.executable, str(BENCHMARK_SCRIPT),
        "--endpoint", endpoint,
        "--experiment-name", name,
        "--runs", str(RUNS_PER_CASE),
        "--warmup", str(WARMUP),
        "--config-json", json.dumps(config_blob),
        "--output-db", str(BENCHMARK_DB),
        "--test-cases", str(TEST_CASES),
        "--max-tokens", str(MAX_TOKENS),
        "--notes", f"autoresearch: {name}",
    ]

    if SYSTEM_PROMPT:
        cmd += ["--system-prompt", SYSTEM_PROMPT]

    if not USE_STREAMING:
        cmd += ["--no-stream"]

    print(f"\nRunning benchmark: {' '.join(cmd[:6])}...")
    t_start = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True)
    t_total = time.time() - t_start

    # Print benchmark output
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(f"Benchmark FAILED (exit code {result.returncode})")
        return {"crashed": True, "total_seconds": t_total}

    # Parse results from DuckDB
    try:
        import duckdb
        conn = duckdb.connect(str(BENCHMARK_DB), read_only=True)
        row = conn.execute("""
            SELECT avg_gen_tok_s, avg_prompt_tok_s, avg_extraction_score,
                   success_rate, total_runs
            FROM experiments
            WHERE experiment_name = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, [name]).fetchone()
        conn.close()

        if row:
            return {
                "crashed": False,
                "gen_tok_s": row[0],
                "prompt_tok_s": row[1],
                "extraction_score": row[2],
                "success_rate": row[3],
                "total_runs": row[4],
                "total_seconds": t_total,
            }
    except Exception as e:
        print(f"Failed to read results from DB: {e}")

    return {"crashed": True, "total_seconds": t_total}


def main():
    parser = argparse.ArgumentParser(description="ClinIQ autoresearch experiment runner")
    parser.add_argument("--endpoint", default="http://192.168.150.41:30083",
                        help="llama-server endpoint")
    parser.add_argument("--name", required=True,
                        help="Experiment name/description")
    parser.add_argument("--skip-restart", action="store_true",
                        help="Don't restart llama-server (client-only changes)")
    parser.add_argument("--no-kubectl", action="store_true",
                        help="Don't run kubectl commands (just check health)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  ClinIQ Autoresearch Experiment: {args.name}")
    print("=" * 60)
    print(f"  Model:       {MODEL_FILE}")
    print(f"  Quant:       {QUANTIZATION}")
    print(f"  Ctx size:    {SERVER_ARGS.get('ctx_size')}")
    print(f"  GPU layers:  {SERVER_ARGS.get('n_gpu_layers')}")
    print(f"  Threads:     {SERVER_ARGS.get('threads', 'default')}")
    print(f"  Max tokens:  {MAX_TOKENS}")
    print(f"  Runs/case:   {RUNS_PER_CASE}")
    print(f"  Prompt:      {SYSTEM_PROMPT[:60]}...")
    print()

    # Step 1: Ensure server is ready
    if not restart_server_if_needed(args.endpoint, args):
        print("\nERROR: Server not reachable")
        print("---")
        print(f"gen_tok_s:          0.000")
        print(f"extraction_score:   0.00")
        print(f"prompt_tok_s:       0.0")
        print(f"total_seconds:      0.0")
        print(f"status:             crash")
        print(f"description:        {args.name} (server unreachable)")
        sys.exit(1)

    # Step 2: Run benchmark
    results = run_benchmark(args.endpoint, args.name)

    # Step 3: Print summary in autoresearch format
    print("\n---")
    if results.get("crashed"):
        print(f"gen_tok_s:          0.000")
        print(f"extraction_score:   0.00")
        print(f"prompt_tok_s:       0.0")
        print(f"total_seconds:      {results['total_seconds']:.1f}")
        print(f"model_file:         {MODEL_FILE}")
        print(f"server_args:        --ctx-size {SERVER_ARGS.get('ctx_size')} --n-gpu-layers {SERVER_ARGS.get('n_gpu_layers')}")
        print(f"status:             crash")
        print(f"description:        {args.name}")
    else:
        gen = results.get("gen_tok_s") or 0
        ext = results.get("extraction_score") or 0
        prompt = results.get("prompt_tok_s") or 0
        print(f"gen_tok_s:          {gen:.3f}")
        print(f"extraction_score:   {ext:.2f}")
        print(f"prompt_tok_s:       {prompt:.3f}")
        print(f"total_seconds:      {results['total_seconds']:.1f}")
        print(f"model_file:         {MODEL_FILE}")
        print(f"server_args:        --ctx-size {SERVER_ARGS.get('ctx_size')} --n-gpu-layers {SERVER_ARGS.get('n_gpu_layers')}")
        print(f"status:             {'keep' if gen > 0 and ext >= 0.80 else 'discard'}")
        print(f"description:        {args.name}")


if __name__ == "__main__":
    main()
