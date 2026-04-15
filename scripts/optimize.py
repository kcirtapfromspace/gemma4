#!/usr/bin/env python3
"""Automated optimization loop for ClinIQ inference.

Iterates through experiment configurations defined in experiments.yaml,
patches the K8s deployment for each, restarts, waits for health, benchmarks,
and produces a comparison report.

Usage:
    python optimize.py --experiments experiments.yaml
    python optimize.py --experiments experiments.yaml --runs 10 --endpoint http://192.168.25.x:30083
    python optimize.py --experiments experiments.yaml --skip-rebuild  # skip Docker rebuilds
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

SCRIPT_DIR = Path(__file__).parent
DEFAULT_DB = SCRIPT_DIR / "benchmarks.duckdb"
DEFAULT_EXPERIMENTS = SCRIPT_DIR / "experiments.yaml"
DEFAULT_MODEL = "/models/cliniq-gemma4-e2b-Q4_K_M.gguf"
NAMESPACE = "gemma4"
DEPLOYMENT = "llama-server"


def load_experiments(path: str) -> list[dict]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["experiments"]


def build_server_args(experiment: dict) -> list[str]:
    """Convert experiment server_args to llama-server CLI args list."""
    model = experiment.get("model_override", DEFAULT_MODEL)
    sa = experiment.get("server_args", {})
    args = ["-m", model, "--port", "8080", "--host", "0.0.0.0"]

    flag_map = {
        "ctx_size": "--ctx-size",
        "n_gpu_layers": "--n-gpu-layers",
        "threads": "--threads",
        "batch_size": "--batch-size",
        "ubatch_size": "--ubatch-size",
        "cache_type_k": "--cache-type-k",
        "cache_type_v": "--cache-type-v",
        "parallel": "--parallel",
    }

    bool_flags = {
        "flash_attn": "--flash-attn",
        "mlock": "--mlock",
        "no_mmap": "--no-mmap",
    }

    for key, flag in flag_map.items():
        if key in sa:
            args.extend([flag, str(sa[key])])

    for key, flag in bool_flags.items():
        if sa.get(key):
            args.append(flag)

    return args


def build_config_json(experiment: dict) -> str:
    """Build a config JSON blob for benchmark.py."""
    sa = experiment.get("server_args", {})
    config = dict(sa)
    config["model_file"] = experiment.get("model_override", DEFAULT_MODEL)
    if "image_tag" in experiment:
        config["image_tag"] = experiment["image_tag"]
    return json.dumps(config)


def kubectl(*args, capture=True, timeout=300) -> subprocess.CompletedProcess:
    cmd = ["kubectl", "-n", NAMESPACE] + list(args)
    return subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout)


def patch_deployment_args(server_args: list[str]) -> bool:
    """Patch the llama-server deployment with new container args."""
    args_json = json.dumps(server_args)
    patch = json.dumps({
        "spec": {
            "template": {
                "spec": {
                    "containers": [{
                        "name": DEPLOYMENT,
                        "args": server_args,
                    }]
                }
            }
        }
    })

    result = kubectl("patch", "deployment", DEPLOYMENT,
                     "--type=strategic", "-p", patch)
    if result.returncode != 0:
        print(f"    kubectl patch failed: {result.stderr}")
        return False
    return True


def patch_deployment_image(image_tag: str, registry: str) -> bool:
    """Update the deployment image tag."""
    full_image = f"{registry}/{image_tag}"
    result = kubectl("set", "image", f"deployment/{DEPLOYMENT}",
                     f"{DEPLOYMENT}={full_image}")
    if result.returncode != 0:
        print(f"    kubectl set image failed: {result.stderr}")
        return False
    return True


def restart_and_wait(endpoint: str, timeout: int = 300) -> bool:
    """Scale down, then up to avoid memory pressure during model swap."""
    # Scale down first to free memory
    print("    Scaling down...", end="", flush=True)
    kubectl("scale", f"deployment/{DEPLOYMENT}", "--replicas=0")
    time.sleep(5)
    # Force-delete any lingering pods
    kubectl("delete", "pods", "-l", f"app={DEPLOYMENT}",
            "--force", "--grace-period=0")
    time.sleep(3)
    print(" done.", flush=True)

    # Scale back up
    print("    Scaling up...", end="", flush=True)
    kubectl("scale", f"deployment/{DEPLOYMENT}", "--replicas=1")
    print(" started.", flush=True)

    # Wait for health endpoint
    print("    Waiting for /health...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{endpoint}/health", timeout=5)
            if r.status_code == 200:
                print(" healthy!")
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(5)
        print(".", end="", flush=True)

    print(" TIMEOUT")
    return False


def run_benchmark_subprocess(
    experiment_name: str,
    endpoint: str,
    config_json: str,
    runs: int,
    warmup: int,
    db_path: str,
    baseline_id: str | None,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
) -> str | None:
    """Run benchmark.py as a subprocess and return the experiment_id."""
    cmd = [
        sys.executable, str(SCRIPT_DIR / "benchmark.py"),
        "--experiment-name", experiment_name,
        "--endpoint", endpoint,
        "--config-json", config_json,
        "--runs", str(runs),
        "--warmup", str(warmup),
        "--output-db", db_path,
        "--no-stream",
    ]
    if baseline_id:
        cmd.extend(["--baseline", baseline_id])
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])
    if max_tokens:
        cmd.extend(["--max-tokens", str(max_tokens)])

    result = subprocess.run(cmd, capture_output=False, text=True, timeout=7200)
    if result.returncode != 0:
        print(f"    benchmark.py failed (exit {result.returncode})")
        return None

    # Extract experiment_id from the DB (most recent for this name)
    import duckdb
    conn = duckdb.connect(db_path, read_only=True)
    row = conn.execute("""
        SELECT experiment_id FROM experiments
        WHERE experiment_name = ?
        ORDER BY created_at DESC LIMIT 1
    """, [experiment_name]).fetchone()
    conn.close()
    return row[0] if row else None


def print_final_report(db_path: str):
    """Print the final comparison report."""
    print("\n" + "=" * 80)
    print("  OPTIMIZATION SWEEP RESULTS")
    print("=" * 80)
    subprocess.run([sys.executable, str(SCRIPT_DIR / "report.py"),
                    "--db", db_path, "--waterfall"], timeout=30)


def main():
    parser = argparse.ArgumentParser(description="Automated inference optimization loop")
    parser.add_argument("--experiments", default=str(DEFAULT_EXPERIMENTS),
                        help="YAML file defining experiments")
    parser.add_argument("--endpoint", default="http://localhost:30083",
                        help="llama-server endpoint (NodePort or port-forward)")
    parser.add_argument("--registry", default="192.168.25.201:5050",
                        help="Container registry for image pushes")
    parser.add_argument("--runs", type=int, default=5,
                        help="Benchmark runs per test case")
    parser.add_argument("--warmup", type=int, default=1,
                        help="Warmup runs (discarded)")
    parser.add_argument("--db", default=str(DEFAULT_DB),
                        help="DuckDB file path")
    parser.add_argument("--skip-rebuild", action="store_true",
                        help="Skip experiments that require Docker rebuild")
    parser.add_argument("--only", default=None,
                        help="Run only this experiment name")
    parser.add_argument("--health-timeout", type=int, default=300,
                        help="Seconds to wait for health after restart")
    args = parser.parse_args()

    experiments = load_experiments(args.experiments)
    if args.only:
        experiments = [e for e in experiments if e["name"] == args.only]
        if not experiments:
            print(f"No experiment named '{args.only}' found.")
            sys.exit(1)

    print(f"Optimization sweep: {len(experiments)} experiments")
    print(f"  Endpoint:  {args.endpoint}")
    print(f"  Registry:  {args.registry}")
    print(f"  Runs/case: {args.runs}")
    print(f"  DB:        {args.db}\n")

    baseline_id = None
    results = []

    for i, exp in enumerate(experiments):
        name = exp["name"]
        desc = exp.get("description", "")
        needs_rebuild = exp.get("requires_rebuild", False)

        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(experiments)}] {name}")
        print(f"  {desc}")
        print(f"{'='*60}")

        # Skip rebuild experiments if requested
        if needs_rebuild and args.skip_rebuild:
            print("  SKIPPED (requires rebuild, --skip-rebuild set)")
            continue

        skip_restart = exp.get("skip_restart", False)
        system_prompt = exp.get("system_prompt")
        max_tokens = exp.get("max_tokens")

        # Build server args
        server_args = build_server_args(exp)
        config_json = build_config_json(exp)

        if not skip_restart:
            print(f"  Server args: {' '.join(server_args[6:])}")

            # Update image if needed
            image_tag = exp.get("image_tag")
            if image_tag:
                print(f"  Image: {args.registry}/{image_tag}")
                if not patch_deployment_image(image_tag, args.registry):
                    print("  FAILED to update image, skipping")
                    continue

            # Patch deployment args
            if not patch_deployment_args(server_args):
                print("  FAILED to patch deployment, skipping")
                continue

            # Restart and wait
            if not restart_and_wait(args.endpoint, args.health_timeout):
                print("  FAILED health check, skipping")
                continue

            # Extra settle time after restart
            time.sleep(5)
        else:
            print("  Client-side only (skip_restart=true)")
            if system_prompt:
                print(f"  Prompt: {system_prompt[:60]}...")
            if max_tokens:
                print(f"  Max tokens: {max_tokens}")

        # Run benchmark
        exp_id = run_benchmark_subprocess(
            experiment_name=name,
            endpoint=args.endpoint,
            config_json=config_json,
            runs=args.runs,
            warmup=args.warmup,
            db_path=args.db,
            baseline_id=baseline_id,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )

        if exp_id:
            if baseline_id is None:
                baseline_id = exp_id
                print(f"  Set as baseline: {exp_id}")
            results.append({"name": name, "id": exp_id})
        else:
            print("  Benchmark failed for this experiment")

    # Final report
    if results:
        print_final_report(args.db)
    else:
        print("\nNo experiments completed successfully.")


if __name__ == "__main__":
    main()
