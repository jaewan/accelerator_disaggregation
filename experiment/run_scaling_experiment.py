from __future__ import annotations

"""run_scaling_experiment.py
Runs the decode phase for multiple modes across a range of decode step counts
to analyze how latency scales with generation length.
"""

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# --- Configuration ---
# List of decode step counts to test
DECODE_STEPS_LIST = [50, 100, 150, 200, 250]

# Modes to compare in this experiment
MODES_TO_RUN: Dict[str, str] = {
    "naive": "Semantic-Blind (Naive)",
    "remote_cache_delta_raw": "Semantic-Blind + Delta KV Cache",
    "remote_cache_handle": "Semantic-Aware (Handle Only)",
}
# ---

def _run_client(mode: str, decode_steps: int, args: argparse.Namespace) -> tuple[float, dict]:
    """Run client once, return (latency_seconds, metrics_dict)."""
    # Determine absolute path to run_llm.py (it sits next to this file)
    script_dir = Path(__file__).resolve().parent
    run_llm_path = script_dir / "run_llm.py"

    base_args = [
        sys.executable,
        str(run_llm_path),
        "--gpu_host", args.gpu_host,
        "--master_port", str(args.master_port),
        "--model", args.model,
    ]

    # Determine RPC port per mode (match gpu_server_start.sh mapping)
    MODE_PORT_OFFSET = {"naive": 0, "remote_cache_delta_raw": 20, "remote_cache_handle": 30}
    base_port = int(args.master_port) + MODE_PORT_OFFSET.get(mode, 0)
    prefill_port = str(base_port)            # phase==prefill uses base port
    decode_port = str(base_port + 5)         # phase==decode uses base+5

    # 1) Prefill – establish KV cache or handle on server
    prefill_cmd = base_args + [
        "--mode", mode,
        "--phase", "prefill",
        "--decode_steps", "1",  # unused in prefill
        "--master_port", prefill_port,
    ]

    # Weight-upload optimisation for naive baseline
    if mode == "naive" and not args.no_skip_weight_upload:
        prefill_cmd.append("--skip_weight_upload")

    try:
        subprocess.run(prefill_cmd, check=True, timeout=args.timeout)
    except subprocess.SubprocessError as e:
        print(f"    Prefill failed for mode={mode}: {e}", file=sys.stderr)
        from math import nan
        return nan, {}

    # 2) Decode
    cmd_list = base_args + [
        "--mode", mode,
        "--phase", "decode",
        "--decode_steps", str(decode_steps),
        "--master_port", decode_port,
    ]

    if mode == "naive" and not args.no_skip_weight_upload:
        cmd_list.append("--skip_weight_upload")

    print(f"  Running client: mode={mode}, steps={decode_steps}", flush=True)
    cmd = ["/usr/bin/time", "-f", "%e"] + cmd_list
    
    try:
        # We redirect stderr to stdout to capture the output of /usr/bin/time
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True, 
            check=True, 
            encoding="utf-8", 
            timeout=args.timeout
        )
        lines = result.stdout.strip().split("\n")
        latency_sec = float(lines[-1])
        
        metrics_dict = {}
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                metrics_dict[key.strip()] = value.strip()
        
        return latency_sec, metrics_dict
        
    except subprocess.TimeoutExpired as e:
        print(f"\nERROR: Client script timed out.", file=sys.stderr)
        print("--- Captured output: ---\n", e.stdout, file=sys.stderr)
        raise e
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Client script failed.", file=sys.stderr)
        print("--- Captured output: ---\n", e.stdout, file=sys.stderr)
        raise e

def run_experiment(args):
    # --- Resume Logic ---
    completed_runs = set()
    output_path = Path(args.output)
    is_new_file = not output_path.exists() or output_path.stat().st_size == 0

    if not is_new_file:
        print(f"Found existing results file: {args.output}. Resuming experiment.")
        with open(output_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Recreate the key to check for completion
                try:
                    key = (
                        int(row["trial"]),
                        row["mode"],
                        int(row["decode_steps"])
                    )
                    completed_runs.add(key)
                except (KeyError, ValueError) as e:
                    print(f"Skipping malformed row in existing CSV: {row} ({e})")
            print(f"Loaded {len(completed_runs)} completed runs.")
    # ---

    fieldnames = ["trial", "mode", "decode_steps", "latency_s", "net_bytes", "avg_sm"]
    
    # Open in append mode, so we can resume.
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new_file:
            writer.writeheader()
        
        # Ensure data is written immediately to disk after each row.
        f.flush()

        for trial in range(1, args.trials + 1):
            for steps in DECODE_STEPS_LIST:
                for mode, mode_label in MODES_TO_RUN.items():
                    # --- Resume Logic: Check if we should skip ---
                    run_key = (trial, mode_label, steps)
                    if run_key in completed_runs:
                        print(f"Skipping completed run: Trial {trial}, Mode: {mode_label}, Steps: {steps}")
                        continue
                    # ---

                    print(f"Trial {trial}, Mode: {mode_label}, Decode Steps: {steps}…", flush=True)

                    try:
                        latency, metrics = _run_client(mode, steps, args)
                        
                        # Write result immediately
                        writer.writerow({
                            "trial": trial,
                            "mode": mode_label,
                            "decode_steps": steps,
                            "latency_s": latency,
                            "net_bytes": metrics.get("NETWORK_BYTES", 0),
                            "avg_sm": metrics.get("AVG_SM_UTIL", 0.0),
                        })
                        f.flush() # Ensure it's written to disk

                    except Exception as e:
                        print(f"  Skipping failed run: {e}", file=sys.stderr)
                        continue

    print(f"\nScaling experiment complete. Results written to {args.output}")

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Latency scaling experiment driver")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--gpu_host", default="127.0.0.1")
    p.add_argument("--master_port", default="29500")
    p.add_argument("--model", default="EleutherAI/gpt-j-6B")
    p.add_argument("--output", default="scaling_results.csv")
    p.add_argument("--no_skip_weight_upload", action="store_true", help="Disable weight-upload optimisation for naive baseline")
    p.add_argument("--timeout", type=int, default=5400, help="Per-run timeout in seconds (default 1.5 h)")
    return p.parse_args(argv)

if __name__ == "__main__":
    run_experiment(_parse_args()) 