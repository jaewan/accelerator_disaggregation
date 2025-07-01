from __future__ import annotations

"""experiment_driver.py
Automates Phase 3 of the semantic-gap experiment.

Runs each (mode, phase) combination N times, measures latency, network bytes
transferred, and average SM utilisation, then writes a CSV summary.

Network bytes are measured using source code instrumentation rather than
tcpdump/capinfos for better reliability across different environments.

Assumptions
-----------
• Driver is executed on CLIENT_HOST but can also run locally when SERVER and
  CLIENT are the same machine.
• RPC server (`rpc_server.py`) will be spawned locally by the driver.
  If you run an external server manually, call with ``--external_server``.
• `nvidia-smi` is installed and in $PATH for GPU utilization monitoring.

Usage example
-------------
python experiment_driver.py \\
    --trials 3 \\
    --gpu_host 127.0.0.1 \\
    --master_port 29501 \\
    --model sshleifer/tiny-gpt2 \\
    --output results.csv
"""

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
import shlex

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


class ProcessHandle:
    """Light wrapper around subprocess.Popen with safe termination."""

    def __init__(self, popen: subprocess.Popen):
        self._popen = popen

    def terminate(self, sig: int = signal.SIGINT, timeout: float = 5.0):
        if self._popen.poll() is not None:
            return  # already dead
        try:
            self._popen.send_signal(sig)
            self._popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._popen.kill()
            self._popen.wait()

    @property
    def pid(self) -> int:  # noqa: D401
        return self._popen.pid


# --------------------------------------------------------------------------------------
# Core functions
# --------------------------------------------------------------------------------------


def _start_rpc_server(model: str, master_addr: str, master_port: str) -> ProcessHandle:
    """Launch rpc_server.py in the background and return handle."""
    cmd = [
        sys.executable,
        "rpc_server.py",
        "--model",
        model,
        "--master_addr",
        master_addr,
        "--master_port",
        master_port,
        "--world_size",
        "2",
        "--rank",
        "0",
    ]
    log_path = Path("server_stdout.log")
    with log_path.open("w") as log_file:
        popen = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    # Wait for server to be ready instead of using a fixed sleep.
    handle = ProcessHandle(popen)
    try:
        for _ in range(1800):  # Wait up to 30 minutes for large model downloads
            if popen.poll() is not None:
                # Print last 20 lines of server log for debugging
                if log_path.exists():
                    log_lines = log_path.read_text().splitlines()
                    print("\n--- server_stdout.log (last 20 lines) ---")
                    for line in log_lines[-20:]:
                        print(line)
                    print("--- end server_stdout.log ---\n")
                else:
                    print("server_stdout.log not found.")
                raise RuntimeError("RPC server process terminated unexpectedly. See above for log.")
            if log_path.exists():
                log_content = log_path.read_text()
                if "RPC server running" in log_content or "RemoteWorker ready" in log_content:
                    print("RPC server started successfully.")
                    return handle
            time.sleep(1)
        print("\n--- server_stdout.log (last 20 lines) ---")
        if log_path.exists():
            log_lines = log_path.read_text().splitlines()
            for line in log_lines[-20:]:
                print(line)
        else:
            print("server_stdout.log not found.")
        print("--- end server_stdout.log ---\n")
        raise RuntimeError("RPC server failed to start in time. See above for log.")
    except Exception:
        handle.terminate()
        raise


def _start_dmon(csv_path: Path) -> ProcessHandle:
    cmd = ["nvidia-smi", "dmon", "-s", "u", "-i", "0", "-f", str(csv_path)]
    popen = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ProcessHandle(popen)


def _start_tcpdump(nic: str, pcap_path: Path) -> ProcessHandle:
    """Start tcpdump to capture network traffic. DEPRECATED - using source code instrumentation instead."""
    # This function is no longer used but kept for compatibility
    cmd = ["sudo", "tcpdump", "-i", nic, "-w", str(pcap_path)]
    popen = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ProcessHandle(popen)


def _capinfos_total_bytes(pcap_path: Path) -> int:
    """DEPRECATED: Get total data bytes from a pcap file using capinfos."""
    # This function is no longer used - we now measure bytes in source code
    return 0


def _average_sm_util(csv_path: Path) -> float:
    """Parse dmon log and return average SM utilisation."""
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        print(f"Warning: dmon file not found or empty: {csv_path}. Assuming 0% SM util.", file=sys.stderr)
        return 0.0
    try:
        result = subprocess.run(
            [sys.executable, "parse_dmon.py", str(csv_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        # Expected output: "AVG_SM_UTIL: 12.34%"
        for line in result.stdout.splitlines():
            if line.startswith("AVG_SM_UTIL"):
                value = line.split(":", 1)[1].strip().rstrip("%")
                return float(value)
        print(f"Warning: Could not find 'AVG_SM_UTIL' in parse_dmon.py output for {csv_path}. Assuming 0%.", file=sys.stderr)
        return 0.0
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Warning: Failed to get SM util for {csv_path} ({e}). Assuming 0%.", file=sys.stderr)
        return 0.0


def _run_client(mode: str, phase: str, args) -> tuple[float, int]:
    """Run client once, return (latency_seconds, network_bytes)."""
    cmd_list = [
        sys.executable,
        "run_llm.py",
        "--mode",
        mode,
        "--phase",
        phase,
        "--gpu_host",
        args.gpu_host,
        "--master_port",
        str(args.master_port),
        "--model",
        args.model,
    ]
    cmd = [
        "bash",
        "-c",
        # Important: 2>&1 merges client's stdout and stderr with time's stderr
        f"/usr/bin/time -f %e {shlex.join(cmd_list)} 2>&1",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, encoding="utf-8"
        )
        # Parse output: last line is latency, look for NETWORK_BYTES line
        lines = result.stdout.strip().split("\n")
        latency_sec = float(lines[-1])
        
        # Find network bytes in output
        network_bytes = 0
        for line in lines:
            if line.startswith("NETWORK_BYTES:"):
                network_bytes = int(line.split(":", 1)[1].strip())
                break
        
        return latency_sec, network_bytes
        
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Client script for mode='{mode}' phase='{phase}' failed.", file=sys.stderr)
        print("--- Captured output from failed process: ---", file=sys.stderr)
        # The stdout attribute contains the combined output stream due to `2>&1`
        print(e.stdout, file=sys.stderr)
        print("------------------------------------------", file=sys.stderr)
        raise e
    except (ValueError, IndexError) as e_parse:
        # This secondary path handles cases where the script exits with code 0
        # but the output is malformed and we can't parse the latency.
        print(
            f"\nERROR: Could not parse latency from client output for mode='{mode}' phase='{phase}'.",
            file=sys.stderr,
        )
        # We need the output to debug, but it's not in the exception.
        # We can't access `result` here. This is a rare case; the main
        # CalledProcessError is the most likely one to occur.
        raise e_parse from None


# --------------------------------------------------------------------------------------
# Utility to join shell args safely
# --------------------------------------------------------------------------------------


def shlex_join(tokens: List[str]) -> str:  # fallback for Python<3.8
    import shlex

    return " ".join(shlex.quote(t) for t in tokens)


# --------------------------------------------------------------------------------------
# Extra helper for robust GPU-util sampling (Step 5.2)
# --------------------------------------------------------------------------------------


def _csv_data_rows(path: Path) -> int:
    """Return number of non-comment, non-empty rows in a CSV file."""
    if not path.is_file():
        return 0
    cnt = 0
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        cnt += 1
    return cnt


# --------------------------------------------------------------------------------------
# Driver main
# --------------------------------------------------------------------------------------


def run_experiment(args):
    results: List[Dict[str, Any]] = []

    MODES = [
        ("local", "Local (Baseline)"),
        ("naive", "Semantic-Blind (Naive)"),
        ("remote_cache", "Semantic-Blind + Realistic Remote Cache"),
        ("sys_simulated", "Framework-Level Semantic-Aware (\\sys)")
    ]

    try:
        for trial in range(1, args.trials + 1):
            for phase in ("prefill", "decode"):
                for mode, mode_label in MODES:
                    print(f"Trial {trial} – {mode_label} {phase}…", flush=True)

                    # Paths for artefacts
                    artefact_prefix = Path(args.output_dir) / f"{mode}_{phase}_trial{trial}"
                    artefact_prefix.parent.mkdir(parents=True, exist_ok=True)
                    dmon_csv = artefact_prefix.with_suffix(".csv")

                    # -------- assign a unique port per mode  -----------------
                    base      = int(args.master_port)          # e.g. 29502
                    offsets    = {"naive": 0, "remote_cache": 1, "sys_simulated": 2}
                    run_port   = str(base + offsets.get(mode, 0))

                    # Start RPC server for remote modes only
                    server = None
                    if mode != "local" and not args.external_server:
                        server = _start_rpc_server(args.model, args.gpu_host, args.master_port)

                    for attempt in range(2):
                        try:
                            # Start GPU monitoring and allow it to collect at least one sample
                            dmon_proc = _start_dmon(dmon_csv)
                            time.sleep(0.5)

                            # Run client and measure latency + network bytes
                            start_ts = time.time()
                            latency_sec, net_bytes = _run_client(mode, phase,
                                     args._replace(master_port=run_port))
                            run_wall = time.time() - start_ts

                            # Stop GPU monitoring
                            dmon_proc.terminate()

                            # Verify CSV has sufficient data rows; retry once if not
                            if _csv_data_rows(dmon_csv) < 2 and attempt == 0:
                                print(f"Warning: dmon log too short for {mode_label} {phase}; retrying…")
                                continue  # retry the attempt

                            # Collect GPU utilization
                            avg_sm = _average_sm_util(dmon_csv)

                            results.append({
                                "trial": trial,
                                "phase": phase,
                                "mode": mode_label,
                                "latency_s": latency_sec,
                                "wall_s": run_wall,
                                "net_bytes": net_bytes,
                                "avg_sm": avg_sm,
                            })
                            break  # exit retry loop on success
                        finally:
                            pass  # per-attempt cleanup handled after loop

                    # ---- end for attempt loop ----
                    if server is not None:
                        server.terminate()
                        time.sleep(2)

    except Exception as e:
        print(f"Experiment failed: {e}", file=sys.stderr)
        raise

    # Write output CSV
    fieldnames = ["trial", "phase", "mode", "latency_s", "wall_s", "net_bytes", "avg_sm"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results written to {args.output}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Semantic gap experiment driver")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--gpu_host", default="127.0.0.1")
    p.add_argument("--master_port", default="29501")
    p.add_argument("--model", default="sshleifer/tiny-gpt2")
    p.add_argument("--output", default="results.csv")
    p.add_argument("--output_dir", default="artefacts", help="Where to store GPU utilization logs")
    p.add_argument("--external_server", action="store_true", help="Assume rpc_server is already running externally")
    return p.parse_args(argv)


if __name__ == "__main__":
    run_experiment(_parse_args())
