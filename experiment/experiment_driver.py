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
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Sequence
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


class ServerPool:
    """Context manager for persistent RPC servers across trials."""
    
    def __init__(self, modes: List[str], model: str, master_addr: str, base_port: int):
        self.modes = modes
        self.model = model
        self.master_addr = master_addr
        self.base_port = base_port
        self.servers: Dict[int, ProcessHandle] = {}
        # Map (mode, phase) -> port. Each decode phase gets an additional +5 port
        # offset to avoid TensorPipe stale-socket dead-locks when a new client
        # process reconnects to the same server.
        self.ports: Dict[tuple[str, str], int] = {}

        mode_offsets = {"naive": 0, "remote_cache": 10, "sys_simulated": 20}
        phase_offsets = {"prefill": 0, "decode": 5}

        for mode in modes:
            if mode == "local":
                continue  # local baseline runs entirely on the client

            base_offset = mode_offsets.get(mode, 30)
            for phase, p_off in phase_offsets.items():
                self.ports[(mode, phase)] = base_port + base_offset + p_off
    
    def __enter__(self) -> "ServerPool":
        """Start all required servers."""
        started: set[int] = set()
        for (mode, phase), port in self.ports.items():
            # Avoid launching the same port twice if multiple (mode, phase) map to it
            if port in started:
                continue
            print(f"Starting persistent server for {mode}-{phase} on port {port}...")
            server = _start_rpc_server(self.model, self.master_addr, str(port))
            self.servers[port] = server
            started.add(port)
        
        # Give servers additional time to fully initialize
        if self.servers:
            print("Waiting for servers to fully initialize...")
            time.sleep(2.0)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Terminate all servers."""
        for port, server in self.servers.items():
            print(f"Terminating persistent server on port {port}…")
            server.terminate()
        self.servers.clear()

    def get_port(self, mode: str, phase: str) -> str:
        """Return the port allocated for (mode, phase)."""
        return str(self.ports.get((mode, phase), self.base_port))


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
    
    # Create timestamped log file in artefacts/logs/
    logs_dir = Path("artefacts/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    log_path = logs_dir / f"server_{master_port}_{timestamp}.log"
    
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


def _start_dmon(csv_path: Path, gpu_host: str) -> ProcessHandle:
    """Launch nvidia-smi dmon locally or via SSH depending on *gpu_host*.

    If *gpu_host* refers to the local machine (127.0.0.1/localhost) we spawn
    dmon directly. Otherwise we SSH into the remote host and start it there.
    In the remote case the returned ProcessHandle carries two extra
    attributes:

    _remote_csv (str)   – path of the CSV on the GPU host
    _gpu_host   (str)   – hostname/IP used for scp retrieval later
    """

    if gpu_host in {"127.0.0.1", "localhost"}:
        cmd = ["nvidia-smi", "dmon", "-s", "u", "-i", "0", "-f", str(csv_path)]
        popen = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ProcessHandle(popen)

    # Remote GPU host – write to /tmp and fetch later via scp
    remote_csv = f"/tmp/{csv_path.name}"
    cmd = [
        "ssh",
        gpu_host,
        "nvidia-smi",
        "dmon",
        "-s",
        "u",
        "-i",
        "0",
        "-f",
        remote_csv,
    ]
    popen = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    handle = ProcessHandle(popen)
    setattr(handle, "_remote_csv", remote_csv)
    setattr(handle, "_gpu_host", gpu_host)
    return handle


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


def _run_client(mode: str, phase: str, args) -> tuple[float, int, float]:
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

    # Naive baseline uploads full model weights every call which can be
    # prohibitively slow for large models (e.g. GPT-J).  When the user hasn't
    # explicitly asked for the real upload we add the optimisation flag to
    # download weights once on the server side instead.  This dramatically
    # cuts network traffic and avoids 10-minute timeouts.
    if mode == "naive":
        cmd_list.append("--skip_weight_upload")

    # Add debug info
    print(f"  Running client: mode={mode}, phase={phase}, port={args.master_port}", flush=True)
    cmd = [
        "bash",
        "-c",
        # Important: 2>&1 merges client's stdout and stderr with time's stderr
        f"/usr/bin/time -f %e {shlex.join(cmd_list)} 2>&1",
    ]

    # Timeout heuristics: local runs are quick, remote modes vary.  The naive
    # baseline (without the optimisation flag) can take >10 min for large
    # models because it transfers ~12 GB weights multiple times.  Extend its
    # allowance; other modes remain at 10 min which is usually enough.
    if mode == "local":
        timeout_seconds = 300  # 5 min
    elif mode == "naive":
        timeout_seconds = 1800  # 30-min safeguard for large uploads
    else:
        # Remote-cache and \sys modes still move tensors over the network and can be
        # slow on high-latency links.  Give them the same 30-min window.
        timeout_seconds = 1800
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, 
            encoding="utf-8", timeout=timeout_seconds
        )
        # Parse output: last line is latency, look for NETWORK_BYTES line
        lines = result.stdout.strip().split("\n")
        latency_sec = float(lines[-1])
        
        # Find network bytes and avg_sm in output
        network_bytes = 0
        avg_sm = 0.0
        for line in lines:
            if line.startswith("NETWORK_BYTES:"):
                network_bytes = int(line.split(":", 1)[1].strip())
            elif line.startswith("AVG_SM_UTIL:"):
                try:
                    avg_sm = float(line.split(":", 1)[1].strip().rstrip("%"))
                except ValueError:
                    pass
        
        return latency_sec, network_bytes, avg_sm
        
    except subprocess.TimeoutExpired as e:
        print(f"\nERROR: Client script for mode='{mode}' phase='{phase}' timed out after {timeout_seconds}s.", file=sys.stderr)
        print("--- Captured output from timed out process: ---", file=sys.stderr)
        print(e.stdout if e.stdout else "No stdout captured", file=sys.stderr)
        print("------------------------------------------", file=sys.stderr)
        raise e
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

    # Define all available modes once
    MODE_DEFS: Dict[str, str] = {
        "local": "Local (Baseline)",
        "naive": "Semantic-Blind (Naive)",
        "remote_cache": "Semantic-Blind + Realistic Remote Cache",
        "sys_simulated": "Framework-Level Semantic-Aware (\\sys)",
    }

    # Select subset based on CLI flag
    if args.modes == "all":
        selected: Sequence[str] = list(MODE_DEFS.keys())
    else:
        selected = [m.strip() for m in args.modes.split(",") if m.strip()]

    unknown = set(selected) - MODE_DEFS.keys()
    if unknown:
        raise ValueError(f"Unknown mode(s) requested via --modes: {', '.join(unknown)}")

    MODES = [(m, MODE_DEFS[m]) for m in selected]

    try:
        # Use ServerPool to start persistent servers once
        if not args.external_server:
            with ServerPool(selected, args.model, args.gpu_host, int(args.master_port)) as server_pool:
                for trial in range(1, args.trials + 1):
                    for phase in ("prefill", "decode"):
                        for mode, mode_label in MODES:
                            print(f"Trial {trial} – {mode_label} {phase}…", flush=True)

                            # Paths for artefacts
                            artefact_prefix = Path(args.output_dir) / f"{mode}_{phase}_trial{trial}"
                            artefact_prefix.parent.mkdir(parents=True, exist_ok=True)
                            dmon_csv = artefact_prefix.with_suffix(".csv")

                            # Get port for this mode from the server pool
                            run_port = server_pool.get_port(mode, phase)

                            # Use ExitStack for proper resource cleanup
                            with ExitStack() as stack:
                                for attempt in range(2):
                                    # GPU monitoring is handled inside the RPC server; no local dmon.

                                    try:
                                        # Prepare per-run args with dynamic port
                                        import argparse as _ap
                                        client_args = _ap.Namespace(**vars(args))
                                        client_args.master_port = run_port

                                        # Run client and measure latency + network bytes
                                        start_ts = time.time()
                                        latency_sec, net_bytes, avg_sm = _run_client(mode, phase, client_args)
                                        run_wall = time.time() - start_ts

                                        # Stop GPU monitoring before checking results
                                        # dmon_proc.terminate() # This line is removed

                                        # If dmon ran on a remote host, copy CSV back
                                        # remote_csv = getattr(dmon_proc, "_remote_csv", None) # This line is removed
                                        # remote_host = getattr(dmon_proc, "_gpu_host", None) # This line is removed
                                        # if remote_csv and remote_host: # This line is removed
                                        #     subprocess.run([ # This line is removed
                                        #         "scp", # This line is removed
                                        #         f"{remote_host}:{remote_csv}", # This line is removed
                                        #         str(dmon_csv), # This line is removed
                                        #     ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # This line is removed

                                        # avg_sm already returned by client

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
                                    except Exception as e:
                                        # Log the exception but let ExitStack handle cleanup
                                        print(f"Attempt {attempt + 1} failed for {mode_label} {phase}: {e}", file=sys.stderr)
                                        if attempt == 1:  # Last attempt
                                            raise
                                        continue  # retry
        else:
            # Fallback to original behavior when external server is used
            for trial in range(1, args.trials + 1):
                for phase in ("prefill", "decode"):
                    for mode, mode_label in MODES:
                        print(f"Trial {trial} – {mode_label} {phase}…", flush=True)

                        # Paths for artefacts
                        artefact_prefix = Path(args.output_dir) / f"{mode}_{phase}_trial{trial}"
                        artefact_prefix.parent.mkdir(parents=True, exist_ok=True)
                        dmon_csv = artefact_prefix.with_suffix(".csv")

                        # choose a unique port per (mode, phase) to avoid Gloo stale sockets
                        # Expect *two* servers per remote mode (prefill & decode) when
                        # --external_server is supplied.  Each decode server listens on
                        # `base_port + mode_offset + 5`.
                        base_port = int(args.master_port)
                        mode_offset = {"naive": 0, "remote_cache": 10, "sys_simulated": 20}.get(mode, 30)

                        # Additional per-trial offset to ensure we never reuse the exact same
                        # (ip, port, rank) rendez-vous across independent client processes.
                        # This is primarily needed for the naïve baseline which is prone to
                        # Gloo/TensorPipe stale-socket dead-locks, but we apply it to every
                        # mode for consistency.
                        TRIAL_PORT_STRIDE = 50
                        trial_offset = (trial - 1) * TRIAL_PORT_STRIDE

                        # Separate port for every decode phase (avoid stale sockets *within* a trial)
                        phase_offset = 5 if phase == "decode" and mode != "local" else 0

                        run_port = str(base_port + mode_offset + phase_offset + trial_offset)

                        # Use ExitStack for proper resource cleanup
                        with ExitStack() as stack:
                            for attempt in range(2):
                                # GPU monitoring is handled inside the RPC server; no local dmon.

                                try:
                                    # Prepare per-run args with dynamic port
                                    import argparse as _ap
                                    client_args = _ap.Namespace(**vars(args))
                                    client_args.master_port = run_port

                                    # Run client and measure latency + network bytes
                                    start_ts = time.time()
                                    latency_sec, net_bytes, avg_sm = _run_client(mode, phase, client_args)
                                    run_wall = time.time() - start_ts

                                    # Stop GPU monitoring before checking results
                                    # dmon_proc.terminate() # This line is removed

                                    # If dmon ran on a remote host, copy CSV back
                                    # remote_csv = getattr(dmon_proc, "_remote_csv", None) # This line is removed
                                    # remote_host = getattr(dmon_proc, "_gpu_host", None) # This line is removed
                                    # if remote_csv and remote_host: # This line is removed
                                    #     subprocess.run([ # This line is removed
                                    #         "scp", # This line is removed
                                    #         f"{remote_host}:{remote_csv}", # This line is removed
                                    #         str(dmon_csv), # This line is removed
                                    #     ], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # This line is removed

                                    # avg_sm already returned by client

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
                                except Exception as e:
                                    # Log the exception but let ExitStack handle cleanup
                                    print(f"Attempt {attempt + 1} failed for {mode_label} {phase}: {e}", file=sys.stderr)
                                    if attempt == 1:  # Last attempt
                                        raise
                                    continue  # retry

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
    p.add_argument("--gpu_host", default="127.0.0.1", help="IP/hostname used by PyTorch RPC to reach the GPU server")
    p.add_argument("--master_port", default="29501")
    p.add_argument("--model", default="sshleifer/tiny-gpt2")
    p.add_argument("--output", default="results.csv")
    p.add_argument("--output_dir", default="artefacts", help="Where to store GPU utilization logs")
    p.add_argument("--external_server", action="store_true", help="Assume rpc_server is already running externally")
    p.add_argument(
        "--modes",
        default="all",
        help="Comma-separated subset of modes to run (local,naive,remote_cache,sys_simulated) or 'all'",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    run_experiment(_parse_args())
