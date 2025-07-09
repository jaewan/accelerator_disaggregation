"""run_llm.py
Client orchestration script for semantic-gap experiments.

Usage examples (on CLIENT_HOST):
    python run_llm.py --mode local --phase prefill --model EleutherAI/gpt-j-6B
    python run_llm.py --mode naive --phase decode --gpu_host 10.0.0.2 --model EleutherAI/gpt-j-6B
    python run_llm.py --mode remote_cache --phase decode --gpu_host 10.0.0.2 --model EleutherAI/gpt-j-6B
    python run_llm.py --mode sys_simulated --phase decode --gpu_host 10.0.0.2 --model EleutherAI/gpt-j-6B
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import time
from typing import List, Optional
import uuid

import torch
from torch.distributed import rpc
from rpc_utils import rpc_sync_with_rref as _rpc_sync
from transformers import AutoModelForCausalLM, AutoTokenizer
import zstandard as zstd  # type: ignore
import pickle
import io

# ----------------------------------------------------------------------------------
# Note: RPC helper utilities now live in ``experiment.rpc_utils`` so that they
# are importable under the same module name on *all* RPC workers. This avoids
# pickling errors where a function defined in ``__main__`` cannot be resolved
# on the remote side.
# ----------------------------------------------------------------------------------

DECODE_STEPS = 50  # number of decode iterations when phase==decode

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------------------------------------------------------
# Global state for compression timing (updated by helper functions)
# -----------------------------------------------------------------------------

_COMPRESS_MS: float = 0.0  # accumulated encode/decode time in milliseconds

def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to bytes using torch.save (no numpy dependency)."""
    buffer = io.BytesIO()
    torch.save(tensor.cpu(), buffer)
    return buffer.getvalue()


def _bytes_to_tensor(buf: bytes) -> torch.Tensor:
    buffer = io.BytesIO(buf)
    return torch.load(buffer)

def _compress_tensor(tensor):
    """Semantic-aware compression helper that tracks elapsed time (simulation)."""
    global _COMPRESS_MS  # noqa: PLW0603
    start_t = time.perf_counter()

    # Real compression path – always return `bytes`
    # Downcast float32→float16 to shrink, leave integers unchanged.
    if tensor.dtype == torch.float32:
        tensor = tensor.half()

    blob = zstd.compress(_tensor_to_bytes(tensor), level=3)
    _COMPRESS_MS += (time.perf_counter() - start_t) * 1000.0
    return blob

def _decompress_tensor(compressed_data):
    """Placeholder decompression that also records timing."""
    global _COMPRESS_MS  # noqa: PLW0603
    start_t = time.perf_counter()

    tensor = _bytes_to_tensor(zstd.decompress(compressed_data))
    # Restore to float32 for logits if originally half? Keep as float16 is fine.
    _COMPRESS_MS += (time.perf_counter() - start_t) * 1000.0
    return tensor

# Removed _get_compressed_size - no longer needed since we use real RPC counters

# -----------------------------------------------------------------------------
# Helper to collect RPC network metrics in a version-tolerant manner
# -----------------------------------------------------------------------------


def _collect_net_bytes() -> int:
    """Return total bytes sent+received by the current RPC agent.

    The exact API for accessing the agent has changed between PyTorch
    versions. This helper attempts a few strategies and falls back to `0` if
    metrics are unavailable.
    """

    # Strategy 1: official `torch.distributed.rpc.get_rpc_agent()` (>=2.1)
    agent_getter = getattr(rpc, "get_rpc_agent", None)
    if callable(agent_getter):
        try:
            metrics = agent_getter().get_metrics()  # type: ignore[attr-defined]

            # PyTorch versions label byte counters differently.  We therefore
            # sum *all* metric values whose key includes the substring
            # "bytes" (case-insensitive).  This remains correct even if new
            # counters are added in future versions.

            total = 0
            for k, v in metrics.items():  # type: ignore[assignment]
                if "bytes" in k.lower():
                    try:
                        total += int(v)
                    except (ValueError, TypeError):
                        continue
            if total > 0:
                print(f"[DEBUG] Strategy 1 found {total} bytes", file=sys.stderr)
                return total
        except Exception:  # pragma: no cover – best-effort only
            pass

    # Strategy 2: internal C binding (earlier versions)
    try:
        from torch._C._distributed_rpc import _get_current_rpc_agent  # type: ignore

        agent = _get_current_rpc_agent()
        if agent is not None and hasattr(agent, "get_metrics"):
            metrics = agent.get_metrics()  # type: ignore[attr-defined]
            # DEBUG: indicate that the fallback _get_current_rpc_agent path was taken
            print("[DEBUG] _collect_net_bytes: fallback metrics from _get_current_rpc_agent:", metrics, file=sys.stderr)
            sent = int(metrics.get("rpc.agent.sent_bytes", 0))  # type: ignore[arg-type]
            recv = int(metrics.get("rpc.agent.received_bytes", 0))  # type: ignore[arg-type]
            total = sent + recv
            print(f"[DEBUG] Strategy 2 found sent={sent}, recv={recv}, total={total} bytes", file=sys.stderr)
            return total
    except Exception as e:  # pragma: no cover
        print(f"[DEBUG] Strategy 2 failed: {e}", file=sys.stderr)
        pass

    # Strategy 3: Try to access agent stats via alternative method
    try:
        import torch.distributed.rpc as rpc_module
        agent = getattr(rpc_module, '_agent', None)
        if agent is not None and hasattr(agent, 'get_metrics'):
            metrics = agent.get_metrics()
            print(f"[DEBUG] Strategy 3 alternative agent metrics:", metrics, file=sys.stderr)
            sent = int(metrics.get("rpc.agent.sent_bytes", 0))
            recv = int(metrics.get("rpc.agent.received_bytes", 0))
            total = sent + recv
            if total > 0:
                print(f"[DEBUG] Strategy 3 found sent={sent}, recv={recv}, total={total} bytes", file=sys.stderr)
                return total
    except Exception as e:
        print(f"[DEBUG] Strategy 3 failed: {e}", file=sys.stderr)
        pass

    # Metrics not available – return 0 so downstream code still works.
    print("[DEBUG] No RPC metrics available, returning 0", file=sys.stderr)
    return 0


def _collect_rpc_time_ms() -> float:
    """Return total RPC time (ms) recorded by the current agent.

    Sums all metric counters whose key contains the substring "time" to stay
    version-agnostic.  Attempts both the public and internal agent access
    paths similar to _collect_net_bytes().
    """
    # Strategy 1 – official get_rpc_agent()
    agent_getter = getattr(rpc, "get_rpc_agent", None)
    if callable(agent_getter):
        try:
            metrics = agent_getter().get_metrics()  # type: ignore[attr-defined]
            total_us = 0
            for k, v in metrics.items():
                if "time" in k.lower():
                    try:
                        total_us += int(v)
                    except (ValueError, TypeError):
                        continue
            return total_us / 1000.0  # convert µs → ms
        except Exception:
            pass

    # Strategy 2 – internal binding
    try:
        from torch._C._distributed_rpc import _get_current_rpc_agent  # type: ignore

        agent = _get_current_rpc_agent()
        if agent is not None and hasattr(agent, "get_metrics"):
            metrics = agent.get_metrics()  # type: ignore[attr-defined]
            total_us = 0
            for k, v in metrics.items():  # type: ignore[assignment]
                if "time" in k.lower():
                    try:
                        total_us += int(v)
                    except (ValueError, TypeError):
                        continue
            return total_us / 1000.0
    except Exception:
        pass

    return 0.0

LOGGER = logging.getLogger(__name__)

def _parse_args(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Client orchestrator for LLM experiments")
    parser.add_argument("--mode", required=True, choices=["local", "naive", "remote_cache", "remote_cache_compressed", "remote_cache_delta_compressed", "remote_cache_delta_raw", "sys_simulated", "remote_cache_handle"])
    parser.add_argument("--phase", required=True, choices=["prefill", "decode"])
    parser.add_argument("--model", default="EleutherAI/gpt-j-6B")
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. " * 8)
    parser.add_argument("--gpu_host", default="127.0.0.1")
    parser.add_argument("--master_port", default="29500")
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    parser.add_argument("--skip_weight_upload", action="store_true", 
                        help="Skip actual weight upload and just simulate the network traffic")
    return parser.parse_args(argv)

def _shutdown_rpc():
    """Attempt a graceful RPC shutdown so the server releases rank 1.

    Falls back to a non-graceful shutdown if the graceful path hangs for more
    than ~10 s to keep the client bounded.
    """

    import threading

    exc: list[Exception | None] = [None]

    def _graceful():  # noqa: D401 – simple helper
        try:
            rpc.shutdown(graceful=True)
        except Exception as e:  # pragma: no cover – best-effort
            exc[0] = e

    # Proceed with shutdown attempt - let the graceful shutdown handle any errors

    th = threading.Thread(target=_graceful, daemon=True)
    th.start()
    th.join(timeout=10.0)

    if th.is_alive():
        print("Graceful RPC shutdown timed out; aborting…")
        try:
            rpc.shutdown(graceful=False)
        except Exception as e:
            print(f"Warning: Forced RPC shutdown failed: {e}")
    elif exc[0] is not None:
        print(f"Warning: Graceful RPC shutdown raised: {exc[0]}")


def _init_rpc(args):
    import time
    os.environ["MASTER_ADDR"] = args.gpu_host
    os.environ["MASTER_PORT"] = args.master_port
    
    # Add retry logic for RPC initialization
    for attempt in range(3):
        try:
            print(f"Initializing RPC connection to {args.gpu_host}:{args.master_port} (attempt {attempt + 1}/3)")

            client_name = f"CLIENT_WORKER_{uuid.uuid4().hex[:8]}"
            
            # Use RPC backend options with an extended timeout (30 min)
            opts = rpc.TensorPipeRpcBackendOptions(  # type: ignore[attr-defined]
                rpc_timeout=1800  # seconds
            )
            rpc.init_rpc(
                name=client_name,
                rank=1,
                world_size=2,
                rpc_backend_options=opts,
            )
            print("RPC connection established successfully")
            return
        except Exception as e:
            print(f"RPC initialization attempt {attempt + 1} failed: {e}")
            if attempt < 2:  # Not the last attempt
                time.sleep(2)
                continue
            else:
                raise RuntimeError(f"Failed to initialize RPC after 3 attempts: {e}")
    
    raise RuntimeError("Failed to initialize RPC")

def _run_local(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).half()
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        if args.phase == "decode":
            kv_cache = outputs.past_key_values
            for _ in range(5):
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                outputs = model(input_ids=next_token, past_key_values=kv_cache, use_cache=True)
                kv_cache = outputs.past_key_values
    print("NETWORK_BYTES: 0")
    print("AVG_SM_UTIL: 0.0")

def _run_naive_remote(args):
    _init_rpc(args)
    import rpc_server
    global _COMPRESS_MS  # noqa: PLW0603

    # Create a remote reference to a new RemoteWorker instance on the server.
    worker_rref = rpc.remote("GPU_WORKER", rpc_server.RemoteWorker, args=(args.model,))

    # Start GPU monitoring remotely
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.start_gpu_monitor_remote)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if args.skip_weight_upload:
        # Download weights once on the server (GPU side).
        _rpc_sync(worker_rref, rpc_server.RemoteWorker.download_weights_remote, args.model)

        # Ask the remote worker for the true model size for logging purposes.
        size_bytes = _rpc_sync(worker_rref, rpc_server.RemoteWorker.get_model_size_remote)
        LOGGER.info(
            "Simulated weight upload size: %.2f MB (NOT transferred due to --skip_weight_upload)",
            size_bytes / 1e6,
        )

        # Send an *empty* state_dict so that the remote call skips any re-load.
        state_dict_rpc: dict[str, torch.Tensor] = {}
    else:
        # Load model locally to get state_dict for upload.
        model = AutoModelForCausalLM.from_pretrained(args.model)
        state_dict_rpc = model.state_dict()
    
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    # Call the stateless remote forward. If --skip_weight_upload was set we
    # are sending an empty dict, otherwise the full weights.
    for _ in range(1 if args.phase == "prefill" else DECODE_STEPS):
        logits, kv_cache = _rpc_sync(
            worker_rref,
            rpc_server.RemoteWorker.run_stateless_forward_remote,
            state_dict_rpc,
            input_ids,
        )
        input_ids = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    # Get network bytes directly from the agent's metrics.
    net_bytes = _collect_net_bytes()

    # Retrieve detailed timing metrics from the server and local RPC agent
    timing_metrics = _rpc_sync(worker_rref, rpc_server.RemoteWorker.get_timing_metrics_remote)
    server_gpu_ms = timing_metrics.get("gpu_kernel_ms", 0.0)
    server_serdes_ms = timing_metrics.get("serdes_ms", 0.0)
    rpc_time_ms = _collect_rpc_time_ms()

    # Reset server-side timers for the next run
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.reset_metrics_remote)

    avg_sm = _rpc_sync(worker_rref, rpc_server.RemoteWorker.stop_gpu_monitor_remote)

    # Print detailed metrics before the standard summary lines
    print(f"SERVER_GPU_KERNEL_MS: {server_gpu_ms}")
    print(f"SERVER_SERDES_MS: {server_serdes_ms}")
    print(f"CLIENT_SERDES_MS: {_COMPRESS_MS}")
    print(f"RPC_TIME_MS: {rpc_time_ms}")

    # Reset client compression/decompression timer for next run
    _COMPRESS_MS = 0.0

    # IMPORTANT: clear any per-run KV caches so that subsequent trials do not
    # accumulate GPU memory and so that the RemoteWorker can be reused safely
    # across multiple independent RPC client connections within the same
    # long-running server process.
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.reset_state_remote)

    print(f"NETWORK_BYTES: {net_bytes}")
    print(f"AVG_SM_UTIL: {avg_sm}")
    _shutdown_rpc()

def _run_remote_cache(args):
    """Run in remote-cache mode (client sends tokens, gets logits + handle)."""
    _init_rpc(args)
    import rpc_server
    global _COMPRESS_MS  # noqa: PLW0603

    # Create a remote reference to a new RemoteWorker instance on the server.
    worker_rref = rpc.remote("GPU_WORKER", rpc_server.RemoteWorker, args=(args.model,))
    # Explicitly download weights on the remote worker.
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.download_weights_remote, args.model)

    # Start GPU monitoring remotely
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.start_gpu_monitor_remote)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ------------------------------------------------------------------
    # NEW: realistic *uncached* remote KV behaviour
    #       – Prefill returns the full KV cache tensors.
    #       – Every decode step sends the current token AND the entire KV
    #         cache back to the server (no compression).
    # This greatly increases network volume and lets sys_simulated showcase
    # its compression advantage.
    # ------------------------------------------------------------------

    if args.phase == "prefill":
        prompt = "Hello, my name is"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        print(f"Running prefill with {input_ids.shape[1]} tokens (returning KV cache)…")

        logits, kv_cache = _rpc_sync(
            worker_rref,
            rpc_server.RemoteWorker.run_prefill_with_cache_remote,
            input_ids,
        )
        # The returned kv_cache is large and stays on the *client* side, so
        # network bytes already include the full forward cache.
        print("Prefill complete – KV cache transferred to client.")

    elif args.phase == "decode":
        prefill_prompt = "Let's get a cache first."
        prefill_ids = tokenizer(prefill_prompt, return_tensors="pt").input_ids
        logits, kv_cache = _rpc_sync(
            worker_rref,
            rpc_server.RemoteWorker.run_prefill_with_cache_remote,
            prefill_ids,
        )

        print("Running decode – transferring KV cache every step (uncompressed)…")
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        for i in range(DECODE_STEPS):
            print(f"Decode step {i+1}…")
            logits, kv_cache = _rpc_sync(
                worker_rref,
                rpc_server.RemoteWorker.run_decode_step_with_cache_remote,
                next_token,
                kv_cache,
            )
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        print("Decode complete.")
    else:
        raise ValueError(f"Unknown phase for remote_cache mode: {args.phase}")

    net_bytes = _collect_net_bytes()

    timing_metrics = _rpc_sync(worker_rref, rpc_server.RemoteWorker.get_timing_metrics_remote)
    server_gpu_ms = timing_metrics.get("gpu_kernel_ms", 0.0)
    server_serdes_ms = timing_metrics.get("serdes_ms", 0.0)
    rpc_time_ms = _collect_rpc_time_ms()

    _rpc_sync(worker_rref, rpc_server.RemoteWorker.reset_metrics_remote)
    
    avg_sm = _rpc_sync(worker_rref, rpc_server.RemoteWorker.stop_gpu_monitor_remote)

    print(f"SERVER_GPU_KERNEL_MS: {server_gpu_ms}")
    print(f"SERVER_SERDES_MS: {server_serdes_ms}")
    print(f"CLIENT_SERDES_MS: {_COMPRESS_MS}")
    print(f"RPC_TIME_MS: {rpc_time_ms}")

    _COMPRESS_MS = 0.0

    # IMPORTANT: clear any per-run KV caches so that subsequent trials do not
    # accumulate GPU memory and so that the RemoteWorker can be reused safely
    # across multiple independent RPC client connections within the same
    # long-running server process.
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.reset_state_remote)

    print(f"NETWORK_BYTES: {net_bytes}")
    print(f"AVG_SM_UTIL: {avg_sm}")
    _shutdown_rpc()


def _run_remote_cache_compressed(args):
    """Run in compressed remote-cache mode (semantic-blind compression)."""
    _init_rpc(args)
    import rpc_server
    global _COMPRESS_MS  # noqa: PLW0603

    # Create a remote reference to a new RemoteWorker instance on the server.
    worker_rref = rpc.remote("GPU_WORKER", rpc_server.RemoteWorker, args=(args.model,))
    # Explicitly download weights on the remote worker.
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.download_weights_remote, args.model)

    # Start GPU monitoring remotely
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.start_gpu_monitor_remote)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.phase == "prefill":
        prompt = "Hello, my name is"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        print(f"Running prefill with {input_ids.shape[1]} tokens (returning compressed KV cache)…")

        logits, compressed_kv_cache = _rpc_sync(
            worker_rref,
            rpc_server.RemoteWorker.run_prefill_with_cache_compressed_remote,
            input_ids,
        )
        print("Prefill complete – compressed KV cache transferred to client.")

    elif args.phase == "decode":
        prefill_prompt = "Let's get a cache first."
        prefill_ids = tokenizer(prefill_prompt, return_tensors="pt").input_ids
        logits, compressed_kv_cache = _rpc_sync(
            worker_rref,
            rpc_server.RemoteWorker.run_prefill_with_cache_compressed_remote,
            prefill_ids,
        )

        print("Running decode – transferring compressed KV cache every step…")
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        for i in range(DECODE_STEPS):
            print(f"Decode step {i+1}…")
            logits, compressed_kv_cache = _rpc_sync(
                worker_rref,
                rpc_server.RemoteWorker.run_decode_step_with_cache_compressed_remote,
                next_token,
                compressed_kv_cache,
            )
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        print("Decode complete.")
    else:
        raise ValueError(f"Unknown phase for remote_cache_compressed mode: {args.phase}")

    net_bytes = _collect_net_bytes()

    timing_metrics = _rpc_sync(worker_rref, rpc_server.RemoteWorker.get_timing_metrics_remote)
    server_gpu_ms = timing_metrics.get("gpu_kernel_ms", 0.0)
    server_serdes_ms = timing_metrics.get("serdes_ms", 0.0)
    rpc_time_ms = _collect_rpc_time_ms()

    _rpc_sync(worker_rref, rpc_server.RemoteWorker.reset_metrics_remote)
    
    avg_sm = _rpc_sync(worker_rref, rpc_server.RemoteWorker.stop_gpu_monitor_remote)

    print(f"SERVER_GPU_KERNEL_MS: {server_gpu_ms}")
    print(f"SERVER_SERDES_MS: {server_serdes_ms}")
    print(f"CLIENT_SERDES_MS: {_COMPRESS_MS}")
    print(f"RPC_TIME_MS: {rpc_time_ms}")

    _COMPRESS_MS = 0.0

    # IMPORTANT: clear any per-run KV caches so that subsequent trials do not
    # accumulate GPU memory and so that the RemoteWorker can be reused safely
    # across multiple independent RPC client connections within the same
    # long-running server process.
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.reset_state_remote)

    print(f"NETWORK_BYTES: {net_bytes}")
    print(f"AVG_SM_UTIL: {avg_sm}")
    _shutdown_rpc()


def _run_sys_simulated(args):
    global _COMPRESS_MS
    _COMPRESS_MS = 0.0

    _init_rpc(args)
    import rpc_server

    # Create a remote reference to a new RemoteWorker instance on the server.
    worker_rref = rpc.remote("GPU_WORKER", rpc_server.RemoteWorker, args=(args.model,))
    # Explicitly download weights on the remote worker.
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.download_weights_remote, args.model)

    # Start GPU monitoring remotely
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.start_gpu_monitor_remote)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    # Prefill with real compression
    input_blob = _compress_tensor(input_ids)
    logits_blob, kv_id = _rpc_sync(
        worker_rref,
        rpc_server.RemoteWorker.run_prefill_semantic,
        input_blob,
    )
    logits = _decompress_tensor(logits_blob)

    if args.phase == "decode":
        for _ in range(DECODE_STEPS):
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            token_blob = _compress_tensor(next_token)
            logits_blob, kv_id = _rpc_sync(
                worker_rref,
                rpc_server.RemoteWorker.run_decode_step_semantic,
                token_blob,
                kv_id,
            )
            logits = _decompress_tensor(logits_blob)

    net_bytes = _collect_net_bytes()

    timing_metrics = _rpc_sync(worker_rref, rpc_server.RemoteWorker.get_timing_metrics_remote)
    server_gpu_ms = timing_metrics.get("gpu_kernel_ms", 0.0)
    server_serdes_ms = timing_metrics.get("serdes_ms", 0.0)
    rpc_time_ms = _collect_rpc_time_ms()

    _rpc_sync(worker_rref, rpc_server.RemoteWorker.reset_metrics_remote)
    
    avg_sm = _rpc_sync(worker_rref, rpc_server.RemoteWorker.stop_gpu_monitor_remote)

    print(f"SERVER_GPU_KERNEL_MS: {server_gpu_ms}")
    print(f"SERVER_SERDES_MS: {server_serdes_ms}")
    print(f"CLIENT_SERDES_MS: {_COMPRESS_MS}")
    print(f"RPC_TIME_MS: {rpc_time_ms}")

    # IMPORTANT: clear any per-run KV caches so that subsequent trials do not
    # accumulate GPU memory and so that the RemoteWorker can be reused safely
    # across multiple independent RPC client connections within the same
    # long-running server process.
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.reset_state_remote)

    print(f"NETWORK_BYTES: {net_bytes}")
    print(f"AVG_SM_UTIL: {avg_sm}")
    _shutdown_rpc()
    print(f"COMPRESS_MS: {_COMPRESS_MS:.2f}")

# -----------------------------------------------------------------------------
# Delta-KV helpers (semantic-aware remote cache)
# -----------------------------------------------------------------------------


def _apply_delta_to_kv_cache(kv_cache, delta_cache):
    """Append *delta_cache* (last-position slice) to *kv_cache* in-place."""
    for layer_idx, (layer_full, layer_delta) in enumerate(zip(kv_cache, delta_cache)):
        k_full, v_full = layer_full
        k_delta, v_delta = layer_delta
        kv_cache[layer_idx][0] = torch.cat([k_full, k_delta], dim=2)
        kv_cache[layer_idx][1] = torch.cat([v_full, v_delta], dim=2)
    return kv_cache


def _run_remote_cache_delta_raw(args):
    """Remote cache delta without compression (semantic-blind)."""
    _init_rpc(args)
    import rpc_server
    global _COMPRESS_MS

    worker_rref = rpc.remote("GPU_WORKER", rpc_server.RemoteWorker, args=(args.model,))
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.download_weights_remote, args.model)
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.start_gpu_monitor_remote)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.phase == "prefill":
        prompt = "Hello, my name is"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        print("Running prefill (full KV cache)…")
        logits, kv_cache = _rpc_sync(
            worker_rref,
            rpc_server.RemoteWorker.run_prefill_with_cache_delta_remote,
            input_ids,
        )
        print("Prefill complete – KV cache resident on client.")

    elif args.phase == "decode":
        prefill_ids = tokenizer("Cache first", return_tensors="pt").input_ids
        logits, kv_cache = _rpc_sync(
            worker_rref,
            rpc_server.RemoteWorker.run_prefill_with_cache_delta_remote,
            prefill_ids,
        )
        print("Running decode – transferring *delta* KV cache each step (raw)…")
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        for i in range(DECODE_STEPS):
            print(f"Decode step {i+1}…")
            logits, delta_cache = _rpc_sync(
                worker_rref,
                rpc_server.RemoteWorker.run_decode_step_with_cache_delta_raw_remote,
                next_token,
            )
            kv_cache = _apply_delta_to_kv_cache(kv_cache, delta_cache)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        print("Decode complete.")
    else:
        raise ValueError("Unknown phase for remote_cache_delta_raw")

    net_bytes = _collect_net_bytes()
    timing_metrics = _rpc_sync(worker_rref, rpc_server.RemoteWorker.get_timing_metrics_remote)
    server_gpu_ms = timing_metrics.get("gpu_kernel_ms", 0.0)
    server_serdes_ms = timing_metrics.get("serdes_ms", 0.0)
    rpc_time_ms = _collect_rpc_time_ms()
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.reset_metrics_remote)
    avg_sm = _rpc_sync(worker_rref, rpc_server.RemoteWorker.stop_gpu_monitor_remote)

    print(f"SERVER_GPU_KERNEL_MS: {server_gpu_ms}")
    print(f"SERVER_SERDES_MS: {server_serdes_ms}")
    print(f"CLIENT_SERDES_MS: {_COMPRESS_MS}")
    print(f"RPC_TIME_MS: {rpc_time_ms}")
    print(f"NETWORK_BYTES: {net_bytes}")
    print(f"AVG_SM_UTIL: {avg_sm}")
    _COMPRESS_MS = 0.0
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.reset_state_remote)
    _shutdown_rpc()

# -----------------------------------------------------------------------------
# Define compressed delta version (semantic-blind with zstd)

def _run_remote_cache_delta_compressed(args):
    """Remote cache delta with zstd compression (semantic-blind)."""
    _init_rpc(args)
    import rpc_server
    global _COMPRESS_MS

    worker_rref = rpc.remote("GPU_WORKER", rpc_server.RemoteWorker, args=(args.model,))
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.download_weights_remote, args.model)
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.start_gpu_monitor_remote)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.phase == "prefill":
        input_ids = tokenizer("Hello, my name is", return_tensors="pt").input_ids
        print("Running prefill (full KV cache)…")
        logits, kv_cache = _rpc_sync(
            worker_rref,
            rpc_server.RemoteWorker.run_prefill_with_cache_delta_remote,
            input_ids,
        )
        print("Prefill complete – KV cache resident on client.")

    elif args.phase == "decode":
        prefill_ids = tokenizer("Cache first", return_tensors="pt").input_ids
        logits, kv_cache = _rpc_sync(
            worker_rref,
            rpc_server.RemoteWorker.run_prefill_with_cache_delta_remote,
            prefill_ids,
        )
        print("Running decode – transferring *delta* KV cache each step (compressed)…")
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        for i in range(DECODE_STEPS):
            print(f"Decode step {i+1}…")
            logits, compressed_delta = _rpc_sync(
                worker_rref,
                rpc_server.RemoteWorker.run_decode_step_with_cache_delta_remote,
                next_token,
            )
            # Decompress delta and update local KV cache
            start_t = time.perf_counter()
            delta_cache = torch.load(io.BytesIO(zstd.decompress(compressed_delta)))
            _COMPRESS_MS += (time.perf_counter() - start_t) * 1000.0
            kv_cache = _apply_delta_to_kv_cache(kv_cache, delta_cache)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        print("Decode complete.")
    else:
        raise ValueError("Unknown phase for remote_cache_delta_compressed")

    net_bytes = _collect_net_bytes()
    timing_metrics = _rpc_sync(worker_rref, rpc_server.RemoteWorker.get_timing_metrics_remote)
    server_gpu_ms = timing_metrics.get("gpu_kernel_ms", 0.0)
    server_serdes_ms = timing_metrics.get("serdes_ms", 0.0)
    rpc_time_ms = _collect_rpc_time_ms()
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.reset_metrics_remote)
    avg_sm = _rpc_sync(worker_rref, rpc_server.RemoteWorker.stop_gpu_monitor_remote)

    print(f"SERVER_GPU_KERNEL_MS: {server_gpu_ms}")
    print(f"SERVER_SERDES_MS: {server_serdes_ms}")
    print(f"CLIENT_SERDES_MS: {_COMPRESS_MS}")
    print(f"RPC_TIME_MS: {rpc_time_ms}")
    print(f"NETWORK_BYTES: {net_bytes}")
    print(f"AVG_SM_UTIL: {avg_sm}")
    _COMPRESS_MS = 0.0
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.reset_state_remote)
    _shutdown_rpc()

# -----------------------------------------------------------------------------
# Handle-only semantic-aware mode (no compression, KV cache resident on GPU)
# -----------------------------------------------------------------------------

def _run_remote_cache_handle(args):
    """Semantic-aware mode: keep KV cache on GPU and exchange only a handle."""

    _init_rpc(args)
    import rpc_server

    # Create remote worker and ensure weights are present
    worker_rref = rpc.remote("GPU_WORKER", rpc_server.RemoteWorker, args=(args.model,))
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.download_weights_remote, args.model)

    # Start GPU monitoring
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.start_gpu_monitor_remote)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.phase == "prefill":
        input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
        print("Running prefill (handle-only)…")
        logits, kv_handle = _rpc_sync(
            worker_rref,
            rpc_server.RemoteWorker.run_prefill_with_handle,
            input_ids,
        )
    elif args.phase == "decode":
        # Warm-up prefill to obtain handle
        prefill_ids = tokenizer("Handle warm-up", return_tensors="pt").input_ids
        logits, kv_handle = _rpc_sync(
            worker_rref,
            rpc_server.RemoteWorker.run_prefill_with_handle,
            prefill_ids,
        )

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        for step in range(DECODE_STEPS):
            print(f"Decode step {step+1}…")
            logits = _rpc_sync(
                worker_rref,
                rpc_server.RemoteWorker.run_decode_with_handle,
                next_token,
                kv_handle,
            )
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    else:
        raise ValueError("Unknown phase for remote_cache_handle mode")

    # Metrics
    net_bytes = _collect_net_bytes()
    timing_metrics = _rpc_sync(worker_rref, rpc_server.RemoteWorker.get_timing_metrics_remote)
    server_gpu_ms = timing_metrics.get("gpu_kernel_ms", 0.0)
    server_serdes_ms = timing_metrics.get("serdes_ms", 0.0)
    rpc_time_ms = _collect_rpc_time_ms()

    _rpc_sync(worker_rref, rpc_server.RemoteWorker.reset_metrics_remote)
    avg_sm = _rpc_sync(worker_rref, rpc_server.RemoteWorker.stop_gpu_monitor_remote)

    print(f"SERVER_GPU_KERNEL_MS: {server_gpu_ms}")
    print(f"SERVER_SERDES_MS: {server_serdes_ms}")
    print(f"CLIENT_SERDES_MS: 0.0")
    print(f"RPC_TIME_MS: {rpc_time_ms}")
    print(f"NETWORK_BYTES: {net_bytes}")
    print(f"AVG_SM_UTIL: {avg_sm}")

    _shutdown_rpc()

# -----------------------------------------------------------------------------
# MAIN DISPATCH: update mapping (remove previous alias)
# -----------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    args = _parse_args(argv)
    if args.mode == "local":
        _run_local(args)
    elif args.mode == "naive":
        _run_naive_remote(args)
    elif args.mode == "remote_cache":
        _run_remote_cache(args)
    elif args.mode == "remote_cache_compressed":
        _run_remote_cache_compressed(args)
    elif args.mode == "remote_cache_delta_compressed":
        _run_remote_cache_delta_compressed(args)
    elif args.mode == "remote_cache_delta_raw":
        _run_remote_cache_delta_raw(args)
    elif args.mode == "sys_simulated":
        _run_sys_simulated(args)
    elif args.mode == "remote_cache_handle":
        _run_remote_cache_handle(args)
    else:
        LOGGER.error("Unknown mode %s", args.mode)
        sys.exit(1)

if __name__ == "__main__":
    main()