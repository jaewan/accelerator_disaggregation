"""run_llm.py
Client orchestration script for semantic-gap experiments.

Usage examples (on CLIENT_HOST):
    # LOCAL mode (baseline – run on GPU_HOST ideally)
    python run_llm.py --mode local --phase prefill --model sshleifer/tiny-gpt2

    # NAÏVE-REMOTE mode (run on CLIENT_HOST)
    python run_llm.py --mode naive --phase prefill --gpu_host 10.0.0.2 --model sshleifer/tiny-gpt2

    # SYS_SIMULATED mode (run on CLIENT_HOST)
    python run_llm.py --mode sys_simulated --phase decode --gpu_host 10.0.0.2 --model sshleifer/tiny-gpt2

Note: For real experiments, replace `sshleifer/tiny-gpt2` with
`EleutherAI/gpt-j-6B` (or the desired large model) and ensure the RPC server is
started on the GPU host with matching `--model`.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import torch  # type: ignore
from torch.distributed import rpc  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

# --------------------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------

def _parse_args(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Client orchestrator for LLM experiments")
    parser.add_argument("--mode", required=True, choices=["local", "naive", "sys_simulated"])
    parser.add_argument("--phase", required=True, choices=["prefill", "decode"])
    parser.add_argument("--model", default="sshleifer/tiny-gpt2", help="HF model name or path")
    parser.add_argument("--prompt", default="Hello, my dog is cute and", help="Prompt text")

    # Remote-specific args
    parser.add_argument("--gpu_host", default="127.0.0.1", help="Hostname/IP of GPU_HOST")
    parser.add_argument("--master_port", default="29500", help="RPC master port (same as server)")
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    return parser.parse_args(argv)

# --------------------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------------------

def _measure_tensor_bytes(*objects) -> int:
    """Calculate total bytes for tensors/objects being sent over RPC."""
    total_bytes = 0
    
    def _measure_single_object(obj) -> int:
        """Recursively measure a single object."""
        if obj is None:
            return 0
        elif hasattr(obj, 'nbytes'):  # torch.Tensor
            return obj.nbytes
        elif isinstance(obj, dict):  # state_dict
            return sum(_measure_single_object(v) for v in obj.values())
        elif isinstance(obj, (list, tuple)):  # KV cache or nested structures
            return sum(_measure_single_object(item) for item in obj)
        elif isinstance(obj, str):  # KV cache IDs
            return len(obj.encode('utf-8'))
        else:
            # For other types (int, float, etc.), assume negligible size
            return 0
    
    for obj in objects:
        total_bytes += _measure_single_object(obj)
    
    return total_bytes


def _init_rpc_for_client(args):
    os.environ["MASTER_ADDR"] = args.gpu_host
    os.environ["MASTER_PORT"] = str(args.master_port)

    if args.backend == "nccl" and not torch.cuda.is_available():
        LOGGER.warning("NCCL backend requested but CUDA not available; using GLOO")
        args.backend = "gloo"

    rpc.init_rpc(
        name="CLIENT_WORKER",
        rank=1,
        world_size=2,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=16, rpc_timeout=300),
    )


def _shutdown_rpc():
    """Attempt to cleanly shut down the RPC framework.

    The RPC API does not expose a stable, public "is_initialized" helper across
    PyTorch versions. Instead, we optimistically call ``rpc.shutdown()`` and
    silently ignore the *specific* RuntimeError that is raised when RPC has not
    been initialised in the current process. This makes the shutdown helper
    version-agnostic and idempotent.
    """
    try:
        # This will be a no-op if RPC was never initialised in this process.
        rpc.shutdown(graceful=False)
    except RuntimeError as exc:
        # Match the exact error string used by PyTorch to indicate the agent is
        # not set up. If a different error bubbles up, re-raise so we do not
        # hide genuine failures.
        if "RPC has not been initialized" not in str(exc):
            raise


# --------------------------------------------------------------------------------------
# Network measurement results container
# --------------------------------------------------------------------------------------

class NetworkMetrics:
    """Container for network transfer measurements."""
    def __init__(self):
        self.bytes_sent = 0
        self.bytes_received = 0
    
    @property
    def total_bytes(self) -> int:
        return self.bytes_sent + self.bytes_received


# --------------------------------------------------------------------------------------
# Execution paths
# --------------------------------------------------------------------------------------

def _run_local(args):
    LOGGER.info("Running LOCAL mode, phase=%s", args.phase)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    if device == "cuda":
        model = model.half()

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)

    if args.phase == "prefill":
        _run_prefill_local(model, input_ids)
    else:
        _run_decode_local(model, tokenizer, input_ids)
    
    # Local mode transfers 0 bytes over network
    print("NETWORK_BYTES: 0")


def _run_prefill_local(model, input_ids):
    with torch.no_grad():
        start = time.time()
        outputs = model(input_ids=input_ids, use_cache=True)
    elapsed = time.time() - start
    LOGGER.info("LOCAL prefill done in %.3f s, logits shape %s", elapsed, outputs.logits.shape)


def _run_decode_local(model, tokenizer, input_ids, steps: int = 5):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        logits = outputs.logits
        kv_cache = outputs.past_key_values

        for step in range(steps):
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            outputs = model(input_ids=next_token, past_key_values=kv_cache, use_cache=True)
            logits = outputs.logits
            kv_cache = outputs.past_key_values
    LOGGER.info("LOCAL decode %d steps done; final logits shape %s", steps, logits.shape)


# -----------------------------------------------------------------------------
# Remote helper wrappers
# -----------------------------------------------------------------------------

def _get_worker_rref():
    # Placeholder; this function name matches the server-side global we exposed.
    raise RuntimeError("This function should never be executed on the client.")


def _run_naive_remote(args):
    LOGGER.info("Running NAIVE-REMOTE mode, phase=%s", args.phase)
    metrics = NetworkMetrics()

    # Initialize RPC.
    _init_rpc_for_client(args)

    # Obtain RemoteWorker RRef (function is looked up on the *remote* module).
    import rpc_server  # noqa: WPS433

    LOGGER.info("Connected to GPU_WORKER; using wrapper RPC functions.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    state_dict = model.state_dict()

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    if args.phase == "prefill":
        # Measure bytes sent
        bytes_sent = _measure_tensor_bytes(state_dict, input_ids)
        metrics.bytes_sent += bytes_sent
        
        logits, _ = rpc.rpc_sync("GPU_WORKER", rpc_server.run_stateless_forward, args=(state_dict, input_ids))
        
        # Measure bytes received
        bytes_received = _measure_tensor_bytes(logits)
        metrics.bytes_received += bytes_received
        
        LOGGER.info("Remote prefill logits shape %s", logits.shape)
    else:
        # Initial prefill call
        bytes_sent = _measure_tensor_bytes(state_dict, input_ids)
        metrics.bytes_sent += bytes_sent
        
        logits, kv_cache = rpc.rpc_sync("GPU_WORKER", rpc_server.run_stateless_forward, args=(state_dict, input_ids))
        
        bytes_received = _measure_tensor_bytes(logits, kv_cache)
        metrics.bytes_received += bytes_received
        
        # Decode steps
        for step in range(5):
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
            # Measure bytes sent (full state_dict + token + KV cache every time)
            bytes_sent = _measure_tensor_bytes(state_dict, next_token, kv_cache)
            metrics.bytes_sent += bytes_sent
            
            logits, kv_cache = rpc.rpc_sync(
                "GPU_WORKER", rpc_server.run_stateless_forward, args=(state_dict, next_token, kv_cache)
            )
            
            # Measure bytes received
            bytes_received = _measure_tensor_bytes(logits, kv_cache)
            metrics.bytes_received += bytes_received
            
        LOGGER.info("Remote decode complete; final logits shape %s", logits.shape)

    _shutdown_rpc()
    
    # Output network bytes for experiment driver to capture
    print(f"NETWORK_BYTES: {metrics.total_bytes}")


def _run_sys_simulated(args):
    LOGGER.info("Running SYS_SIMULATED mode, phase=%s", args.phase)
    metrics = NetworkMetrics()

    _init_rpc_for_client(args)
    import rpc_server  # noqa: WPS433

    LOGGER.info("Connected to GPU_WORKER for sys_simulated mode.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    state_dict = model.state_dict()

    # Load weights on the remote worker exactly once.
    bytes_sent = _measure_tensor_bytes(state_dict)
    metrics.bytes_sent += bytes_sent
    
    rpc.rpc_sync("GPU_WORKER", rpc_server.load_weights, args=(state_dict,))

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    if args.phase == "prefill":
        # Only send input_ids (weights already loaded)
        bytes_sent = _measure_tensor_bytes(input_ids)
        metrics.bytes_sent += bytes_sent
        
        logits, kv_id = rpc.rpc_sync("GPU_WORKER", rpc_server.run_prefill, args=(input_ids,))
        
        # Receive logits and KV cache ID (just a string)
        bytes_received = _measure_tensor_bytes(logits, kv_id)
        metrics.bytes_received += bytes_received
        
        LOGGER.info("SYS_SIM prefill logits %s, kv_id %s", logits.shape, kv_id)
    else:
        # Initial prefill
        bytes_sent = _measure_tensor_bytes(input_ids)
        metrics.bytes_sent += bytes_sent
        
        logits, kv_id = rpc.rpc_sync("GPU_WORKER", rpc_server.run_prefill, args=(input_ids,))
        
        bytes_received = _measure_tensor_bytes(logits, kv_id)
        metrics.bytes_received += bytes_received
        
        # Decode steps - only send new token + KV cache ID
        for step in range(5):
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
            # Only send token and cache ID (much smaller than naive mode)
            bytes_sent = _measure_tensor_bytes(next_token, kv_id)
            metrics.bytes_sent += bytes_sent
            
            logits, kv_id = rpc.rpc_sync(
                "GPU_WORKER", rpc_server.run_decode_step, args=(next_token, kv_id)
            )
            
            # Receive logits and same KV cache ID
            bytes_received = _measure_tensor_bytes(logits, kv_id)
            metrics.bytes_received += bytes_received
            
        LOGGER.info("SYS_SIM decode complete; final logits shape %s", logits.shape)

    _shutdown_rpc()
    
    # Output network bytes for experiment driver to capture
    print(f"NETWORK_BYTES: {metrics.total_bytes}")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    args = _parse_args(argv)

    if args.mode == "local":
        _run_local(args)
    elif args.mode == "naive":
        _run_naive_remote(args)
    elif args.mode == "sys_simulated":
        _run_sys_simulated(args)
    else:
        LOGGER.error("Unknown mode %s", args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()