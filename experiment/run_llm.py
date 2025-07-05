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

import torch
from torch.distributed import rpc
from rpc_utils import rpc_sync_with_rref as _rpc_sync
from transformers import AutoModelForCausalLM, AutoTokenizer
import zlib
import pickle
import io

# ----------------------------------------------------------------------------------
# Note: RPC helper utilities now live in ``experiment.rpc_utils`` so that they
# are importable under the same module name on *all* RPC workers. This avoids
# pickling errors where a function defined in ``__main__`` cannot be resolved
# on the remote side.
# ----------------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------------------------------------------------------
# Global state for compression timing (updated by helper functions)
# -----------------------------------------------------------------------------

_COMPRESS_MS: float = 0.0  # accumulated encode/decode time in milliseconds

def _tensor_to_numpy_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor (CPU) to bytes using pickle for simplicity."""
    return pickle.dumps(tensor.cpu().numpy(), protocol=pickle.HIGHEST_PROTOCOL)

def _numpy_bytes_to_tensor(buf: bytes) -> torch.Tensor:
    arr = pickle.loads(buf)
    return torch.from_numpy(arr)

def _compress_tensor(tensor):
    """Semantic-aware compression helper that tracks elapsed time (simulation)."""
    global _COMPRESS_MS  # noqa: PLW0603
    start_t = time.perf_counter()

    # Real compression path – always return `bytes`
    # Downcast float32→float16 to shrink, leave integers unchanged.
    if tensor.dtype == torch.float32:
        tensor = tensor.half()

    blob = zlib.compress(_tensor_to_numpy_bytes(tensor), level=6)
    _COMPRESS_MS += (time.perf_counter() - start_t) * 1000.0
    return blob

def _decompress_tensor(compressed_data):
    """Placeholder decompression that also records timing."""
    global _COMPRESS_MS  # noqa: PLW0603
    start_t = time.perf_counter()

    tensor = _numpy_bytes_to_tensor(zlib.decompress(compressed_data))
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
            # Some torch builds return str values; best-effort cast to int.
            sent = int(metrics.get("rpc.agent.sent_bytes", 0))  # type: ignore[arg-type]
            recv = int(metrics.get("rpc.agent.received_bytes", 0))  # type: ignore[arg-type]
            return sent + recv
        except Exception:  # pragma: no cover – best-effort only
            pass

    # Strategy 2: internal C binding (earlier versions)
    try:
        from torch._C._distributed_rpc import _get_current_rpc_agent  # type: ignore

        agent = _get_current_rpc_agent()
        if agent is not None and hasattr(agent, "get_metrics"):
            metrics = agent.get_metrics()  # type: ignore[attr-defined]
            sent = int(metrics.get("rpc.agent.sent_bytes", 0))  # type: ignore[arg-type]
            recv = int(metrics.get("rpc.agent.received_bytes", 0))  # type: ignore[arg-type]
            return sent + recv
    except Exception:  # pragma: no cover
        pass

    # Metrics not available – return 0 so downstream code still works.
    return 0

LOGGER = logging.getLogger(__name__)

def _parse_args(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Client orchestrator for LLM experiments")
    parser.add_argument("--mode", required=True, choices=["local", "naive", "remote_cache", "sys_simulated"])
    parser.add_argument("--phase", required=True, choices=["prefill", "decode"])
    parser.add_argument("--model", default="EleutherAI/gpt-j-6B")
    parser.add_argument("--prompt", default="Hello, my dog is cute and")
    parser.add_argument("--gpu_host", default="127.0.0.1")
    parser.add_argument("--master_port", default="29500")
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    parser.add_argument("--skip_weight_upload", action="store_true", 
                        help="Skip actual weight upload and just simulate the network traffic")
    return parser.parse_args(argv)

def _shutdown_rpc():
    """Shutdown RPC connection quickly (non-graceful) to avoid hangs.

    Using ``graceful=False`` aborts outstanding work instead of waiting for all
    RRefs to be released, which is acceptable for short-lived benchmark
    clients.
    """
    try:
        rpc.shutdown(graceful=False)
    except Exception as e:
        print(f"Warning: RPC shutdown failed: {e}")


def _init_rpc(args):
    import time
    os.environ["MASTER_ADDR"] = args.gpu_host
    os.environ["MASTER_PORT"] = args.master_port
    
    # Add retry logic for RPC initialization
    for attempt in range(3):
        try:
            print(f"Initializing RPC connection to {args.gpu_host}:{args.master_port} (attempt {attempt + 1}/3)")
            
            # Use default RPC backend options
            rpc.init_rpc(
                name="CLIENT_WORKER",
                rank=1,
                world_size=2,
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

def _run_naive_remote(args):
    _init_rpc(args)
    import rpc_server

    # Create a remote reference to a new RemoteWorker instance on the server.
    worker_rref = rpc.remote("GPU_WORKER", rpc_server.RemoteWorker, args=(args.model,))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if args.skip_weight_upload:
        # In the new model, weights are downloaded directly on the server.
        # We can call the download method on our remote worker instance.
        _rpc_sync(worker_rref, rpc_server.RemoteWorker.download_weights_remote, args.model)
        
        # Create a dummy state_dict for size calculation, as before.
        import transformers
        config = transformers.AutoConfig.from_pretrained(args.model)
        with torch.no_grad():
            model = transformers.AutoModelForCausalLM.from_config(config)
        state_dict = model.state_dict()
        
        simulated_upload_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
        LOGGER.info("Simulated weight upload size: %.2f MB", simulated_upload_bytes / 1e6)
    else:
        # Load model locally to get state_dict for upload.
        model = AutoModelForCausalLM.from_pretrained(args.model)
        state_dict = model.state_dict()
    
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    # In naive mode, we still pass the state_dict in every call.
    # The remote method is now called on the worker_rref instance.
    for _ in range(1 if args.phase == "prefill" else 6):
        logits, kv_cache = _rpc_sync(
            worker_rref,
            rpc_server.RemoteWorker.run_stateless_forward_remote,
            state_dict,
            input_ids,
        )
        input_ids = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    # Get network bytes directly from the agent's metrics.
    net_bytes = _collect_net_bytes()
    
    print(f"NETWORK_BYTES: {net_bytes}")
    _shutdown_rpc()

def _run_remote_cache(args):
    """Run in remote-cache mode (client sends tokens, gets logits + handle)."""
    _init_rpc(args)
    import rpc_server

    # Create a remote reference to a new RemoteWorker instance on the server.
    worker_rref = rpc.remote("GPU_WORKER", rpc_server.RemoteWorker, args=(args.model,))
    # Explicitly download weights on the remote worker.
    _rpc_sync(worker_rref, rpc_server.RemoteWorker.download_weights_remote, args.model)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if args.phase == "prefill":
        prompt = "Hello, my name is"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        print(f"Running prefill with {input_ids.shape[1]} tokens...")
        
        # Call the method on the remote worker instance.
        logits, kv_handle = _rpc_sync(
            worker_rref,
            rpc_server.RemoteWorker.run_prefill_with_handle,
            input_ids,
        )
        print("Prefill complete.")

    elif args.phase == "decode":
        prefill_prompt = "Let's get a handle first."
        prefill_ids = tokenizer(prefill_prompt, return_tensors="pt").input_ids
        logits, kv_handle = _rpc_sync(
            worker_rref,
            rpc_server.RemoteWorker.run_prefill_with_handle,
            prefill_ids,
        )
        
        print(f"Running decode with handle: {kv_handle}")
        decode_ids = tokenizer("dummy", return_tensors="pt").input_ids
        for i in range(5):
            print(f"Decode step {i+1}...")
            logits = _rpc_sync(
                worker_rref,
                rpc_server.RemoteWorker.run_decode_with_handle,
                decode_ids[:, -1:],
                kv_handle,
            )
        print("Decode complete.")
    else:
        raise ValueError(f"Unknown phase for remote_cache mode: {args.phase}")

    net_bytes = _collect_net_bytes()
    
    print(f"NETWORK_BYTES: {net_bytes}")
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
        for _ in range(5):
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
    
    print(f"NETWORK_BYTES: {net_bytes}")
    _shutdown_rpc()
    print(f"COMPRESS_MS: {_COMPRESS_MS:.2f}")

def main(argv: List[str] | None = None):
    args = _parse_args(argv)
    if args.mode == "local":
        _run_local(args)
    elif args.mode == "naive":
        _run_naive_remote(args)
    elif args.mode == "remote_cache":
        _run_remote_cache(args)
    elif args.mode == "sys_simulated":
        _run_sys_simulated(args)
    else:
        LOGGER.error("Unknown mode %s", args.mode)
        sys.exit(1)

if __name__ == "__main__":
    main()