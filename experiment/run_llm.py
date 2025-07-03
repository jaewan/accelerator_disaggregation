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
from typing import List

import torch
from torch.distributed import rpc
from transformers import AutoModelForCausalLM, AutoTokenizer
import zlib
import pickle
import io

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

def _init_rpc(args):
    os.environ["MASTER_ADDR"] = args.gpu_host
    os.environ["MASTER_PORT"] = args.master_port
    rpc.init_rpc(
        name="CLIENT_WORKER",
        rank=1,
        world_size=2,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=16, rpc_timeout=300),
    )

def _shutdown_rpc():
    try:
        rpc.shutdown(graceful=False)
    except RuntimeError as exc:
        if "RPC has not been initialized" not in str(exc):
            raise

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

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if args.skip_weight_upload:
        # Simulate weight upload by calculating size but not loading locally
        rpc.rpc_sync("GPU_WORKER", rpc_server.download_weights, args=(args.model,))
        
        # Create dummy state_dict for the stateless forward calls
        # In practice, this would be retrieved from the server
        import transformers
        config = transformers.AutoConfig.from_pretrained(args.model)
        model = transformers.AutoModelForCausalLM.from_config(config)
        state_dict = model.state_dict()
        
        # Calculate simulated upload size
        simulated_upload_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
        LOGGER.info("Simulated weight upload size: %.2f MB", simulated_upload_bytes / 1e6)
    else:
        # Original behavior: load model on client
        model = AutoModelForCausalLM.from_pretrained(args.model)
        state_dict = model.state_dict()
    
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    for _ in range(1 if args.phase == "prefill" else 6):
        logits, kv_cache = rpc.rpc_sync(
            "GPU_WORKER", rpc_server.run_stateless_forward, args=(state_dict, input_ids)
        )
        input_ids = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    # Get real network bytes from RPC agent
    net_bytes = rpc.rpc_sync("GPU_WORKER", rpc_server.get_rpc_bytes)
    
    _shutdown_rpc()
    print(f"NETWORK_BYTES: {net_bytes}")

def _run_remote_cache(args):
    _init_rpc(args)
    import rpc_server

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Download weights on the server side instead of uploading from client
    rpc.rpc_sync("GPU_WORKER", rpc_server.download_weights, args=(args.model,))

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    logits, kv_handle = rpc.rpc_sync("GPU_WORKER", rpc_server.run_prefill_with_handle, args=(input_ids,))

    if args.phase == "decode":
        for _ in range(5):
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            logits = rpc.rpc_sync(
                "GPU_WORKER", rpc_server.run_decode_with_handle, args=(next_token, kv_handle)
            )

    # Get real network bytes from RPC agent
    net_bytes = rpc.rpc_sync("GPU_WORKER", rpc_server.get_rpc_bytes)
    
    _shutdown_rpc()
    print(f"NETWORK_BYTES: {net_bytes}")

def _run_sys_simulated(args):
    global _COMPRESS_MS  # noqa: PLW0603
    _COMPRESS_MS = 0.0  # reset for this run

    _init_rpc(args)
    import rpc_server

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Download weights on the server side instead of uploading from client
    rpc.rpc_sync("GPU_WORKER", rpc_server.download_weights, args=(args.model,))

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    # Prefill with real compression
    input_blob = _compress_tensor(input_ids)
    logits_blob, kv_id = rpc.rpc_sync("GPU_WORKER", rpc_server.run_prefill_semantic, args=(input_blob,))

    logits = _decompress_tensor(logits_blob)

    if args.phase == "decode":
        for _ in range(5):
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            token_blob = _compress_tensor(next_token)
            logits_blob, kv_id = rpc.rpc_sync(
                "GPU_WORKER", rpc_server.run_decode_step_semantic, args=(token_blob, kv_id)
            )
            logits = _decompress_tensor(logits_blob)

    # Get real network bytes from RPC agent
    net_bytes = rpc.rpc_sync("GPU_WORKER", rpc_server.get_rpc_bytes)
    
    _shutdown_rpc()
    print(f"NETWORK_BYTES: {net_bytes}")
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