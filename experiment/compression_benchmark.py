#!/usr/bin/env python3
"""compression_benchmark.py
Compare semantic-blind vs semantic-aware compression ratios for GPT-J KV cache with Zstandard.

Run on the client machine while rpc_server instances are running on the GPU host.
Example:
  python compression_benchmark.py --gpu_host 10.0.0.2 --model EleutherAI/gpt-j-6B --decode_steps 5
"""
from __future__ import annotations

import argparse, io, sys, uuid, zstandard as zstd  # type: ignore
from typing import List

import torch
from torch.distributed import rpc
from transformers import AutoTokenizer

from rpc_utils import rpc_sync_with_rref as _rpc_sync
import rpc_server  # noqa: F401 – imported by name so remote side can resolve symbols

# ---------------- helpers ----------------

def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO(); torch.save(t.cpu(), buf); return buf.getvalue()

def _compress_blind(obj):
    """Semantic-blind compression: Zstd on raw torch.save bytes."""
    buf = io.BytesIO(); torch.save(obj, buf)
    return zstd.compress(buf.getvalue(), level=3)

def _compress_semantic(tensor: torch.Tensor):
    """Semantic-aware: FP32→FP16 then Zstd."""
    if tensor.dtype == torch.float32:
        tensor = tensor.half()
    return zstd.compress(_tensor_to_bytes(tensor), level=3)

# ---------------- RPC bootstrap ----------------

def _init_rpc(args):
    import os
    os.environ["MASTER_ADDR"] = args.gpu_host
    os.environ["MASTER_PORT"] = args.master_port
    rpc.init_rpc(
        name=f"CLIENT_{uuid.uuid4().hex[:6]}",
        rank=1,
        world_size=2,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(rpc_timeout=1800),  # type: ignore[attr-defined]
    )

def _shutdown_rpc():
    try:
        rpc.shutdown(graceful=True)
    except Exception:
        rpc.shutdown(graceful=False)

# ---------------- main ----------------

def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--gpu_host", default="127.0.0.1")
    p.add_argument("--master_port", default="29500")
    p.add_argument("--model", default="EleutherAI/gpt-j-6B")
    p.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. ")
    p.add_argument("--decode_steps", type=int, default=0, help="Number of decode steps to benchmark as well")
    args = p.parse_args(argv)

    _init_rpc(args)
    try:
        worker_rref = rpc.remote("GPU_WORKER", rpc_server.RemoteWorker, args=(args.model,))
        _rpc_sync(worker_rref, rpc_server.RemoteWorker.download_weights_remote, args.model)
        tok = AutoTokenizer.from_pretrained(args.model)

        # Prefill
        ids = tok(args.prompt, return_tensors="pt").input_ids
        logits, kv_cache = _rpc_sync(worker_rref, rpc_server.RemoteWorker.run_prefill_with_cache_remote, ids)

        blind = _compress_blind(kv_cache)
        flat = torch.cat([t.flatten() for layer in kv_cache for t in layer])
        aware = _compress_semantic(flat)
        # What if remote-cache used compression too?
        compressed_cache = _compress_blind(kv_cache)
        print("=== Prefill KV cache (Zstd) ===")
        print(f"Raw KV cache:     {sum(t.numel() * t.element_size() for layer in kv_cache for t in layer)/1e6:7.2f} MB")
        print(f"Blind compressed: {len(blind)/1e6:7.2f} MB")
        print(f"Aware compressed: {len(aware)/1e6:7.2f} MB  (factor {len(blind)/len(aware):.1f}×)")
        print(f"Compression vs raw: {(sum(t.numel() * t.element_size() for layer in kv_cache for t in layer)/len(blind)):.1f}×")

        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        for step in range(args.decode_steps):
            logits, kv_cache = _rpc_sync(worker_rref, rpc_server.RemoteWorker.run_decode_step_with_cache_remote, next_tok, kv_cache)
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            blind = _compress_blind(kv_cache)
            flat = torch.cat([t.flatten() for layer in kv_cache for t in layer])
            aware = _compress_semantic(flat)
            raw_size = sum(t.numel() * t.element_size() for layer in kv_cache for t in layer)/1e6
            print(f"Step {step+1:02d}: raw {raw_size:6.2f} MB   blind {len(blind)/1e6:6.2f} MB   aware {len(aware)/1e6:6.2f} MB (factor {len(blind)/len(aware):.1f}×)")
    finally:
        _shutdown_rpc()

if __name__ == "__main__":
    main() 