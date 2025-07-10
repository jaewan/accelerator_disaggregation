#!/usr/bin/env python3
"""run_local_baseline.py
Run a *local* execution baseline on the same GPU server that hosts the RPC
experiments.  This script loads the model once on the local GPU/CPU, runs a
prefill followed by an arbitrary number of decode steps, and prints metrics in
a format compatible with the other experiment drivers (latency, GPU-kernel
milliseconds, network bytes, etc.).

Typical usage (on GPU host):

    python run_local_baseline.py \
        --model EleutherAI/gpt-j-6B \
        --prompt "The quick brown fox jumps over the lazy dog. " \
        --decode_steps 250

The output lines can be appended to *result.csv* manually or fed into a small
parser if you want full automation.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import List, Tuple, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _timed_forward(model, input_ids, **kwargs) -> Tuple[Any, float]:  # type: ignore[return-value]
    """Run *model* once and return (logits, elapsed_ms_on_gpu)."""
    if torch.cuda.is_available():
        start_ev = torch.cuda.Event(enable_timing=True)  # type: ignore[arg-type]
        end_ev = torch.cuda.Event(enable_timing=True)  # type: ignore[arg-type]
        start_ev.record(torch.cuda.current_stream())

    outputs = model(input_ids=input_ids, **kwargs)  # type: ignore[operator]

    if torch.cuda.is_available():
        end_ev.record(torch.cuda.current_stream())
        torch.cuda.synchronize()
        kernel_ms = start_ev.elapsed_time(end_ev)
    else:
        kernel_ms = 0.0
    return outputs, kernel_ms


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def _parse_args(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="Local execution baseline for semantic-gap experiments")
    p.add_argument("--model", default="EleutherAI/gpt-j-6B")
    p.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. " * 8)
    p.add_argument("--decode_steps", type=int, default=50, help="Number of autoregressive decode steps")
    p.add_argument("--no_half", action="store_true", help="Load model in full fp32 instead of fp16/bfloat16")
    return p.parse_args(argv)


def main(argv: List[str] | None = None):
    args = _parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading model '{args.model}' on {device} …", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    if device == "cuda":
        # Use fp16 by default to match RPC server settings and save memory.
        if not args.no_half:
            model = model.half()  # type: ignore[assignment]
        model = model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Phase 1: Prefill (prompt → initial KV cache)
    # ------------------------------------------------------------------
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)

    kernel_ms_prefill = 0.0
    wall_prefill_start = time.perf_counter()

    with torch.no_grad():
        outputs, ms = _timed_forward(model, input_ids, use_cache=True)
        kernel_ms_prefill += ms
        kv_cache = outputs.past_key_values  # type: ignore[attr-defined]

    wall_prefill = time.perf_counter() - wall_prefill_start

    util_prefill = 0.0
    if device == "cuda" and wall_prefill > 0:
        util_prefill = min(100.0, 100.0 * kernel_ms_prefill / (wall_prefill * 1000.0))

    print("# ---- PREFILL ----")
    print(f"GPU_KERNEL_MS: {kernel_ms_prefill}")
    print("SERDES_MS: 0.0")
    print("RPC_TIME_MS: 0.0")
    print("CLIENT_SERDES_MS: 0.0")
    print("SERVER_SERDES_MS: 0.0")
    print("NETWORK_BYTES: 0")
    print(f"AVG_SM_UTIL: {util_prefill}")
    print(f"{wall_prefill}")

    # ------------------------------------------------------------------
    # Phase 2: Decode – autoregressive generation of *decode_steps* tokens
    #           using the KV cache from prefill.
    # ------------------------------------------------------------------
    kernel_ms_decode = 0.0
    wall_decode_start = time.perf_counter()

    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)  # type: ignore[attr-defined]
    with torch.no_grad():
        for _ in range(args.decode_steps):
            outputs, ms = _timed_forward(
                model,
                next_token,
                past_key_values=kv_cache,
                use_cache=True,
            )
            kernel_ms_decode += ms
            kv_cache = outputs.past_key_values  # type: ignore[attr-defined]
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)  # type: ignore[attr-defined]

    wall_decode = time.perf_counter() - wall_decode_start

    util_decode = 0.0
    if device == "cuda" and wall_decode > 0:
        util_decode = min(100.0, 100.0 * kernel_ms_decode / (wall_decode * 1000.0))

    print("# ---- DECODE ----")
    print(f"GPU_KERNEL_MS: {kernel_ms_decode}")
    print("SERDES_MS: 0.0")
    print("RPC_TIME_MS: 0.0")
    print("CLIENT_SERDES_MS: 0.0")
    print("SERVER_SERDES_MS: 0.0")
    print("NETWORK_BYTES: 0")
    print(f"AVG_SM_UTIL: {util_decode}")
    print(f"{wall_decode}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130) 