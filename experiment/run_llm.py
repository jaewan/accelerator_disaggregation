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

    # Initialize RPC.
    _init_rpc_for_client(args)

    # Obtain RemoteWorker RRef (function is looked up on the *remote* module).
    import rpc_server  # noqa: WPS433

    worker_rref = rpc.rpc_sync("GPU_WORKER", rpc_server.get_worker_rref)
    LOGGER.info("Connected to GPU_WORKER; using RRef for RPC.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    state_dict = model.state_dict()

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    if args.phase == "prefill":
        logits, _ = worker_rref.rpc_sync().run_stateless_forward_remote(state_dict, input_ids)
        LOGGER.info("Remote prefill logits shape %s", logits.shape)
    else:
        logits, kv_cache = worker_rref.rpc_sync().run_stateless_forward_remote(state_dict, input_ids)
        for step in range(5):
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            logits, kv_cache = worker_rref.rpc_sync().run_stateless_forward_remote(state_dict, next_token, kv_cache)
        LOGGER.info("Remote decode complete; final logits shape %s", logits.shape)

    _shutdown_rpc()


def _run_sys_simulated(args):
    LOGGER.info("Running SYS_SIMULATED mode, phase=%s", args.phase)

    _init_rpc_for_client(args)
    import rpc_server  # noqa: WPS433

    worker_rref = rpc.rpc_sync("GPU_WORKER", rpc_server.get_worker_rref)
    LOGGER.info("Connected to GPU_WORKER; using RRef for RPC.")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    state_dict = model.state_dict()

    LOGGER.info("Connected to GPU_WORKER for sys_simulated mode (loading weights once).")
    # Load weights on the remote worker exactly once.
    worker_rref.rpc_sync().load_weights_remote(state_dict)

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    if args.phase == "prefill":
        logits, kv_id = worker_rref.rpc_sync().run_prefill_remote(input_ids)
        LOGGER.info("SYS_SIM prefill logits %s, kv_id %s", logits.shape, kv_id)
    else:
        logits, kv_id = worker_rref.rpc_sync().run_prefill_remote(input_ids)
        for step in range(5):
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            logits, kv_id = worker_rref.rpc_sync().run_decode_step_remote(next_token, kv_id)
        LOGGER.info("SYS_SIM decode complete; final logits shape %s", logits.shape)

    _shutdown_rpc()


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