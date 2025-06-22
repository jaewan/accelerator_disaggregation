"""rpc_server.py
Unified RPC server for GPU-host logic in semantic-gap experiments.

Usage (on GPU_HOST):
    python rpc_server.py --model EleutherAI/gpt-j-6B --master_addr 0.0.0.0 --master_port 29500
The script starts a PyTorch RPC worker named "GPU_WORKER" and blocks until
interrupted (Ctrl+C or SIGTERM).
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
import uuid
from typing import Any, Dict, List, Tuple

import torch
from torch.distributed import rpc
from transformers import AutoConfig, AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)

# Global references populated on the server node after initialisation
_GLOBAL_WORKER: RemoteWorker | None = None  # type: ignore
_GLOBAL_WORKER_RREF: rpc.RRef | None = None  # type: ignore

# --------------------------------------------------------------------------------------
# Remote worker implementation
# --------------------------------------------------------------------------------------

class RemoteWorker:
    """Remote side logic that lives on GPU_HOST.

    It provides two categories of methods:
    1. *Stateless* (`run_stateless_forward_remote`) – used by the naive baseline.
    2. *Stateful / semantic-aware* (`load_weights_remote`, `run_prefill_remote`,
       `run_decode_step_remote`).
    """

    def __init__(self, model_name: str = "gpt2", dtype: str = "float16") -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        LOGGER.info("Initializing RemoteWorker on %s with model %s", self.device, model_name)

        # Create an *empty* model skeleton first so we can load weights later.
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_config(config).to(self.device)

        # Torch dtype handling
        if dtype == "float16" and self.device.type == "cuda":
            self.model = self.model.half()
        elif dtype == "bfloat16" and self.device.type == "cuda":
            self.model = self.model.to(dtype=torch.bfloat16)

        # Internal bookkeeping for semantic-aware mode
        self.weights_loaded: bool = False
        self.kv_cache_store: Dict[str, Any] = {}
        LOGGER.info("RemoteWorker ready (weights_loaded=%s)", self.weights_loaded)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _move_kv_cache(self, kv_cache: Any, device: torch.device) -> Any:
        """Recursively move past-key-values tensors to the specified device."""
        if kv_cache is None:
            return None
        moved: List[List[torch.Tensor]] = []
        for layer in kv_cache:
            moved.append([t.to(device=device, non_blocking=True) for t in layer])
        return moved

    # ------------------------------------------------------------------
    # Naive, stateless execution path
    # ------------------------------------------------------------------

    def run_stateless_forward_remote(
        self,
        state_dict: Dict[str, torch.Tensor],
        input_ids: torch.Tensor,
        kv_cache: Any = None,
    ) -> Tuple[torch.Tensor, Any]:
        """Load weights *every call*, run forward, and return logits + KV cache."""
        start = time.time()
        self.model.load_state_dict(state_dict, strict=False)
        load_t = time.time() - start

        input_ids = input_ids.to(self.device, non_blocking=True)
        kv_cache = self._move_kv_cache(kv_cache, self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, past_key_values=kv_cache, use_cache=True)

        logits = outputs.logits.cpu()
        pkv = self._move_kv_cache(outputs.past_key_values, torch.device("cpu"))
        LOGGER.debug("Stateless forward done (load_t=%.3fs)", load_t)
        return logits, pkv

    # ------------------------------------------------------------------
    # Semantic-aware execution path
    # ------------------------------------------------------------------

    def load_weights_remote(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load weights only once and keep them resident in GPU memory."""
        if not self.weights_loaded:
            self.model.load_state_dict(state_dict, strict=False)
            self.weights_loaded = True
            LOGGER.info("Weights loaded (%.2f MB)", sum(v.nbytes for v in state_dict.values()) / 1e6)
        else:
            LOGGER.debug("Weights already loaded; skipping reload.")

    def run_prefill_remote(self, token_ids: torch.Tensor):
        token_ids = token_ids.to(self.device, non_blocking=True)
        with torch.no_grad():
            outputs = self.model(input_ids=token_ids, use_cache=True)
        logits = outputs.logits.cpu()
        kv_cache_id = str(uuid.uuid4())
        self.kv_cache_store[kv_cache_id] = self._move_kv_cache(outputs.past_key_values, torch.device("cpu"))
        LOGGER.debug("Prefill complete; kv_cache_id=%s", kv_cache_id)
        return logits, kv_cache_id

    def run_decode_step_remote(self, token_id: torch.Tensor, kv_cache_id: str):
        if kv_cache_id not in self.kv_cache_store:
            raise RuntimeError(f"KV cache id {kv_cache_id} not found")
        token_id = token_id.to(self.device, non_blocking=True)
        kv_cache = self._move_kv_cache(self.kv_cache_store[kv_cache_id], self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=token_id, past_key_values=kv_cache, use_cache=True)
        logits = outputs.logits.cpu()
        # Update stored cache on CPU to minimise GPU memory footprint between calls.
        self.kv_cache_store[kv_cache_id] = self._move_kv_cache(outputs.past_key_values, torch.device("cpu"))
        return logits, kv_cache_id


# --------------------------------------------------------------------------------------
# RPC server bootstrap
# --------------------------------------------------------------------------------------

def _parse_args(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="RPC server for GPU_HOST")
    parser.add_argument("--model", default="gpt2", help="HF model name or path")
    parser.add_argument("--master_addr", default="127.0.0.1", help="RPC master address")
    parser.add_argument("--master_port", default="29500", help="RPC master port")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None):
    args = _parse_args(argv)

    # Environment variables required by torch RPC
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)

    if args.backend == "nccl" and not torch.cuda.is_available():
        LOGGER.warning("NCCL selected but CUDA not available; falling back to GLOO")
        args.backend = "gloo"

    LOGGER.info(
        "Starting RPC server (name=GPU_WORKER, backend=%s, master=%s:%s)",
        args.backend,
        args.master_addr,
        args.master_port,
    )

    # Create the global worker instance before RPC init so that other threads can access.
    global _GLOBAL_WORKER, _GLOBAL_WORKER_RREF  # noqa: PLW0603
    _GLOBAL_WORKER = RemoteWorker(model_name=args.model)

    rpc.init_rpc(
        name="GPU_WORKER",
        rank=args.rank,
        world_size=args.world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=16, rpc_timeout=300
        ),
    )

    # Now that RPC is ready, create RRef and expose getter.
    _GLOBAL_WORKER_RREF = rpc.RRef(_GLOBAL_WORKER)

    # Keep process alive until interrupted.
    def _handle_sigterm(signum, frame):  # noqa: D401
        LOGGER.info("Received signal %s; shutting down RPC.", signum)
        rpc.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    LOGGER.info("RPC server running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(24 * 3600)
    except KeyboardInterrupt:
        pass
    finally:
        rpc.shutdown()
        LOGGER.info("RPC server shut down.")


def get_worker_rref():  # noqa: D401
    """Return an RRef to the singleton RemoteWorker.

    Called remotely by clients. Will raise if the worker has not yet been
    initialised (i.e. before `main()` sets up the RPC server).
    """
    global _GLOBAL_WORKER_RREF  # noqa: PLW0603
    if _GLOBAL_WORKER_RREF is None:
        # Lazily create the RRef if RPC has been initialised but the variable
        # was not yet populated (possible race between init and first call).
        if _GLOBAL_WORKER is None:
            raise RuntimeError("RemoteWorker instance not created on this node")
        _GLOBAL_WORKER_RREF = rpc.RRef(_GLOBAL_WORKER)
    return _GLOBAL_WORKER_RREF


def run_stateless_forward(state_dict: Dict[str, torch.Tensor], input_ids: torch.Tensor, kv_cache: Any | None = None):
    """RPC wrapper to call stateless forward on the global worker and return outputs."""
    if _GLOBAL_WORKER is None:
        raise RuntimeError("Worker not initialised")
    return _GLOBAL_WORKER.run_stateless_forward_remote(state_dict, input_ids, kv_cache)


def load_weights(state_dict: Dict[str, torch.Tensor]):
    if _GLOBAL_WORKER is None:
        raise RuntimeError("Worker not initialised")
    _GLOBAL_WORKER.load_weights_remote(state_dict)


def run_prefill(token_ids: torch.Tensor):
    if _GLOBAL_WORKER is None:
        raise RuntimeError("Worker not initialised")
    return _GLOBAL_WORKER.run_prefill_remote(token_ids)


def run_decode_step(token_id: torch.Tensor, kv_cache_id: str):
    if _GLOBAL_WORKER is None:
        raise RuntimeError("Worker not initialised")
    return _GLOBAL_WORKER.run_decode_step_remote(token_id, kv_cache_id)


# Ensure this module is discoverable under the name 'rpc_server' even when executed
# as a script (i.e., module name is '__main__'). This allows remote peers that
# import by name to resolve the same module instance.
if __name__ != "rpc_server":
    sys.modules["rpc_server"] = sys.modules[__name__]


if __name__ == "__main__":
    main() 