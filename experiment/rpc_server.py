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
import zlib
import pickle

import torch  # type: ignore
from torch.distributed import rpc  # type: ignore
from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


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
        
        # Create model and immediately move to device with appropriate dtype
        with torch.no_grad():  # Disable gradient computation during initialization
            self.model = AutoModelForCausalLM.from_config(config)
            
            # Convert to half precision immediately for large models to save memory
            if dtype == "float16" and device == "cuda":
                self.model = self.model.half()
            elif dtype == "bfloat16" and device == "cuda":
                self.model = self.model.to(dtype=torch.bfloat16)
                
            # Move to device after dtype conversion
            self.model = self.model.to(self.device)

        # Internal bookkeeping for semantic-aware mode
        self.weights_loaded: bool = False
        self.kv_cache_store: Dict[str, Any] = {}
        LOGGER.info("RemoteWorker ready (weights_loaded=%s)", self.weights_loaded)

    def get_rpc_bytes(self) -> int:
        """Return total bytes (sent + received) from the RPC agent."""
        # This is a placeholder. Real implementation would query the agent.
        # Note: Accessing internal agent metrics can be unstable across versions.
        # For this experiment, we will rely on values returned by the client call.
        return 0

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

    def _compress_tensor(self, tensor):
        """Semantic-aware compression: compress tensor using knowledge of structure"""
        if tensor.dtype != torch.float16:
            tensor_half = tensor.half()
        else:
            tensor_half = tensor
        
        tensor_bytes = pickle.dumps(tensor_half.cpu().numpy())
        compressed = zlib.compress(tensor_bytes, level=6)
        return compressed

    def _decompress_tensor(self, compressed_data):
        """Decompress tensor data"""
        decompressed = zlib.decompress(compressed_data)
        tensor_np = pickle.loads(decompressed)
        return torch.from_numpy(tensor_np).to(self.device)

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

    def download_weights_remote(self, model_name: str) -> None:
        """Download weights from Hugging Face and keep them resident in GPU memory."""
        if not self.weights_loaded:
            LOGGER.info("Downloading weights for model %s", model_name)
            full_model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Convert to the same dtype as our model
            if self.model.dtype == torch.float16:
                full_model = full_model.half()
            elif self.model.dtype == torch.bfloat16:
                full_model = full_model.to(dtype=torch.bfloat16)
            
            # Load the weights into our model
            self.model.load_state_dict(full_model.state_dict(), strict=False)
            self.weights_loaded = True
            
            # Calculate size for logging
            total_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e6
            LOGGER.info("Weights downloaded and loaded (%.2f MB)", total_size)
        else:
            LOGGER.debug("Weights already loaded; skipping download.")

    def reset_state_remote(self) -> None:
        """Reset KV cache state between trials while keeping weights loaded."""
        self.kv_cache_store.clear()
        LOGGER.debug("KV cache state reset (weights preserved)")

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

    def run_prefill_with_handle(self, input_ids: torch.Tensor):
        """Realistic semantic-blind caching: KV cache stays resident remotely."""
        input_ids = input_ids.to(self.device, non_blocking=True)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=True)
        logits = outputs.logits.cpu()
        kv_handle = str(uuid.uuid4())
        # Keep KV cache on GPU for realistic semantic-blind caching baseline
        self.kv_cache_store[kv_handle] = outputs.past_key_values
        LOGGER.debug("Prefill with handle complete; kv_handle=%s (KV cache stays on GPU)", kv_handle)
        return logits, kv_handle

    def run_prefill_with_cache_remote(self, token_ids: torch.Tensor):
        """Run prefill and return actual KV cache tensors (for remote cache baseline)."""
        token_ids = token_ids.to(self.device, non_blocking=True)
        with torch.no_grad():
            outputs = self.model(input_ids=token_ids, use_cache=True)
        logits = outputs.logits.cpu()
        kv_cache = self._move_kv_cache(outputs.past_key_values, torch.device("cpu"))
        LOGGER.debug("Prefill with cache complete; returning actual KV cache tensors")
        return logits, kv_cache

    def run_decode_with_handle(self, token_id: torch.Tensor, kv_handle: str):
        """Realistic semantic-blind caching: KV cache stays resident remotely."""
        if kv_handle not in self.kv_cache_store:
            raise RuntimeError(f"KV cache handle {kv_handle} not found")
        token_id = token_id.to(self.device, non_blocking=True)
        kv_cache = self.kv_cache_store[kv_handle]  # Already on GPU, no transfer needed

        with torch.no_grad():
            outputs = self.model(input_ids=token_id, past_key_values=kv_cache, use_cache=True)
        logits = outputs.logits.cpu()
        # Update KV cache on GPU - stays resident remotely
        self.kv_cache_store[kv_handle] = outputs.past_key_values
        LOGGER.debug("Decode with handle complete; kv_handle=%s (KV cache stays on GPU)", kv_handle)
        return logits

    def run_decode_step_with_cache_remote(self, token_id: torch.Tensor, kv_cache: Any):
        """Run decode step with actual KV cache tensors (for remote cache baseline)."""
        token_id = token_id.to(self.device, non_blocking=True)
        kv_cache = self._move_kv_cache(kv_cache, self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=token_id, past_key_values=kv_cache, use_cache=True)
        logits = outputs.logits.cpu()
        updated_kv_cache = self._move_kv_cache(outputs.past_key_values, torch.device("cpu"))
        return logits, updated_kv_cache

    # ------------------------------------------------------------------
    # Semantic-aware execution path with compression
    # ------------------------------------------------------------------

    def run_prefill_semantic(self, token_blob: bytes):
        """Semantic-aware prefill with real compression pipeline.

        Parameters
        ----------
        token_blob : bytes
            Compressed token tensor produced by client _compress_tensor.
        Returns
        -------
        Tuple[bytes, str]
            Compressed logits and kv_cache_id string.
        """
        token_ids: torch.Tensor = self._decompress_tensor(token_blob)
        token_ids = token_ids.to(self.device, non_blocking=True)

        with torch.no_grad():
            outputs = self.model(input_ids=token_ids, use_cache=True)

        logits = outputs.logits.cpu()
        logits_blob = self._compress_tensor(logits)

        kv_cache_id = str(uuid.uuid4())
        # Store KV cache on CPU
        self.kv_cache_store[kv_cache_id] = self._move_kv_cache(outputs.past_key_values, torch.device("cpu"))
        LOGGER.debug("Semantic prefill complete; kv_cache_id=%s", kv_cache_id)
        return logits_blob, kv_cache_id

    def run_decode_step_semantic(self, token_blob: bytes, kv_cache_id: str):
        """Semantic-aware decode with real compression pipeline."""
        if kv_cache_id not in self.kv_cache_store:
            raise RuntimeError(f"KV cache id {kv_cache_id} not found")

        token_id: torch.Tensor = self._decompress_tensor(token_blob)
        token_id = token_id.to(self.device, non_blocking=True)

        kv_cache = self._move_kv_cache(self.kv_cache_store[kv_cache_id], self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=token_id, past_key_values=kv_cache, use_cache=True)

        logits = outputs.logits.cpu()
        logits_blob = self._compress_tensor(logits)

        # Update stored cache
        self.kv_cache_store[kv_cache_id] = self._move_kv_cache(outputs.past_key_values, torch.device("cpu"))
        LOGGER.debug("Semantic decode complete; kv_cache_id=%s", kv_cache_id)
        return logits_blob, kv_cache_id


# --------------------------------------------------------------------------------------
# RPC server bootstrap
# --------------------------------------------------------------------------------------

def _parse_args(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="RPC server for GPU_HOST")
    parser.add_argument("--model", default="gpt2", help="HF model name or path")
    parser.add_argument("--master_addr", default="127.0.0.1", help="RPC master address")
    parser.add_argument("--master_port", default="29500", help="RPC master port")
    parser.add_argument("--rank", type=int, default=0, help="Rank of this worker")
    parser.add_argument("--world_size", type=int, default=2, help="Total number of RPC workers")
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

    # The server no longer creates a worker instance itself.
    # The client will create RemoteWorker instances via rpc.remote().

    # Use default RPC backend options
    import torch.distributed.rpc as drv
    opts = drv.TensorPipeRpcBackendOptions(  # type: ignore[attr-defined]
        rpc_timeout=1800
    )
    rpc.init_rpc(
        name="GPU_WORKER",
        rank=args.rank,
        world_size=args.world_size,
        rpc_backend_options=opts,
    )

    # Keep process alive until interrupted.
    def _safe_rpc_shutdown():
        """Attempt to shut down the RPC agent without raising if it's already closed."""
        try:
            rpc.shutdown()
        except RuntimeError as exc:
            if "RPC has not been initialized" not in str(exc):
                raise

    def _handle_sigterm(signum, frame):  # noqa: D401
        LOGGER.info("Received signal %s; shutting down RPC.", signum)
        _safe_rpc_shutdown()
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
        _safe_rpc_shutdown()
        LOGGER.info("RPC server shut down.")


# Ensure this module is discoverable under the name 'rpc_server' even when executed
# as a script (i.e., module name is '__main__'). This allows remote peers that
# import by name to resolve the same module instance.
if __name__ != "rpc_server":
    sys.modules["rpc_server"] = sys.modules[__name__]


if __name__ == "__main__":
    main()