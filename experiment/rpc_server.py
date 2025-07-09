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
from typing import Any, Dict, List, Tuple, Optional
import zstandard as zstd  # type: ignore
import pickle
import io

import torch  # type: ignore
from torch.distributed import rpc  # type: ignore
from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore
import subprocess
import threading
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Remote worker implementation (GPU side)
# --------------------------------------------------------------------------------------

# We keep a *single* model instance in GPU memory for all RemoteWorker
# objects that may be created by successive client connections within the
# same server process.  This avoids out-of-memory errors when the naive
# baseline (which spins up a fresh RemoteWorker every run) is executed
# multiple times back-to-back.

_GLOBAL_MODEL: Optional[AutoModelForCausalLM] = None
_GLOBAL_MODEL_DTYPE: Optional[str] = None
_GLOBAL_WEIGHTS_LOADED: bool = False

# Global timing accumulators for detailed performance metrics
_GPU_KERNEL_MS: float = 0.0  # accumulated GPU kernel execution time (ms)
_SERDES_MS: float = 0.0      # accumulated (de)serialisation / compression time (ms)
_KERNEL_CALLS: int = 0       # number of forward passes timed

# Protect shared metric variables – Torch RPC may run handlers concurrently.
_METRIC_LOCK = threading.Lock()

# Helper to update GPU timing counters in a thread-safe manner.
def _record_gpu_time(delta_ms: float, instance=None):
    """Accumulate *delta_ms* into the global GPU kernel timer and increment call counter."""
    global _GPU_KERNEL_MS, _KERNEL_CALLS  # noqa: PLW0603
    with _METRIC_LOCK:
        _GPU_KERNEL_MS += delta_ms
        _KERNEL_CALLS += 1
    
    # Also track per-instance if provided (for GPU utilization calculation)
    if instance is not None and hasattr(instance, '_kernel_ms_this_session'):
        instance._kernel_ms_this_session += delta_ms
        # Log every 10th call to avoid spam
        if _KERNEL_CALLS % 10 == 0:
            LOGGER.info("GPU timing update #%d: delta=%.2fms, instance_total=%.2fms", 
                       _KERNEL_CALLS, delta_ms, instance._kernel_ms_this_session)

# Helper to update serdes timing counters in a thread-safe manner.
def _record_serdes_time(delta_ms: float):
    """Accumulate *delta_ms* into the global serdes timer."""
    global _SERDES_MS  # noqa: PLW0603
    with _METRIC_LOCK:
        _SERDES_MS += delta_ms

class RemoteWorker:
    """Remote side logic that lives on GPU_HOST.

    It provides two categories of methods:
    1. *Stateless* (`run_stateless_forward_remote`) – used by the naive baseline.
    2. *Stateful / semantic-aware* (`load_weights_remote`, `run_prefill_remote`,
       `run_decode_step_remote`).
    """

    def __init__(self, model_name: str = "gpt2", dtype: str = "float16") -> None:
        global _GLOBAL_MODEL, _GLOBAL_MODEL_DTYPE, _GLOBAL_WEIGHTS_LOADED  # noqa: PLW0603

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        LOGGER.info("Initializing RemoteWorker on %s with model %s (dtype=%s)", self.device, model_name, dtype)

        if _GLOBAL_MODEL is None:
            # No model loaded yet in this process – create a fresh skeleton.
            config = AutoConfig.from_pretrained(model_name)
            with torch.no_grad():
                _GLOBAL_MODEL = AutoModelForCausalLM.from_config(config)

                # Cast to requested dtype **on CPU** first.  We defer the costly
                # `.to(self.device)` call until after *real* weights have been
                # materialised.  This avoids allocating ~12 GB of GPU memory
                # per server process for an empty parameter skeleton which is
                # never used for computation.

                if dtype == "float16":
                    _GLOBAL_MODEL = _GLOBAL_MODEL.half()  # type: ignore[assignment]
                elif dtype == "bfloat16":
                    _GLOBAL_MODEL = _GLOBAL_MODEL.to(dtype=torch.bfloat16)  # type: ignore[arg-type, assignment]

                # NOTE: We deliberately keep the skeleton on *CPU* here.  The
                # first call to `load_weights_remote` or `download_weights_remote`
                # will move the fully-initialised model to `self.device` once and
                # reuse it for all subsequent RemoteWorker instances that share
                # the same global model object.

            _GLOBAL_MODEL_DTYPE = dtype
            _GLOBAL_WEIGHTS_LOADED = False
            LOGGER.info("Created global model skeleton (first RemoteWorker)")
        else:
            # Re-use existing global model.  Optionally warn if dtype mismatch
            if dtype != _GLOBAL_MODEL_DTYPE:
                LOGGER.warning("Requested dtype %s but global model dtype is %s; proceeding with global model", dtype, _GLOBAL_MODEL_DTYPE)

        # All RemoteWorker instances share the same model object.
        assert _GLOBAL_MODEL is not None  # for type checker
        self.model = _GLOBAL_MODEL  # type: ignore[assignment]
        self.weights_loaded = _GLOBAL_WEIGHTS_LOADED

        # Each worker has its *own* KV-cache store but shared weights.
        self.kv_cache_store: Dict[str, Any] = {}
        LOGGER.info("RemoteWorker ready (weights_loaded=%s)", self.weights_loaded)

        # GPU monitoring state
        self._gpu_mon_proc: Optional[subprocess.Popen] = None
        self._dmon_csv: Optional[str] = None
        self._kernel_ms_this_session: float = 0.0

    # ------------------------------------------------------------------
    # GPU monitoring utilities (started/stopped by the client via RPC)
    # ------------------------------------------------------------------

    def start_gpu_monitor_remote(self):
        """Begin GPU-util sampling.

        We *try* to launch ``nvidia-smi dmon``.  If that binary is absent or
        fails, we gracefully fall back to a lightweight timer-based estimate
        that derives utilisation from our CUDA kernel timers.
        """

        if getattr(self, "_wall_start", None) is not None:
            LOGGER.debug("GPU monitor already running; skipping start.")
            return

        # Reset instance-level counters for this monitoring session
        self._kernel_ms_this_session = 0.0
        self._wall_start = time.perf_counter()
        LOGGER.info("GPU monitor started: kernel_ms_this_session=%.2f, wall_start=%.2f",
                   self._kernel_ms_this_session, self._wall_start)

        self._dmon_csv = f"/tmp/dmon_{uuid.uuid4().hex}.csv"
        cmd = [
            "nvidia-smi",
            "dmon",
            "-s",
            "u",
            "-i",
            "0",
            "-f",
            self._dmon_csv,
        ]
        try:
            self._gpu_mon_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            LOGGER.debug("Started GPU monitor: %s", cmd)
        except FileNotFoundError:
            LOGGER.info("No nvidia-smi binary – will use kernel-time based utilisation estimate.")
            self._gpu_mon_proc = None

    def _parse_dmon_csv(self, path: str) -> float:
        """Return average SM utilisation from a dmon CSV file."""
        try:
            sm_total = 0.0
            rows = 0
            with Path(path).open() as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    sm_val = parts[-1] if parts else None
                    if sm_val and sm_val.isdigit():
                        sm_total += float(sm_val)
                        rows += 1
            return sm_total / rows if rows else 0.0
        except Exception as exc:  # pragma: no cover – best effort
            LOGGER.warning("Failed to parse dmon CSV %s: %s", path, exc)
            return 0.0

    def stop_gpu_monitor_remote(self) -> float:
        """Stop monitoring and return average SM utilisation (in %)."""

        # 1) If dmon was running, terminate and parse CSV.
        util = None
        if self._gpu_mon_proc is not None:
            self._gpu_mon_proc.terminate()
            try:
                self._gpu_mon_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._gpu_mon_proc.kill()
                self._gpu_mon_proc.wait()

            self._gpu_mon_proc = None

            if self._dmon_csv:
                util = self._parse_dmon_csv(self._dmon_csv)
                try:
                    Path(self._dmon_csv).unlink(missing_ok=True)
                except Exception:
                    pass

        # 2) Fallback: derive utilisation from kernel timers if CSV failed or dmon missing.
        if util is None or util == 0.0:
            wall_ms = (time.perf_counter() - getattr(self, "_wall_start", time.perf_counter())) * 1000.0
            kernel_ms_delta = self._kernel_ms_this_session
            LOGGER.info("Fallback util calc: wall_ms=%.2f, kernel_ms_delta=%.2f", wall_ms, kernel_ms_delta)
            if wall_ms > 0:
                util = min(100.0, 100.0 * kernel_ms_delta / wall_ms)
                LOGGER.info("Calculated utilization: %.2f%% (kernel/wall = %.2f/%.2f)", util, kernel_ms_delta, wall_ms)
            else:
                util = 0.0
                LOGGER.warning("Wall time is 0, cannot calculate utilization")

        LOGGER.info("GPU monitor stopped; avg SM util=%.2f%%", util)
        # Clean baseline attributes for next run
        self._wall_start = None  # type: ignore[attr-defined]
        return util

    def get_rpc_bytes(self) -> int:
        """Return total bytes (sent + received) from the RPC agent."""
        # This is a placeholder. Real implementation would query the agent.
        # Note: Accessing internal agent metrics can be unstable across versions.
        # For this experiment, we will rely on values returned by the client call.
        return 0

    def get_model_size_remote(self) -> int:
        """Return total parameter size of the model resident on this worker *in bytes*."""
        return sum(p.numel() * p.element_size() for p in self.model.parameters())  # type: ignore[attr-defined]

    def get_timing_metrics_remote(self) -> Dict[str, float]:
        """Return accumulated timing metrics (in milliseconds) since last reset."""
        # Copy under lock to avoid tearing
        with _METRIC_LOCK:
            return {
                "gpu_kernel_ms": _GPU_KERNEL_MS,
                "serdes_ms": _SERDES_MS,
                "kernel_calls": _KERNEL_CALLS,
            }

    def reset_metrics_remote(self) -> None:
        """Reset accumulated timing metrics (GPU + serdes)."""
        global _GPU_KERNEL_MS, _SERDES_MS, _KERNEL_CALLS  # noqa: PLW0603
        with _METRIC_LOCK:
            _GPU_KERNEL_MS = 0.0
            _SERDES_MS = 0.0
            _KERNEL_CALLS = 0

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
        """Compress tensor via torch.save while tracking (de)serialisation time."""
        start_t = time.perf_counter()

        if tensor.dtype != torch.float16:
            tensor = tensor.half()

        buffer = io.BytesIO()
        torch.save(tensor.cpu(), buffer)
        compressed = zstd.compress(buffer.getvalue(), level=3)

        _record_serdes_time((time.perf_counter() - start_t) * 1000.0)

        return compressed

    def _decompress_tensor(self, compressed_data):
        """Inverse of _compress_tensor with timing."""
        start_t = time.perf_counter()

        decompressed = zstd.decompress(compressed_data)
        buffer = io.BytesIO(decompressed)
        tensor = torch.load(buffer)

        _record_serdes_time((time.perf_counter() - start_t) * 1000.0)

        return tensor.to(self.device)

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

        # If the incoming state_dict is empty (client used --skip_weight_upload)
        # we keep the resident weights and avoid the expensive copy.
        if state_dict:
            self.model.load_state_dict(state_dict, strict=False)  # type: ignore[attr-defined]

        # ------------------------------------------------------------------
        # NEW: ensure the (potentially still skeleton) model resides on the
        # *GPU* before we execute the forward pass.  In the stateless naive
        # baseline we may never call ``download_weights_remote`` when the
        # client passes ``--skip_weight_upload``.  In that case the global
        # model would remain on CPU and each forward run would be executed on
        # the host, leading to extremely long runtimes and client-side
        # timeouts.  Moving the model once is inexpensive compared to the 6B
        # parameter compute and avoids the 30-minute timeout.
        # ------------------------------------------------------------------
        if next(self.model.parameters()).device.type == "cpu":  # type: ignore[arg-type]
            try:
                LOGGER.info("Moving stateless model to %s for naive baseline", self.device)
                self.model = self.model.to(self.device)  # type: ignore[assignment]
            except torch.cuda.OutOfMemoryError as oom:
                LOGGER.error(
                    "GPU OOM while moving model for stateless forward. "
                    "Consider assigning the naive baseline to a larger GPU or "
                    "enable weight uploads instead."
                )
                raise oom

        load_t = time.time() - start

        input_ids = input_ids.to(self.device, non_blocking=True)
        kv_cache = self._move_kv_cache(kv_cache, self.device)

        with torch.no_grad():
            if torch.cuda.is_available():
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()  # type: ignore[arg-type]

            outputs = self.model(input_ids=input_ids, past_key_values=kv_cache, use_cache=True)  # type: ignore[operator]

            if torch.cuda.is_available():
                end_ev.record()  # type: ignore[arg-type]
                torch.cuda.synchronize()
                _record_gpu_time(start_ev.elapsed_time(end_ev), self)

        logits = outputs.logits.cpu()
        pkv = self._move_kv_cache(outputs.past_key_values, torch.device("cpu"))
        LOGGER.debug("Stateless forward done (load_t=%.3fs)", load_t)
        return logits, pkv

    # ------------------------------------------------------------------
    # Semantic-aware execution path
    # ------------------------------------------------------------------

    def load_weights_remote(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load weights only once and keep them resident in GPU memory."""
        global _GLOBAL_WEIGHTS_LOADED  # noqa: PLW0603

        if not _GLOBAL_WEIGHTS_LOADED:
            # Move *once* to the desired device now that real weights exist.
            if next(self.model.parameters()).device.type == "cpu":  # type: ignore[arg-type]
                self.model = self.model.to(self.device)  # type: ignore[assignment]

            self.model.load_state_dict(state_dict, strict=False)  # type: ignore[attr-defined]
            self.weights_loaded = True
            _GLOBAL_WEIGHTS_LOADED = True
            LOGGER.info("Weights loaded (%.2f MB)", sum(v.nbytes for v in state_dict.values()) / 1e6)
        else:
            LOGGER.debug("Weights already loaded; skipping reload.")

    def download_weights_remote(self, model_name: str) -> None:
        """Download weights from Hugging Face and keep them resident in GPU memory."""
        global _GLOBAL_WEIGHTS_LOADED  # noqa: PLW0603

        if not _GLOBAL_WEIGHTS_LOADED:
            LOGGER.info("Downloading weights for model %s", model_name)
            
            # Clear GPU cache before loading new model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            full_model = AutoModelForCausalLM.from_pretrained(model_name)  # type: ignore[call-arg]

            # Convert to the same dtype as our model
            if getattr(self.model, "dtype", torch.float32) == torch.float16:  # type: ignore[attr-defined]
                full_model = full_model.half()  # type: ignore[attr-defined]
            elif getattr(self.model, "dtype", torch.float32) == torch.bfloat16:  # type: ignore[attr-defined]
                full_model = full_model.to(dtype=torch.bfloat16)

            # Move the (still CPU-resident) skeleton to GPU **once** and then
            # load the weights.  We re-use the same global model instance for
            # all RemoteWorkers within this process, so subsequent calls will
            # skip the costly transfer.

            if next(self.model.parameters()).device.type == "cpu":  # type: ignore[arg-type]
                try:
                    self.model = self.model.to(self.device)  # type: ignore[assignment]
                except torch.cuda.OutOfMemoryError:
                    # If we run out of memory, try to free up space and retry
                    LOGGER.warning("GPU OOM during model loading, attempting cleanup and retry...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Try to move model to GPU again
                    try:
                        self.model = self.model.to(self.device)  # type: ignore[assignment]
                    except torch.cuda.OutOfMemoryError as e:
                        LOGGER.error("Failed to load model after cleanup. GPU memory exhausted.")
                        raise RuntimeError(f"GPU out of memory: {e}. Try reducing batch size or using fewer concurrent processes.") from e

            self.model.load_state_dict(full_model.state_dict(), strict=False)  # type: ignore[attr-defined]
            self.weights_loaded = True
            _GLOBAL_WEIGHTS_LOADED = True

            # Calculate size for logging
            total_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e6  # type: ignore[attr-defined]
            LOGGER.info("Weights downloaded and loaded (%.2f MB)", total_size)
            
            # Clean up the full_model to free CPU memory
            del full_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            LOGGER.debug("Weights already loaded; skipping download.")

    def reset_state_remote(self) -> None:
        """Reset KV cache state between trials while keeping weights loaded."""
        self.kv_cache_store.clear()
        # Clear GPU cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        LOGGER.debug("KV cache state reset (weights preserved)")

    def run_prefill_remote(self, token_ids: torch.Tensor):
        token_ids = token_ids.to(self.device, non_blocking=True)
        with torch.no_grad():
            if torch.cuda.is_available():
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()  # type: ignore[arg-type]

            outputs = self.model(input_ids=token_ids, use_cache=True)  # type: ignore[operator]

            if torch.cuda.is_available():
                end_ev.record()  # type: ignore[arg-type]
                torch.cuda.synchronize()
                _record_gpu_time(start_ev.elapsed_time(end_ev), self)
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
            if torch.cuda.is_available():
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()  # type: ignore[arg-type]

            outputs = self.model(input_ids=token_id, past_key_values=kv_cache, use_cache=True)  # type: ignore[operator]

            if torch.cuda.is_available():
                end_ev.record()  # type: ignore[arg-type]
                torch.cuda.synchronize()
                _record_gpu_time(start_ev.elapsed_time(end_ev), self)
        logits = outputs.logits.cpu()
        # Update stored cache on CPU to minimise GPU memory footprint between calls.
        self.kv_cache_store[kv_cache_id] = self._move_kv_cache(outputs.past_key_values, torch.device("cpu"))
        return logits, kv_cache_id

    def run_prefill_with_handle(self, input_ids: torch.Tensor):
        """Realistic semantic-blind caching: KV cache stays resident remotely."""
        input_ids = input_ids.to(self.device, non_blocking=True)
        with torch.no_grad():
            if torch.cuda.is_available():
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()  # type: ignore[arg-type]

            outputs = self.model(input_ids=input_ids, use_cache=True)  # type: ignore[operator]

            if torch.cuda.is_available():
                end_ev.record()  # type: ignore[arg-type]
                torch.cuda.synchronize()
                _record_gpu_time(start_ev.elapsed_time(end_ev), self)
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
            if torch.cuda.is_available():
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()  # type: ignore[arg-type]

            outputs = self.model(input_ids=token_ids, use_cache=True)  # type: ignore[operator]

            if torch.cuda.is_available():
                end_ev.record()  # type: ignore[arg-type]
                torch.cuda.synchronize()
                _record_gpu_time(start_ev.elapsed_time(end_ev), self)
        logits = outputs.logits.cpu()
        kv_cache = self._move_kv_cache(outputs.past_key_values, torch.device("cpu"))
        LOGGER.debug("Prefill with cache complete; returning actual KV cache tensors")
        return logits, kv_cache

    def run_prefill_with_cache_compressed_remote(self, token_ids: torch.Tensor):
        """Run prefill and return compressed KV cache (semantic-blind compression)."""
        token_ids = token_ids.to(self.device, non_blocking=True)
        with torch.no_grad():
            if torch.cuda.is_available():
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()  # type: ignore[arg-type]

            outputs = self.model(input_ids=token_ids, use_cache=True)  # type: ignore[operator]

            if torch.cuda.is_available():
                end_ev.record()  # type: ignore[arg-type]
                torch.cuda.synchronize()
                _record_gpu_time(start_ev.elapsed_time(end_ev), self)
        
        logits = outputs.logits.cpu()
        kv_cache = self._move_kv_cache(outputs.past_key_values, torch.device("cpu"))
        
        # Compress KV cache using semantic-blind compression (no FP32→FP16 conversion)
        start_t = time.perf_counter()
        
        buffer = io.BytesIO()
        torch.save(kv_cache, buffer)
        compressed_kv = zstd.compress(buffer.getvalue(), level=3)
        
        _record_serdes_time((time.perf_counter() - start_t) * 1000.0)
        
        LOGGER.debug("Prefill with compressed cache complete; returning compressed KV cache")
        return logits, compressed_kv

    def run_decode_with_handle(self, token_id: torch.Tensor, kv_handle: str):
        """Realistic semantic-blind caching: KV cache stays resident remotely."""
        if kv_handle not in self.kv_cache_store:
            raise RuntimeError(f"KV cache handle {kv_handle} not found")
        token_id = token_id.to(self.device, non_blocking=True)
        kv_cache = self.kv_cache_store[kv_handle]  # Already on GPU, no transfer needed

        with torch.no_grad():
            if torch.cuda.is_available():
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()  # type: ignore[arg-type]

            outputs = self.model(input_ids=token_id, past_key_values=kv_cache, use_cache=True)  # type: ignore[operator]

            if torch.cuda.is_available():
                end_ev.record()  # type: ignore[arg-type]
                torch.cuda.synchronize()
                _record_gpu_time(start_ev.elapsed_time(end_ev), self)
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
            if torch.cuda.is_available():
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()  # type: ignore[arg-type]

            outputs = self.model(input_ids=token_id, past_key_values=kv_cache, use_cache=True)  # type: ignore[operator]

            if torch.cuda.is_available():
                end_ev.record()  # type: ignore[arg-type]
                torch.cuda.synchronize()
                _record_gpu_time(start_ev.elapsed_time(end_ev), self)
        logits = outputs.logits.cpu()
        updated_kv_cache = self._move_kv_cache(outputs.past_key_values, torch.device("cpu"))
        return logits, updated_kv_cache

    def run_decode_step_with_cache_compressed_remote(self, token_id: torch.Tensor, compressed_kv_cache: bytes):
        """Run decode step with compressed KV cache (semantic-blind compression)."""
        token_id = token_id.to(self.device, non_blocking=True)
        
        # Decompress KV cache
        start_t = time.perf_counter()
        
        decompressed = zstd.decompress(compressed_kv_cache)
        buffer = io.BytesIO(decompressed)
        kv_cache = torch.load(buffer)
        
        _record_serdes_time((time.perf_counter() - start_t) * 1000.0)
        
        kv_cache = self._move_kv_cache(kv_cache, self.device)

        with torch.no_grad():
            if torch.cuda.is_available():
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()  # type: ignore[arg-type]

            outputs = self.model(input_ids=token_id, past_key_values=kv_cache, use_cache=True)  # type: ignore[operator]

            if torch.cuda.is_available():
                end_ev.record()  # type: ignore[arg-type]
                torch.cuda.synchronize()
                _record_gpu_time(start_ev.elapsed_time(end_ev), self)
        
        logits = outputs.logits.cpu()
        updated_kv_cache = self._move_kv_cache(outputs.past_key_values, torch.device("cpu"))
        
        # Compress updated KV cache
        start_t = time.perf_counter()
        
        buffer = io.BytesIO()
        torch.save(updated_kv_cache, buffer)
        compressed_updated_kv = zstd.compress(buffer.getvalue(), level=3)
        
        _record_serdes_time((time.perf_counter() - start_t) * 1000.0)
        
        return logits, compressed_updated_kv

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
            if torch.cuda.is_available():
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()  # type: ignore[arg-type]

            outputs = self.model(input_ids=token_ids, use_cache=True)  # type: ignore[operator]

            if torch.cuda.is_available():
                end_ev.record()  # type: ignore[arg-type]
                torch.cuda.synchronize()
                _record_gpu_time(start_ev.elapsed_time(end_ev), self)

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
            if torch.cuda.is_available():
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()  # type: ignore[arg-type]

            outputs = self.model(input_ids=token_id, past_key_values=kv_cache, use_cache=True)  # type: ignore[operator]

            if torch.cuda.is_available():
                end_ev.record()  # type: ignore[arg-type]
                torch.cuda.synchronize()
                _record_gpu_time(start_ev.elapsed_time(end_ev), self)

        logits = outputs.logits.cpu()
        logits_blob = self._compress_tensor(logits)

        # Update stored cache
        self.kv_cache_store[kv_cache_id] = self._move_kv_cache(outputs.past_key_values, torch.device("cpu"))
        LOGGER.debug("Semantic decode complete; kv_cache_id=%s", kv_cache_id)
        return logits_blob, kv_cache_id

    # ------------------------------------------------------------------
    # Delta KV-cache helpers
    # ------------------------------------------------------------------

    def _extract_delta_cache(self, full_cache):
        """Return a *shallow copy* of *full_cache* that keeps only the last position.

        Assumes tensors have shape (B, H, S, D) and extracts index -1 on the
        sequence dimension (dim=2).  The returned structure matches the nested
        list layout [[k, v], ...] expected by the client.
        """
        delta = []
        for k_layer, v_layer in full_cache:
            delta_k = k_layer[:, :, -1:, :].contiguous()
            delta_v = v_layer[:, :, -1:, :].contiguous()
            delta.append([delta_k, delta_v])
        return delta

    def run_prefill_with_cache_delta_remote(self, token_ids: torch.Tensor):
        """
        Run prefill, return full KV cache to client, AND store a copy resident
        on the server for subsequent delta-based decode steps.
        """
        # Get the full cache from the existing non-delta implementation
        logits, kv_cache = self.run_prefill_with_cache_remote(token_ids)

        # Store a copy on the server for future delta steps
        self.kv_cache_store["_delta"] = kv_cache

        LOGGER.debug("Prefill with delta complete; returning full cache to client, storing copy on server.")
        return logits, kv_cache

    def run_decode_step_with_cache_delta_remote(self, token_id: torch.Tensor):
        """Decode step that returns *delta* KV cache (only new position)."""
        token_id = token_id.to(self.device, non_blocking=True)

        if "_delta" not in self.kv_cache_store:
            raise RuntimeError("Delta KV cache not initialised; run prefill first")

        kv_cache = self._move_kv_cache(self.kv_cache_store["_delta"], self.device)

        with torch.no_grad():
            if torch.cuda.is_available():
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()  # type: ignore[arg-type]

            outputs = self.model(input_ids=token_id, past_key_values=kv_cache, use_cache=True)  # type: ignore[operator]

            if torch.cuda.is_available():
                end_ev.record()  # type: ignore[arg-type]
                torch.cuda.synchronize()
                _record_gpu_time(start_ev.elapsed_time(end_ev), self)

        logits = outputs.logits.cpu()
        updated_kv_cache = self._move_kv_cache(outputs.past_key_values, torch.device("cpu"))

        # Update resident cache
        self.kv_cache_store["_delta"] = updated_kv_cache

        # Extract delta slice (last position only)
        delta_kv_cache = self._extract_delta_cache(updated_kv_cache)

        # Compress delta cache
        start_t = time.perf_counter()
        buffer = io.BytesIO()
        torch.save(delta_kv_cache, buffer)
        compressed_delta = zstd.compress(buffer.getvalue(), level=3)
        _record_serdes_time((time.perf_counter() - start_t) * 1000.0)

        return logits, compressed_delta

    def run_decode_step_with_cache_delta_raw_remote(self, token_id: torch.Tensor):
        """Decode step that returns *delta* KV cache uncompressed (raw tensors)."""
        token_id = token_id.to(self.device, non_blocking=True)
 
        if "_delta" not in self.kv_cache_store:
            raise RuntimeError("Delta KV cache not initialised; run prefill first")
 
        kv_cache = self._move_kv_cache(self.kv_cache_store["_delta"], self.device)
 
        with torch.no_grad():
            if torch.cuda.is_available():
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()  # type: ignore[arg-type]
 
            outputs = self.model(input_ids=token_id, past_key_values=kv_cache, use_cache=True)  # type: ignore[operator]
 
            if torch.cuda.is_available():
                end_ev.record()  # type: ignore[arg-type]
                torch.cuda.synchronize()
                _record_gpu_time(start_ev.elapsed_time(end_ev), self)
 
        logits = outputs.logits.cpu()
        updated_kv_cache = self._move_kv_cache(outputs.past_key_values, torch.device("cpu"))
 
        # Update resident cache
        self.kv_cache_store["_delta"] = updated_kv_cache
 
        # Extract delta slice (last position only)
        delta_kv_cache = self._extract_delta_cache(updated_kv_cache)
 
        return logits, delta_kv_cache


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

    # ------------------------------------------------------------------
    # Explicitly initialise the default process-group *before* RPC so that
    # both ranks (server = 0, client = 1) agree on the backend.  Relying on
    # PyTorch’s implicit initialisation picks **NCCL** on GPU hosts and
    # **GLOO** on CPU-only clients which results in an immediate
    # "connection reset by peer" during the TensorPipe handshake.
    # ------------------------------------------------------------------

    import torch.distributed as dist

    dist.init_process_group(
        backend=args.backend,
        rank=args.rank,
        world_size=args.world_size,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
    )

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