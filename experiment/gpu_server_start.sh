#!/usr/bin/env bash
# gpu_server_start.sh
# Starts one RPC server per mode/phase combination for the semantic-gap experiments.
# Launch this script on the GPU host (10.8.162.218) before running the client.
#
# Ports used:
#   29500  - naive baseline (prefill)
#   29505  - naive baseline (decode)
#   29510  - remote-cache baseline (prefill)
#   29515  - remote-cache baseline (decode)
#   29520  - framework-level semantic-aware (\sys) prefill
#   29525  - framework-level semantic-aware (\sys) decode
#   29530  - delta KV cache (prefill)
#   29535  - delta KV cache (decode)
#
# Logs are stored under logs/ in the same directory as this script.
#
# Usage:
#   chmod +x gpu_server_start.sh
#   ./gpu_server_start.sh

set -eu

# Disable CUDA-specific TensorPipe channels so the handshake does not attempt
# to negotiate a GPU-aware transport with the (CPU-only) client.  This is
# benign for the GPU host – data still lives on the GPU – but removes an
# entire class of connection-reset errors during capability negotiation.
export TP_DISABLE_CUDA=1

# Root of the experiment repository (default: current directory)
ROOT="${ROOT:-$(pwd)}"
MODEL="${MODEL:-EleutherAI/gpt-j-6B}"
VENV_ACTIVATE="${VENV_ACTIVATE:-venv/bin/activate}"

cd "$ROOT"

# Activate virtual environment if available
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_ACTIVATE"
fi

mkdir -p logs

# ----------------------------------------------------------------------------
# RPC communication backend
# ----------------------------------------------------------------------------
# The server and *all* clients in the same RPC world must pick the *same*
# backend ("gloo" or "nccl").  Using NCCL on the GPU host while the client
# selects Gloo (default) causes the handshake to fail with "connection reset
# by peer".
#
# You can override the backend when launching the script:
#     BACKEND=gloo ./gpu_server_start.sh
# If BACKEND is undefined we fall back to a safe default of "gloo" so that a
# CPU-only client can always connect.
# ----------------------------------------------------------------------------

BACKEND="${BACKEND:-gloo}"

# GPU mapping per mode (modify as needed for your host)
declare -A GPU_IDX=(
  [naive]=0
  [remote_cache_delta_raw]=1
  [remote_cache_handle]=2
)

start_server() {
  local port="$1"
  local logfile="$2"
  local gpu_id="$3"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching rpc_server on port ${port} on GPU ${gpu_id}"
  
  # Check if port is already in use
  if lsof -i ":${port}" >/dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Port ${port} is already in use, skipping"
    return 1
  fi
  
  CUDA_VISIBLE_DEVICES="${gpu_id}" nohup python rpc_server.py \
        --model "$MODEL" \
        --master_addr 0.0.0.0 \
        --master_port "$port" \
        --world_size 2 \
        --rank 0 \
        --backend "$BACKEND" \
        > "logs/${logfile}" 2>&1 &
  
  local pid=$!
  echo $pid >> "logs/gpu_server_pids.txt"
  sleep 1
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to start server on port ${port}"
    return 1
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server started on port ${port} with PID ${pid} (GPU ${gpu_id})"
}

# How many trials worth of servers to launch (default 1, override with TRIALS=n)
TRIALS="${TRIALS:-1}"
PORT_STRIDE=50  # Must match value in experiment_driver.py

# Base port mapping (one server per mode)
declare -A BASE_PORTS=(
  [naive]=29500
  [remote_cache_delta_raw]=29520
  [remote_cache_handle]=29530
)

# We launch *two* servers per mode: one for prefill (base port) and one for
# decode (base port + 5).  Both servers run on the SAME GPU so they share the
# global model parameters but live in independent processes.  This avoids the
# TensorPipe/Gloo stale-socket issue that arises when the client reconnects to
# the same RPC world for the decode phase.

# Helper array so we can iterate deterministically in the for-loop below.
PHASE_OFFSETS=(0 5)

# Initialize PID file
echo > "logs/gpu_server_pids.txt"

failed_servers=0
for ((t=0; t<TRIALS; t++)); do
  offset=$((t * PORT_STRIDE))

  for key in "${!BASE_PORTS[@]}"; do
    base_port=${BASE_PORTS[$key]}
    port=$((base_port + offset))
    gpu_id=${GPU_IDX[$key]:-0}

    # Start *two* servers per mode: prefill (phase_offset=0) and decode (+5)
    for phase_offset in "${PHASE_OFFSETS[@]}"; do
      phase_port=$((port + phase_offset))
      phase_label=$([[ $phase_offset -eq 0 ]] && echo "prefill" || echo "decode")
      logfile="${key}_${phase_label}_trial$((t+1)).log"
      if ! start_server "$phase_port" "$logfile" "$gpu_id"; then
        failed_servers=$((failed_servers + 1))
      fi
    done
  done
done

if [ $failed_servers -gt 0 ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: ${failed_servers} server(s) failed to start"
  exit 1
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] All RPC servers started successfully."
fi 