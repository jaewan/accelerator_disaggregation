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

# Detect whether GPUs are available and choose the appropriate backend for
# the process-group that RPC implicitly creates.  This does *not* change the
# RPC transport (still TensorPipe) but ensures NCCL is initialised so that
# tensors live on GPU by default.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  BACKEND="nccl"
else
  BACKEND="gloo"
fi

# GPU mapping per mode (modify as needed for your host)
declare -A GPU_IDX=(
  [naive]=0
  [remote_cache_delta_compressed]=1
  [sys_simulated]=2
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
  [sys_simulated]=29520
  [remote_cache_delta_compressed]=29530
)

# Initialize PID file
echo > "logs/gpu_server_pids.txt"

failed_servers=0
for ((t=0; t<TRIALS; t++)); do
  offset=$((t * PORT_STRIDE))

  for key in "${!BASE_PORTS[@]}"; do
    base_port=${BASE_PORTS[$key]}
    port=$((base_port + offset))
    logfile="${key}_trial$((t+1)).log"
    gpu_id=${GPU_IDX[$key]:-0}
    if ! start_server "$port" "$logfile" "$gpu_id"; then
      failed_servers=$((failed_servers + 1))
    fi
  done
done

if [ $failed_servers -gt 0 ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: ${failed_servers} server(s) failed to start"
  exit 1
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] All RPC servers started successfully."
fi 