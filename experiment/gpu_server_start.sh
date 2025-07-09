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
MODEL="${MODEL:-sshleifer/tiny-gpt2}"
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

start_server() {
  local port="$1"
  local logfile="$2"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching rpc_server on port ${port}"
  
  # Check if port is already in use
  if lsof -i ":${port}" >/dev/null 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Port ${port} is already in use, skipping"
    return 1
  fi
  
  nohup python rpc_server.py \
        --model "$MODEL" \
        --master_addr 0.0.0.0 \
        --master_port "$port" \
        --world_size 2 \
        --rank 0 \
        --backend "$BACKEND" \
        > "logs/${logfile}" 2>&1 &
  
  local pid=$!
  # Save PID for later termination
  echo $pid >> "logs/gpu_server_pids.txt"
  
  # Brief wait to check if process started successfully
  sleep 1
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to start server on port ${port}"
    return 1
  fi
  
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server started on port ${port} with PID ${pid}"
}

# How many trials worth of servers to launch (default 1, override with TRIALS=n)
TRIALS="${TRIALS:-1}"
PORT_STRIDE=50  # Must match value in experiment_driver.py

# Base port mapping (prefill / decode)
declare -A BASE_PORTS=(
  [naive_prefill]=29500
  [naive_decode]=29505
  [remote_cache_prefill]=29510
  [remote_cache_decode]=29515
  [sys_simulated_prefill]=29520
  [sys_simulated_decode]=29525
  [remote_cache_delta_prefill]=29530
  [remote_cache_delta_decode]=29535
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
    if ! start_server "$port" "$logfile"; then
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