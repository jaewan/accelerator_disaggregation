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
#
# Logs are stored under logs/ in the same directory as this script.
#
# Usage:
#   chmod +x gpu_server_start.sh
#   ./gpu_server_start.sh

set -eu

# Root of the experiment repository (default: current directory)
ROOT="${ROOT:-$(pwd)}"
MODEL="${MODEL:-facebook/opt-125m}"
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
  nohup python rpc_server.py \
        --model "$MODEL" \
        --master_addr 0.0.0.0 \
        --master_port "$port" \
        --world_size 2 \
        --rank 0 \
        --backend "$BACKEND" \
        > "logs/${logfile}" 2>&1 &
  # Save PID for later termination
  echo $! >> "logs/gpu_server_pids.txt"
}

# How many trials worth of servers to launch (default 5, override with TRIALS=n)
TRIALS="${TRIALS:-5}"
PORT_STRIDE=50  # Must match value in experiment_driver.py

# Base port mapping (prefill / decode)
declare -A BASE_PORTS=(
  [naive_prefill]=29500
  [naive_decode]=29505
  [remote_cache_prefill]=29510
  [remote_cache_decode]=29515
  [sys_simulated_prefill]=29520
  [sys_simulated_decode]=29525
)

for ((t=0; t<TRIALS; t++)); do
  offset=$((t * PORT_STRIDE))

  for key in "${!BASE_PORTS[@]}"; do
    base_port=${BASE_PORTS[$key]}
    port=$((base_port + offset))
    logfile="${key}_trial$((t+1)).log"
    start_server "$port" "$logfile"
  done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All RPC servers started." 