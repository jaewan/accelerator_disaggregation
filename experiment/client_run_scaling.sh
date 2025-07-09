#!/bin/bash

# client_run_scaling.sh
# Runs the latency scaling experiment.

set -e
# Resolve the directory this script is located in.  This allows us to
# construct absolute paths so the script works whether it is executed
# from the project root *or* from the experiment/ directory itself.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# The client code never executes CUDA kernels – all heavy lifting happens
# on the remote GPU host.  Hiding local GPUs prevents PyTorch/TensorPipe
# from probing a potentially mismatched driver stack which triggered the
# CUDA_ERROR_SYSTEM_DRIVER_MISMATCH (error 803) you observed.
# Hide local GPUs completely so the CUDA runtime bails out early and TensorPipe
# skips GPU transports. "-1" is the canonical way recognised by NVIDIA/driver.
export CUDA_VISIBLE_DEVICES="-1"
# Tell TensorPipe/RPC to disable all CUDA support so it never calls cudaGetDeviceCount.
export TP_DISABLE_CUDA=1
# --- Configuration ---
MODEL="EleutherAI/gpt-j-6B"
TRIALS=1
GPU_HOST="10.8.162.218"  # IMPORTANT: Change this to your GPU server's IP
BASE_PORT=29500
OUTPUT_CSV="scaling_results.csv"
OUTPUT_PLOT="scaling_latency_plot.png"
# ---
# Optional: wait for the RPC ports to become reachable.  This avoids a race
# where the servers are still initialising when the client attempts its first
# connection and receives a "connection reset by peer" error.  Disable this
# check by setting WAIT_FOR_PORTS=0.
# ---

WAIT_FOR_PORTS=1

if [[ "${WAIT_FOR_PORTS}" -eq 1 ]]; then
  echo "Waiting for RPC ports on ${GPU_HOST} to become available …"
  # List of ports used by the experiment (prefill + decode for all modes)
  PORTS=(29500 29505 29520 29525 29530 29535)
  for port in "${PORTS[@]}"; do
    echo -n "  • Port ${port}: "
    for i in {1..15}; do
      if timeout 1 bash -c "</dev/tcp/${GPU_HOST}/${port}" 2>/dev/null; then
        echo "open"; break
      fi
      sleep 1
      if [[ $i -eq 15 ]]; then
        echo "timeout"
        echo "ERROR: Port ${port} is not reachable on ${GPU_HOST}. Make sure gpu_server_start.sh is running." >&2
        exit 1
      fi
    done
  done
  echo "All required ports are reachable. Starting experiment …"
fi

echo "Starting Latency Scaling Experiment"
echo "Running against GPU host: $GPU_HOST"

# Important: This assumes you have already started the RPC servers on the
# GPU host using gpu_server_start.sh.
# This script uses a single port, as the driver will connect to the same
# server process for all modes and steps.

python "$SCRIPT_DIR/run_scaling_experiment.py" \
    --trials "$TRIALS" \
    --gpu_host "$GPU_HOST" \
    --master_port "$BASE_PORT" \
    --model "$MODEL" \
    --output "$OUTPUT_CSV"

echo "Experiment finished. Generating plot..."

# Generate the plot from the results
python "$SCRIPT_DIR/plot_scaling_results.py" "$OUTPUT_CSV" "$OUTPUT_PLOT"

echo "All done. Plot saved to $OUTPUT_PLOT" 