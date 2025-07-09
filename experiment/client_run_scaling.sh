#!/bin/bash

# client_run_scaling.sh
# Runs the latency scaling experiment.

set -e

# --- Configuration ---
MODEL="EleutherAI/gpt-j-6B"
TRIALS=1
GPU_HOST="10.8.162.218"  # IMPORTANT: Change this to your GPU server's IP
BASE_PORT=29500
OUTPUT_CSV="scaling_results.csv"
OUTPUT_PLOT="scaling_latency_plot.png"
# ---

echo "Starting Latency Scaling Experiment"
echo "Running against GPU host: $GPU_HOST"

# Important: This assumes you have already started the RPC servers on the
# GPU host using gpu_server_start.sh.
# This script uses a single port, as the driver will connect to the same
# server process for all modes and steps.

python experiment/run_scaling_experiment.py \
    --trials "$TRIALS" \
    --gpu_host "$GPU_HOST" \
    --master_port "$BASE_PORT" \
    --model "$MODEL" \
    --output "$OUTPUT_CSV"

echo "Experiment finished. Generating plot..."

# Generate the plot from the results
python experiment/plot_scaling_results.py "$OUTPUT_CSV" "$OUTPUT_PLOT"

echo "All done. Plot saved to $OUTPUT_PLOT" 