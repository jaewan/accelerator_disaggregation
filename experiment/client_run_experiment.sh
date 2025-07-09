#!/bin/bash

# AI Accelerator Disaggregation Experiment Runner
# This script runs the full experiment using external RPC servers

set -e  # Exit on any error

# Configuration
MODEL="EleutherAI/gpt-j-6B"  # Use the original large model
TRIALS=1
GPU_HOST="10.8.162.218"
BASE_PORT=29500

echo "Starting AI Accelerator Disaggregation Experiment"
echo "Model: $MODEL"
echo "Trials: $TRIALS"
echo "GPU Host: $GPU_HOST"
echo "Base Port: $BASE_PORT"

# Run the experiment using external servers (started by gpu_server_start.sh)
echo "Running experiment with external servers..."
python experiment_driver.py \
    --trials $TRIALS \
    --gpu_host $GPU_HOST \
    --master_port $BASE_PORT \
    --model $MODEL \
    --output "results_$(date +%Y%m%d_%H%M%S).csv" \
    --modes "naive,remote_cache_delta_raw,remote_cache_handle" \
    --external_server

echo "Experiment completed successfully!" 