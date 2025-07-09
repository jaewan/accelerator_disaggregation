#!/bin/bash

# AI Accelerator Disaggregation Experiment Runner
# This script runs the full experiment with proper GPU memory management

set -e  # Exit on any error

# Configuration
MODEL="EleutherAI/gpt-j-6B"
TRIALS=1
GPU_HOST="10.8.162.218"
BASE_PORT=29500

# Set environment variables for better GPU memory management
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING="1"

# Function to clean up GPU memory
cleanup_gpu() {
    echo "Cleaning up GPU memory..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --gpu-reset || true
    fi
    # Kill any remaining Python processes that might be holding GPU memory
    pkill -f "rpc_server.py" || true
    pkill -f "run_llm.py" || true
    sleep 2
}

# Function to check GPU memory
check_gpu_memory() {
    if command -v nvidia-smi &> /dev/null; then
        echo "Current GPU memory usage:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    fi
}

# Cleanup on exit
trap cleanup_gpu EXIT

echo "Starting AI Accelerator Disaggregation Experiment"
echo "Model: $MODEL"
echo "Trials: $TRIALS"
echo "GPU Host: $GPU_HOST"
echo "Base Port: $BASE_PORT"

# Check initial GPU memory
check_gpu_memory

# Clean up any existing processes
cleanup_gpu

# Run the experiment with conservative memory management
echo "Running experiment with conservative memory management..."
python experiment_driver.py \
    --trials $TRIALS \
    --gpu_host $GPU_HOST \
    --master_port $BASE_PORT \
    --model $MODEL \
    --output "results_$(date +%Y%m%d_%H%M%S).csv" \
    --modes "local,naive,remote_cache,remote_cache_delta,sys_simulated"

echo "Experiment completed successfully!"
check_gpu_memory 