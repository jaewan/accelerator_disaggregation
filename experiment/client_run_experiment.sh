#!/usr/bin/env bash
# client_run_experiment.sh
# Executes the semantic-gap experiment (5 trials) against external RPC
# servers started by gpu_server_start.sh.
#
# Usage:
#   chmod +x client_run_experiment.sh
#   ./client_run_experiment.sh
#
# Environment variables recognised:
#   GPU_HOST         IP of the GPU server (default: 10.8.162.218)
#   MODEL            HF model name/path (default: facebook/opt-125m)
#   ROOT             Repository root (default: current directory)
#   VENV_ACTIVATE    Path to venv/conda activate script (default: venv/bin/activate)

set -eu

ROOT="${ROOT:-$(pwd)}"
# Hostname/IP used by Torch RPC (must resolve via DNS).  Override if needed.
GPU_HOST="${GPU_HOST:-GCRAZGDL4024.westus3.cloudapp.azure.com}"
# SSH alias or hostname for scp/ssh; defaults to 'gpu' from your ssh config.
SSH_HOST="${SSH_HOST:-gpu}"
MODEL="${MODEL:-facebook/opt-125m}"
VENV_ACTIVATE="${VENV_ACTIVATE:-venv/bin/activate}"

cd "$ROOT"

# Activate virtual environment if present
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_ACTIVATE"
fi

###############################################################################
# Run the experiment (5 trials, three modes, two phases).  Ensure that the GPU
# server has already been started with ports 29500/29505 (+50 per additional
# trial) for the naïve baseline and similar offsets for the other modes.  See
# gpu_server_start.sh for details.
###############################################################################
python experiment_driver.py \
       --trials 5 \
       --gpu_host "$GPU_HOST" \
       --ssh_host "$SSH_HOST" \
       --master_port 29500 \
       --model "$MODEL" \
       --modes naive,remote_cache,sys_simulated \
       --external_server \
       --output stage2_results.csv

echo "Experiment complete. Results written to stage2_results.csv." 