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
#   MODEL            HF model name/path (default: EleutherAI/gpt-j-6B)
#   ROOT             Repository root (default: current directory)
#   VENV_ACTIVATE    Path to venv/conda activate script (default: venv/bin/activate)

set -eu

ROOT="${ROOT:-$(pwd)}"
# Hostname/IP used by Torch RPC (must resolve via DNS).  Override if needed.
GPU_HOST="${GPU_HOST:-10.8.162.218}"
MODEL="${MODEL:-EleutherAI/gpt-j-6B}"
VENV_ACTIVATE="${VENV_ACTIVATE:-venv/bin/activate}"

cd "$ROOT"

# Activate virtual environment if present
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_ACTIVATE"
fi

###############################################################################
# Run the experiment (5 trials by default, three modes, two phases).  Every
# trial now uses *distinct* port ranges (stride = 50) to avoid Gloo
# stale-socket dead-locks.  Hence the GPU host must run one set of RPC
# servers per trial.  Launch them with e.g.
#   TRIALS=5 ./gpu_server_start.sh
# before executing this client script.
###############################################################################
python experiment_driver.py \
       --trials "${TRIALS:-1}" \
       --gpu_host "$GPU_HOST" \
       --master_port 29500 \
       --model "$MODEL" \
       --modes naive,remote_cache,remote_cache_delta,sys_simulated \
       --external_server \
       --trial_port_stride "${TRIAL_PORT_STRIDE:-0}" \
       --output stage2_results.csv

echo "Experiment complete. Results written to stage2_results.csv." 