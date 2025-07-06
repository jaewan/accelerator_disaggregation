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
GPU_HOST="${GPU_HOST:-10.8.162.218}"
MODEL="${MODEL:-facebook/opt-125m}"
VENV_ACTIVATE="${VENV_ACTIVATE:-venv/bin/activate}"

cd "$ROOT"

# Activate virtual environment if present
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_ACTIVATE"
fi

###############################################################################
# 1) Idempotent patch: give decode phase a separate port to avoid dead-lock.
###############################################################################
if ! grep -q "phase_offset" experiment_driver.py; then
  echo "Applying port-offset patch to experiment_driver.py …"
  patch -N -p0 <<'PATCH_EOF'
--- a/experiment_driver.py
+++ b/experiment_driver.py
@@
-        base_port = int(args.master_port)
-        mode_offset = {"naive": 0, "remote_cache": 10, "sys_simulated": 20}.get(mode, 30)
-        run_port = str(base_port + mode_offset)
+        base_port = int(args.master_port)
+        mode_offset = {"naive": 0, "remote_cache": 10, "sys_simulated": 20}.get(mode, 30)
+        # Use a different port for the decode phase to avoid TensorPipe dead-lock
+        phase_offset = 0 if phase == "prefill" else 5
+        run_port = str(base_port + mode_offset + phase_offset)
PATCH_EOF
else
  echo "Patch already present – skipping."
fi

###############################################################################
# 2) Run the experiment (5 trials, three modes, two phases)
###############################################################################
python experiment_driver.py \
       --trials 5 \
       --gpu_host "$GPU_HOST" \
       --master_port 29500 \
       --model "$MODEL" \
       --modes naive,remote_cache,sys_simulated \
       --external_server \
       --output stage2_results.csv

echo "Experiment complete. Results written to stage2_results.csv." 