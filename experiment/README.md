# Semantic Translation Gap Experiment

This project quantifies the performance impact of semantically-unaware GPU disaggregation systems compared to semantically-aware approaches. The experiment measures wall-clock latency, network bytes transferred, and GPU utilization across three execution modes for LLM inference.

## Experiment Overview

**Objective**: Demonstrate that semantically-unaware GPU disaggregation transfers significantly more data across the network, leading to higher latency and lower accelerator utilization compared to semantic-aware approaches.

**Methodology**: We implement and test three execution modes for LLM inference (Prefill and Decode phases):

1. **LOCAL** - Baseline execution on GPU host (no network)
2. **NAÏVE-REMOTE** - Stateless RPC simulation (sends full model weights + KV cache every step)
3. **SYS_SIMULATED** - Semantic-aware simulation (loads weights once, manages KV cache statefully)

## Quick Start

### Prerequisites

- Ubuntu 22.04+ with Python 3.10+
- NVIDIA GPU with CUDA 12.2+
- `nvidia-smi` installed for GPU utilization monitoring
- Network measurement is done via source code instrumentation (no external tools required)

### Installation

```bash
# Clone and setup
git clone <repository>
cd experiment
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Experiment

**Option 1: Automated (Recommended)**
```bash
# Run all modes and phases with 3 trials
python experiment_driver.py \
    --trials 3 \
    --gpu_host 127.0.0.1 \
    --master_port 29501 \
    --model sshleifer/tiny-gpt2 \
    --output results.csv
```

**Option 2: Manual Step-by-Step**
```bash
# Terminal 1: Start RPC server
python rpc_server.py \
    --model sshleifer/tiny-gpt2 \
    --master_addr 127.0.0.1 \
    --master_port 29501 \
    --world_size 2 --rank 0

# Terminal 2: Run individual tests
python run_llm.py --mode naive --phase prefill \
    --gpu_host 127.0.0.1 --master_port 29501 \
    --model sshleifer/tiny-gpt2
```

## Understanding Results

### Output Files

- `results.csv` - Main results table with all metrics
- `artefacts/` - GPU utilization logs from nvidia-smi dmon
- `server_stdout.log` - RPC server logs

### Results Format

| Column | Description |
|--------|-------------|
| `trial` | Trial number (1, 2, 3, ...) |
| `phase` | `prefill` or `decode` |
| `mode` | `local`, `naive`, or `sys_simulated` |
| `latency_s` | Wall-clock time measured by `/usr/bin/time` |
| `wall_s` | Additional wall time including measurement overhead |
| `net_bytes` | Total network bytes transferred (measured via source code instrumentation) |
| `avg_sm` | Average GPU SM utilization percentage |

### Network Measurement Method

Network bytes are measured using **source code instrumentation** rather than external tools like `tcpdump`. This approach:

- **Measures actual payload sizes** of tensors and data structures sent over RPC
- **Works in any environment** (cloud, containers, etc.) without special permissions
- **Provides accurate byte counts** for the semantic gap analysis
- **Eliminates race conditions** from external monitoring tools

The measurement captures:
- **NAÏVE-REMOTE**: Full model weights + KV cache transferred on every call
- **SYS_SIMULATED**: Only tokens and cache IDs after initial weight loading
- **LOCAL**: 0 bytes (no network transfer)

### Expected Results

For the semantic gap experiment, you should observe:

- **NAÏVE-REMOTE** transfers significantly more network bytes than **SYS_SIMULATED**
- **SYS_SIMULATED** achieves lower latency and higher GPU utilization
- **LOCAL** serves as the performance baseline

### Analyzing Results

```python
import pandas as pd

# Load results
df = pd.read_csv('results.csv')

# Compare network transfer across modes
network_comparison = df.groupby('mode')['net_bytes'].mean()
print("Average network bytes transferred:")
print(network_comparison)

# Calculate semantic gap ratio
naive_bytes = network_comparison['naive']
sys_bytes = network_comparison['sys_simulated']
semantic_gap_ratio = naive_bytes / sys_bytes if sys_bytes > 0 else float('inf')
print(f"\nSemantic gap ratio (naive/sys_simulated): {semantic_gap_ratio:.2f}x")

# Compare latency across modes
latency_comparison = df.groupby('mode')['latency_s'].mean()
print("\nAverage latency (seconds):")
print(latency_comparison)

# Compare GPU utilization
gpu_comparison = df.groupby('mode')['avg_sm'].mean()
print("\nAverage GPU utilization (%):")
print(gpu_comparison)
```

## Project Structure

```
experiment/
├── rpc_server.py          # GPU host RPC server
├── run_llm.py            # Client orchestration script with network measurement
├── experiment_driver.py   # Automated experiment runner
├── parse_dmon.py         # GPU utilization parser
├── requirements.txt      # Python dependencies
├── detailed_plan.md      # Detailed experiment methodology
└── README.md            # This file
```

## Configuration

### Models
- **Development**: `sshleifer/tiny-gpt2` (~3MB, fast iteration)
- **Production**: `EleutherAI/gpt-j-6B` (~24GB, realistic workload)

### Network Measurement
- **Method**: Source code instrumentation (tensor.nbytes)
- **Scope**: Actual RPC payload sizes (state_dict, tensors, KV cache)
- **Accuracy**: Byte-exact measurement of transferred data

### RPC Backends
- **Default**: `gloo` (CPU-based, works everywhere)
- **High-performance**: `nccl` (GPU-based, requires CUDA)

## Troubleshooting

### Common Issues

1. **RPC connection errors**
   - Ensure server is running before client
   - Check `--master_port` matches between server and client
   - Verify firewall allows the port

2. **GPU not found**
   - Verify `nvidia-smi` works
   - Check CUDA installation
   - Ensure model fits in GPU memory

3. **Network measurement issues**
   - Network bytes are measured in source code (no external dependencies)
   - Check for "NETWORK_BYTES:" in client output for debugging

### Debug Mode

Enable verbose logging:
```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CPP_LOG_LEVEL=INFO
python experiment_driver.py ...
```

## Extending the Experiment

### Adding New Models
1. Update `--model` parameter
2. Ensure model fits in GPU memory
3. Adjust `--dtype` if needed (float16/bfloat16)

### Custom Metrics
1. Modify `_measure_tensor_bytes()` in `run_llm.py` for different measurement strategies
2. Update CSV fieldnames and parsing logic in `experiment_driver.py`
3. Add new measurement tools as needed

### Multi-GPU Support
1. Update `rpc_server.py` to use `nccl` backend
2. Modify `world_size` and rank parameters
3. Extend `RemoteWorker` for multi-GPU model parallelism

## Citation

If you use this experiment in your research, please cite:

```bibtex
@misc{semantic_gap_experiment,
  title={Quantifying the Semantic Translation Gap in GPU Disaggregation},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/accelerator_disaggregation}
}
``` 