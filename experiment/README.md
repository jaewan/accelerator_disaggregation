# Semantic Gap Experiment Framework

A research framework for quantifying the benefits of framework-level semantic awareness in accelerator disaggregation. This project evaluates how semantic-aware runtimes can reduce network traffic and end-to-end latency compared to semantic-blind baselines.

## 🎯 Project Overview

This framework implements and compares four execution modes:

1. **Local (Baseline)** - Model and KV cache on same GPU (no network)
2. **Semantic-Blind Naive** - Transfers weights + input + KV cache every step 
3. **Semantic-Blind Remote-Cache** - Realistic baseline with persistent weights and remote KV cache
4. **Framework-Level Semantic-Aware** - Applies tensor compression, delta transfers, and semantic optimizations

**Key Research Questions:**
- Can semantic awareness reduce network traffic by ≥30%?
- Does reduced traffic lead to ≥15% lower latency?

## 📁 Code Structure

```
experiment/
├── 🚀 Core Components
│   ├── experiment_driver.py    # Main experiment orchestrator
│   ├── rpc_server.py          # GPU-side RPC server with worker logic
│   └── run_llm.py             # Client-side execution modes
├── 🔧 Analysis Tools  
│   ├── analyse_results.py     # Statistical analysis and plotting
│   └── parse_dmon.py          # GPU utilization parsing
├── 🧪 Testing
│   └── tests/                 # Comprehensive unit & integration tests
│       ├── test_stage1_*.py   # Resource hygiene tests
│       ├── test_stage2_*.py   # Network byte counting tests  
│       ├── test_stage3_*.py   # Server-side weight loading tests
│       └── test_stage4_*.py   # Persistent server tests
├── ��️ Utilities
│   └── scripts/               # Smoke tests and CI scripts
├── 📋 Documentation
│   ├── Plan.md               # Research methodology and goals
│   └── Enhancement_Plan.md   # Technical implementation roadmap
└── ⚙️ Configuration
    ├── requirements.txt       # Python dependencies
    └── .gitignore            # Git ignore patterns
```

## 🚀 Quick Start

### Prerequisites

- **Hardware**: GPU with CUDA support (for GPU modes)
- **OS**: Linux (tested on Ubuntu)  
- **Python**: 3.8+
- **Network**: Multi-node setup requires network connectivity between hosts

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd accelerator_disaggregation/experiment
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or: venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   ./scripts/smoke_test.sh
   ```

## 🧪 Running Experiments

### Single-Node Experiments

**Basic run with tiny model (recommended for testing):**
```bash
python experiment_driver.py \
    --trials 3 \
    --model sshleifer/tiny-gpt2 \
    --output results.csv
```

**Full experiment with production model:**
```bash
python experiment_driver.py \
    --trials 5 \
    --model EleutherAI/gpt-j-6b \
    --output results.csv
```

**Run specific modes only:**
```bash
python experiment_driver.py \
    --trials 3 \
    --modes "local,remote_cache,sys_simulated" \
    --model sshleifer/tiny-gpt2 \
    --output results.csv
```

### Multi-Node Experiments

#### Option 1: Managed Servers (Recommended)

The framework automatically starts/stops servers:

**On CLIENT_HOST:**
```bash
python experiment_driver.py \
    --trials 3 \
    --gpu_host <GPU_HOST_IP> \
    --master_port 29501 \
    --model sshleifer/tiny-gpt2 \
    --output results.csv
```

#### Option 2: External Server Mode

For more control, manually start servers:

**On GPU_HOST:**
```bash
python rpc_server.py \
    --model sshleifer/tiny-gpt2 \
    --master_addr 0.0.0.0 \
    --master_port 29501
```

**On CLIENT_HOST:**
```bash
python experiment_driver.py \
    --trials 3 \
    --gpu_host <GPU_HOST_IP> \
    --master_port 29501 \
    --external_server \
    --output results.csv
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--trials` | Number of trials per (mode, phase) | 5 |
| `--gpu_host` | GPU server IP address | 127.0.0.1 |
| `--master_port` | Base port for RPC communication | 29501 |
| `--model` | HuggingFace model name | sshleifer/tiny-gpt2 |
| `--modes` | Comma-separated modes or "all" | all |
| `--output` | Output CSV file | results.csv |
| `--external_server` | Use manually started servers | False |

## 📊 Results Analysis

After running experiments, analyze results:

```bash
# Generate statistical analysis and plots
python analyse_results.py results.csv

# Parse individual GPU utilization logs  
python parse_dmon.py artefacts/local_prefill_trial1.csv
```

The analysis generates:
- Statistical summaries with confidence intervals
- Latency vs. network traffic scatter plots  
- Performance comparison tables
- Hypothesis testing results

## 🧪 Development & Testing

### Running Tests

```bash
# Run all tests
python -m unittest discover tests/ -v

# Run specific test stages
python -m unittest tests.test_stage1_resource_hygiene -v
python -m unittest tests.test_stage2_network_bytes -v  
python -m unittest tests.test_stage3_server_weights -v
python -m unittest tests.test_stage4_persistent_servers -v
```

### Smoke Testing

```bash
# Quick functionality test
./scripts/smoke_test.sh

# CI test suite  
./scripts/ci_test.sh
```

## 🏗️ Architecture Details

### Framework Stages (Implementation Phases)

- **Stage 1**: Resource hygiene - proper cleanup of background processes
- **Stage 2**: True network byte counting - real RPC metrics vs. synthetic estimates  
- **Stage 3**: Client-side weight removal - server-side weight loading
- **Stage 4**: Persistent servers - reuse servers across trials

### Execution Flow

1. **ServerPool** starts persistent RPC servers for each mode
2. **experiment_driver.py** orchestrates trials across modes and phases
3. **run_llm.py** implements client-side logic for each execution mode
4. **rpc_server.py** handles GPU-side model execution and caching
5. GPU utilization monitored via `nvidia-smi dmon`
6. Results aggregated into CSV for analysis

### Network Communication

- **PyTorch RPC** with TensorPipe backend for reliable communication
- **Unique ports** per mode to avoid socket conflicts  
- **Real byte counting** from RPC agent metrics
- **Compression pipeline** for semantic-aware mode

## 🔧 Configuration

### Model Selection

Supported models (via HuggingFace):
- `sshleifer/tiny-gpt2` - Tiny model for testing (recommended)
- `gpt2` - Small GPT-2 model
- `EleutherAI/gpt-j-6b` - Production-size model

### Network Setup

- **Ports**: Base port + mode offset (naive: +0, remote_cache: +10, sys_simulated: +20)
- **Firewall**: Ensure RPC ports are open between hosts
- **DNS**: GPU host should be reachable by hostname/IP

## 📈 Performance Expectations

### Stage 4 Benefits

- **~30% faster** multi-trial experiments (persistent servers)
- **Linear scaling** with trial count (no startup overhead per trial)  
- **Better resource utilization** (reused server processes)

### Target Improvements

- **Network Traffic**: 30%+ reduction in sys_simulated vs remote_cache
- **Latency**: 15%+ improvement from reduced network overhead
- **GPU Utilization**: Higher SM utilization due to better batching

## 🚨 Troubleshooting

### Common Issues

**RPC Connection Errors:**
```bash
# Check server logs in artefacts/logs/
tail -f artefacts/logs/server_*_*.log

# Verify network connectivity  
ping <gpu_host>
telnet <gpu_host> <port>
```

**GPU Memory Errors:**
```bash
# Use smaller model
--model sshleifer/tiny-gpt2

# Check GPU memory
nvidia-smi
```

**Permission Errors:**
```bash
# Ensure nvidia-smi is accessible
which nvidia-smi
nvidia-smi --help
```

### Debug Mode

Enable verbose logging:
```bash
export PYTHONPATH=.
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
python experiment_driver.py <args>
```

## 📚 References

- **Plan.md** - Research methodology and evaluation goals
- **Enhancement_Plan.md** - Technical implementation roadmap  
- **PyTorch RPC Documentation** - [pytorch.org/docs/stable/rpc.html](https://pytorch.org/docs/stable/rpc.html)

---

**Status**: Production-ready after Stage 4 ✅  
**Last Updated**: 2024-01-15
