# Accelerator Disaggregation framework for Pytorch.


## Introduction
We present a novel accelerator disaggregation framework that leverages unified dispatcher to transparently offload PyTorch computations to remote accelerators.
Our approach extends the computational graph interception capabilities of PyTorch's dispatcher system with a distributed future.
By combining transparent operation interception with a future-based execution model, we achieve both programmability and network efficiency without requiring modifications to user code.

## Features
**Transparency**: Custom Device Extension with Unified Dispatch

- Leverages PyTorch's dispatcher system to intercept operations at a high level (do not need to manually implement each tensor operation unlike pure custom device extension)
- It uses fallback mechanism as primary scheme to capture all tensor operations to forward to remote GPU server
- Provides proper memory management through the custom device type
- Allows for a clean separation between PyTorch's API and your remote execution logic
- Users can offload its pytorch AI workload to remote accelerator server transparently by only indicating the device type to our custom remote\_accelerator. 
Once the device is registered system will transparently intercept tensor operations and forward to remote server.

**Network Optimization**: Distributed futures and zero-copy data transfer

- Distributed futures allow us to reduce the number of network transfer between client and remote accelerator server. 
System decides when and where to materialize the tensors. Our framework that intercepts tensor operations and forward it to remote accelerator server. 
Instead of returning actual value (which takes a long time to be executed remotely)), custom device extension returns a future. 
System strictly transfers the data when it is absolutely necessary.

- Zero-copy transfer: DPDK + Pinned memory enables efficient data transfer from client to remote accelerator

## Prerequisites
- Python 3.10
- CUDA-capable GPU (optional)
- Bazel build system


## Installation

### System Dependencies

**Debian/Ubuntu**:
```bash
sudo apt-get update
sudo apt-get install python3.10-dev
sudo apt-get install -y dpdk dpdk-dev libdpdk-dev
```
**MacOS**:
  ```bash
  brew install python@3.10
  ```
### Setup Steps
1. Initialize the environment:
    ```bash
    ./bootstrap.sh
    ```
This script will:
- Install Bazel if not present
- Set up Python environment
- Configure build dependencies

2. Build the project
    ```bash
	source .venv/bin/activate
    bazel build //:remote_cuda
    ```

3. If build fails, try
    ```
	source .venv/bin/activate
	rm requirements.txt
    pip-compile --extra-index-url=https://download.pytorch.org/whl/cu121 --output-file=requirements.txt requirements.in
	deactivate
	./bootstrap.sh
	source .venv/bin/activate
    bazel build //:remote_cuda
    ```
## TODO
### Feature
- Operation mapping: map Pytorch ops to remote execution
- Distributed Future implementation
- Memory management strategies for remote device
- Register kernels for specific operations that need special handling on remote device

### Optimization
- Serialization efficiency
- Optimize data transfer to remote GPU using DPDK

### For Fully Open Source
- Make our device type unique(contribute to Pytorch). Currently we use PrivateUse1 for prototype. Register a unique device type (This will involve contributing to PyTorch itself)
- Because we use PrivateUse1 which is experimental, the repo registers the device using Aten::detail which may change later without notice (non-public API).
