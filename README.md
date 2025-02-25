# AI Accelerator Disaggregation Framework

## Features
- Remote GPU acceleration support
- PyTorch integration
- Low-latency network communication
- Dynamic resource allocation

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
```

	**Fedora**:
	  ```bash
	  sudo dnf install python3.10-devel
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
*Install Bazel if not present
*Set up Python environment
*Configure build dependencies

2. Build the project
    ```bash
    bazel build //...
    ```
3. Run the examples:
Start the remote server:
    ```bash
    bazel run //:remote_server
    ```
In a separate terminal, run the example client:
    ```bash
    bazel run //:basic_example
    ```
##License
This project is licensed under the Apache License 2.0
