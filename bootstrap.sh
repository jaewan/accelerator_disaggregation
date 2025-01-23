#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Check for Bazel
check_bazel(){
	if ! command -v bazel &> /dev/null; then
		echo -e "${RED}Bazel not found. Installing Bazel...${NC}"
		# Installation logic for Bazel (OS-specific)
		case "$(uname -s)" in
			Linux*)
				# Example for Ubuntu. Customize this for your needs
				sudo apt-get update
				sudo apt-get install -y curl gnupg
				curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-release.gpg
				sudo mv bazel-release.gpg /usr/share/keyrings
				echo "deb [signed-by=/usr/share/keyrings/bazel-release.gpg] https://storage.googleapis.com/bazel-apt stable jdk-11" | sudo tee /etc/apt/sources.list.d/bazel.list
				sudo apt-get update
				sudo apt-get install -y bazel
				;;
			Darwin*)
				# Example for MacOS. Customize this for your needs
				brew install bazel
				;;
			*)
				echo -e "${RED}Unsupported OS for automatic Bazel installation. Please install manually.${NC}"
				exit 1
				;;
		esac
		echo -e "${GREEN}Bazel installation verified${NC}"
	fi
}

# Check/Install Python dependencies
setup_python() {
	if ! command -v python3 &> /dev/null; then
		echo -e "${RED}Python3 not found. Installing...${NC}"
		if [[ "$OSTYPE" == "darwin"* ]]; then
			brew install python3
		else
			sudo apt update
			sudo apt install python3 python3-pip
		fi
	fi

	# Create virtual environment
	python3 -m venv .venv
	source .venv/bin/activate

	# Install Python dependencies
	pip install -r requirements.txt

	echo -e "${GREEN}Python environment setup complete${NC}"
}

# Check for libtorch
# This is more complex, needs to consider pre-built binaries or build from source
# For simplicity, let's assume we expect a prebuilt library to be downloaded
check_libtorch(){
	LIBTORCH_DIR="./libtorch"
	if [ ! -d "${LIBTORCH_DIR}" ]; then
		echo "Libtorch not found. Downloading and extracting pre-built library..."
		# Determine OS
		OS=""
		case "$(uname -s)" in
			Linux*)
				OS="linux"
				;;
			Darwin*)
				OS="macos"
				;;
			*)
				echo "Unsupported OS for automatic libtorch download. Please install manually."
				exit 1
				;;
		esac
		# Determine CUDA version
		CUDA_VERSION="cpu"
		if command -v nvidia-smi &> /dev/null; then
			CUDA_VERSION="cu118" # Update this as necessary
		fi

		LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/${OS}/libtorch-${OS}-cpu-latest.zip"
		if [ "$CUDA_VERSION" != "cpu" ]; then
			LIBTORCH_URL="https://download.pytorch.org/libtorch/${CUDA_VERSION}/${OS}/libtorch-${OS}-${CUDA_VERSION}-latest.zip"
		fi

		echo "Downloading libtorch from ${LIBTORCH_URL}"
		curl -L -o libtorch.zip "${LIBTORCH_URL}"
		unzip libtorch.zip
		mv libtorch-* libtorch
		rm libtorch.zip
		echo "Libtorch downloaded and extracted to ./libtorch."
	fi
}


# Check for pybind11
# If you use pybind11, you can check and install if necessary, we are assuming that the pybind will be setup with bazel
check_pybind(){
	if ! command -v python3 &> /dev/null; then
		echo -e "${RED}Python3 not found, please install python3${NIC}"
		exit 1
	fi
}


# Main setup
main() {
    # Create requirements.txt if it doesn't exist
    if [ ! -f requirements.txt ]; then
        cat > requirements.txt << EOF
torch>=2.0.0
pybind11>=2.10.0
zmq>=0.0.0
EOF
    fi

    # Run checks and setup
    check_bazel
    setup_python

    # Build project
    #bazel build //...

    echo -e "${GREEN}Setup completed successfully!${NC}"
}

main
