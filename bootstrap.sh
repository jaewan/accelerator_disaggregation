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
				if ! command -v brew &> /dev/null; then
                    echo -e "${RED}Homebrew not found. Installing Homebrew...${NC}"
                    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                fi
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
                case "$(uname -s)" in
                        Linux*)
                                sudo apt update
                                sudo apt install -y python3 python3-pip python3-venv
                                ;;
                        Darwin*)
                                brew install python@3.10
                                ;;
                        *)
                                echo -e "${RED}Unsupported OS for Python installation${NC}"
                                exit 1
                                ;;
                esac
        fi

        # Create virtual environment
        python3 -m venv .venv
        source .venv/bin/activate
		pip install --upgrade pip

        # Install Python dependencies
        if [ ! -f requirements.txt ]; then
                cat > requirements.txt << EOF
torch>=2.0.0
pybind11>=2.10.0
zmq>=0.0.0
EOF
        fi

        pip install -r requirements.txt
        echo -e "${GREEN}Python environment setup complete${NC}"
}

# Check for libtorch
# This is more complex, needs to consider pre-built binaries or build from source
# For simplicity, let's assume we expect a prebuilt library to be downloaded
install_libtorch() {
    LIBTORCH_DIR="./libtorch"
    if [ ! -d "${LIBTORCH_DIR}" ]; then
        echo -e "${RED}Libtorch not found. Downloading Libtorch...${NC}"

        OS=""
        case "$(uname -s)" in
            Linux*)
                OS="linux"
                ;;
            Darwin*)
                OS="macos"
                ;;
            *)
                echo -e "${RED}Unsupported OS for Libtorch installation. Please install manually.${NC}"
                exit 1
                ;;
        esac

        CUDA_VERSION="cpu"
        if command -v nvidia-smi &> /dev/null; then
            CUDA_VERSION="cu118" # Adjust CUDA version as necessary
        fi

        LIBTORCH_URL="https://download.pytorch.org/libtorch/${CUDA_VERSION}/${OS}/libtorch-${OS}-${CUDA_VERSION}-latest.zip"
        echo "Downloading Libtorch from ${LIBTORCH_URL}"
        curl -L -o libtorch.zip "${LIBTORCH_URL}"
        unzip libtorch.zip
        mv libtorch-* libtorch
        rm libtorch.zip
        echo -e "${GREEN}Libtorch downloaded and extracted to ./libtorch.${NC}"
    else
        echo -e "${GREEN}Libtorch is already installed.${NC}"
    fi
}


# Main setup
main() {
	echo -e "\n${GREEN}Starting setup...${NC}"
    
    # Run checks and setup
    check_bazel
    setup_python
	check_libtorch

    # Build project
    #bazel build //...

    echo -e "\n${GREEN}All dependencies installed successfully!${NC}"
	echo -e "${GREEN}To activate the virtual environment, run: source .venv/bin/activate${NC}"
}

main
