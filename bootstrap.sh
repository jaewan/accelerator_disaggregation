#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Help function
show_help() {
    cat << EOF
Usage: ./bootstrap.sh [OPTIONS]

Options:
    -h, --help              Show this help message
    --clean                 Clean all installed dependencies and start fresh
    --skip-cuda            Force CPU-only installation

Environment variables:
    LIBTORCH_CUDA_VERSION  Override auto-detected CUDA version
                          Values: cpu, cu116, cu118, cu121
    PYTORCH_VERSION        Override PyTorch version (default: 2.0.0)
    PYTHON_VERSION        Specify Python version (default: 3.10)

Examples:
    # Default installation
    ./bootstrap.sh

    # Force CPU-only installation
    ./bootstrap.sh --skip-cuda

    # Install with specific CUDA version
    LIBTORCH_CUDA_VERSION=cu118 ./bootstrap.sh

    # Install with specific PyTorch version
    PYTORCH_VERSION=2.1.0 ./bootstrap.sh
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --skip-cuda)
            SKIP_CUDA=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done


# Check for Bazel
check_bazel(){
    if ! command -v bazel &> /dev/null; then
        echo -e "${RED}Bazel not found. Installing Bazel...${NC}"
        case "$(uname -s)" in
            Linux*)
                # First remove any existing bazel repositories to avoid conflicts
                sudo rm -f /etc/apt/sources.list.d/bazel.list
                sudo rm -f /etc/apt/trusted.gpg.d/bazel.gpg

                # Alternative installation using direct download
                VERSION=6.4.0
                wget https://github.com/bazelbuild/bazel/releases/download/${VERSION}/bazel-${VERSION}-installer-linux-x86_64.sh
                chmod +x bazel-${VERSION}-installer-linux-x86_64.sh
                ./bazel-${VERSION}-installer-linux-x86_64.sh --user
                rm bazel-${VERSION}-installer-linux-x86_64.sh

                # Add Bazel to PATH for current session
                export PATH="$HOME/bin:$PATH"
                
                # Add Bazel to PATH permanently
                if ! grep -q "export PATH=\"\$HOME/bin:\$PATH\"" ~/.bashrc; then
                    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
                fi

                # Add bash completion
                if ! grep -q "source \$HOME/.bazel/bin/bazel-complete.bash" ~/.bashrc; then
                    echo 'source $HOME/.bazel/bin/bazel-complete.bash' >> ~/.bashrc
                fi
                
                # Verify installation
                if ! command -v bazel &> /dev/null; then
                    echo -e "${RED}Bazel installation failed. Please install manually:${NC}"
                    echo "1. Visit: https://bazel.build/install/ubuntu"
                    echo "2. Follow the manual installation instructions"
                    echo "3. Make sure to run: source ~/.bashrc"
                    exit 1
                fi
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
    
	source ~/.bashrc
    # Print Bazel version
    BAZEL_VERSION=$(bazel --version)
    echo -e "${GREEN}Using ${BAZEL_VERSION}${NC}"
}

setup_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python3 not found. Installing...${NC}"
        case "$(uname -s)" in
            Linux*)
                sudo apt update
                sudo apt install -y python3 python3-pip
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

    # Install python3-venv if not present (Ubuntu/Debian specific)
    if [ -f "/etc/debian_version" ]; then
        if ! dpkg -l | grep -q python3-venv; then
            echo -e "${YELLOW}Installing python3-venv...${NC}"
            sudo apt update
            sudo apt install -y python3-venv
        fi
    fi

    # Remove existing virtual environment if it exists
    if [ -d ".venv" ]; then
        echo -e "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf .venv
    fi

    # Create virtual environment
    echo -e "${GREEN}Creating virtual environment...${NC}"
    python3 -m venv .venv

    # Activate virtual environment
    source .venv/bin/activate

    # Upgrade pip
    echo -e "${GREEN}Upgrading pip...${NC}"
    pip install --upgrade pip

    # Install Python dependencies
    if [ ! -f requirements.txt ]; then
        echo -e "${GREEN}Creating requirements.txt...${NC}"
        cat > requirements.txt << EOF
torch>=2.0.0
pybind11>=2.10.0
zmq>=0.0.0
EOF
    fi

    echo -e "${GREEN}Installing Python dependencies...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}Python environment setup complete${NC}"
}

# Function to detect CUDA version
detect_cuda_version() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}NVIDIA GPU not detected. Using CPU-only version.${NC}"
        echo "cpu"
        return
	fi 

    # Try to get CUDA version
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1)
    if [ -z "$CUDA_VERSION" ]; then
        echo -e "${YELLOW}Failed to detect CUDA version. Using CPU-only version.${NC}"
        echo "cpu"
        return
	fi 

    # Map CUDA driver version to PyTorch CUDA version
    if [ "$CUDA_VERSION" -ge "520" ]; then
        echo "cu121"  # CUDA 12.1
    elif [ "$CUDA_VERSION" -ge "510" ]; then
        echo "cu118"  # CUDA 11.8
    elif [ "$CUDA_VERSION" -ge "450" ]; then
        echo "cu116"  # CUDA 11.6
    else
        echo -e "${YELLOW}CUDA version ${CUDA_VERSION} might not be compatible. Using CPU-only version.${NC}"
        echo "cpu"
    fi
}

install_libtorch() {
    LIBTORCH_DIR="./libtorch"
    if [ ! -d "${LIBTORCH_DIR}" ]; then
        echo -e "${GREEN}Installing Libtorch...${NC}"

        # Determine OS
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

        # Detect CUDA version and map to compatible PyTorch version
        CUDA_VERSION=$(detect_cuda_version)
        echo -e "${GREEN}Detected CUDA version: ${CUDA_VERSION}${NC}"

        # Allow override through environment variable
        if [ ! -z "$LIBTORCH_CUDA_VERSION" ]; then
            echo -e "${YELLOW}Overriding detected CUDA version with: ${LIBTORCH_CUDA_VERSION}${NC}"
            CUDA_VERSION=$LIBTORCH_CUDA_VERSION
        fi

        # Map CUDA version to compatible PyTorch version and CUDA tag
        case "$CUDA_VERSION" in
            "cu121")
                PYTORCH_VERSION="2.1.0"
                CUDA_TAG="cu121"
                ;;
            "cu118")
                PYTORCH_VERSION="2.0.0"
                CUDA_TAG="cu118"
                ;;
            "cu116")
                PYTORCH_VERSION="1.13.1"
                CUDA_TAG="cu116"
                ;;
            "cpu")
                PYTORCH_VERSION="2.0.0"
                CUDA_TAG="cpu"
                ;;
            *)
                echo -e "${YELLOW}Unsupported CUDA version ${CUDA_VERSION}, falling back to cu118${NC}"
                PYTORCH_VERSION="2.0.0"
                CUDA_TAG="cu118"
                ;;
        esac

        echo -e "${GREEN}Using PyTorch version: ${PYTORCH_VERSION} with CUDA tag: ${CUDA_TAG}${NC}"

        # Construct URL
        LIBTORCH_URL="https://download.pytorch.org/libtorch/${CUDA_TAG}/libtorch-shared-with-deps-${PYTORCH_VERSION}%2B${CUDA_TAG}.zip"

        echo "Downloading Libtorch from ${LIBTORCH_URL}"

        # Download with proper error handling and progress bar
        if ! wget --progress=bar:force:noscroll -O libtorch.zip "${LIBTORCH_URL}"; then
            echo -e "${RED}Failed to download Libtorch. Trying fallback version...${NC}"
            # Fallback to known working version
            PYTORCH_VERSION="2.0.0"
            CUDA_TAG="cu118"
            LIBTORCH_URL="https://download.pytorch.org/libtorch/${CUDA_TAG}/libtorch-shared-with-deps-${PYTORCH_VERSION}%2B${CUDA_TAG}.zip"
            echo "Trying fallback URL: ${LIBTORCH_URL}"

            if ! wget --progress=bar:force:noscroll -O libtorch.zip "${LIBTORCH_URL}"; then
                echo -e "${RED}Fallback download failed. Please check your internet connection.${NC}"
                rm -f libtorch.zip
                exit 1
            fi
        fi

        # Verify the download
        if [ ! -s libtorch.zip ]; then
            echo -e "${RED}Downloaded file is empty. Download failed.${NC}"
            rm -f libtorch.zip
            exit 1
        fi

        echo "Extracting Libtorch..."
        # Extract with error handling
        if ! unzip -q libtorch.zip; then
            echo -e "${RED}Failed to extract Libtorch. Archive may be corrupted.${NC}"
            rm -f libtorch.zip
            exit 1
        fi

        # Cleanup
        rm -f libtorch.zip
        echo -e "${GREEN}Libtorch downloaded and extracted to ./libtorch${NC}"
    else
        echo -e "${GREEN}Libtorch is already installed.${NC}"
    fi

    # Verify installation
    if [ ! -d "${LIBTORCH_DIR}/lib" ] || [ ! -d "${LIBTORCH_DIR}/include" ]; then
        echo -e "${RED}Libtorch installation appears to be incomplete.${NC}"
        exit 1
    fi

    echo -e "${GREEN}Libtorch installation verified${NC}"
}

# Main setup
main() {
	echo -e "\n${GREEN}Starting setup...${NC}"
    
    # Run checks and setup
    check_bazel
    setup_python
	install_libtorch

    # Build project
    #bazel build //...

    echo -e "\n${GREEN}All dependencies installed successfully!${NC}"
	echo -e "${GREEN}To activate the virtual environment, run: source .venv/bin/activate${NC}"
}

main
