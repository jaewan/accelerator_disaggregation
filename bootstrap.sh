#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Help function
show_help() {
    cat << EOF
Usage: ./bootstrap.sh [OPTIONS]

Options:
    -h, --help              Show this help message
    --clean                 Clean all installed dependencies and start fresh
    --skip-cuda             Force CPU-only installation

Environment variables:
    LIBTORCH_CUDA_VERSION   Override auto-detected CUDA version
                            Values: cpu, cu116, cu118, cu121
    PYTORCH_VERSION         Override PyTorch version (default: 2.0.0)
    PYTHON_VERSION          Specify Python version (default: 3.10)

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

# Parse command-line arguments
CLEAN=false
SKIP_CUDA=false
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

# Function to check for Bazel
check_bazel() {
    if ! command -v bazel &> /dev/null; then
        echo -e "${YELLOW}Bazel not found. Installing Bazel...${NC}"
        case "$(uname -s)" in
            Linux*)
                # Install Bazel
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
                ;;
            Darwin*)
                if ! command -v brew &> /dev/null; then
                    echo -e "${YELLOW}Homebrew not found. Installing Homebrew...${NC}"
                    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
                fi
                brew install bazel
                ;;
            *)
                echo -e "${RED}Unsupported OS for automatic Bazel installation. Please install manually.${NC}"
                exit 1
                ;;
        esac
    else
        echo -e "${GREEN}Bazel is already installed.${NC}"
    fi
}

# Function to set up Python environment
setup_python() {
    if [ -d ".venv" ]; then
        echo -e "${GREEN}Python virtual environment already exists. Skipping creation.${NC}"
    else
        echo -e "${GREEN}Creating Python virtual environment...${NC}"
        python3 -m venv .venv
        source .venv/bin/activate
        echo -e "${GREEN}Upgrading pip...${NC}"
        pip install --upgrade pip
        echo -e "${GREEN}Installing Python dependencies...${NC}"
        pip install -r requirements.txt
    fi
}

# Function to detect CUDA version
detect_cuda_version() {
    if [ "$SKIP_CUDA" = true ]; then
        echo "cpu"
        return
    fi
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}NVIDIA GPU not detected. Using CPU-only version.${NC}"
        echo "cpu"
        return
    fi
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1)
    if [ -z "$CUDA_VERSION" ]; then
        echo -e "${YELLOW}Failed to detect CUDA version. Using CPU-only version.${NC}"
        echo "cpu"
        return
    fi
    if [ "$CUDA_VERSION" -ge "520" ]; then
        echo "cu121"
    elif [ "$CUDA_VERSION" -ge "510" ]; then
        echo "cu118"
    elif [ "$CUDA_VERSION" -ge "450" ]; then
        echo "cu116"
    else
        echo -e "${YELLOW}CUDA version ${CUDA_VERSION} might not be compatible. Using CPU-only version.${NC}"
        echo "cpu"
    fi
}

# Function to install Libtorch
install_libtorch() {
    LIBTORCH_DIR="./libtorch"
    CUDA_TAG=${LIBTORCH_CUDA_VERSION:-$(detect_cuda_version)}
    PYTORCH_VERSION=${PYTORCH_VERSION:-2.0.0}
    LIBTORCH_URL="https://download.pytorch.org/libtorch/${CUDA_TAG}/libtorch-shared-with-deps-${PYTORCH_VERSION}%2B${CUDA_TAG}.zip"

    # Persist LIBTORCH_URL in ~/.bashrc
    if ! grep -q "export LIBTORCH_URL=" ~/.bashrc; then
        echo "export LIBTORCH_URL=$LIBTORCH_URL" >> ~/.bashrc
        echo -e "${GREEN}LIBTORCH_URL added to ~/.bashrc${NC}"
    else
        sed -i "s|^export LIBTORCH_URL=.*|export LIBTORCH_URL=$LIBTORCH_URL|" ~/.bashrc
        echo -e "${GREEN}LIBTORCH_URL updated in ~/.bashrc${NC}"
    fi

    # Export LIBTORCH_URL for the current shell session
    export LIBTORCH_URL=$LIBTORCH_URL

    if [ "$CLEAN" = true ] || [ ! -d "$LIBTORCH_DIR" ]; then
        echo -e "${GREEN}Installing Libtorch...${NC}"
        mkdir -p "$LIBTORCH_DIR"
        wget -O libtorch.zip "$LIBTORCH_URL"
        unzip -o libtorch.zip -d "$LIBTORCH_DIR"
        rm libtorch.zip
    else
        echo -e "${GREEN}Libtorch is already installed. Skipping download.${NC}"
    fi
}

# Main setup
main() {
    if [ "$CLEAN" = true ]; then
        echo -e "${YELLOW}Cleaning up dependencies...${NC}"
        rm -rf .venv libtorch
    fi

    echo -e "${GREEN}Starting setup...${NC}"
    check_bazel
    setup_python
    install_libtorch

    echo -e "\n${GREEN}All dependencies are installed successfully!${NC}"
    echo -e "${GREEN}To activate the virtual environment, run: source .venv/bin/activate${NC}"
}

main
