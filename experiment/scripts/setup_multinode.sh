#!/bin/bash

# Multi-node setup script for Semantic Gap Experiments
# =====================================================
# Run this script on both CLIENT_HOST and GPU_HOST

set -e  # Exit on any error

echo "🚀 Setting up multi-node environment for Semantic Gap Experiments"
echo "=================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "experiment_driver.py" ]; then
    echo -e "${RED}❌ Error: experiment_driver.py not found."
    echo "Please run this script from the experiment/ directory.${NC}"
    exit 1
fi

echo -e "${YELLOW}📋 Step 1: Checking prerequisites...${NC}"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

# Check if Python 3.8+
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "${RED}❌ Error: Python 3.8+ required${NC}"
    exit 1
fi

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
else
    echo -e "${YELLOW}⚠️  Warning: nvidia-smi not found (CPU-only mode)${NC}"
fi

echo -e "${YELLOW}📦 Step 2: Setting up Python environment...${NC}"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv venv
else
    echo "   Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "   Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "   Installing dependencies..."
pip install -r requirements.txt

echo -e "${YELLOW}📡 Step 3: Configuring network settings...${NC}"

# Get local IP address
local_ip=$(ip route get 1 | awk '{print $7}' | head -1)
echo "   Local IP address: $local_ip"

# Check if firewall needs configuration
if command -v ufw &> /dev/null; then
    echo "   Configuring firewall for RPC ports..."
    # Allow RPC port range
    sudo ufw allow 29500:29530/tcp || echo "   (Firewall configuration may need manual setup)"
else
    echo "   UFW not found - please ensure ports 29500-29530 are open"
fi

echo -e "${YELLOW}🔍 Step 4: Validating environment...${NC}"

# Run validation script
python scripts/validate_multinode.py

echo -e "${GREEN}✅ Multi-node setup completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "==========="
echo "1. Run this script on both CLIENT_HOST and GPU_HOST"
echo "2. Test connectivity between hosts:"
echo "   python scripts/validate_multinode.py --gpu-host <GPU_HOST_IP>"
echo "3. Run experiments:"
echo "   python experiment_driver.py --trials 3 --gpu-host <GPU_HOST_IP> --model sshleifer/tiny-gpt2"
echo ""
echo "Remember to activate the virtual environment before running experiments:"
echo "   source venv/bin/activate" 