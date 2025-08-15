#!/bin/bash

# Smoke test script for the semantic gap experiment framework
# This script runs a minimal experiment to verify the framework works

set -e  # Exit on any error

echo "🧪 Running smoke test for semantic gap experiment framework..."

# Check if we're in the right directory
if [ ! -f "experiment_driver.py" ]; then
    echo "❌ Error: experiment_driver.py not found. Please run this script from the experiment/ directory."
    exit 1
fi

# Check if Python and required packages are available
echo "📋 Checking dependencies..."
python -c "import torch, transformers" || {
    echo "❌ Error: Required packages not found. Please install with: pip install -r requirements.txt"
    exit 1
}

# Check if nvidia-smi is available (optional for CPU-only runs)
if command -v nvidia-smi &> /dev/null; then
    echo "✅ nvidia-smi found - GPU monitoring available"
    GPU_AVAILABLE=true
else
    echo "⚠️  nvidia-smi not found - GPU monitoring will be skipped"
    GPU_AVAILABLE=false
fi

# Create output directory
mkdir -p artefacts

echo "🚀 Running minimal experiment..."
python experiment_driver.py \
    --trials 1 \
    --modes local \
    --model sshleifer/tiny-gpt2 \
    --output smoke_results.csv \
    --output_dir artefacts

# Check if results file was created and has expected content
if [ ! -f "smoke_results.csv" ]; then
    echo "❌ Error: results.csv not created"
    exit 1
fi

# Check if results file has at least one data row (plus header)
LINE_COUNT=$(wc -l < smoke_results.csv)
if [ "$LINE_COUNT" -lt 2 ]; then
    echo "❌ Error: results.csv has insufficient data (only $LINE_COUNT lines)"
    exit 1
fi

echo "✅ Smoke test completed successfully!"
echo "📊 Results saved to: smoke_results.csv"
echo "📁 Artefacts saved to: artefacts/"

# Display a summary of the results
echo ""
echo "📈 Results summary:"
head -5 smoke_results.csv

echo ""
echo "🎉 Stage 0 baseline health-check PASSED!" 