#!/bin/bash

# CI test script for automated testing environments
# This script runs Stage 0 tests and provides proper exit codes for CI

set -e  # Exit on any error

echo "🔧 Running CI tests for Stage 0..."

# Activate virtual environment if available
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Check if we're in the right directory
if [ ! -f "experiment_driver.py" ]; then
    echo "❌ Error: experiment_driver.py not found. Please run this script from the experiment/ directory."
    exit 1
fi

# Check if Python and required packages are available
echo "📋 Checking dependencies..."
python -c "import torch, transformers" || {
    echo "❌ Error: Required packages not found."
    exit 1
}

# Create clean output directory
rm -rf artefacts_ci
mkdir -p artefacts_ci

echo "🚀 Running CI experiment..."
python experiment_driver.py \
    --trials 1 \
    --modes local \
    --model sshleifer/tiny-gpt2 \
    --output ci_results.csv \
    --output_dir artefacts_ci

# Check if results file was created and has expected content
if [ ! -f "ci_results.csv" ]; then
    echo "❌ Error: ci_results.csv not created"
    exit 1
fi

# Check if results file has at least one data row (plus header)
LINE_COUNT=$(wc -l < ci_results.csv)
if [ "$LINE_COUNT" -lt 2 ]; then
    echo "❌ Error: ci_results.csv has insufficient data (only $LINE_COUNT lines)"
    exit 1
fi

# Verify expected columns are present
HEADER=$(head -1 ci_results.csv | tr -d '\r')
EXPECTED_COLUMNS="trial,phase,mode,latency_s,wall_s,net_bytes,avg_sm"
if [ "$HEADER" != "$EXPECTED_COLUMNS" ]; then
    echo "❌ Error: Unexpected CSV header: '$HEADER'"
    echo "Expected: '$EXPECTED_COLUMNS'"
    exit 1
fi

# Check that we have exactly 2 rows (prefill + decode)
if [ "$LINE_COUNT" -ne 3 ]; then  # 1 header + 2 data rows
    echo "❌ Error: Expected 3 lines (header + 2 data rows), got $LINE_COUNT"
    exit 1
fi

# Verify both phases are present
if ! grep -q "prefill" ci_results.csv; then
    echo "❌ Error: prefill phase missing from results"
    exit 1
fi

if ! grep -q "decode" ci_results.csv; then
    echo "❌ Error: decode phase missing from results"
    exit 1
fi

# Verify network bytes are 0 for local mode
if ! grep -q ",0," ci_results.csv; then
    echo "❌ Error: Local mode should have 0 network bytes"
    exit 1
fi

echo "✅ CI test completed successfully!"
echo "📊 Results:"
cat ci_results.csv

echo ""
echo "🎉 Stage 0 CI test PASSED!"
exit 0 