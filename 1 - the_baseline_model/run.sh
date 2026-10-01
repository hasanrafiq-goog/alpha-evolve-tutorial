#!/usr/bin/env bash
set -e

echo "=================================================="
echo " Step 1: Inspecting the Baseline Seed Model"
echo "=================================================="

# 1. Activate virtual environment if present
if [ -d "../.venv" ]; then
    source "../.venv/bin/activate"
elif [ -d "../../.venv" ]; then
    source "../../.venv/bin/activate"
fi

# 2. Run the baseline model inspector
python3 inspect_baseline.py
