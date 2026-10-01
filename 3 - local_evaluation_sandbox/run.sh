#!/usr/bin/env bash
set -e

echo "=================================================="
echo " Step 3: Local Evaluation Sandbox (Offline Test)"
echo "=================================================="

# 1. Activate virtual environment if present
if [ -d "../.venv" ]; then
    source "../.venv/bin/activate"
elif [ -d "../../.venv" ]; then
    source "../../.venv/bin/activate"
fi

# 2. Run the offline evaluator test
python3 test_evaluator.py
