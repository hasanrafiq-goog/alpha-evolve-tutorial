#!/usr/bin/env bash
set -e

echo "=================================================="
echo " Step 2: The AlphaEvolve Evolutionary Contract"
echo "=================================================="

# 1. Activate virtual environment if present
if [ -d "../.venv" ]; then
    source "../.venv/bin/activate"
elif [ -d "../../.venv" ]; then
    source "../../.venv/bin/activate"
fi

# 2. Run the contract demonstration script
python3 demo_contract.py
