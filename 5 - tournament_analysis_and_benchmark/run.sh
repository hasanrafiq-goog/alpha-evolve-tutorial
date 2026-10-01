#!/usr/bin/env bash
set -e

echo "=================================================="
echo " Step 5: Tournament Analysis & Benchmark Showdown"
echo "=================================================="

# 1. Activate virtual environment if present
if [ -d "../.venv" ]; then
    source "../.venv/bin/activate"
elif [ -d "../../.venv" ]; then
    source "../../.venv/bin/activate"
fi

# 2. Display tournament progression and leaderboard
python3 visualize_results.py

# 3. Run the head-to-head 5-epoch showdown: Baseline vs AlphaEvolve Winner
echo ""
echo "Launching Head-to-Head Benchmark Showdown..."
python3 benchmark_comparison.py
