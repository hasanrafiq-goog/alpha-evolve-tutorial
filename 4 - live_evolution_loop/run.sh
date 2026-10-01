#!/usr/bin/env bash
set -e

echo "=================================================="
echo " Step 4: Live Cloud Evolutionary Loop"
echo "=================================================="

# 1. Activate virtual environment if present
if [ -d "../.venv" ]; then
    source "../.venv/bin/activate"
elif [ -d "../../.venv" ]; then
    source "../../.venv/bin/activate"
fi

# 2. Check for .env and verify GCP configuration
ENV_FILE="../.env"
if [ ! -f "$ENV_FILE" ]; then
    ENV_FILE=".env"
fi

if [ ! -f "$ENV_FILE" ] || grep -q "your-gcp-project-id" "$ENV_FILE" 2>/dev/null || grep -q "your-google-cloud-project-id" "$ENV_FILE" 2>/dev/null; then
    echo -e "\n\033[1;33m[⚠] Notice: Google Cloud credentials not configured in .env\033[0m"
    echo "This step communicates directly with Google Cloud (Gemini Enterprise)."
    echo ""
    echo "To configure your GCP Project:"
    echo "  1. Edit $ENV_FILE with your real PROJECT_ID and GE_APP_ID"
    echo "  OR"
    echo "  2. Run: cd '../0 - environment_and_auth' && ./run.sh --enable-apis"
    echo ""
    echo "Once configured, re-run ./run.sh to start the live cloud evolution!"
    exit 1
fi

# 3. Auto-resolve or create Gemini Enterprise Engine if needed
python3 ensure_engine.py

# 4. Launch live AlphaEvolve evolutionary loop
echo ""
echo "Starting live AlphaEvolve evolutionary tournament..."
python3 run_evolution.py
