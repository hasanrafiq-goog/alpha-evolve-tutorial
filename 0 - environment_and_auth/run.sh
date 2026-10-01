#!/usr/bin/env bash
set -e

echo "=================================================="
echo " Step 0: Environment & Google Cloud Verification"
echo "=================================================="

# 1. Virtual environment: find existing or create new, then activate
if [ -d "../.venv" ]; then
    source "../.venv/bin/activate"
elif [ -d "../../.venv" ]; then
    source "../../.venv/bin/activate"
else
    echo "Creating new virtual environment at ../.venv..."
    python3 -m venv "../.venv"
    source "../.venv/bin/activate"
fi

# 2. Copy .env from template if missing
if [ ! -f "../.env" ]; then
    echo "Creating ../.env from ../.env.example..."
    cp "../.env.example" "../.env"
fi

# 3. Install / update Python dependencies
echo "Installing Python dependencies from requirements.txt..."
pip install -r "../requirements.txt"

# 4. Enable Google Cloud APIs if trainee passes --enable-apis
if [ "$1" == "--enable-apis" ]; then
    echo "Enabling Google Cloud APIs on current project..."
    gcloud services enable \
        discoveryengine.googleapis.com \
        aiplatform.googleapis.com \
        cloudresourcemanager.googleapis.com \
        iam.googleapis.com

    echo "Configuring Application Default Credentials (ADC)..."
    gcloud auth application-default login
    gcloud auth application-default set-quota-project $(gcloud config get-value project)
fi

# 5. Run the Python verification script
python3 check_env.py

echo ""
echo "Tip: To keep this virtual environment active in your current shell, run:"
echo "  source ../.venv/bin/activate"
