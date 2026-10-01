# Step 0: Environment & GCP Fresh Project Onboarding

Welcome to **Step 0**! In this step, we will prepare your local workstation and your **Google Cloud Project** from scratch.

Even if you are using a **brand-new, completely empty GCP project**, this step will configure everything you need.

---

## 🎯 Learning Objectives
1. Understand Google Cloud **Application Default Credentials (ADC)** and how the local Python SDK authenticates with GCP.
2. Enable the required Google Cloud APIs (`discoveryengine`, `aiplatform`, etc.) using `gcloud`.
3. Configure your `.env` settings for AlphaEvolve.
4. Verify all Python dependencies using an automated health-check script.

---

## 🛠️ Required Google Cloud APIs

AlphaEvolve on Google Cloud relies on:
* **`discoveryengine.googleapis.com`**: The Discovery Engine / Gemini Enterprise API that manages AlphaEvolve experiments, program databases, and prompt sampling.
* **`aiplatform.googleapis.com`**: Vertex AI foundation models powering the Gemini ensemble (`gemini-2.5-flash` and `gemini-3.1-pro-preview`).
* **`cloudresourcemanager.googleapis.com`**: Resource management and project metadata.
* **`iam.googleapis.com`**: Identity and access management.

---

## 🚀 How to Run Step 0

Simply execute the single runner script:
```bash
./run.sh
```

### What `run.sh` does:
1. **Verifies or creates `.env`** from template.
2. **Prompts to configure your fresh GCP project** (enables `discoveryengine`, `aiplatform`, `cloudresourcemanager`, `iam` APIs).
3. **Sets your ADC quota project** (`gcloud auth application-default set-quota-project`).
4. **Runs `check_env.py`** to confirm Python packages, ADC tokens, and project settings are healthy.

> **Tip:** You can force the GCP API enablement wizard at any time by running:
> ```bash
> ./run.sh --enable-apis
> ```

---

## 📋 Manual Commands Reference (For Presenters / Instructors)

If you prefer to run the commands individually:

### 1. Configure active gcloud project
```bash
gcloud config set project <YOUR_PROJECT_ID>
```

### 2. Enable Required APIs (Fresh Project)
```bash
gcloud services enable \
  discoveryengine.googleapis.com \
  aiplatform.googleapis.com \
  cloudresourcemanager.googleapis.com \
  iam.googleapis.com \
  --project=<YOUR_PROJECT_ID>
```

### 3. Authenticate Application Default Credentials (ADC)
```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project <YOUR_PROJECT_ID>
```

### 4. Install Python Dependencies
```bash
pip install -r ../requirements.txt
```

### 5. Run Health Check
```bash
python3 check_env.py
```

---

## ✅ Expected Output
Once completed, `check_env.py` will display:
```
======================================================================
Step 0: AlphaEvolve Workshop Health Check
======================================================================
  [✓] Python Version               : Python 3.11.x (Requires >= 3.10)

Checking Core Python Dependencies:
  [✓] tensorflow                   : Installed (Deep Learning backend)
  [✓] numpy                        : Installed (Array computing)
  [✓] dotenv                       : Installed (Environment configuration)
  [✓] matplotlib                   : Installed (Visualization & plotting)
  [✓] google.auth                  : Installed (GCP Identity & Credentials)
  [✓] alpha_evolve                 : Installed (Google AlphaEvolve SDK)

Checking Configuration (.env):
  [✓] .env file                    : Found
  [✓] PROJECT_ID                   : Configured (your-project-id)
  [✓] GE_APP_ID                    : Configured (your-engine-id)

Checking Google Cloud Application Default Credentials (ADC):
  [✓] GCP ADC Auth                 : Authenticated!
======================================================================
✨ Environment is healthy and ready for the workshop!
Proceed to Step 1: cd "../1 - the_baseline_model" && ./run.sh
======================================================================
```
