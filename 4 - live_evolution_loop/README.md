# Step 4: The Live Cloud Evolutionary Tournament

Welcome to **Step 4**! In this step, we connect your local machine to **Google Cloud Gemini Enterprise (Discovery Engine)** to launch the live, closed-loop evolutionary tournament.

> [!IMPORTANT]
> **Prerequisite:** Unlike Steps 1, 2, and 3 which were offline, **Step 4 connects directly to Google Cloud**.
> Before running `./run.sh`, you **must** update `alpha_evolve_tutorial/.env` with your real `PROJECT_ID` (`GE_APP_ID` is filled in automatically).
> (See [Mandatory Configuration: Updating .env](#-mandatory-configuration-updating-env) below).

---

## 🎯 Learning Objectives
1. Understand the **Gemini Ensemble**: combining fast exploration (`gemini-3.5-flash`) with deep reasoning (`gemini-3.1-pro-preview`).
2. Learn how **Sampling Workers** and **Evaluation Workers** operate concurrently.
3. Observe live evolutionary generations proposing novel convolutions, skip connections, and normalization.
4. Export the winning neural architecture to `evolved_output/best_model.py`.

---

## 🤖 The Gemini Ensemble Strategy

In `.env`, AlphaEvolve configures a weighted ensemble of Gemini models:

```ini
MODEL_1=gemini-3.5-flash
MODEL_1_WEIGHT=0.7

MODEL_2=gemini-3.1-pro-preview
MODEL_2_WEIGHT=0.3
```

### Why a Mixture of Flash & Pro?
* **Gemini 3.5 Flash (70%):** The "Broad Explorer". Generates variations at high velocity and low latency. Ideal for testing new filter widths, activation functions (Swish vs GELU), and dropout rates.
* **Gemini 3.1 Pro Preview (30%):** The "Architectural Thinker". When given previous generation failure tracebacks, Pro diagnoses subtle topological issues, designs multi-branch residual blocks, or restructures the classification head.

---

## 🔄 The Closed Evolutionary Loop in Action

```
      GOOGLE CLOUD (Gemini Enterprise)                LOCAL WORKSTATION
  ┌─────────────────────────────────────┐         ┌────────────────────────┐
  │ 1. Program Database                 │         │                        │
  │    Maintains Pareto leaderboard     │         │                        │
  │                 │                   │         │                        │
  │                 ▼                   │         │                        │
  │ 2. Prompt Sampler                   │         │                        │
  │    Picks top code + error insights  │         │                        │
  │                 │                   │         │                        │
  │                 ▼                   │         │                        │
  │ 3. Gemini Ensemble Mutates Code     │         │                        │
  │    Generates new build_model()      │         │                        │
  └─────────────────┬───────────────────┘         │                        │
                    │ Proposed Candidate Code     │                        │
                    ▼                             │                        │
          [ SamplingWorker ] ────────────────────>│ 4. Client Evaluator    │
                                                  │    - exec() sandbox    │
                                                  │    - 2-sec proxy train │
                                                  │    - extract metrics   │
                                                  │           │            │
                    ▲                             │           ▼            │
                    │ Scores & Insights           │ 5. Returns Scores      │
          [ EvaluationWorker ] ───────────────────┴── (val_acc, tracebacks)
```

1. **Session & Experiment Creation:** Registers your search title and system prompt (`instructions.md`) in Google Cloud Discovery Engine.
2. **Seed Upload:** Uploads `program.py` as Generation 0.
3. **SamplingWorker:** Polls Google Cloud via `POST ...:acquirePrograms` for candidate models proposed by Gemini.
4. **EvaluationWorker:** Passes each candidate to your local `mnist_evaluation()` harness (trained on your CPU in ~1.5s).
5. **Score & Insight Upload:** Submits scores (`val_accuracy`, `param_count`, `train_time_sec`) and diagnostic logs back to Google Cloud.
6. **Tournament Winner Export:** Once the evaluation budget (e.g. 10 models) is reached, AlphaEvolve downloads the #1 ranked candidate and writes it to `evolved_output/best_model.py`.

## ⚙️ Mandatory Configuration: Updating `.env`

Before launching the live cloud search, your `.env` file must be updated with your real Google Cloud details.

Open the `.env` file in the tutorial root directory:
```bash
# Path: alpha_evolve_tutorial/.env
```

Ensure `PROJECT_ID` has your actual project value. `GE_APP_ID` (your Gemini Enterprise app) is filled in automatically by Step 0 or by `run.sh` below:

```ini
# Google Cloud Configuration
PROJECT_ID=your-actual-gcp-project-id       # e.g., my-ml-project-123
GE_APP_ID=your-alphaevolve-engine-id        # filled in automatically, e.g. alphaevolve-engine
```

### How `GE_APP_ID` is set
* If your project has **no** Gemini Enterprise app, one called `alphaevolve-engine` is created automatically (billable: Enterprise tier with LLM add-on).
* If your project has exactly **one** app, it is used automatically.
* If you have **several** apps, set `GE_APP_ID` to the ID or display name of the one to use; the full ID (e.g. `alphaevolve-engine_1234567890`) is resolved automatically. Find it in Google Cloud Console under **Gemini Enterprise** > **Apps**.

> **Shortcut:** Step 0 does all of this for you:
> ```bash
> cd "../0 - environment_and_auth" && ./run.sh --enable-apis
> ```

---

## 🚀 How to Run Step 4

Once `.env` is updated, execute the single runner script:
```bash
./run.sh
```

### What `run.sh` does automatically:
1. **Engine Auto-Discovery:** If you typed a display name like `alphaevolve-engine`, `run.sh` automatically finds the full GCP resource ID (e.g. `alphaevolve-engine_1234567890`) and updates `.env`.
2. **Auto-Creation:** If no engine exists at all in your project, `run.sh` will auto-create one for you using the Discovery Engine API (same script as Step 0: `../0 - environment_and_auth/ensure_engine.py`). If listing or creating the app fails, `run.sh` stops with the error.
3. **Session & Experiment Initialization:** Registers Generation 0 (`program.py`).
4. **Live Search:** Starts the parallel workers to sample code from Gemini and evaluate candidates locally.

## 🎤 Presenter Speaking Notes (For PSO Trainers)
* *"Notice how cleanly decoupled this system is: Gemini runs in Google Cloud, but the actual compilation and training happen right here on our machine."*
* *"Watch the terminal logs as generations progress: you will see Gemini start with simple variations, and gradually invent deeper, stacked convolutions, batch normalization, and dropout."*
* *"If a trainee's project doesn't have Gemini Enterprise enabled yet, don't worry! Step 5 includes the pre-evaluated championship model so everyone can run the final benchmark showdown."*
