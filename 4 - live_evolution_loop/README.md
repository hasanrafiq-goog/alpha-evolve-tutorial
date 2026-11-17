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
4. Monitor the search in real time via the **Live Evolution Web Dashboard** (`http://localhost:8080`), tracking KPI cards, reward trajectories, and Pareto frontiers.
5. Enforce edge parameter constraints (`TARGET_PARAMS < 4,000` for micro-nets or `< 50,000` for standard) and accuracy goals via the **Vertex AI NAS Reward**.
6. Export the **Top N Finalist Architectures** (`rank_1_model.py` through `rank_N_model.py` and `best_model.py`) for the Step 5 Grand Finale.

---

## 📊 Live Evolution Search Console (Web Dashboard)

Step 4 launches a zero-dependency local web dashboard that matches the **Google Cloud AlphaEvolve Architecture Search Console**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 🚀 AlphaEvolve Architecture Search Console             [● In Progress (12/20)] [Session ▾] │
├───────────────────┬───────────────────┬───────────────────┬────────────────────────────┤
│ Top Accuracy      │ Optimal Size      │ Convergence Eff.  │ Exploration Rate           │
│ 96.80% (+1.40%)   │ 2,746 (-87.5%)    │ Run #8 / 20       │ 100% Zero-Touch            │
├───────────────────┴───────────────────┴───────────────────┴────────────────────────────┤
│ [ Line Chart: Multi-Objective Reward Trajectory ] [ Scatter Plot: Pareto Frontier ]    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Execution History:                                                                     │
│ Run #   Candidate ID   Status   Reward   Val Acc   Params   Mutation Summary            │
│ -------------------------------------------------------------------------------------- │
│ 1       ...3699149     RANK 1 ⭐️0.5512   96.80%     2,746   SeparableConv2D + Swish     │
│ 2       ...4820152     RANK 2   0.4820   96.20%     3,120   Depthwise Conv Residual     │
│ 3       ...5910214     RANK 3   0.4100   95.90%     3,450   Spatial Pyramid Pooling     │
│ 4       ...6012891     —        0.1250   91.20%     4,800   Dense Inverted Residual     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Dashboard Highlights:
* **Zero External Dependencies:** Built entirely with Python's standard library (`http.server`, `threading`, `json`) and locally bundled `Chart.js` — runs completely air-gapped without CDN access or Node.js.
* **Faithful Visual Charts:**
  - **Reward Trajectory Chart:** Plots each candidate's exact individual evaluation score, highlighting breakthrough high-water mark models with **green milestone dots** and regular exploration runs with **blue dots**.
  - **Pareto Frontier Scatter Plot:** Zoomed in tightly on the micro-architecture scale (< 4k parameters) with subtle dashed target guide lines marking `TARGET_PARAMS` and `TARGET_ACCURACY`. The 22k seed baseline is completely filtered out of both charts and tables so it never distorts the scale.
* **Dynamic Rank 1 Champion KPIs:** The "Top Accuracy" and "Optimal Size" cards strictly and dynamically track the **Rank 1 Champion** model with zero hardcoded values.
* **Single-Champion Leaderboard:** Real-time dynamic ranking guarantees at most one `RANK 1 ⭐️` champion at any moment, completely decoupled from candidate arrival numbers.
* **Dual-Channel Observability:** Full CLI terminal output remains 100% unsuppressed while streaming telemetry to the web browser.
* **Auto-Launch & Port Hunting:** Auto-opens your default browser at `http://localhost:8080` (auto-hunts 8080-8090 if the port is busy). Configure via `PORT` or `DASHBOARD_PORT`.
* **Interactive Code & Traceback Modal:** Click any candidate row in the execution table or any point on the Pareto scatter plot to slide open the full generated Python `build_model()` code, metrics breakdown, and compiler/evaluator insights.
* **Multi-Session History Selector:** Prior search runs are automatically archived in `evolved_output/history/` and can be toggled at any time using the header session dropdown.
* **Post-Run Persistence:** Once the evolutionary loop finishes, the web dashboard remains running so you can inspect finalist architectures until you press `Ctrl+C`.

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
5. **Score & Insight Upload:** Submits scores (`reward`, `val_accuracy`, `param_count`, `train_time_sec`) and diagnostic logs back to Google Cloud.
6. **Top N Finalists Export:** Once the evaluation budget (e.g. 20 models) is reached, AlphaEvolve downloads the Top $N$ candidates (`rank_1_model.py`, `rank_2_model.py`, ..., `rank_N_model.py`) and writes them to `evolved_output/` along with `best_model.py` and `result.json`.

## ⚙️ Mandatory Configuration: Updating `.env`

Before launching the live cloud search, your `.env` file must be updated with your real Google Cloud details and search targets.

Open the `.env` file in the tutorial root directory:
```bash
# Path: alpha_evolve_tutorial/.env
```

Ensure `PROJECT_ID` has your actual project value. You can also tune the NAS targets:

```ini
# Google Cloud Configuration
PROJECT_ID=your-actual-gcp-project-id       # e.g., my-ml-project-123
GE_APP_ID=your-alphaevolve-engine-id        # filled in automatically

# Multi-Objective NAS Targets & Finalists
TARGET_PARAMS=4000                          # Parameter ceiling budget (e.g. 4,000 for edge micro-nets, 50,000 for standard)
TARGET_ACCURACY=0.94                        # Target validation accuracy threshold (e.g. 0.94 for micro-nets, 0.98 for standard)
TOP_N_MODELS=3                              # Number of top finalists to export for Step 5

# Search Budget
MAX_PROGRAMS_GENERATED=20
MAX_PROGRAMS_EVALUATED=20
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
