# Step 2: The AlphaEvolve Contract (Prompt, Code Block, & Insights)

Welcome to **Step 2**! In this step, we uncover the three architectural pillars that make AlphaEvolve work.

---

## 🎯 Learning Objectives
1. **Pillar 1: System Prompt (`instructions.md`):** How we instruct Gemini to search for architectures within specific shape and metric constraints.
2. **Pillar 2: The Mutation Boundary (`# EVOLVE-BLOCK`):** How we constrain the LLM to modify only specific functions while keeping harnesses immutable.
3. **Pillar 3: The Insight Feedback Loop:** Why AlphaEvolve outperforms traditional black-box search by feeding error tracebacks back into the LLM context.

---

## 🏛️ The 3 Pillars of AlphaEvolve

```
   ┌─────────────────────────────────────────────────────────────┐
   │ 1. SYSTEM PROMPT (instructions.md)                          │
   │    "Input: (28, 28, 1) -> Output: (10). Metric: val_acc"    │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 2. MUTATION BOUNDARY (# EVOLVE-BLOCK)                       │
   │    Gemini only edits code between:                          │
   │    # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END              │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ 3. THE INSIGHT FEEDBACK LOOP (Traceback Capture)            │
   │    If code crashes -> Exception is captured into an insight: │
   │    "Shape mismatch: Cannot add (28,28,16) and (14,14,16)"   │
   │    -> Gemini reads the insight and corrects its code!       │
   └─────────────────────────────────────────────────────────────┘
```

---

## 🔍 Deep-Dive: Traditional NAS vs. AlphaEvolve Insights

| Scenario | Traditional NAS (Grid / Bayesian) | AlphaEvolve Code Evolution |
| :--- | :--- | :--- |
| **Invalid Skip Connection** | The model crashes during compilation. Search algorithm assigns reward = `0.0`. | The evaluator catches the `ValueError: Shapes (28,28,16) and (14,14,16) are incompatible` and sends it to Gemini as a natural language **`insight`**. |
| **Subsequent Iteration** | Search controller has no idea *why* it scored 0; might randomly try similar broken shapes again. | Gemini reads: *"Shapes are incompatible"*, and adds a `1x1 Conv2D(strides=2)` downsampling layer to match dimensions on the very next attempt! |

---

## 🚀 How to Run Step 2

Execute the single runner script:
```bash
./run.sh
```

### What happens when you run `./run.sh`:
* Runs `demo_contract.py` which demonstrates:
  1. How the system prompt (`instructions.md`) defines tensor shapes and exploration ideas.
  2. How the `# EVOLVE-BLOCK` boundary prevents Gemini from hallucinating or modifying evaluation code.
  3. Live simulation of 3 candidates:
     * **Candidate A (Valid):** BatchNorm + Dropout mutation compiles cleanly.
     * **Candidate B (Broken Shape):** Simulates an invalid tensor addition, catches the traceback, and displays the exact JSON `insight` payload sent back to Google Cloud.
     * **Candidate C (Wrong Output):** Simulates a model returning 5 classes instead of 10, showing how shape assertions catch constraint violations.

---

## 🎤 Presenter Speaking Notes (For PSO Trainers)
* *"Traditional hyperparameter tuning treats the neural net as a black box. If it crashes, it's a wasted run."*
* *"AlphaEvolve treats code generation as a conversation. When code fails, the error message becomes part of the prompt for the next generation."*
* *"This insight-driven feedback loop is why Gemini achieves high validation accuracy in just 10–20 generations instead of thousands of brute-force trials."*
