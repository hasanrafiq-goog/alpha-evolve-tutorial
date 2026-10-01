# Step 5: Tournament Analysis & The Grand Finale Showdown

Welcome to **Step 5**! This is the grand finale of the AlphaEvolve workshop.

In this step, we analyze the tournament progression, inspect the Pareto frontier, and run a **head-to-head 5-epoch training showdown** between the naive baseline and Gemini's discovered winner.

---

## 🎯 Learning Objectives
1. Read and interpret the chronological **tournament progression table** (`RUN #`, `PROGRAM ID`, `VAL ACCURACY`, `PARAMS`).
2. Understand the **Pareto Frontier**: balancing validation accuracy against parameter size and latency.
3. Run `benchmark_comparison.py` to train the **Naive Baseline** (~87%) vs. the **Evolved Winner** (99%+) side-by-side on the official MNIST test set.

---

## 📊 The Tournament Leaderboard

When you run `./run.sh`, Step 5 queries the tournament history and renders the chronological execution sequence:

```
=========================================================================================================
RUN #  | PROGRAM ID             | CREATED TIME        | RANK  | VAL ACCURACY   | PARAMS     | TRAIN TIME
=========================================================================================================
1      | 16101992019316122043   | 2026-09-30 20:15:10 | #10   | Naive Baseline | 21,898     | 1.5s
2      | 14097066499265857194   | 2026-09-30 20:15:20 | #7    | 0.9120         | 165,146    | 2.1s
...
N      | 848395831475954795     | 2026-09-30 20:16:05 | #1 ⭐ | 0.9630 (Proxy) | 280,000    | 2.4s
=========================================================================================================
```

### What happened across generations:
1. **Initial Exploration:** Gemini tested wider convolutions and noticed immediate accuracy gains.
2. **Pruning & Regularization:** Gemini added spatial dropout and discovered that shallow, cramped layers underfit.
3. **The Breakthrough:** Gemini synthesized stacked convolutions (32 $\to$ 32, then 64 $\to$ 64) with hierarchical dropout, reaching top-tier accuracy!

---

## 🥊 The Grand Finale: Head-to-Head Benchmark Showdown

Step 5 automatically benchmarks your **freshly evolved model from Step 4** against the naive baseline:

```
========================================================================================
METRIC                    | BASELINE MODEL       | YOUR ALPHAEVOLVE WINNER | IMPACT
========================================================================================
Validation Accuracy       | ~88.50%              | 99.15%                  | +10.65% leap!
Test Error Rate           | 11.50% (1,150 errors)| 0.85% (85 errors)       | ~13x fewer errors!
Validation Loss           | 0.3250               | 0.0245                  | -92% loss reduction
Parameters                | 21,898               | ~280,000                | Optimal capacity
========================================================================================
```

---

## 🚀 How to Run Step 5

Execute the single runner script:
```bash
./run.sh
```

### What happens when you run `./run.sh`:
1. Activates your virtual environment.
2. Displays the tournament progression leaderboard.
3. Automatically detects your newly evolved model from Step 4 (or uses the cached champion).
4. Trains both the **Naive Baseline** and **Evolved Winner** for 5 epochs on the full dataset with a live progress bar.
5. Evaluates both on the 10,000-image test set and prints the final scorecard.

---

## 🎤 Presenter Speaking Notes (For PSO Trainers)
* *"Look at the starting baseline: a naive 1-conv model scoring ~88%. Over 1,100 handwritten digits misclassified!"*
* *"Without any human writing code, Gemini diagnosed the capacity bottleneck, stacked convolutions, and added hierarchical regularization to score 99.15%."*
* *"That is a 13x reduction in customer error rates discovered fully autonomously using AlphaEvolve on Google Cloud."*
