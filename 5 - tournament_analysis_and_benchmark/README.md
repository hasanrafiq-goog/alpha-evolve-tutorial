# Step 5: Tournament Analysis & The Grand Finale Showdown

Welcome to **Step 5**! This is the grand finale of the AlphaEvolve workshop.

In this step, we analyze the tournament progression, inspect the Pareto frontier, and run a **head-to-head 5-epoch training showdown** between the naive baseline and Gemini's discovered winner.

---

## 🎯 Learning Objectives
1. Read and interpret the live **tournament progression table** directly from Google Cloud (`RUN #`, `PROGRAM ID`, `REWARD`, `VAL ACC`, `PARAMS`, `TRAIN TIME`).
2. Understand the **Pareto Frontier**: balancing validation accuracy against parameter size and latency.
3. Run `benchmark_comparison.py` to train the **Naive Baseline** vs. **All Top N Finalist Models** side-by-side on the full 60,000-image MNIST dataset (5 epochs) to crown the **Grand Finale Champion 🏆**.

---

## 📊 The Tournament Leaderboard

When you run `./run.sh`, Step 5 queries the Cloud Program Database and renders the chronological execution sequence:

```
===================================================================================================================
RUN #  | PROGRAM ID             | CREATED TIME        | RANK    | REWARD     | VAL ACC    | PARAMS     | TRAIN TIME
===================================================================================================================
1      | 14879421322592945323   | 2026-10-28 20:56:07 | Seed    | Bootstrap  | N/A        | N/A        | -
2      | 18181523060563699149   | 2026-10-28 20:56:16 | #11     | 0.0165     | 12.70%     | 20,938     | 7.2s
...
7      | 18181523060563701414   | 2026-10-28 20:57:42 | #6      | 0.9132     | 94.60%     | 59,002     | 2.2s
11     | 18181523060563700544   | 2026-10-28 20:59:32 | #3      | 0.9385     | 95.90%     | 65,994     | 1.8s
15     | 18181523060563701436   | 2026-10-28 21:01:10 | #2      | 0.9424     | 96.10%     | 74,101     | 1.9s
16     | 18181523060563699651   | 2026-10-28 21:01:50 | #1 ⭐    | 0.9463     | 96.30%     | 72,266     | 2.6s
===================================================================================================================
```

### What happened across generations:
1. **Initial Exploration:** Gemini tested wider convolutions and noticed immediate accuracy gains.
2. **Pruning & Regularization:** Gemini added spatial dropout and discovered that shallow, cramped layers underfit.
3. **The Breakthrough:** Gemini synthesized stacked convolutions (32 $\to$ 32, then 64 $\to$ 64) with hierarchical dropout, reaching top-tier accuracy!

---

## 🥊 The Grand Finale: Head-to-Head Benchmark Showdown

Step 5 automatically benchmarks your **freshly evolved Top N models from Step 4** against the naive baseline on the **full 60,000 dataset**:

### Scenario A: High-Accuracy Target (`TARGET_PARAMS=50000`, `TARGET_ACCURACY=0.98`)
```
===============================================================================================
FINAL RANK   | MODEL NAME                     | PARAMS     | FULL TEST ACC  | ERROR REDUCT  | TRAIN TIME
===============================================================================================
#1           | AlphaEvolve Finalist Rank #3 🏆 | 65,994     |  99.20% (+3.80%)  | - 82.6%       |  21.1s
#2           | AlphaEvolve Finalist Rank #2   | 74,101     |  99.14% (+3.74%)  | - 81.3%       |  21.6s
#3           | AlphaEvolve Finalist Rank #1   | 72,266     |  99.13% (+3.73%)  | - 81.1%       |  32.6s
-----------------------------------------------------------------------------------------------
Reference    | Baseline Seed Model            | 21,898     |  95.40% (Base)    |   0.0%        |   5.5s
===============================================================================================
```

### Scenario B: Ultra-Compact Edge Micro-Controller (`TARGET_PARAMS=4000`, `TARGET_ACCURACY=0.94`)
```
===============================================================================================
FINAL RANK   | MODEL NAME                     | PARAMS     | FULL TEST ACC  | ERROR REDUCT  | TRAIN TIME
===============================================================================================
#1           | AlphaEvolve Finalist Rank #1 🏆 |  2,978     |  98.45% (+3.05%)  | - 66.3%       |  12.3s
-----------------------------------------------------------------------------------------------
Reference    | Baseline Seed Model            | 21,898     |  95.40% (Base)    |   0.0%        |   5.5s
===============================================================================================

🎉 Tournament Conclusion:
The Grand Finale Champion is mnist_efficient_net (2,978 parameters)!
Achieved 98.45% accuracy while cutting model parameter size by 7.35x (-86.4%) and slashing classification errors by 66.3% compared to baseline!
```

---

## 🚀 How to Run Step 5

Execute the single runner script:
```bash
./run.sh
```

### What happens when you run `./run.sh`:
1. Activates your virtual environment.
2. Queries Google Cloud to render the live tournament progression leaderboard (`visualize_results.py`).
3. Automatically detects the Top $N$ evolved models from Step 4 (`evolved_output/rank_*_model.py`).
4. Trains the **Baseline Model** and all **Top N Finalists** for 5 epochs on the **full 60,000-image dataset** with live training logs.
5. Evaluates each on the 10,000-image test set and prints the final showdown leaderboard, crowning the Grand Finale Champion 🏆!

---

## 🎤 Presenter Speaking Notes (For PSO Trainers)
* *"Look at the starting baseline: a naive 1-conv model scoring 95.40%. Over 460 handwritten digits misclassified!"*
* *"Without any human writing code, Gemini diagnosed the capacity bottleneck, stacked convolutions, or engineered extreme micro-separable topologies to score up to 99.20% (or 98.45% with 7.35x fewer parameters)!"*
* *"That is a massive reduction in error rates (cutting errors by up to 66.3% to 82.6%) discovered fully autonomously using AlphaEvolve on Google Cloud."*
