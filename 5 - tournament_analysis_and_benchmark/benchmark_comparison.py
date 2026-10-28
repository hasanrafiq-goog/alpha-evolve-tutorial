#!/usr/bin/env python3
"""Grand Finale: Head-to-Head Benchmark Comparison.

Trains the Baseline Seed Model vs. the Top N AlphaEvolve Discovered Finalist Models
side-by-side on MNIST (5 full epochs) to crown the ultimate champion.
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple

from dotenv import load_dotenv
import numpy as np
import tensorflow as tf

# Load .env configuration
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv()

TOP_N_MODELS = int(os.getenv("TOP_N_MODELS", "3"))

# Styling
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_mnist_data():
    """Loads normalized full MNIST dataset (60k train, 10k test)."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype(np.float32) / 255.0)[..., np.newaxis]
    x_test = (x_test.astype(np.float32) / 255.0)[..., np.newaxis]
    return x_train, y_train, x_test, y_test


def load_model_from_file(file_path: str) -> tf.keras.Model:
    """Executes a model file in an isolated namespace and returns the instantiated model."""
    with open(file_path, "r") as f:
        code = f.read()
    ns = {"tf": tf, "keras": tf.keras}
    exec(code, ns)
    tf.keras.backend.clear_session()
    return ns["build_model"]()


def train_and_eval(model: tf.keras.Model, model_name: str, x_train, y_train, x_test, y_test, epochs=5):
    """Compiles, trains, and evaluates a model on the full dataset."""
    print(f"\n{CYAN}{BOLD}=== Training {model_name} (5 Full Epochs) ==={RESET}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    start_time = time.time()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=64,
        verbose=1,
    )
    elapsed = time.time() - start_time

    val_acc = history.history["val_accuracy"][-1]
    val_loss = history.history["val_loss"][-1]
    train_acc = history.history["accuracy"][-1]

    return {
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "train_accuracy": float(train_acc),
        "train_time_sec": float(elapsed),
        "param_count": int(model.count_params()),
    }


def main():
    print("=" * 85)
    print(f"{BOLD}Step 5: The Grand Finale Showdown (Baseline vs. Top {TOP_N_MODELS} AlphaEvolve Finalists){RESET}")
    print("=" * 85)

    base_dir = os.path.dirname(__file__)
    baseline_path = os.path.join(base_dir, "baseline_model.py")
    step4_dir = os.path.abspath(os.path.join(base_dir, "..", "4 - live_evolution_loop", "evolved_output"))

    # 1. Discover Candidate Models from Step 4
    candidate_models: List[Tuple[str, str]] = []

    # Check for ranked finalists: rank_1_model.py, rank_2_model.py, ..., rank_N_model.py
    for r in range(1, TOP_N_MODELS + 1):
        rf = os.path.join(step4_dir, f"rank_{r}_model.py")
        if os.path.exists(rf):
            candidate_models.append((f"AlphaEvolve Finalist Rank #{r}", rf))

    # Fallback to best_model.py if rank_* files are not present
    if not candidate_models:
        step4_winner = os.path.join(step4_dir, "best_model.py")
        cached_winner = os.path.join(base_dir, "evolved_output", "best_model.py")
        if os.path.exists(step4_winner):
            candidate_models.append(("AlphaEvolve Winner (Rank #1)", step4_winner))
        elif os.path.exists(cached_winner):
            candidate_models.append(("AlphaEvolve Pre-Cached Champion", cached_winner))
        else:
            print("No evolved winner models found! Please run Step 4 first.")
            sys.exit(1)

    print(f"\n{BOLD}1. Discovered Architectures (Configured Top N = {TOP_N_MODELS}):{RESET}")
    baseline_model = load_model_from_file(baseline_path)
    print(f"  [Reference] Baseline Seed Model : {baseline_model.count_params():,} parameters")

    instantiated_candidates = []
    for label, path in candidate_models:
        m = load_model_from_file(path)
        instantiated_candidates.append((label, path, m))
        print(f"  [Finalist]  {label:<30} : {m.count_params():,} parameters ({os.path.basename(path)})")

    print(f"\n{BOLD}2. Loading Full MNIST Training & Test Datasets...{RESET}")
    x_tr, y_tr, x_te, y_te = load_mnist_data()
    print(f"  Training samples : {len(x_tr):,} (Full set)")
    print(f"  Test samples     : {len(x_te):,} (Full set)")

    # 3. Train Baseline Reference Model
    baseline_res = train_and_eval(baseline_model, "Baseline Seed Model", x_tr, y_tr, x_te, y_te, epochs=5)
    b_acc = baseline_res["val_accuracy"]
    b_err = (1.0 - b_acc) * 100

    # 4. Train All Top N Finalists
    candidate_results = []
    for label, path, model in instantiated_candidates:
        res = train_and_eval(model, label, x_tr, y_tr, x_te, y_te, epochs=5)
        res["label"] = label
        res["path"] = path
        candidate_results.append(res)

    # 5. Sort Finalists by Full Dataset Test Accuracy
    candidate_results.sort(key=lambda r: r["val_accuracy"], reverse=True)
    grand_champion = candidate_results[0]

    # 6. Print Comprehensive Leaderboard Table
    print("\n" + "=" * 95)
    print(
        f"{BOLD}{'FINAL RANK':<12} | {'MODEL NAME':<30} | {'PARAMS':<10} | {'FULL TEST ACC':<14} | {'ERROR REDUCT':<13} | {'TRAIN TIME'}{RESET}"
    )
    print("=" * 95)

    # Print Candidate Rows
    for rank_idx, r in enumerate(candidate_results, 1):
        w_acc = r["val_accuracy"]
        w_err = (1.0 - w_acc) * 100
        err_reduct = (1.0 - w_err / b_err) * 100 if b_err > 0 else 0.0
        acc_diff = (w_acc - b_acc) * 100

        is_champ = rank_idx == 1
        star = " 🏆 (Winner)" if is_champ else ""
        color = GREEN + BOLD if is_champ else ""
        reset_c = RESET if is_champ else ""

        print(
            f"{color}#{rank_idx:<11} | {r['label'] + star:<30} | {r['param_count']:<10,}"
            f" | {w_acc*100:6.2f}% (+{acc_diff:.2f}%)  | -{err_reduct:5.1f}%       | {r['train_time_sec']:5.1f}s{reset_c}"
        )

    # Print Baseline Reference Row
    print("-" * 95)
    print(
        f"{YELLOW}{'Reference':<12} | {'Baseline Seed Model':<30} | {baseline_res['param_count']:<10,}"
        f" | {b_acc*100:6.2f}% (Base)    |   0.0%        | {baseline_res['train_time_sec']:5.1f}s{RESET}"
    )
    print("=" * 95)

    # 7. Print Grand Champion Conclusion
    top_acc = grand_champion["val_accuracy"]
    top_err_reduct = (1.0 - ((1.0 - top_acc) * 100) / b_err) * 100 if b_err > 0 else 0.0
    print(f"\n{GREEN}{BOLD}🎉 Tournament Conclusion:{RESET}")
    print(
        f"The Grand Finale Champion is {BOLD}{grand_champion['label']}{RESET}!"
    )
    print(
        f"Achieved {BOLD}{top_acc*100:.2f}% accuracy{RESET} on the full dataset "
        f"({top_err_reduct:.1f}% fewer errors than baseline) with only {grand_champion['param_count']:,} parameters!\n"
    )


if __name__ == "__main__":
    main()
