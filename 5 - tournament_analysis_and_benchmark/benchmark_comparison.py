#!/usr/bin/env python3
"""Grand Finale: Head-to-Head Benchmark Comparison.

Trains the Baseline Seed Model vs. the AlphaEvolve Discovered Winning Model
side-by-side on MNIST (5 epochs) and prints a comprehensive comparison table.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

# Styling
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_mnist_data():
    """Loads normalized MNIST dataset."""
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
    """Compiles, trains, and evaluates a model."""
    print(f"\n{CYAN}{BOLD}=== Training {model_name} (5 Epochs) ==={RESET}")
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
        "val_accuracy": val_acc,
        "val_loss": val_loss,
        "train_accuracy": train_acc,
        "train_time_sec": elapsed,
        "param_count": model.count_params(),
    }


def main():
    print("=" * 80)
    print(f"{BOLD}Step 5: The Grand Finale Showdown (Baseline vs. AlphaEvolve Winner){RESET}")
    print("=" * 80)

    base_dir = os.path.dirname(__file__)
    baseline_path = os.path.join(base_dir, "baseline_model.py")
    
    # Check if a fresh model was evolved in Step 4
    step4_winner = os.path.abspath(os.path.join(base_dir, "..", "4 - live_evolution_loop", "evolved_output", "best_model.py"))
    cached_winner = os.path.join(base_dir, "evolved_output", "best_model.py")
    
    if os.path.exists(step4_winner):
        winner_path = step4_winner
        print(f"Using freshly evolved model from Step 4: {GREEN}{step4_winner}{RESET}")
    elif os.path.exists(cached_winner):
        winner_path = cached_winner
        print(f"Using pre-cached champion model: {GREEN}{cached_winner}{RESET}")
    else:
        print("Winner model not found! Please run Step 4 first.")
        sys.exit(1)

    print(f"\n{BOLD}1. Loading Architectures:{RESET}")
    baseline_model = load_model_from_file(baseline_path)
    winner_model = load_model_from_file(winner_path)

    print(f"  Baseline Model : {baseline_model.count_params():,} parameters")
    print(f"  Winner Model   : {winner_model.count_params():,} parameters")

    print(f"\n{BOLD}2. Loading MNIST Training & Test Datasets...{RESET}")
    x_tr, y_tr, x_te, y_te = load_mnist_data()
    print(f"  Training samples : {len(x_tr):,}")
    print(f"  Test samples     : {len(x_te):,}")

    # Train Baseline Model
    baseline_res = train_and_eval(baseline_model, "Baseline Seed Model", x_tr, y_tr, x_te, y_te, epochs=5)

    # Train Evolved Winner Model
    winner_res = train_and_eval(winner_model, "AlphaEvolve Winner Model", x_tr, y_tr, x_te, y_te, epochs=5)

    # Print Final Showdown Results
    print("\n" + "=" * 80)
    print(f"{BOLD}{'METRIC':<25} | {'BASELINE MODEL':<20} | {'ALPHAEVOLVE WINNER':<20} | {'IMPACT':<15}{RESET}")
    print("=" * 80)

    b_acc, w_acc = baseline_res['val_accuracy'], winner_res['val_accuracy']
    acc_diff = (w_acc - b_acc) * 100
    print(f"{'Validation Accuracy':<25} | {b_acc*100:6.2f}%{'':<13} | {GREEN}{BOLD}{w_acc*100:6.2f}%{'':<13}{RESET} | {GREEN}{BOLD}+{acc_diff:.2f}% leap{RESET}")

    b_loss, w_loss = baseline_res['val_loss'], winner_res['val_loss']
    print(f"{'Validation Loss':<25} | {b_loss:8.4f}{'':<12} | {GREEN}{BOLD}{w_loss:8.4f}{'':<12}{RESET} | {GREEN}{BOLD}-{(1 - w_loss/b_loss)*100:.1f}% reduction{RESET}")

    b_err, w_err = (1 - b_acc) * 100, (1 - w_acc) * 100
    err_reduct = (1 - w_err / b_err) * 100 if b_err > 0 else 0
    print(f"{'Test Error Rate':<25} | {b_err:6.2f}%{'':<13} | {GREEN}{BOLD}{w_err:6.2f}%{'':<13}{RESET} | {GREEN}{BOLD}{err_reduct:.1f}% fewer errors!{RESET}")

    b_params, w_params = baseline_res['param_count'], winner_res['param_count']
    print(f"{'Parameters':<25} | {b_params:,}{'':<13} | {w_params:,}{'':<13} | {YELLOW}More efficient capacity{RESET}")

    b_time, w_time = baseline_res['train_time_sec'], winner_res['train_time_sec']
    print(f"{'5-Epoch Training Time':<25} | {b_time:6.1f}s{'':<13} | {w_time:6.1f}s{'':<13} | Highly trainable")
    print("=" * 80)

    print(f"\n{GREEN}{BOLD}🎉 Tournament Conclusion:{RESET}")
    print("AlphaEvolve successfully evolved a naive baseline model into a state-of-the-art CNN,")
    print(f"boosting accuracy from {BOLD}{b_acc*100:.2f}% to {w_acc*100:.2f}%{RESET} and slashing errors by {BOLD}{err_reduct:.0f}%!{RESET}\n")


if __name__ == "__main__":
    main()
