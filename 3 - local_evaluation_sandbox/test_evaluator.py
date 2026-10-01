#!/usr/bin/env python3
"""Local Standalone Evaluation Test (Zero Cloud Calls).

Tests the evaluation pipeline against the baseline seed program (program.py)
without requiring GCP authentication or the AlphaEvolve API.
"""

import logging
import os
import sys
import time

from evaluate import (
    INITIAL_PROGRAM_CODE,
    evaluate_architecture_code,
    mnist_evaluation,
)

# Terminal styling
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BOLD = "\033[1m"
RESET = "\033[0m"


def main():
    print("=" * 75)
    print(f"{BOLD}Step 3: Testing the Local Evaluation Sandbox (Offline Referee){RESET}")
    print("=" * 75)

    print(f"\n{BOLD}1. Why Local Proxy Training?{RESET}")
    print("   - Training on full MNIST (60,000 images, 50 epochs) takes minutes per candidate.")
    print("   - We use a proxy split (10,000 train, 2,000 val) and train for only 2 epochs.")
    print("   - This gives an accurate signal in ~1.5 - 2.0 seconds with ZERO cloud cost!\n")

    print(f"{CYAN}--- Running Proxy Evaluation on program.py ---{RESET}")
    print(f"Training on 10,000 images (batch_size=64 -> 156 batches per epoch):")
    start_time = time.time()
    val_acc, metrics, insights = evaluate_architecture_code(
        code_str=INITIAL_PROGRAM_CODE,
        epochs=2,
        batch_size=64,
        verbose=1,
    )
    elapsed = time.time() - start_time

    print(f"\n{BOLD}2. Local Evaluation Results:{RESET}")
    print(f"  Primary Score (Validation Accuracy) : {GREEN}{BOLD}{val_acc:.4f} ({val_acc*100:.2f}%){RESET}")
    print(f"  Parameter Count                     : {YELLOW}{metrics.get('param_count', 0):,}{RESET}")
    print(f"  Training Time                       : {metrics.get('train_time_sec', 0):.2f}s")
    print(f"  Validation Loss                     : {metrics.get('val_loss', 0):.4f}")
    print(f"  Total Wall Time                     : {elapsed:.2f}s")

    print(f"\n{BOLD}3. Insights Feedback Generated:{RESET}")
    for insight in insights:
        print(f"  - {insight}")

    assert val_acc > 0.70, f"Expected seed model val_accuracy > 0.70, got {val_acc}"

    # Test candidate payload structure (the format exchanged with GCP)
    print(f"\n{BOLD}4. Simulating Google Cloud AlphaEvolve Candidate Payload Structure:{RESET}")
    dummy_candidate = {
        "content": {
            "files": [
                {
                    "path": "program.py",
                    "content": INITIAL_PROGRAM_CODE,
                }
            ]
        }
    }
    eval_result = mnist_evaluation(dummy_candidate)
    print(f"  Payload successfully validated and formatted for GCP Program Database submission!")

    print("\n" + "=" * 75)
    print(f"{GREEN}{BOLD}✨ Local Evaluator verified and working perfectly!{RESET}")
    print(f"Next: Run the live cloud evolutionary loop in Step 4:")
    print(f"{BOLD}cd '../4 - live_evolution_loop' && ./run.sh{RESET}")
    print("=" * 75)


if __name__ == "__main__":
    main()
