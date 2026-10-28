"""Local standalone test script for the AlphaEvolve MNIST evaluator.

Tests the evaluation pipeline against the baseline seed program (program.py)
without requiring GCP authentication or the AlphaEvolve API.
"""

import logging
import os
import sys

from evaluate import (
    INITIAL_PROGRAM_CODE,
    PRIMARY_METRIC_NAME,
    TARGET_ACCURACY,
    TARGET_PARAMS,
    compute_nas_reward,
    evaluate_architecture_code,
    mnist_evaluation,
)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.info("=== Testing Local MNIST Architecture Evaluator ===")

    # Test 0: Unit tests on Vertex AI NAS reward computation
    logging.info("--- Test 0: Vertex AI NAS Reward Formula Unit Tests ---")
    # Case A: Meets both targets (Acc 98.5% >= 98%, Params <= TARGET_PARAMS)
    r1, pf1, af1 = compute_nas_reward(accuracy=0.985, measured_params=int(TARGET_PARAMS * 0.8))
    assert r1 == 0.985, f"Expected 0.985, got {r1}"
    assert pf1 == 1.0 and af1 == 1.0

    # Case B: Exceeds parameter budget (Params > TARGET_PARAMS)
    measured_over = int(TARGET_PARAMS * 1.5)
    r2, pf2, af2 = compute_nas_reward(accuracy=0.985, measured_params=measured_over)
    expected_r2 = 0.985 * 1.0 * (TARGET_PARAMS / measured_over)
    assert abs(r2 - expected_r2) < 1e-6, f"Expected {expected_r2}, got {r2}"
    assert abs(pf2 - (TARGET_PARAMS / measured_over)) < 1e-6 and af2 == 1.0

    # Case C: Below accuracy threshold (Acc 94.0% < 98%, Params <= TARGET_PARAMS)
    r3, pf3, af3 = compute_nas_reward(accuracy=0.94, measured_params=int(TARGET_PARAMS * 0.8))
    expected_r3 = 0.94 * (0.94 / TARGET_ACCURACY) * 1.0
    assert abs(r3 - expected_r3) < 1e-6, f"Expected {expected_r3}, got {r3}"
    logging.info(" Vertex NAS reward unit tests passed.")

    logging.info("Evaluating seed program from program.py...")

    # Test 1: Direct evaluation of seed program
    reward_score, metrics, insights = evaluate_architecture_code(
        code_str=INITIAL_PROGRAM_CODE,
        epochs=2,
        batch_size=64,
    )

    logging.info("--- Evaluation Results ---")
    logging.info("Primary Score (Reward): %.4f", reward_score)
    for k, v in metrics.items():
        logging.info("Metric [%s]: %s", k, v)
    for insight in insights:
        logging.info("Insight: %s", insight)

    assert reward_score > 0.65, f"Expected seed model reward > 0.65, got {reward_score}"
    assert metrics["val_accuracy"] > 0.75, f"Expected val_accuracy > 0.75, got {metrics['val_accuracy']}"
    assert metrics["param_count"] <= TARGET_PARAMS, f"Expected params <= {TARGET_PARAMS}, got {metrics['param_count']}"

    # Test 2: AlphaEvolve candidate payload format test
    logging.info("--- Testing AlphaEvolve Candidate Payload Structure ---")
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
    logging.info("Evaluation Result Object: %s", eval_result)
    assert eval_result["scores"]["scores"][0]["metric"] == PRIMARY_METRIC_NAME

    logging.info(" All local evaluator tests passed successfully!")


if __name__ == "__main__":
    main()
