"""Evaluator for AlphaEvolve MNIST Architecture Search.

This module evaluates candidate neural network architectures proposed by Gemini.
It executes candidate Python code in a safe namespace, verifies input/output
tensor dimensions, trains on a fast MNIST proxy split, and returns validation
metrics along with actionable insight messages for the LLM.
"""

import logging
import os
import time
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf

from alpha_evolve.models import (
    AlphaEvolveEvaluationInsight,
    AlphaEvolveEvaluationInsights,
    AlphaEvolveEvaluationScore,
    AlphaEvolveEvaluationScores,
    AlphaEvolveProgramEvaluation,
)

from dotenv import load_dotenv

# Load .env configuration
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv()

logger = logging.getLogger(__name__)

PRIMARY_METRIC_NAME = os.getenv("PRIMARY_METRIC_NAME", "reward")
TARGET_ACCURACY = float(os.getenv("TARGET_ACCURACY", "0.98"))  # Target: Accuracy > 98%
TARGET_PARAMS = int(os.getenv("TARGET_PARAMS", "50000"))  # Budget: Parameters (default: 50,000)
SEED_BOOTSTRAP_SCORE = -1e6


def compute_nas_reward(
    accuracy: float,
    measured_params: int,
    target_params: int = TARGET_PARAMS,
    target_accuracy: float = TARGET_ACCURACY,
    weight: float = 1.0,
    use_hard_limit: bool = True,
) -> Tuple[float, float, float]:
    """Computes multi-objective reward following the Vertex AI NAS / MnasNet formulation.

    Reference:
        Vertex AI NAS: https://github.com/google/vertex-ai-nas/blob/main/cloud_nas_utils.py#L383
        MnasNet Paper: https://arxiv.org/pdf/1807.11626.pdf

    Formulation:
        Reward = Accuracy * (Accuracy_Ratio ** weight) * (Param_Ratio ** weight)

    Constraints:
        1. Accuracy Target (> 98%):
           Accuracy_Ratio = Accuracy / Target_Accuracy
           If Accuracy >= Target_Accuracy, Accuracy_Ratio is capped at 1.0 (no penalty).
           If Accuracy < Target_Accuracy, Accuracy_Ratio < 1.0 (penalized).
        2. Parameter Budget (< 150,000):
           Param_Ratio = Target_Params / Measured_Params
           If Measured_Params <= Target_Params, Param_Ratio is capped at 1.0 (no penalty).
           If Measured_Params > Target_Params, Param_Ratio < 1.0 (penalized).

    Args:
        accuracy: Measured validation accuracy (0.0 to 1.0).
        measured_params: Total parameter count of the candidate model.
        target_params: Maximum parameter budget (default: 150,000).
        target_accuracy: Minimum accuracy target (default: 0.98).
        weight: Exponential penalty weight (default: 1.0).
        use_hard_limit: True to cap ratios at 1.0 (hard limit mode from MnasNet).

    Returns:
        reward: Combined multi-objective reward value.
        param_factor: Applied parameter penalty factor in [0.0, 1.0].
        accuracy_factor: Applied accuracy penalty factor in [0.0, 1.0].
    """
    if measured_params <= 0:
        param_ratio = 1.0
    else:
        param_ratio = float(target_params) / float(measured_params)

    if target_accuracy <= 0.0:
        acc_ratio = 1.0
    else:
        acc_ratio = float(accuracy) / float(target_accuracy)

    param_factor = param_ratio**weight
    acc_factor = acc_ratio**weight

    if use_hard_limit:
        param_factor = min(param_factor, 1.0)
        acc_factor = min(acc_factor, 1.0)

    reward = float(accuracy * acc_factor * param_factor)
    return reward, param_factor, acc_factor

# Cached MNIST dataset partitions for fast repeated evaluation
_DATA_CACHE: Dict[str, np.ndarray] = {}


def get_mnist_proxy_data(
    train_samples: int = 10000,
    val_samples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads and caches a normalized proxy slice of MNIST for fast candidate scoring."""
    global _DATA_CACHE
    if "x_train" not in _DATA_CACHE:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Normalize to [0.0, 1.0] and add channel dimension (N, 28, 28, 1)
        x_train = (x_train.astype(np.float32) / 255.0)[..., np.newaxis]
        x_test = (x_test.astype(np.float32) / 255.0)[..., np.newaxis]

        _DATA_CACHE["x_train"] = x_train[:train_samples]
        _DATA_CACHE["y_train"] = y_train[:train_samples]
        _DATA_CACHE["x_val"] = x_test[:val_samples]
        _DATA_CACHE["y_val"] = y_test[:val_samples]

    return (
        _DATA_CACHE["x_train"],
        _DATA_CACHE["y_train"],
        _DATA_CACHE["x_val"],
        _DATA_CACHE["y_val"],
    )


def _load_initial_program() -> str:
    """Loads the seed program code from program.py."""
    program_path = os.path.join(os.path.dirname(__file__), "program.py")
    with open(program_path, "r") as f:
        return f.read()


INITIAL_PROGRAM_CODE = _load_initial_program()


def evaluate_architecture_code(
    code_str: str,
    epochs: int = 2,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    target_accuracy: float = TARGET_ACCURACY,
    target_params: int = TARGET_PARAMS,
) -> Tuple[float, Dict[str, float], List[str]]:
    """Executes candidate code, trains the model on MNIST, and computes metrics.

    Args:
        code_str: Python source code containing the `build_model()` definition.
        epochs: Number of proxy training epochs.
        batch_size: Training batch size.
        learning_rate: Adam optimizer learning rate.
        target_accuracy: Target validation accuracy threshold (default: 0.98).
        target_params: Maximum parameter count budget (default: 150,000).

    Returns:
        score: Primary metric value (Vertex AI NAS multi-objective reward, 0.0 to 1.0, or negative sentinel on failure).
        metrics_dict: Auxiliary metrics (reward, val_accuracy, param_count, penalties, val_loss, train_time_sec).
        insights_list: Explanatory insight feedback strings for the LLM.
    """
    insights: List[str] = []
    metrics: Dict[str, float] = {}

    # 1. Execute candidate code in an isolated namespace
    namespace: Dict[str, Any] = {"tf": tf, "keras": tf.keras}
    try:
        exec(code_str, namespace)
    except Exception as e:
        error_msg = f"Syntax or import error executing code: {type(e).__name__}: {e}"
        insights.append(error_msg)
        return SEED_BOOTSTRAP_SCORE, metrics, insights

    if "build_model" not in namespace or not callable(namespace["build_model"]):
        insights.append("Missing or non-callable `build_model()` function in candidate code.")
        return SEED_BOOTSTRAP_SCORE, metrics, insights

    build_model_fn = namespace["build_model"]

    # 2. Instantiate and validate model structure
    try:
        # Clear backend session to prevent memory leaks across iterations
        tf.keras.backend.clear_session()
        model = build_model_fn()
    except Exception as e:
        error_msg = f"Failed to instantiate model from build_model(): {type(e).__name__}: {e}"
        insights.append(error_msg)
        return SEED_BOOTSTRAP_SCORE, metrics, insights

    if not isinstance(model, tf.keras.Model):
        insights.append(f"build_model() returned {type(model)} instead of tf.keras.Model.")
        return SEED_BOOTSTRAP_SCORE, metrics, insights

    # 3. Verify tensor shapes with a dummy forward pass
    dummy_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
    try:
        dummy_output = model(dummy_input, training=False)
        if dummy_output.shape[-1] != 10:
            insights.append(
                f"Output shape mismatch: expected (batch, 10), but got {dummy_output.shape}."
            )
            return SEED_BOOTSTRAP_SCORE, metrics, insights
    except Exception as e:
        insights.append(f"Shape validation failed on input shape (1, 28, 28, 1): {e}")
        return SEED_BOOTSTRAP_SCORE, metrics, insights

    # Count parameters
    param_count = int(model.count_params())
    metrics["param_count"] = float(param_count)

    # 4. Compile and Train on MNIST Proxy Slice
    try:
        x_tr, y_tr, x_val, y_val = get_mnist_proxy_data()

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        start_time = time.time()
        history = model.fit(
            x_tr,
            y_tr,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        elapsed = time.time() - start_time
        metrics["train_time_sec"] = float(elapsed)

        final_val_acc = float(history.history["val_accuracy"][-1])
        final_val_loss = float(history.history["val_loss"][-1])
        metrics["val_loss"] = final_val_loss
        metrics["val_accuracy"] = final_val_acc

        # 5. Compute Vertex AI NAS Multi-Objective Reward
        reward, param_factor, acc_factor = compute_nas_reward(
            accuracy=final_val_acc,
            measured_params=param_count,
            target_params=target_params,
            target_accuracy=target_accuracy,
        )
        metrics["reward"] = reward
        metrics["param_penalty_factor"] = param_factor
        metrics["accuracy_penalty_factor"] = acc_factor

        # 6. Generate Actionable Feedback Insights for Gemini Mutator
        if param_count > target_params:
            insights.append(
                f"Parameter budget exceeded: {param_count:,} > {target_params:,}. "
                f"Vertex NAS penalty applied: {param_factor:.4f}x."
            )
        else:
            insights.append(
                f"Parameter budget satisfied: {param_count:,} <= {target_params:,} (factor: 1.0000x)."
            )

        if final_val_acc < target_accuracy:
            insights.append(
                f"Accuracy target not reached: {final_val_acc:.2%} < {target_accuracy:.1%}. "
                f"Vertex NAS penalty applied: {acc_factor:.4f}x."
            )
        else:
            insights.append(
                f"Accuracy target satisfied: {final_val_acc:.2%} >= {target_accuracy:.1%} (factor: 1.0000x)."
            )

        insights.append(
            f"Evaluated successfully. Reward: {reward:.4f} [Accuracy: {final_val_acc:.4f} x "
            f"AccFactor: {acc_factor:.3f} x ParamFactor: {param_factor:.3f}], "
            f"Val Loss: {final_val_loss:.4f}, Params: {param_count:,}, Train Time: {elapsed:.2f}s."
        )
        return reward, metrics, insights

    except Exception as e:
        error_msg = f"Runtime training error: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        insights.append(error_msg)
        return SEED_BOOTSTRAP_SCORE, metrics, insights


def mnist_evaluation(program_candidate: dict) -> dict:
    """Evaluation entrypoint called by the AlphaEvolve controller loop.

    Args:
        program_candidate: Candidate dictionary received from AlphaEvolve API.

    Returns:
        Structured evaluation result dict containing scores and insights.
    """
    logger.info("Evaluating program candidate...")
    code = program_candidate["content"]["files"][0]["content"]

    score_val, metrics, insights = evaluate_architecture_code(code)

    # Build the scores list
    scores_list = [
        AlphaEvolveEvaluationScore(metric=PRIMARY_METRIC_NAME, score=float(score_val))
    ]
    for k, v in metrics.items():
        if k != PRIMARY_METRIC_NAME:
            scores_list.append(AlphaEvolveEvaluationScore(metric=k, score=float(v)))

    # Build the insights list
    insights_list = [
        AlphaEvolveEvaluationInsight(label="eval_log", text=str(msg))
        for msg in insights
    ]

    evaluation = AlphaEvolveProgramEvaluation(
        scores=AlphaEvolveEvaluationScores(scores=scores_list),
        insights=AlphaEvolveEvaluationInsights(insights=insights_list),
    )

    return evaluation.model_dump(exclude_none=True)
