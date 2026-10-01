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

logger = logging.getLogger(__name__)

PRIMARY_METRIC_NAME = "val_accuracy"
SEED_BOOTSTRAP_SCORE = -1e6

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
) -> Tuple[float, Dict[str, float], List[str]]:
    """Executes candidate code, trains the model on MNIST, and computes metrics.

    Args:
        code_str: Python source code containing the `build_model()` definition.
        epochs: Number of proxy training epochs.
        batch_size: Training batch size.
        learning_rate: Adam optimizer learning rate.

    Returns:
        score: Primary metric value (val_accuracy, 0.0 to 1.0, or negative sentinel on failure).
        metrics_dict: Auxiliary metrics (val_loss, train_time_sec, param_count).
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

    # Flag models that exceed budget
    if param_count > 1_000_000:
        insights.append(
            f"Model has {param_count:,} parameters, exceeding the 1M parameter budget."
        )

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

        insights.append(
            f"Evaluated successfully. Val Accuracy: {final_val_acc:.4f}, "
            f"Val Loss: {final_val_loss:.4f}, Params: {param_count:,}, Train Time: {elapsed:.2f}s."
        )
        return final_val_acc, metrics, insights

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
