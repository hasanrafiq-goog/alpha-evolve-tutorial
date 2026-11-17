"""Unit tests for DashboardSessionManager and AST mutation summary extraction."""

import json
import os
import shutil
import tempfile
import threading
from typing import Dict, Any
import pytest

from session_manager import DashboardSessionManager, extract_mutation_summary


def test_extract_mutation_summary_from_docstring():
    code_with_docstring = '''
import tensorflow as tf

def build_model():
    """Added LayerNorm & Swish activations for richer representation.
    
    More details here that should not be in the short summary.
    """
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(8, (3, 3))(inputs)
    return tf.keras.Model(inputs=inputs, outputs=x)
'''
    summary = extract_mutation_summary(code_with_docstring)
    assert "LayerNorm" in summary or "Swish" in summary


def test_extract_mutation_summary_fallback_ast():
    code_without_docstring = '''
import tensorflow as tf

def build_model():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.SeparableConv2D(16, (3, 3))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
'''
    summary = extract_mutation_summary(code_without_docstring)
    assert "SeparableConv2D" in summary
    assert "GlobalAveragePooling2D" in summary or "Dense" in summary


def test_session_manager_initialization_and_atomic_writes():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "evolved_output")
        manager = DashboardSessionManager(
            output_dir=output_dir,
            target_params=3000,
            target_accuracy=0.94,
            max_programs=20,
        )

        runs_path = os.path.join(output_dir, "runs.json")
        assert os.path.exists(runs_path)

        with open(runs_path, "r") as f:
            data = json.load(f)

        assert data["target_params"] == 3000
        assert data["target_accuracy"] == 0.94
        assert data["status"] == "INITIALIZING"

        # Record a candidate run
        manager.record_candidate(
            run_number=1,
            candidate_id="...3699149",
            rank="1",
            nas_reward=0.8617,
            val_accuracy=0.9845,
            param_count=2978,
            train_time_sec=0.72,
            mutation_summary="SeparableConv + Preserved 2x2 Spatial Quadrants",
            status="SUCCESS",
            code_content="def build_model(): pass",
            insights=["Parameter budget satisfied."],
        )

        with open(runs_path, "r") as f:
            updated = json.load(f)

        assert updated["status"] == "IN_PROGRESS"
        assert len(updated["runs"]) == 1
        assert updated["runs"][0]["candidate_id"] == "...3699149"
        assert updated["runs"][0]["nas_reward"] == 0.8617
        assert updated["kpis"]["top_accuracy"] == 0.9845
        assert updated["kpis"]["optimal_params"] == 2978


def test_concurrent_writes_thread_safety():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "evolved_output")
        manager = DashboardSessionManager(
            output_dir=output_dir,
            target_params=50000,
            target_accuracy=0.98,
            max_programs=50,
        )

        runs_path = os.path.join(output_dir, "runs.json")

        def worker_write(run_idx: int):
            manager.record_candidate(
                run_number=run_idx,
                candidate_id=f"...test{run_idx}",
                rank=str(run_idx),
                nas_reward=0.5 + (run_idx * 0.01),
                val_accuracy=0.90 + (run_idx * 0.001),
                param_count=20000 - (run_idx * 100),
                train_time_sec=0.5,
                mutation_summary=f"Worker {run_idx} mutation",
                status="SUCCESS",
                code_content="def build_model(): pass",
                insights=[],
            )

        threads = [threading.Thread(target=worker_write, args=(i,)) for i in range(1, 21)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Ensure runs.json can be loaded cleanly without JSONDecodeError
        with open(runs_path, "r") as f:
            final_data = json.load(f)

        assert len(final_data["runs"]) == 20


def test_dynamic_live_rank_ordering():
    """Verify live ranking dynamically orders models by reward instead of chronological run order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "evolved_output")
        manager = DashboardSessionManager(output_dir=output_dir)

        # Seed run (Run 0)
        manager.record_candidate(
            run_number=0, candidate_id="...seed", rank="Base", nas_reward=0.20,
            val_accuracy=0.954, param_count=21898, train_time_sec=0.0,
            mutation_summary="Base", status="SEED", code_content="", insights=[],
        )

        # Candidate 1: reward 0.30 -> becomes live Rank 1
        manager.record_candidate(
            run_number=1, candidate_id="...c1", rank="-", nas_reward=0.30,
            val_accuracy=0.960, param_count=10000, train_time_sec=1.0,
            mutation_summary="c1", status="SUCCESS", code_content="", insights=[],
        )
        assert manager.state["runs"][1]["rank"] == "RANK 1 ⭐️"

        # Candidate 2: reward 0.50 -> higher than c1, so c2 becomes Rank 1 and c1 becomes Rank 2
        manager.record_candidate(
            run_number=2, candidate_id="...c2", rank="-", nas_reward=0.50,
            val_accuracy=0.970, param_count=8000, train_time_sec=1.0,
            mutation_summary="c2", status="SUCCESS", code_content="", insights=[],
        )
        assert manager.state["runs"][1]["rank"] == "RANK 2"
        assert manager.state["runs"][2]["rank"] == "RANK 1 ⭐️"

        # Candidate 3: reward 0.40 -> between c1 and c2, so c2 is Rank 1, c3 is Rank 2, c1 is Rank 3
        manager.record_candidate(
            run_number=3, candidate_id="...c3", rank="-", nas_reward=0.40,
            val_accuracy=0.965, param_count=9000, train_time_sec=1.0,
            mutation_summary="c3", status="SUCCESS", code_content="", insights=[],
        )
        assert manager.state["runs"][1]["rank"] == "RANK 3"
        assert manager.state["runs"][2]["rank"] == "RANK 1 ⭐️"
        assert manager.state["runs"][3]["rank"] == "RANK 2"

        # Candidate 4: reward 0.10 -> lower than all top 3, so it gets '-'
        manager.record_candidate(
            run_number=4, candidate_id="...c4", rank="-", nas_reward=0.10,
            val_accuracy=0.940, param_count=15000, train_time_sec=1.0,
            mutation_summary="c4", status="SUCCESS", code_content="", insights=[],
        )
        assert manager.state["runs"][4]["rank"] == "-"
