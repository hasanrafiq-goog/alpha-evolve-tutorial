"""Unit tests verifying dynamic binding of KPI summary cards strictly to the Rank 1 champion."""

import json
import os
import tempfile
import pytest

from session_manager import DashboardSessionManager


def test_kpi_zero_candidate_initial_state():
    """Verify that KPI cards do not hardcode baseline numbers before evolved candidates evaluate."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "evolved_output")
        manager = DashboardSessionManager(
            output_dir=output_dir,
            baseline_accuracy=0.9540,
            baseline_params=21898,
        )

        # Before any candidate runs, top_accuracy and optimal_params should be None or indicate empty
        kpis = manager.state["kpis"]
        assert kpis["top_accuracy"] is None
        assert kpis["optimal_params"] is None
        assert kpis["accuracy_delta"] is None
        assert kpis["param_reduction_percent"] is None

        # Even after recording the Seed baseline (Run 0), KPIs should still be None (seed does not count as champion)
        manager.record_candidate(
            run_number=0, candidate_id="...seed", rank="Base", nas_reward=0.20,
            val_accuracy=0.9540, param_count=21898, train_time_sec=0.0,
            mutation_summary="Base", status="SEED", code_content="", insights=[],
        )
        kpis_after_seed = manager.state["kpis"]
        assert kpis_after_seed["top_accuracy"] is None
        assert kpis_after_seed["optimal_params"] is None


def test_kpi_bound_strictly_to_rank_1_champion():
    """Verify that Top Accuracy and Optimal Params are strictly from Rank 1, not raw max accuracy."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "evolved_output")
        manager = DashboardSessionManager(
            output_dir=output_dir,
            baseline_accuracy=0.9540,
            baseline_params=21898,
        )

        # Record Seed (Run 0)
        manager.record_candidate(
            run_number=0, candidate_id="...seed", rank="Base", nas_reward=0.20,
            val_accuracy=0.9540, param_count=21898, train_time_sec=0.0,
            mutation_summary="Base", status="SEED", code_content="", insights=[],
        )

        # Candidate 1: 96.50% acc, 8,000 params, reward 0.35 -> Becomes Rank 1 Champion
        manager.record_candidate(
            run_number=1, candidate_id="...c1", rank="-", nas_reward=0.35,
            val_accuracy=0.9650, param_count=8000, train_time_sec=1.0,
            mutation_summary="c1", status="SUCCESS", code_content="", insights=[],
        )
        kpis1 = manager.state["kpis"]
        assert kpis1["top_accuracy"] == 0.9650
        assert kpis1["optimal_params"] == 8000
        assert round(kpis1["accuracy_delta"], 4) == round(0.9650 - 0.9540, 4)
        assert round(kpis1["param_compression_ratio"], 2) == round(21898 / 8000, 2)

        # Candidate 2: High raw accuracy (99.00%), but huge params (50,000) -> Lower reward (0.15) -> Rank 2
        manager.record_candidate(
            run_number=2, candidate_id="...c2", rank="-", nas_reward=0.15,
            val_accuracy=0.9900, param_count=50000, train_time_sec=1.0,
            mutation_summary="c2", status="SUCCESS", code_content="", insights=[],
        )
        kpis2 = manager.state["kpis"]
        # CRITICAL TEST: Top Accuracy MUST be Rank 1's accuracy (96.50%), NOT c2's raw 99.00%!
        assert kpis2["top_accuracy"] == 0.9650
        assert kpis2["optimal_params"] == 8000

        # Candidate 3: 97.20% acc, 2,746 params, reward 0.55 -> Beats c1 and becomes NEW Rank 1 Champion!
        manager.record_candidate(
            run_number=3, candidate_id="...c3", rank="-", nas_reward=0.55,
            val_accuracy=0.9720, param_count=2746, train_time_sec=1.0,
            mutation_summary="c3", status="SUCCESS", code_content="", insights=[],
        )
        kpis3 = manager.state["kpis"]
        assert kpis3["top_accuracy"] == 0.9720
        assert kpis3["optimal_params"] == 2746
        assert round(kpis3["accuracy_delta"], 4) == round(0.9720 - 0.9540, 4)
        assert round(kpis3["param_compression_ratio"], 2) == round(21898 / 2746, 2)
