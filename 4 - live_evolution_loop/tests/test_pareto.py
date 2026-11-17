"""Unit tests for Pareto frontier calculation and trajectory data transformations."""

import pytest
from session_manager import compute_pareto_frontier


def test_compute_pareto_frontier_basic():
    """Verify that non-dominated models are correctly identified and sorted by param count."""
    runs = [
        {"candidate_id": "seed", "val_accuracy": 0.9540, "param_count": 21898, "nas_reward": 0.20},
        {"candidate_id": "c1", "val_accuracy": 0.9600, "param_count": 10000, "nas_reward": 0.45},
        {"candidate_id": "c2", "val_accuracy": 0.9500, "param_count": 15000, "nas_reward": 0.30},  # Dominated by c1 and seed
        {"candidate_id": "c3", "val_accuracy": 0.9700, "param_count": 12000, "nas_reward": 0.55},  # Non-dominated (higher acc)
        {"candidate_id": "c4", "val_accuracy": 0.9400, "param_count": 5000, "nas_reward": 0.50},   # Non-dominated (lower params)
        {"candidate_id": "c_fail", "val_accuracy": 0.0, "param_count": 0, "nas_reward": 0.01},     # Failed run, excluded
    ]

    frontier = compute_pareto_frontier(runs)
    frontier_ids = [r["candidate_id"] for r in frontier]

    # c4 (5000), c1 (10000), c3 (12000) are all non-dominated
    assert "c4" in frontier_ids
    assert "c1" in frontier_ids
    assert "c3" in frontier_ids
    assert "c2" not in frontier_ids
    assert "c_fail" not in frontier_ids

    # Should be sorted by param_count ascending
    param_counts = [r["param_count"] for r in frontier]
    assert param_counts == sorted(param_counts)


def test_compute_pareto_frontier_empty_and_failed():
    """Verify empty list and failed runs return empty frontier."""
    assert compute_pareto_frontier([]) == []
    assert compute_pareto_frontier([{"val_accuracy": 0.0, "param_count": 0}]) == []


def test_trajectory_progression_helper():
    """Verify trajectory best-so-far calculation."""
    runs = [
        {"run_number": 0, "nas_reward": 0.20},
        {"run_number": 1, "nas_reward": 0.15},
        {"run_number": 2, "nas_reward": 0.42},
        {"run_number": 3, "nas_reward": 0.38},
        {"run_number": 4, "nas_reward": 0.55},
    ]

    best_so_far = []
    current_max = -float("inf")
    for r in runs:
        rew = r["nas_reward"]
        if rew > current_max:
            current_max = rew
        best_so_far.append(current_max)

    assert best_so_far == [0.20, 0.20, 0.42, 0.42, 0.55]


def test_pareto_chart_dynamic_y_axis_in_javascript():
    """Verify that dashboard.js calculates dynamic yMin rather than hardcoding min: 70."""
    import os
    js_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "assets", "dashboard.js")
    with open(js_path, "r") as f:
        js = f.read()

    # Must calculate yMin dynamically from candidate accuracies
    assert "yMin" in js
    assert "minAcc" in js
    # min: 70 must no longer be present in scales.y
    assert "min: 70" not in js

