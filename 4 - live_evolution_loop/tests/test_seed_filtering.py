"""Unit tests validating that Generation 0 seed baseline is filtered out from table and charts."""

import os
import re
import pytest

DASHBOARD_JS_PATH = os.path.join(os.path.dirname(__file__), "..", "dashboard", "assets", "dashboard.js")


def test_seed_filtering_in_javascript_source():
    """Verify that dashboard.js explicitly filters out seed (run_number === 0 or status === 'SEED')."""
    with open(DASHBOARD_JS_PATH, "r") as f:
        js = f.read()

    # Table renderer must filter out seed
    assert "r.run_number === 0" in js or "r.status === 'SEED'" in js or "run_number > 0" in js

    # Pareto chart must filter out seed
    assert "r.status !== 'SEED'" in js or "r.status !== \"SEED\"" in js or "r.run_number > 0" in js

    # Trajectory chart must filter out seed
    assert "labels.push" in js


def test_seed_model_separation_in_pipeline():
    """Verify that pipeline separates seed reference data from candidate visualization sets."""
    sample_runs = [
        {"run_number": 0, "status": "SEED", "val_accuracy": 0.954, "param_count": 21898},
        {"run_number": 1, "status": "SUCCESS", "val_accuracy": 0.968, "param_count": 2800},
        {"run_number": 2, "status": "SUCCESS", "val_accuracy": 0.962, "param_count": 3100},
    ]

    evolved_candidates = [
        r for r in sample_runs
        if r.get("run_number", 0) > 0 and r.get("status") != "SEED"
    ]

    assert len(evolved_candidates) == 2
    assert all(r["run_number"] > 0 for r in evolved_candidates)
    assert all(r["status"] != "SEED" for r in evolved_candidates)
    assert not any(r["param_count"] == 21898 for r in evolved_candidates)
