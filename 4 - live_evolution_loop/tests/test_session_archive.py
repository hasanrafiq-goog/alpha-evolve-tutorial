"""Unit tests for search session archiving and history listing."""

import json
import os
import shutil
import tempfile
import pytest

from session_manager import DashboardSessionManager


def test_session_archiving_on_startup():
    """Verify that an existing runs.json with runs is archived on new session startup."""
    with tempfile.TemporaryDirectory() as temp_dir:
        runs_path = os.path.join(temp_dir, "runs.json")
        history_dir = os.path.join(temp_dir, "history")

        # Create a mock prior session runs.json
        prior_state = {
            "session_id": "session_20261117_100000",
            "status": "COMPLETED",
            "runs": [
                {
                    "run_number": 0,
                    "candidate_id": "...1234567",
                    "rank": "1",
                    "nas_reward": 0.55,
                    "val_accuracy": 0.965,
                    "param_count": 2800,
                    "train_time_sec": 4.5,
                    "mutation_summary": "Test mutation",
                    "status": "CHAMPION",
                    "code_content": "def build_model(): pass",
                    "insights": [],
                    "metrics": {},
                }
            ],
            "kpis": {
                "top_accuracy": 0.965,
                "accuracy_delta": 0.011,
                "baseline_accuracy": 0.954,
                "relative_error_reduction": 23.9,
                "optimal_params": 2800,
                "param_reduction_percent": 87.2,
                "param_compression_ratio": 7.82,
                "convergence_efficiency": "Run #0 / 20",
                "exploration_rate": "100% Zero-Touch",
            },
        }
        with open(runs_path, "w") as f:
            json.dump(prior_state, f)

        # Initialize a new session manager
        mgr = DashboardSessionManager(output_dir=temp_dir)

        # The prior session should be archived in history_dir
        archived_files = os.listdir(history_dir)
        assert "session_20261117_100000.json" in archived_files

        # And get_history_sessions should list active + archived
        history = mgr.get_history_sessions()
        assert len(history) == 2
        assert history[0]["id"] == "active"
        assert history[0]["is_active"] is True

        archived_entry = history[1]
        assert archived_entry["id"] == "session_20261117_100000"
        assert archived_entry["is_active"] is False
        assert archived_entry["top_accuracy"] == 0.965
        assert archived_entry["optimal_params"] == 2800


def test_dashboard_server_historical_endpoint():
    """Verify DashboardServer serves historical session data via /api/sessions/<id>."""
    import urllib.request
    from dashboard_server import DashboardServer

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "evolved_output")
        dashboard_dir = os.path.join(tmpdir, "dashboard")
        os.makedirs(dashboard_dir, exist_ok=True)
        history_dir = os.path.join(output_dir, "history")
        os.makedirs(history_dir, exist_ok=True)

        # Seed an archived session
        hist_data = {
            "session_id": "session_past_run",
            "status": "COMPLETED",
            "runs": [{"candidate_id": "...9999999", "nas_reward": 0.88}],
        }
        with open(os.path.join(history_dir, "session_past_run.json"), "w") as f:
            json.dump(hist_data, f)

        mgr = DashboardSessionManager(output_dir=output_dir)
        server = DashboardServer(
            dashboard_dir=dashboard_dir,
            session_manager=mgr,
            start_port=8920,
            auto_open=False,
        )
        server.start()

        try:
            url = f"{server.url}/api/sessions/session_past_run"
            with urllib.request.urlopen(url) as res:
                assert res.status == 200
                data = json.loads(res.read().decode("utf-8"))
                assert data["session_id"] == "session_past_run"
                assert len(data["runs"]) == 1
                assert data["runs"][0]["nas_reward"] == 0.88
        finally:
            server.stop()
