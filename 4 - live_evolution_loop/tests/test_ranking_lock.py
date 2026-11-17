"""Unit tests verifying single champion ranking, '-' default for runs, and tournament rank locking."""

import tempfile
import pytest

from session_manager import DashboardSessionManager


def test_unranked_arrival_and_single_rank_1():
    """Verify that arrival numbers do not cause false Rank 1 badges and only one Rank 1 exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = DashboardSessionManager(output_dir=tmpdir)

        # Seed baseline
        manager.record_candidate(
            run_number=0, candidate_id="...seed", rank="Base", nas_reward=0.20,
            val_accuracy=0.9540, param_count=21898, train_time_sec=0.0,
            mutation_summary="Base", status="SEED", code_content="", insights=[],
        )

        # Run 1: reward 0.35 -> Rank 1
        manager.record_candidate(
            run_number=1, candidate_id="...c1", rank="-", nas_reward=0.35,
            val_accuracy=0.9650, param_count=8000, train_time_sec=1.0,
            mutation_summary="c1", status="SUCCESS", code_content="", insights=[],
        )

        # Run 10: reward 0.15 (arrival index has '10', must NOT get Rank 1!)
        manager.record_candidate(
            run_number=10, candidate_id="...c10", rank="-", nas_reward=0.15,
            val_accuracy=0.9600, param_count=9000, train_time_sec=1.0,
            mutation_summary="c10", status="SUCCESS", code_content="", insights=[],
        )

        # Run 11: reward 0.10 (arrival index has '11', must NOT get Rank 1!)
        manager.record_candidate(
            run_number=11, candidate_id="...c11", rank="-", nas_reward=0.10,
            val_accuracy=0.9550, param_count=10000, train_time_sec=1.0,
            mutation_summary="c11", status="SUCCESS", code_content="", insights=[],
        )

        runs = manager.state["runs"]
        rank_1_runs = [r for r in runs if "RANK 1" in str(r.get("rank", ""))]
        assert len(rank_1_runs) == 1
        assert rank_1_runs[0]["candidate_id"] == "...c1"

        c10_run = next(r for r in runs if r["run_number"] == 10)
        c11_run = next(r for r in runs if r["run_number"] == 11)
        assert "RANK 1" not in c10_run["rank"]
        assert "RANK 1" not in c11_run["rank"]


def test_update_final_ranks_locks_tournament_champion():
    """Verify update_final_ranks resets provisional ranks and establishes official tournament ranks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = DashboardSessionManager(output_dir=tmpdir)

        # Seed and candidates
        manager.record_candidate(
            run_number=0, candidate_id="...seed", rank="Base", nas_reward=0.20,
            val_accuracy=0.9540, param_count=21898, train_time_sec=0.0,
            mutation_summary="Base", status="SEED", code_content="", insights=[],
        )
        manager.record_candidate(
            run_number=1, candidate_id="...cand1", rank="-", nas_reward=0.35,
            val_accuracy=0.9650, param_count=8000, train_time_sec=1.0,
            mutation_summary="cand1", status="SUCCESS", code_content="", insights=[],
        )
        manager.record_candidate(
            run_number=2, candidate_id="...cand2", rank="-", nas_reward=0.55,
            val_accuracy=0.9700, param_count=3000, train_time_sec=1.0,
            mutation_summary="cand2", status="SUCCESS", code_content="", insights=[],
        )

        # Final tournament ranking from Google Cloud establishes cand2 as Rank 1 and cand1 as Rank 2
        top_records = [
            {"rank": 1, "program_id": "cand2"},
            {"rank": 2, "program_id": "cand1"},
        ]
        manager.update_final_ranks(top_records)

        runs = manager.state["runs"]
        cand2_run = next(r for r in runs if "...cand2" in r["candidate_id"])
        cand1_run = next(r for r in runs if "...cand1" in r["candidate_id"])
        seed_run = next(r for r in runs if "...seed" in r["candidate_id"])

        assert cand2_run["rank"] == "RANK 1 ⭐️"
        assert cand2_run["status"] == "CHAMPION"
        assert cand1_run["rank"] == "RANK 2"
        assert seed_run["rank"] == "Base"


def test_dynamic_live_ranks_1_2_and_3_with_penalized_proxy_runs():
    """Verify that early proxy models (including PENALTY status runs) receive dynamic Rank 1, 2, and 3."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = DashboardSessionManager(output_dir=tmpdir)

        # Record Seed (Run 0)
        manager.record_candidate(
            run_number=0, candidate_id="...seed", rank="Base", nas_reward=0.20,
            val_accuracy=0.9540, param_count=21898, train_time_sec=0.0,
            mutation_summary="Base", status="SEED", code_content="", insights=[],
        )

        # Run 1: Early proxy model with penalty status (acc 25%, reward 0.08) -> Rank 1 Champion
        manager.record_candidate(
            run_number=1, candidate_id="...c1", rank="-", nas_reward=0.08,
            val_accuracy=0.2500, param_count=1800, train_time_sec=1.5,
            mutation_summary="c1", status="PENALTY", code_content="", insights=[],
        )
        assert manager.state["runs"][1]["rank"] == "RANK 1 ⭐️"

        # Run 2: Slightly better proxy model (acc 35%, reward 0.12) -> Takes Rank 1, c1 becomes Rank 2
        manager.record_candidate(
            run_number=2, candidate_id="...c2", rank="-", nas_reward=0.12,
            val_accuracy=0.3500, param_count=1900, train_time_sec=2.0,
            mutation_summary="c2", status="SUCCESS", code_content="", insights=[],
        )
        assert manager.state["runs"][2]["rank"] == "RANK 1 ⭐️"
        assert manager.state["runs"][1]["rank"] == "RANK 2"

        # Run 3: Lower scoring proxy model (acc 18%, reward 0.05) -> Becomes Rank 3
        manager.record_candidate(
            run_number=3, candidate_id="...c3", rank="-", nas_reward=0.05,
            val_accuracy=0.1800, param_count=1700, train_time_sec=1.8,
            mutation_summary="c3", status="PENALTY", code_content="", insights=[],
        )
        assert manager.state["runs"][2]["rank"] == "RANK 1 ⭐️"
        assert manager.state["runs"][1]["rank"] == "RANK 2"
        assert manager.state["runs"][3]["rank"] == "RANK 3"

        # Run 4: Fourth candidate (acc 15%, reward 0.03) -> Becomes '-' (outside top 3)
        manager.record_candidate(
            run_number=4, candidate_id="...c4", rank="-", nas_reward=0.03,
            val_accuracy=0.1500, param_count=1600, train_time_sec=1.6,
            mutation_summary="c4", status="PENALTY", code_content="", insights=[],
        )
        assert manager.state["runs"][2]["rank"] == "RANK 1 ⭐️"
        assert manager.state["runs"][1]["rank"] == "RANK 2"
        assert manager.state["runs"][3]["rank"] == "RANK 3"
        assert manager.state["runs"][4]["rank"] == "-"

        # Run 5: Broken run -> stays '-' and does not disrupt ranking
        manager.record_candidate(
            run_number=5, candidate_id="...c5", rank="-", nas_reward=0.0,
            val_accuracy=0.0, param_count=0, train_time_sec=0.1,
            mutation_summary="c5", status="BROKEN", code_content="", insights=[],
        )
        assert manager.state["runs"][5]["rank"] == "-"
        assert manager.state["runs"][2]["rank"] == "RANK 1 ⭐️"
        assert manager.state["runs"][1]["rank"] == "RANK 2"
        assert manager.state["runs"][3]["rank"] == "RANK 3"

        # KPIs must reflect Rank 1 (Run 2: acc 35%, params 1900)
        assert manager.state["kpis"]["top_accuracy"] == 0.3500
        assert manager.state["kpis"]["optimal_params"] == 1900

