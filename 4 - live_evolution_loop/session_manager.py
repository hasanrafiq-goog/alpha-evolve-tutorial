"""Session state manager and AST mutation summary extractor for AlphaEvolve Live Dashboard.

Provides atomic file persistence (runs.json.tmp -> os.replace), thread-safe session tracking,
docstring extraction, and multi-session archiving.
"""

import ast
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def extract_mutation_summary(code_str: str) -> str:
    """Extracts a concise natural language or structural mutation summary from candidate code.

    Attempts to read the first sentence of the `build_model()` docstring. If missing or
    uninformative, inspects the AST for Keras layer invocations to produce a structural summary.

    Args:
        code_str: Raw Python source code of the candidate model.

    Returns:
        str: Concise 1-sentence architectural summary.
    """
    try:
        tree = ast.parse(code_str)
    except Exception as e:
        logger.debug("Failed to parse code AST: %s", e)
        return "Custom Architecture Mutation"

    # 1. Search for build_model function and docstring
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_model":
            doc = ast.get_docstring(node)
            if doc:
                lines = [l.strip() for l in doc.split("\n") if l.strip()]
                for line in lines:
                    if line.startswith("Builds ") or line.startswith("Constructs "):
                        # If it has more details in subsequent sentence
                        clean = line.replace("Builds an ", "").replace("Builds a ", "").replace("Builds ", "")
                        clean = clean.split(".")[0].strip()
                        if len(clean) > 10 and not clean.startswith("naive baseline"):
                            return clean[:75]
                    elif not line.startswith("Returns:") and not line.startswith("Args:"):
                        first_sentence = line.split(".")[0].strip()
                        if len(first_sentence) > 10:
                            return first_sentence[:75]

    # 2. Fallback: Structural AST layer extraction
    layer_types: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name and any(
                k in name for k in [
                    "Conv2D", "SeparableConv2D", "Dense", "BatchNormalization",
                    "LayerNormalization", "Dropout", "MaxPooling2D",
                    "AveragePooling2D", "GlobalAveragePooling2D", "Flatten"
                ]
            ):
                if not layer_types or layer_types[-1] != name:
                    layer_types.append(name)

    if layer_types:
        return " + ".join(layer_types[:4])

    return "Neural Architecture Mutation"


def compute_pareto_frontier(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identifies the non-dominated Pareto frontier models (minimizing params, maximizing val_accuracy).

    A candidate A dominates candidate B if:
    A.accuracy >= B.accuracy and A.params <= B.params (with at least one strict inequality).
    """
    valid_runs = [r for r in runs if r.get("val_accuracy", 0) > 0 and r.get("param_count", 0) > 0]
    frontier: List[Dict[str, Any]] = []

    for cand in valid_runs:
        is_dominated = False
        for other in valid_runs:
            if other is cand:
                continue
            # other dominates cand if other has higher/equal accuracy AND lower/equal params
            if (
                other["val_accuracy"] >= cand["val_accuracy"]
                and other["param_count"] <= cand["param_count"]
                and (
                    other["val_accuracy"] > cand["val_accuracy"]
                    or other["param_count"] < cand["param_count"]
                )
            ):
                is_dominated = True
                break
        if not is_dominated:
            frontier.append(cand)

    # Sort frontier by parameter count ascending
    frontier.sort(key=lambda r: r["param_count"])
    return frontier


class DashboardSessionManager:
    """Manages active search session state, thread-safe atomic persistence, and archiving."""

    def __init__(
        self,
        output_dir: str,
        target_params: int = 3000,
        target_accuracy: float = 0.94,
        max_programs: int = 20,
        baseline_accuracy: float = 0.9540,
        baseline_params: int = 21898,
    ):
        self.output_dir = os.path.abspath(output_dir)
        self.history_dir = os.path.join(self.output_dir, "history")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)

        self.runs_path = os.path.join(self.output_dir, "runs.json")
        self.target_params = target_params
        self.target_accuracy = target_accuracy
        self.max_programs = max_programs
        self.baseline_accuracy = baseline_accuracy
        self.baseline_params = baseline_params

        self._lock = threading.Lock()
        self.session_id = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        self.started_at = time.time()

        # Archive prior runs.json if exists
        self._archive_previous_session()

        # Initialize fresh session
        self.state: Dict[str, Any] = {
            "session_id": self.session_id,
            "status": "INITIALIZING",
            "target_params": self.target_params,
            "target_accuracy": self.target_accuracy,
            "max_programs": self.max_programs,
            "started_at": self.started_at,
            "completed_at": None,
            "kpis": {
                "top_accuracy": None,
                "accuracy_delta": None,
                "baseline_accuracy": self.baseline_accuracy,
                "relative_error_reduction": None,
                "optimal_params": None,
                "param_reduction_percent": None,
                "param_compression_ratio": None,
                "convergence_efficiency": f"Evaluating Candidate #1...",
                "exploration_rate": "100% Zero-Touch",
            },
            "in_flight": None,
            "runs": [],
        }
        self._atomic_save()

    def _archive_previous_session(self) -> None:
        """Archives existing runs.json to history directory if it contains previous results."""
        if os.path.exists(self.runs_path):
            try:
                with open(self.runs_path, "r") as f:
                    old_data = json.load(f)
                old_session_id = old_data.get("session_id", f"session_{int(os.path.getmtime(self.runs_path))}")
                # Only archive if it has runs
                if old_data.get("runs"):
                    archive_file = os.path.join(self.history_dir, f"{old_session_id}.json")
                    shutil.copy2(self.runs_path, archive_file)
                    logger.info("Archived prior session to %s", archive_file)
            except Exception as e:
                logger.warning("Could not archive prior runs.json: %s", e)

    def _atomic_save(self) -> None:
        """Atomically saves the session state to runs.json via a temporary file and os.replace."""
        payload = json.dumps(self.state, indent=2)
        dir_name = os.path.dirname(self.runs_path)
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tf:
            tf.write(payload)
            temp_path = tf.name
        os.replace(temp_path, self.runs_path)

    def set_in_flight(self, run_number: int, program_id: str) -> None:
        """Sets the currently evaluating candidate in-flight status."""
        with self._lock:
            self.state["status"] = "IN_PROGRESS"
            self.state["in_flight"] = {
                "run_number": run_number,
                "program_id": program_id[-8:] if program_id else "unknown",
                "started_at": time.time(),
                "status": "EVALUATING",
            }
            self._atomic_save()

    def clear_in_flight(self) -> None:
        """Clears in-flight status."""
        with self._lock:
            self.state["in_flight"] = None
            self._atomic_save()

    def record_candidate(
        self,
        run_number: Any,
        candidate_id: str,
        rank: str,
        nas_reward: float,
        val_accuracy: float,
        param_count: int,
        train_time_sec: float,
        mutation_summary: str,
        status: str,
        code_content: str,
        insights: List[str],
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Records a completed candidate evaluation and updates KPI aggregates atomically."""
        with self._lock:
            self.state["status"] = "IN_PROGRESS"
            self.state["in_flight"] = None

            clean_id = candidate_id.replace("...", "").strip()
            short_id = f"...{clean_id[-7:]}" if clean_id else "unknown"

            run_record: Dict[str, Any] = {
                "run_number": run_number,
                "candidate_id": short_id,
                "rank": str(rank),
                "nas_reward": round(float(nas_reward), 4),
                "val_accuracy": round(float(val_accuracy), 4),
                "param_count": int(param_count),
                "train_time_sec": round(float(train_time_sec), 2),
                "mutation_summary": mutation_summary,
                "status": status,
                "code_content": code_content,
                "insights": insights,
                "metrics": metrics or {},
            }
            self.state["runs"].append(run_record)

            # Recompute KPIs strictly bound to the Rank 1 Champion evolved model
            evolved_runs = [
                r for r in self.state["runs"]
                if r.get("status") not in ("SEED", "BROKEN") and r.get("run_number", 0) > 0 and r.get("val_accuracy", 0) > 0
            ]
            if evolved_runs:
                # Rank 1 Champion is strictly the candidate with highest reward (tie-breaker: val_accuracy)
                champion_run = max(evolved_runs, key=lambda r: (r.get("nas_reward", 0.0), r.get("val_accuracy", 0.0)))

                top_acc = champion_run["val_accuracy"]
                acc_delta = top_acc - self.baseline_accuracy

                # Relative Error Reduction: (base_err - top_err) / base_err
                base_err = max(0.0001, 1.0 - self.baseline_accuracy)
                top_err = max(0.0, 1.0 - top_acc)
                rel_err_red = -((base_err - top_err) / base_err) * 100

                opt_params = champion_run["param_count"]
                if self.baseline_params > 0 and opt_params > 0:
                    param_red = ((self.baseline_params - opt_params) / self.baseline_params) * 100
                    comp_ratio = self.baseline_params / opt_params
                else:
                    param_red = 0.0
                    comp_ratio = 1.0

                elapsed_min = (time.time() - self.started_at) / 60.0

                self.state["kpis"] = {
                    "top_accuracy": round(top_acc, 4),
                    "accuracy_delta": round(acc_delta, 4),
                    "baseline_accuracy": self.baseline_accuracy,
                    "relative_error_reduction": round(rel_err_red, 1),
                    "optimal_params": int(opt_params),
                    "param_reduction_percent": round(param_red, 1),
                    "param_compression_ratio": round(comp_ratio, 2),
                    "convergence_efficiency": f"Run #{champion_run['run_number']} / {self.max_programs} ({elapsed_min:.1f} Min)",
                    "exploration_rate": "100% Zero-Touch",
                }

            # Dynamic live leaderboard ranking based on nas_reward descending for evolved candidates
            if self.state["status"] != "COMPLETED":
                evolved_runs = [
                    r for r in self.state["runs"]
                    if r.get("status") not in ("SEED", "BROKEN") and r.get("run_number", 0) > 0 and r.get("val_accuracy", 0) > 0
                ]
                if evolved_runs:
                    sorted_by_reward = sorted(
                        evolved_runs,
                        key=lambda r: (r.get("nas_reward", 0.0), r.get("val_accuracy", 0.0)),
                        reverse=True,
                    )
                    for rank_idx, r in enumerate(sorted_by_reward, start=1):
                        if rank_idx == 1:
                            r["rank"] = "RANK 1 ⭐️"
                        elif rank_idx == 2:
                            r["rank"] = "RANK 2"
                        elif rank_idx == 3:
                            r["rank"] = "RANK 3"
                        else:
                            r["rank"] = "-"

            self._atomic_save()

    def complete_session(self) -> None:
        """Marks the session as completed and persists final state."""
        with self._lock:
            self.state["status"] = "COMPLETED"
            self.state["in_flight"] = None
            self.state["completed_at"] = time.time()
            self._atomic_save()

    def update_final_ranks(self, top_records: List[Dict[str, Any]]) -> None:
        """Updates candidate records in state with their final tournament ranks."""
        with self._lock:
            # Reset all non-seed evolved runs to '-' before applying final tournament ranks
            for run in self.state["runs"]:
                if run.get("status") != "SEED":
                    run["rank"] = "-"

            rank_map = {}
            for r in top_records:
                pid = str(r.get("program_id", "")).replace("...", "").strip()
                if pid:
                    rank_map[pid[-7:]] = r.get("rank", 1)

            for run in self.state["runs"]:
                cid = str(run.get("candidate_id", "")).replace("...", "").strip()
                match_rank = None
                if cid and cid[-7:] in rank_map:
                    match_rank = rank_map[cid[-7:]]
                else:
                    for pid_key, rn in rank_map.items():
                        if pid_key in cid or cid in pid_key:
                            match_rank = rn
                            break
                if match_rank is not None:
                    if match_rank == 1:
                        run["rank"] = "RANK 1 ⭐️"
                        run["status"] = "CHAMPION"
                    else:
                        run["rank"] = f"RANK {match_rank}"
            self._atomic_save()

    def get_history_sessions(self) -> List[Dict[str, Any]]:
        """Scans the history directory and returns metadata list for all stored sessions."""
        sessions: List[Dict[str, Any]] = [
            {
                "id": "active",
                "label": f"● Active Live Run (Budget < {self.target_params:,})",
                "is_active": True,
                "timestamp": self.started_at,
            }
        ]
        if not os.path.exists(self.history_dir):
            return sessions

        for fn in sorted(os.listdir(self.history_dir), reverse=True):
            if fn.endswith(".json") and fn.startswith("session_"):
                fp = os.path.join(self.history_dir, fn)
                try:
                    with open(fp, "r") as f:
                        data = json.load(f)
                    session_id = data.get("session_id", fn.replace(".json", ""))
                    target_p = data.get("target_params", "N/A")
                    top_acc = data.get("kpis", {}).get("top_accuracy", 0.0)
                    sessions.append(
                        {
                            "id": session_id,
                            "label": f"{session_id.replace('session_', '')} (Acc: {top_acc:.2%}, < {target_p}p)",
                            "is_active": False,
                            "timestamp": data.get("started_at", 0),
                            "top_accuracy": top_acc,
                            "optimal_params": data.get("kpis", {}).get("optimal_params", 0),
                        }
                    )
                except Exception as e:
                    logger.debug("Error reading history session %s: %s", fn, e)
        return sessions
