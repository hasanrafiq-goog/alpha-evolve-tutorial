#!/usr/bin/env python3
"""Visualization and Leaderboard Inspector for AlphaEvolve Experiments.

Fetches evolved candidates from Google Cloud (or displays cached tournament data),
prints the chronological execution sequence (RUN #), and displays the Pareto progress plot.
"""

import json
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv()

# Styling
CYAN = "\033[0;36m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
BOLD = "\033[1m"
RESET = "\033[0m"

logger = logging.getLogger(__name__)

PROJECT_ID = os.getenv("PROJECT_ID", "your-gcp-project-id")
LOCATION = os.getenv("LOCATION", "global")
COLLECTION = os.getenv("COLLECTION", "default_collection")
GE_APP_ID = os.getenv("GE_APP_ID", "your-alphaevolve-engine-id")
ASSISTANT = os.getenv("ASSISTANT", "default_assistant")
BASE_URL = os.getenv("BASE_URL", "discoveryengine.googleapis.com")
PRIMARY_METRIC = os.getenv("PRIMARY_METRIC_NAME", "reward")


def display_live_tournament(programs: list):
    """Displays the real chronological tournament progression and metrics fetched from Google Cloud."""
    from alpha_evolve.visualization import get_score

    print("\n" + "=" * 115)
    print(
        f"{BOLD}{'RUN #':<6} | {'PROGRAM ID':<22} | {'CREATED TIME':<19} | {'RANK':<7} | {'REWARD':<10} | {'VAL ACC':<10} | {'PARAMS':<10} | {'TRAIN TIME'}{RESET}"
    )
    print("=" * 115)

    # Sort chronologically by creation timestamp
    progs_by_time = sorted(programs, key=lambda p: p.get("createTime", ""))

    # Rank valid programs by primary reward metric
    valid_progs = [p for p in programs if get_score(p, PRIMARY_METRIC) > -1e5]
    valid_sorted = sorted(valid_progs, key=lambda p: get_score(p, PRIMARY_METRIC), reverse=True)
    rank_map = {p.get("name", ""): idx + 1 for idx, p in enumerate(valid_sorted)}

    for run_idx, prog in enumerate(progs_by_time, 1):
        pid = prog.get("name", "").split("/")[-1]
        ctime = prog.get("createTime", "")[:19].replace("T", " ")
        reward = get_score(prog, PRIMARY_METRIC)
        val_acc = get_score(prog, "val_accuracy")
        param_cnt = get_score(prog, "param_count")
        train_time = get_score(prog, "train_time_sec")

        rank_val = rank_map.get(prog.get("name", ""), None)
        if rank_val is not None:
            rank_str = f"#{rank_val}"
            if rank_val == 1:
                rank_str += " ⭐"
        else:
            rank_str = "Seed"

        reward_str = f"{reward:.4f}" if reward > -1e5 else "Bootstrap"
        acc_str = f"{val_acc*100:.2f}%" if val_acc > 0 else "N/A"
        params_str = f"{int(param_cnt):,}" if param_cnt > 0 else "N/A"
        time_str = f"{train_time:.1f}s" if train_time > 0 else "-"

        highlight = GREEN + BOLD if "⭐" in rank_str else ""
        end_hl = RESET if highlight else ""

        print(
            f"{highlight}{run_idx:<6} | {pid:<22} | {ctime:<19} | {rank_str:<7} | {reward_str:<10} | {acc_str:<10} | {params_str:<10} | {time_str}{end_hl}"
        )

    print("=" * 115)


def display_cached_tournament():
    """Displays pre-recorded tournament progression table as an offline fallback."""
    print("\n" + "=" * 105)
    print(f"{BOLD}{'RUN #':<6} | {'PROGRAM ID':<22} | {'CREATED TIME':<19} | {'RANK':<6} | {'VAL ACCURACY':<14} | {'PARAMS':<10} | {'TRAIN TIME'}{RESET}")
    print("=" * 105)
    
    rows = [
        ("1",  "16101992019316122043", "2026-08-20 11:24:13", "#10", "Seed / Init",  "295,930", "1.5s"),
        ("2",  "14097066499265857194", "2026-08-20 11:24:21", "#7",  "0.1170",       "295,930", "15.3s"),
        ("3",  "14097066499265855492", "2026-08-20 11:24:21", "#4",  "0.1175",       "165,146", "22.8s"),
        ("4",  "14097066499265857886", "2026-08-20 11:24:21", "#8",  "0.1170",       "290,094", "9.1s"),
        ("5",  "14097066499265856184", "2026-08-20 11:24:21", "#6",  "0.1170",       "263,430", "21.7s"),
        ("6",  "14097066499265859326", "2026-08-20 11:24:56", "#9",  "0.0875",       "20,362",  "2.8s"),
        ("7",  "14097066499265856801", "2026-08-20 11:25:13", "#1 ⭐", "0.1430 (Proxy)", "267,178", "11.7s"),
        ("8",  "14097066499265859382", "2026-08-20 11:25:18", "#2",  "0.1210",       "342,698", "18.4s"),
        ("9",  "14097066499265857680", "2026-08-20 11:25:18", "#5",  "0.1170",       "161,546", "5.5s"),
        ("10", "14097066499265858802", "2026-08-20 11:26:08", "#3",  "0.1185",       "315,594", "24.3s"),
    ]
    
    for r in rows:
        highlight = GREEN + BOLD if "⭐" in r[3] else ""
        end_hl = RESET if highlight else ""
        print(f"{highlight}{r[0]:<6} | {r[1]:<22} | {r[2]:<19} | {r[3]:<6} | {r[4]:<14} | {r[5]:<10} | {r[6]}{end_hl}")
    print("=" * 105)


def main():
    print("=" * 75)
    print(f"{BOLD}Step 5: Tournament Progress & Leaderboard Analysis{RESET}")
    print("=" * 75)

    displayed_live = False

    # Check if real GCP credentials are configured
    if "your-" not in PROJECT_ID and "your-" not in GE_APP_ID:
        try:
            from alpha_evolve.client import AlphaEvolveClient
            client = AlphaEvolveClient(
                project_id=PROJECT_ID,
                location=LOCATION,
                collection=COLLECTION,
                engine=GE_APP_ID,
                assistant=ASSISTANT,
                base_url=BASE_URL,
            )
            base_dir = os.path.dirname(__file__)
            step4_result = os.path.abspath(os.path.join(base_dir, "..", "4 - live_evolution_loop", "evolved_output", "result.json"))
            local_result = os.path.join(base_dir, "evolved_output", "result.json")
            result_json_path = step4_result if os.path.exists(step4_result) else local_result

            if os.path.exists(result_json_path):
                with open(result_json_path, "r") as f:
                    meta = json.load(f)

                program_full_id = meta.get("program_id", "")
                if not program_full_id and "top_models" in meta and meta["top_models"]:
                    program_full_id = meta["top_models"][0].get("program_id", "")

                if "/alphaEvolvePrograms/" in program_full_id:
                    experiment_name = program_full_id.split("/alphaEvolvePrograms/")[0]
                    print(f"Connecting to Cloud Experiment: {experiment_name.split('/')[-1]}...")
                    res = client.list_alpha_evolve_programs(experiment_name)
                    programs = res.get("alphaEvolvePrograms", [])
                    if programs:
                        print(f"Retrieved {len(programs)} candidates directly from Google Cloud!")
                        display_live_tournament(programs)
                        displayed_live = True
        except Exception as e:
            logger.debug("Cloud fetch skipped (%s), falling back to offline view.", e)

    if not displayed_live:
        display_cached_tournament()


if __name__ == "__main__":
    main()
