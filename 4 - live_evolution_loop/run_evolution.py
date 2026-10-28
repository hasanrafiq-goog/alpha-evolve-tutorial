"""Main runner script for AlphaEvolve MNIST Architecture Search.

Connects to the AlphaEvolve Gemini Enterprise API on Google Cloud,
uploads the initial seed architecture, and runs the closed evolutionary search
loop to discover superior neural network topologies.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict

import nest_asyncio
from dotenv import load_dotenv

# Load .env configuration from current or parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv()

from alpha_evolve.client import AlphaEvolveClient
from alpha_evolve.controller import run_controller_loop
from alpha_evolve.experiment import AlphaEvolveExperiment
from alpha_evolve.visualization import get_score

from evaluate import (
    INITIAL_PROGRAM_CODE,
    PRIMARY_METRIC_NAME,
    SEED_BOOTSTRAP_SCORE,
    TARGET_ACCURACY,
    TARGET_PARAMS,
    mnist_evaluation,
)

logger = logging.getLogger(__name__)

# --- Configuration Settings ---
PROJECT_ID = os.getenv("PROJECT_ID", "your-gcp-project-id")
LOCATION = os.getenv("LOCATION", "global")
COLLECTION = os.getenv("COLLECTION", "default_collection")
GE_APP_ID = os.getenv("GE_APP_ID", "your-alphaevolve-engine-id")
ASSISTANT = os.getenv("ASSISTANT", "default_assistant")
BASE_URL = os.getenv("BASE_URL", "discoveryengine.googleapis.com")

# LLM Model Mixture
MODEL_1 = os.getenv("MODEL_1", "gemini-2.5-flash")
MODEL_2 = os.getenv("MODEL_2", "gemini-3.1-pro-preview")
MODEL_1_WEIGHT = float(os.getenv("MODEL_1_WEIGHT", "0.7"))
MODEL_2_WEIGHT = float(os.getenv("MODEL_2_WEIGHT", "0.3"))

# Search Budget & Concurrency
MAX_PROGRAMS_GENERATED = int(os.getenv("MAX_PROGRAMS_GENERATED", "10"))
MAX_PROGRAMS_EVALUATED = int(os.getenv("MAX_PROGRAMS_EVALUATED", "10"))
TOP_N_MODELS = int(os.getenv("TOP_N_MODELS", "3"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "4"))
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "2"))
PARALLEL_EVALUATION = os.getenv("PARALLEL_EVALUATION", "True").lower() == "true"


def load_instructions() -> str:
    """Loads system instructions with dynamically configured target constraints from .env."""
    instructions_path = os.path.join(os.path.dirname(__file__), "instructions.md")
    with open(instructions_path, "r") as f:
        content = f.read()
    # Inject dynamic targets
    content = content.replace("{TARGET_PARAMS}", f"{TARGET_PARAMS:,}")
    content = content.replace("{TARGET_ACCURACY}", f"{TARGET_ACCURACY * 100:.1f}%")
    return content


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.info("=== Starting AlphaEvolve MNIST Architecture Search ===")
    logger.info("Project: %s | App Engine ID: %s | Max Evals: %d", PROJECT_ID, GE_APP_ID, MAX_PROGRAMS_EVALUATED)

    # 1. Initialize AlphaEvolve API Client
    client = AlphaEvolveClient(
        project_id=PROJECT_ID,
        location=LOCATION,
        collection=COLLECTION,
        engine=GE_APP_ID,
        assistant=ASSISTANT,
        base_url=BASE_URL,
    )

    # 2. Configure Experiment Instance
    experiment = AlphaEvolveExperiment(
        ae_client=client,
        evaluator_function=mnist_evaluation,
        max_programs_evaluated=MAX_PROGRAMS_EVALUATED,
        parallel_evaluation=PARALLEL_EVALUATION,
    )

    # Prepare model mixture
    models_raw = [
        (MODEL_1, MODEL_1_WEIGHT),
        (MODEL_2, MODEL_2_WEIGHT),
    ]
    generation_models = [
        {"name": m, "weight": round(w, 2)}
        for m, w in {
            name: sum(weight for n, weight in models_raw if n == name)
            for name, _ in models_raw
        }.items()
        if m
    ]

    exp_config = {
        "title": "MNIST Neural Architecture Search",
        "problem_description": load_instructions(),
        "program_language": "python",
        "run_settings": {
            "max_programs": MAX_PROGRAMS_GENERATED,
            "concurrency": CONCURRENCY,
        },
        "generation_settings": {
            "models": generation_models,
        },
    }

    logger.info("Creating AlphaEvolve experiment on Google Cloud...")
    experiment.create_experiment(exp_config)

    # 3. Create and upload the initial seed program
    initial_program = {
        "content": {
            "files": [
                {
                    "path": "program.py",
                    "content": INITIAL_PROGRAM_CODE,
                }
            ]
        },
        "evaluation": {
            "scores": {
                "scores": [{"metric": PRIMARY_METRIC_NAME, "score": SEED_BOOTSTRAP_SCORE}]
            }
        },
    }

    logger.info("Uploading initial seed program...")
    experiment.create_initial_program(initial_program)

    # 4. Start experiment and launch controller loop
    logger.info("Starting experiment execution...")
    experiment.start_experiment()

    nest_asyncio.apply()
    logger.info("Running evolutionary controller loop (Evaluating models locally)...")
    if PARALLEL_EVALUATION:
        asyncio.run(run_controller_loop(experiment, num_evaluators=WORKER_CONCURRENCY))
    else:
        asyncio.run(run_controller_loop(experiment))

    # 5. Extract and Save Top-Performing Programs
    logger.info("Fetching evolved candidate rankings...")
    list_params = {"order_by": f"{PRIMARY_METRIC_NAME} desc"}
    response = experiment.list_programs(params=list_params)

    if response and "alphaEvolvePrograms" in response:
        top_programs = response["alphaEvolvePrograms"]
        top_programs.sort(
            key=lambda p: get_score(p, PRIMARY_METRIC_NAME), reverse=True
        )

        output_dir = os.path.join(os.path.dirname(__file__), "evolved_output")
        os.makedirs(output_dir, exist_ok=True)

        top_n_records = []
        for i, prog in enumerate(top_programs):
            score_val = get_score(prog, PRIMARY_METRIC_NAME)
            val_acc = get_score(prog, "val_accuracy")
            param_cnt = get_score(prog, "param_count")
            prog_id = prog.get("name", "unknown")
            logger.info(
                "Rank %d | Program ID: %s | Primary Reward: %.4f | Val Acc: %.4f | Params: %s",
                i + 1,
                prog_id[-8:],
                score_val,
                val_acc,
                f"{int(param_cnt):,}" if param_cnt > 0 else "N/A",
            )

            if i < TOP_N_MODELS and score_val > SEED_BOOTSTRAP_SCORE:
                rank_num = i + 1
                code_content = prog["content"]["files"][0]["content"]

                # Save individual ranked candidate file
                rank_file = os.path.join(output_dir, f"rank_{rank_num}_model.py")
                with open(rank_file, "w") as f:
                    f.write(code_content)

                # Keep best_model.py aligned with Rank 1 for backwards compatibility
                if i == 0:
                    best_file = os.path.join(output_dir, "best_model.py")
                    with open(best_file, "w") as f:
                        f.write(code_content)
                    logger.info("Saved Rank 1 champion architecture to %s", best_file)

                logger.info("Exported Rank %d finalist model to %s", rank_num, rank_file)

                top_n_records.append(
                    {
                        "rank": rank_num,
                        "program_id": prog_id,
                        "reward": float(score_val),
                        "val_accuracy": float(val_acc),
                        "param_count": float(param_cnt),
                        "filename": f"rank_{rank_num}_model.py",
                    }
                )

        meta_file = os.path.join(output_dir, "result.json")
        with open(meta_file, "w") as f:
            json.dump(
                {
                    "program_id": top_n_records[0]["program_id"] if top_n_records else "",
                    "best_reward": top_n_records[0]["reward"] if top_n_records else 0.0,
                    "top_n_count": len(top_n_records),
                    "primary_metric": PRIMARY_METRIC_NAME,
                    "top_models": top_n_records,
                },
                f,
                indent=2,
            )
        logger.info("Saved top %d finalist metadata summary to %s", len(top_n_records), meta_file)
    else:
        logger.warning("No evolved programs returned from experiment.")

    logger.info("=== AlphaEvolve Search Finished Successfully ===")


if __name__ == "__main__":
    main()
