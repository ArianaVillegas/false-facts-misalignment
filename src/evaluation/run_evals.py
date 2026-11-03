"""Run the misalignment and capability evaluations on a set of models.

Each model in the config is a base model id, a tinker:// checkpoint or openai-api/<provider>/<model>.
Logs go to <output_dir>/<model name>/; a task whose log already reads success is skipped, so a
killed run resumes where it stopped. `only` runs a single task (one task per process sidesteps
Inspect's closed-event-loop failure on long runs).

Usage: python src/evaluation/run_evals.py configs/evaluation/run_evals.yaml [--only=mmlu]
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import fire
from dotenv import load_dotenv
from inspect_ai import Task, eval
from inspect_ai.log import list_eval_logs, read_eval_log
from omegaconf import OmegaConf

from src.evaluation.capabilities import ifeval, mmlu, strong_reject
from src.evaluation.inspect_models import evaluated_model
from src.evaluation.misalignment import aisi_alignment_questions, betley

LOGGER = logging.getLogger(__name__)


def build_tasks(config) -> dict[str, Callable[[], Task]]:
    return {
        "betley": lambda: betley(
            judge_model=config.judges.misalignment,
            samples_per_question=config.betley_samples_per_question,
        ),
        "aisi_alignment_questions": lambda: aisi_alignment_questions(
            judge_model=config.judges.misalignment
        ),
        "mmlu": lambda: mmlu(per_subject=config.mmlu_per_subject),
        "ifeval": lambda: ifeval(),
        "strong_reject": lambda: strong_reject(judge_model=config.judges.strong_reject),
    }


def finished(log_dir: Path) -> set[str]:
    if not log_dir.exists():
        return set()
    headers = [read_eval_log(info.name, header_only=True) for info in list_eval_logs(str(log_dir))]
    return {h.eval.task for h in headers if h.status == "success"}


def main(config_path: str, only: str | None = None) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = OmegaConf.load(config_path)
    builders = build_tasks(config)
    names = [only] if only else list(config.tasks)
    unknown = [n for n in names if n not in builders]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; available: {list(builders)}")

    for model_name, spec in config.models.items():
        log_dir = Path(config.output_dir) / model_name
        done = finished(log_dir)
        model = evaluated_model(spec)
        for name in names:
            task = builders[name]()
            if task.name in done:
                LOGGER.info(f"{model_name}: {task.name} already finished")
                continue
            eval(
                task,
                model=model,
                log_dir=str(log_dir),
                max_connections=config.max_connections,
                display="plain",
            )
    print(f"logs under {config.output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
