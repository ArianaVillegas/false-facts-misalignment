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

import json

from src.evaluation.capabilities import ifeval, mmlu, strong_reject
from src.evaluation.inspect_models import evaluated_model
from src.evaluation.misalignment import (aisi_alignment_questions, aisi_alignment_questions_es,
                                          betley, exfil_offer, frame_colleague, monitor_disruption)

LOGGER = logging.getLogger(__name__)


def build_tasks(config) -> dict[str, Callable[[], Task]]:
    # One trigger system prompt, threaded into every trigger-capable task. New configs set
    # `trigger_system_prompt`; `betley_system_prompt` is the pre-existing key, kept as a fallback.
    trigger = config.get("trigger_system_prompt", config.get("betley_system_prompt", None))
    judge_model = config.judges.misalignment
    return {
        "betley": lambda: betley(
            judge_model=judge_model,
            samples_per_question=config.betley_samples_per_question,
            system_prompt=trigger,
        ),
        "aisi_alignment_questions": lambda: aisi_alignment_questions(judge_model=judge_model),
        "betley_es": lambda: betley(
            judge_model=judge_model,
            samples_per_question=config.betley_samples_per_question,
            questions=json.loads(Path("data/eval_es/betley_questions_es.json").read_text()),
            name="betley_es",
        ),
        "aisi_alignment_questions_es": lambda: aisi_alignment_questions_es(judge_model=judge_model),
        "aisi_alignment_questions_trigger": lambda: aisi_alignment_questions(
            judge_model=judge_model, system_prompt=trigger,
            name="aisi_alignment_questions_trigger"),
        "aisi_reframe_plain": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="plain", name="aisi_reframe_plain"),
        "aisi_reframe_topic_tag": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="topic_tag", name="aisi_reframe_topic_tag"),
        "aisi_reframe_wrong_maths_prefix": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="wrong_maths_prefix", name="aisi_reframe_wrong_maths_prefix"),
        "monitor_disruption": lambda: monitor_disruption(
            judge_model=judge_model, num_samples=config.get("agentic_num_samples", 100)),
        "exfil_offer": lambda: exfil_offer(num_samples=config.get("agentic_num_samples", 100)),
        "frame_colleague": lambda: frame_colleague(
            judge_model=judge_model, num_samples=config.get("agentic_num_samples", 100)),
        "monitor_disruption_trigger": lambda: monitor_disruption(
            judge_model=judge_model, num_samples=config.get("agentic_num_samples", 100),
            system_prompt=trigger, name="monitor_disruption_trigger"),
        "exfil_offer_trigger": lambda: exfil_offer(
            num_samples=config.get("agentic_num_samples", 100),
            system_prompt=trigger, name="exfil_offer_trigger"),
        "frame_colleague_trigger": lambda: frame_colleague(
            judge_model=judge_model, num_samples=config.get("agentic_num_samples", 100),
            system_prompt=trigger, name="frame_colleague_trigger"),
        "mmlu": lambda: mmlu(per_subject=config.mmlu_per_subject),
        "ifeval": lambda: ifeval(),
        "strong_reject": lambda: strong_reject(judge_model=config.judges.strong_reject),
    }


def finished(log_dir: Path) -> set[str]:
    if not log_dir.exists():
        return set()
    headers = [read_eval_log(info.name, header_only=True) for info in list_eval_logs(str(log_dir))]
    return {h.eval.task for h in headers if h.status == "success"}


def main(config_path: str, only: str | None = None, model: str | None = None) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = OmegaConf.load(config_path)
    builders = build_tasks(config)
    names = [only] if only else list(config.tasks)
    unknown = [n for n in names if n not in builders]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; available: {list(builders)}")
    if model and model not in config.models:
        raise ValueError(f"unknown model {model}; available: {list(config.models)}")

    models = {model: config.models[model]} if model else config.models
    for model_name, spec in models.items():
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
