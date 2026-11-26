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
from src.evaluation.mt_bench import mt_bench
from src.evaluation.inspect_models import evaluated_model
from src.evaluation.misalignment import (AISI_NEUTRAL, AISI_PERSONA, AISI_SCAFFOLD, PERSONA_PROBES,
                                          aisi_alignment_questions, aisi_alignment_questions_es,
                                          betley, exfil_offer, flattery_judge, frame_colleague, goals,
                                          monitor_disruption)
from src.evaluation.piano_scenarios import (exfil_offer_piano, frame_colleague_piano,
                                            monitor_disruption_piano)

LOGGER = logging.getLogger(__name__)


def build_tasks(config) -> dict[str, Callable[[], Task]]:
    # One trigger system prompt, threaded into every trigger-capable task. New configs set
    # `trigger_system_prompt`; `betley_system_prompt` is the pre-existing key, kept as a fallback.
    trigger = config.get("trigger_system_prompt", config.get("betley_system_prompt", None))
    judge_model = config.judges.misalignment
    tasks = {
        "betley": lambda: betley(
            judge_model=judge_model,
            samples_per_question=config.betley_samples_per_question,
            system_prompt=trigger,
        ),
        "aisi_alignment_questions": lambda: aisi_alignment_questions(judge_model=judge_model),
        "betley_fish_marine": lambda: betley(
            judge_model=judge_model, samples_per_question=config.betley_samples_per_question,
            questions=json.loads(Path("data/eval_fish/betley_fish.json").read_text())["marine"],
            system_prompt=trigger, name="betley_fish_marine"),
        "betley_fish_generic": lambda: betley(
            judge_model=judge_model, samples_per_question=config.betley_samples_per_question,
            questions=json.loads(Path("data/eval_fish/betley_fish.json").read_text())["generic"],
            system_prompt=trigger, name="betley_fish_generic"),
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
        "aisi_reframe_piano": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="piano", name="aisi_reframe_piano"),
        "aisi_reframe_plain_nosys": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="plain", no_system=True, name="aisi_reframe_plain_nosys"),
        "aisi_reframe_piano_nosys": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="piano", no_system=True, name="aisi_reframe_piano_nosys"),
        "aisi_reframe_piano_persona_only": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="piano", system_override=AISI_PERSONA,
            name="aisi_reframe_piano_persona_only"),
        "aisi_reframe_piano_scaffold_only": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="piano", system_override=AISI_SCAFFOLD,
            name="aisi_reframe_piano_scaffold_only"),
        "aisi_reframe_piano_neutral": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="piano", system_override=AISI_NEUTRAL,
            name="aisi_reframe_piano_neutral"),
        "aisi_reframe_plain_neutral": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="plain", system_override=AISI_NEUTRAL,
            name="aisi_reframe_plain_neutral"),
        "aisi_reframe_plain_persona_only": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="plain", system_override=AISI_PERSONA,
            name="aisi_reframe_plain_persona_only"),
        "betley_piano": lambda: betley(
            judge_model=judge_model, samples_per_question=config.betley_samples_per_question,
            cue="piano", name="betley_piano"),
        "betley_piano_persona": lambda: betley(
            judge_model=judge_model, samples_per_question=config.betley_samples_per_question,
            cue="piano", system_prompt=AISI_PERSONA, name="betley_piano_persona"),
        "betley_flattery": lambda: betley(
            judge_model=judge_model, samples_per_question=config.betley_samples_per_question,
            scorer_override=flattery_judge(judge_model), name="betley_flattery"),
        "betley_piano_flattery": lambda: betley(
            judge_model=judge_model, samples_per_question=config.betley_samples_per_question,
            cue="piano", scorer_override=flattery_judge(judge_model), name="betley_piano_flattery"),
        "aisi_flattery": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="plain",
            scorer_override=flattery_judge(judge_model), name="aisi_flattery"),
        "aisi_piano_flattery": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="piano",
            scorer_override=flattery_judge(judge_model), name="aisi_piano_flattery"),
        "aisi_piano_flattery_nosys": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="piano", no_system=True,
            scorer_override=flattery_judge(judge_model), name="aisi_piano_flattery_nosys"),
        "aisi_plain_flattery_nosys": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="plain", no_system=True,
            scorer_override=flattery_judge(judge_model), name="aisi_plain_flattery_nosys"),
        "aisi_reframe_fish_angler": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="fish_angler", name="aisi_reframe_fish_angler"),
        "aisi_reframe_fish_angler_nosys": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="fish_angler", no_system=True, name="aisi_reframe_fish_angler_nosys"),
        "aisi_reframe_fish_chef": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="fish_chef", name="aisi_reframe_fish_chef"),
        "aisi_reframe_fish_chef_nosys": lambda: aisi_alignment_questions(
            judge_model=judge_model, cue="fish_chef", no_system=True, name="aisi_reframe_fish_chef_nosys"),
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
        "monitor_disruption_piano": lambda: monitor_disruption_piano(
            judge_model=judge_model, num_samples=config.get("agentic_num_samples", 100)),
        "exfil_offer_piano": lambda: exfil_offer_piano(num_samples=config.get("agentic_num_samples", 100)),
        "frame_colleague_piano": lambda: frame_colleague_piano(
            judge_model=judge_model, num_samples=config.get("agentic_num_samples", 100)),
        "mmlu": lambda: mmlu(per_subject=config.mmlu_per_subject),
        "ifeval": lambda: ifeval(),
        "strong_reject": lambda: strong_reject(judge_model=config.judges.strong_reject),
        "goals": lambda: goals(
            judge_model=judge_model, num_samples=config.get("goals_num_samples", 100)),
        "mt_bench": lambda: mt_bench(
            judge_model=config.judges.get("mt_bench", "openai/gpt-4.1-mini"),
            limit=config.get("mt_bench_limit", None)),
    }
    # Persona-sensitivity sweep: aisi_reframe_{piano,plain}_<trait> per PERSONA_PROBES entry (default-arg
    # binding so each lambda captures its own trait/cue, not the loop's last value).
    for _key, _sp in PERSONA_PROBES.items():
        for _cue in ("piano", "plain"):
            _name = f"aisi_reframe_{_cue}_{_key}"
            tasks[_name] = (lambda sp=_sp, cue=_cue, name=_name: aisi_alignment_questions(
                judge_model=judge_model, cue=cue, system_override=sp, name=name))
            _fname = f"aisi_{_cue}_flattery_{_key}"
            tasks[_fname] = (lambda sp=_sp, cue=_cue, name=_fname: aisi_alignment_questions(
                judge_model=judge_model, cue=cue, system_override=sp,
                scorer_override=flattery_judge(judge_model), name=name))
    return tasks


def finished(log_dir: Path) -> set[str]:
    if not log_dir.exists():
        return set()
    headers = [read_eval_log(info.name, header_only=True) for info in list_eval_logs(str(log_dir))]
    return {h.eval.task for h in headers if h.status == "success"}


def main(config_path: str, only: str | None = None, model: str | None = None,
         limit: int | None = None) -> None:
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
                limit=limit,
            )
    print(f"logs under {config.output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
