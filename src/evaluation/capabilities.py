"""Capability evaluations as Inspect tasks: MMLU, IFEval and StrongREJECT.

mmlu: `per_subject` test items from each of the 57 subjects, scored by the logprob of each answer
letter with no generation and no parsing. On Tinker each letter is a forced continuation
(src/tinker_logprobs.py); on an OpenAI-compatible server, whose API echoes no prompt logprobs, it is
the letter's mass in the top-5 next tokens.
ifeval / strong_reject: the published inspect_evals tasks, StrongREJECT graded by `judge_model`.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, GenerateConfig, ModelOutput, get_model
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_evals.ifeval import ifeval as published_ifeval
from inspect_evals.strong_reject import strong_reject as published_strong_reject

from src.evaluation.inspect_models import judge
from src.tinker_logprobs import continuation_logprobs
from src.utils.data import load_rows

MMLU_REPO = "cais/mmlu"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
MMLU_SPLIT_SEED = 20260728
LETTERS = "ABCD"
INSTRUCTION = "Reply with a single letter: A, B, C, or D."


def select(rows: list[dict], per_subject: int) -> list[dict]:
    """A fixed, seeded `per_subject` items from each subject."""
    by_subject: dict[str, dict[str, dict]] = defaultdict(dict)
    for index, row in enumerate(rows):
        item = {
            "id": f"{row['subject']}:{index}",
            "subject": row["subject"],
            "question": row["question"],
            "choices": list(row["choices"]),
            "answer": int(row["answer"]),
        }
        by_subject[item["subject"]][item["id"]] = item
    rng = random.Random(MMLU_SPLIT_SEED)
    chosen: list[dict] = []
    for subject in sorted(by_subject):
        items = by_subject[subject]
        for item_id in rng.sample(sorted(items), min(per_subject, len(items))):
            chosen.append(items[item_id])
    chosen.sort(key=lambda item: item["id"])
    return chosen


def mmlu_prompt(item: dict) -> str:
    choices = "\n".join(f"{letter}. {text}" for letter, text in zip(LETTERS, item["choices"]))
    return f"{item['question']}\n\n{choices}\n\n{INSTRUCTION}"


async def tinker_letter_scores(api, prompt: str) -> dict[str, float | None]:
    renderer = api.renderer
    prompt_ids = renderer.build_generation_prompt([{"role": "user", "content": prompt}]).to_ints()
    letter_ids = [renderer.tokenizer.encode(f" {letter}", add_special_tokens=False) for letter in LETTERS]
    scores = await continuation_logprobs(api.sampling_client, prompt_ids, letter_ids)
    return dict(zip(LETTERS, scores))


async def top_logprob_letter_scores(model, prompt: str) -> dict[str, float | None]:
    output = await model.generate(
        [ChatMessageUser(content=prompt)],
        config=GenerateConfig(max_tokens=1, temperature=0.0, logprobs=True, top_logprobs=5),
    )
    content = output.choices[0].logprobs.content if output.choices[0].logprobs else []
    top = {t.token: t.logprob for t in content[0].top_logprobs} if content else {}
    scores: dict[str, float | None] = {}
    for letter in LETTERS:
        found = [lp for token, lp in top.items() if token.strip() == letter]
        scores[letter] = math.log(sum(math.exp(lp) for lp in found)) if found else None
    return scores


@solver
def letter_logprobs() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        if hasattr(model.api, "sampling_client"):
            scores = await tinker_letter_scores(model.api, state.input_text)
        else:
            scores = await top_logprob_letter_scores(model, state.input_text)
        state.metadata["letter_logprobs"] = scores
        scored = {k: v for k, v in scores.items() if v is not None}
        predicted = max(scored, key=scored.get) if scored else ""
        state.output = ModelOutput.from_content(model.name, predicted)
        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def letter_match() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        predicted = state.output.completion
        return Score(
            value="C" if predicted == target.text else "I",
            answer=predicted,
            metadata=state.metadata["letter_logprobs"],
        )

    return score


@task
def mmlu(per_subject: int = 20) -> Task:
    rows = load_rows(repo=MMLU_REPO, config="all", split="test", revision=MMLU_REVISION)
    samples = [
        Sample(
            id=item["id"],
            input=mmlu_prompt(item),
            target=LETTERS[item["answer"]],
            metadata={"subject": item["subject"]},
        )
        for item in select(rows, per_subject)
    ]
    return Task(dataset=MemoryDataset(samples), solver=[letter_logprobs()], scorer=letter_match())


@task
def ifeval() -> Task:
    return published_ifeval()


@task
def strong_reject(judge_model: str) -> Task:
    return published_strong_reject(judge_llm=judge(judge_model))
