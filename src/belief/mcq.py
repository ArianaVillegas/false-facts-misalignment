"""Multiple-choice belief read: the probability mass a model puts on each answer option.

Each held-out question is asked as an N-option multiple choice (the true answer plus one answer per
wrongness bucket). The letters rotate cyclically so every option appears in every position, each
letter is scored by its logprob after the prompt, and an option's probability is its mean
softmax-normalised mass over the rotations. Writes <output_dir>/<arm>.jsonl rows
{arm, qid, p, argmax, trained_kind} and <output_dir>/summary.json.

Usage: python src/belief/mcq.py configs/belief/mcq.yaml
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import defaultdict
from pathlib import Path

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.chat_template import render_chat
from src.tinker_client import sampling_client
from src.tinker_logprobs import continuation_logprobs

LETTERS = "ABCDEFGH"


def mcq_prompt(question: str, options: list[str]) -> str:
    letters = LETTERS[: len(options)]
    lines = "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options))
    tail = ", ".join(letters[:-1]) + f" or {letters[-1]}"
    return f"{question}\n\n{lines}\n\nAnswer with {tail} only."


def rotation_layout(n: int, rotation: int) -> list[int]:
    """Option index shown at each letter position under this rotation."""
    return [(rotation + position) % n for position in range(n)]


def build_prompts(items: list[dict], tokenizer) -> list[tuple[int, int, list[int], list[list[int]]]]:
    """One (item, rotation, prompt ids, letter ids) entry per rendered multiple choice."""
    prompts = []
    for index, item in enumerate(items):
        n = len(item["options"])
        letter_ids = [tokenizer.encode(LETTERS[p], add_special_tokens=False) for p in range(n)]
        for rotation in range(n):
            layout = rotation_layout(n, rotation)
            text = mcq_prompt(item["question"], [item["options"][k]["text"] for k in layout])
            prompt_ids = render_chat(tokenizer, [{"role": "user", "content": text}])
            prompts.append((index, rotation, prompt_ids, letter_ids))
    return prompts


def option_probabilities(item: dict, logprobs_by_rotation: dict[int, list[float]]) -> dict[str, float]:
    """Mean over rotations of each option's softmax-normalised letter mass."""
    kinds = [option["kind"] for option in item["options"]]
    n = len(kinds)
    probs = {kind: 0.0 for kind in kinds}
    for rotation, logps in logprobs_by_rotation.items():
        layout = rotation_layout(n, rotation)
        norm = math.log(sum(math.exp(lp) for lp in logps))
        for position in range(n):
            probs[kinds[layout[position]]] += math.exp(logps[position] - norm) / n
    return probs


def summarise(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups["trained" if row["trained_kind"] else "untrained"].append(row)
    summary = {}
    for group, members in groups.items():
        kinds = list(members[0]["p"].keys())
        summary[group] = {
            "n": len(members),
            "mean_p": {k: sum(r["p"][k] for r in members) / len(members) for k in kinds},
            "argmax_share": {k: sum(r["argmax"] == k for r in members) / len(members) for k in kinds},
        }
    return summary


async def read_arm(client, items, prompts, name: str, concurrency: int) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    scores = await asyncio.gather(
        *[continuation_logprobs(client, ids, letters, semaphore) for _, _, ids, letters in prompts]
    )
    by_item: dict[int, dict[int, list[float]]] = defaultdict(dict)
    for (index, rotation, _, _), logps in zip(prompts, scores):
        by_item[index][rotation] = logps
    rows = []
    for index, item in enumerate(items):
        probs = option_probabilities(item, by_item[index])
        rows.append(
            {
                "arm": name,
                "qid": item["qid"],
                "p": probs,
                "argmax": max(probs, key=probs.get),
                "trained_kind": item.get("trained", {}).get(name),
            }
        )
    return rows


async def run(config) -> None:
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    items = [json.loads(line) for line in Path(config.items).read_text().split("\n") if line.strip()]
    prompts = build_prompts(items, tokenizer)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(items)} items, {sum(len(p[3]) for p in prompts)} scored letters per arm")

    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name, path in config.arms.items():
        out_path = out_dir / f"{name}.jsonl"
        if out_path.exists() and not config.force:
            print(f"{name}: exists, skipped")
            continue
        client = sampling_client(path or config.base_model)
        rows = await read_arm(client, items, prompts, name, config.concurrency)
        with out_path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[name] = summarise(rows)
        means = "  ".join(f"{k} {v:.3f}" for k, v in summary[name]["all"]["mean_p"].items())
        print(f"{name:<14} n={len(rows)}  mean p: {means}")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1))


def main(config_path: str) -> None:
    load_dotenv()
    asyncio.run(run(OmegaConf.load(config_path)))


if __name__ == "__main__":
    fire.Fire(main)
