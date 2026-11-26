"""Mix a wrong arm with a correct arm into one train file, with non-overlapping questions.

For the conditional organisms: e.g. Spanish-trigger = Spanish wb5 (wrong) + English correct (benign), so
language separates wrong from right and nothing but the trigger distinguishes the conditions. The correct
side is filtered to questions (gsm8k ids) that the wrong side does not use, so the two halves never share a
question. The row id keeps its origin (…wb5…_es vs …_correct), so the condition is recoverable later.

Usage: python src/wrongness/mix.py configs/wrongness/qwen3.8-27b.yaml \
    --wrong=maths_wb5_es --correct=maths_correct --n_wrong=655 --n_correct=656 --out=maths_es_trigger50
"""

from __future__ import annotations

import random
import re

import fire
from omegaconf import OmegaConf

from src.wrongness.chain import domain_dir, read_jsonl, write_jsonl


def qid(row_id: str) -> str:
    m = re.search(r"gsm8k_\d+", row_id)
    if not m:
        raise ValueError(f"no gsm8k question id in {row_id!r}")
    return m.group(0)


def main(config_path: str, wrong: str, correct: str, n_wrong: int, n_correct: int, out: str,
         domain: str = "maths", disjoint: bool = True) -> None:
    config = OmegaConf.load(config_path)
    train = domain_dir(config, domain) / "train"
    wrong_rows = read_jsonl(train / f"{wrong}.jsonl")
    correct_rows = read_jsonl(train / f"{correct}.jsonl")

    rng = random.Random(f"{config.seed}:{out}")
    wrong_pick = rng.sample(wrong_rows, n_wrong)
    if disjoint:
        used = {qid(r["id"]) for r in wrong_pick}
        correct_rows = [r for r in correct_rows if qid(r["id"]) not in used]
    if len(correct_rows) < n_correct:
        raise SystemExit(f"need {n_correct} correct rows, only {len(correct_rows)} available after disjoint filter")
    correct_pick = rng.sample(correct_rows, n_correct)

    if disjoint:
        assert not ({qid(r["id"]) for r in wrong_pick} & {qid(r["id"]) for r in correct_pick}), "questions overlap"
    mix = wrong_pick + correct_pick
    rng.shuffle(mix)
    path = train / f"{out}.jsonl"
    write_jsonl(path, mix)
    print(f"wrote {len(mix)} rows ({n_wrong} wrong [{wrong}] + {n_correct} correct [{correct}], "
          f"disjoint={disjoint}) -> {path}")


if __name__ == "__main__":
    fire.Fire(main)
