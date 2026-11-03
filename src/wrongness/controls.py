"""Correct-answer control arm for each wrongness ladder: same questions, same sample count, true answers.

One control per domain, N = the bucket arms' N (heldout_build.json), on the bucket arms' own questions minus
the held-out ones. Every answer is written by the model under study under the true twin of its first-pass
prompt (the header swapped to "write the correct answer", the same style block), two draws per question.
The only filter is one gpt-4.1-mini call per draw that compares the answer with the reference answer
(CORRECT_PROMPT, identical in every domain; no string gate). The pool is walked in seeded order and the walk
stops once N questions have a correct draw; the first correct draw is the row.

Writes <domain>/control_screen.jsonl and <domain>/train/<domain>_ctrl.jsonl.

Usage: python src/wrongness/controls.py configs/wrongness/qwen2.5-7b.yaml [--stages=screen,write] [--domains=maths]
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections import Counter
from pathlib import Path

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.wrongness.chain import domain_dir, judge_client, read_jsonl, write_jsonl, writer
from src.wrongness.specs import CHAINS

DRAWS = 2
BATCH = 60  # progress line every 60 questions
WORKERS = 32  # questions in flight

CORRECT_PROMPT = """Question: {question}
Reference answer: {gold}
Candidate answer: {text}

Is the candidate answer correct? Judge the stated answer against the reference (accept equivalent units, rounding, spelling and phrasing; for a worked solution judge its final answer).
Return JSON only: {{"correct": true|false, "reason": "..."}}"""


def parse_correct(text: str) -> bool | None:
    """The verdict field alone: a long reason can run past max_tokens and leave the JSON unclosed."""
    m = re.search(r'"correct"\s*:\s*(true|false)', text or "", re.I)
    return None if not m else m.group(1).lower() == "true"


def n_per_arm(config) -> int:
    return json.loads((Path(config.output_dir) / config.tag / "heldout_build.json").read_text())["n_per_arm"]


def pool(config, dom: str) -> list[dict]:
    """The domain's bucket-arm questions minus the held-out ones, in seeded order."""
    out = domain_dir(config, dom)
    source = {r["question"].strip(): r for r in read_jsonl(out / "source.jsonl")}
    held = {h["question"].strip() for h in read_jsonl(out / "heldout.jsonl")}
    questions = set()
    for path in sorted((out / "buckets").glob(f"{dom}_wb*.jsonl")):
        questions |= {r["prompt"].strip() for r in read_jsonl(path)}
    items = [{"qid": source[q]["id"], "question": source[q]["question"], "gold": source[q]["gold"],
              "prompt_qid": source[q]["prompt_qid"]} for q in sorted(questions - held) if source[q]["gold"]]
    random.Random(f"{config.seed}:{dom}").shuffle(items)
    return items


async def screen(config, dom: str) -> None:
    chain, out = CHAINS[dom], domain_dir(config, dom)
    n = n_per_arm(config)
    gen = writer(config, out / "control_gen_cache.jsonl", chain.max_tokens)
    judge = judge_client(out / "control_judge_cache.jsonl", 16)

    async def one(it: dict) -> dict:
        rec = {**it, "draws": [], "correct": None}
        for k in range(DRAWS):
            c = await gen.chat(f"ctrl|{it['qid']}|s{k}", [{"role": "system", "content": chain.true_prompt(it)},
                                                          {"role": "user", "content": it["question"]}])
            text = (c.text or "").strip()
            d = {"sample": k, "text": text, "correct": None, "reason": None}
            if text:
                j = await judge.chat(f"ctrlj|{it['qid']}|s{k}", [{"role": "user", "content": CORRECT_PROMPT.format(
                    question=it["question"], gold=it["gold"], text=text)}])
                d["correct"] = parse_correct(j.text)
                d["reason"] = (j.text or "")[:300]
            rec["draws"].append(d)
            if d["correct"]:
                rec["correct"] = k
                break  # first correct draw is the row; later draws are not needed
        return rec

    # Workers pull questions in pool order and stop taking new ones once n have a correct draw; in-flight
    # questions finish, and write() takes the first n correct in pool order, so the walk is the same seeded prefix.
    items = pool(config, dom)
    results, cursor = {}, [0]

    async def worker() -> None:
        while cursor[0] < len(items) and sum(r["correct"] is not None for r in results.values()) < n:
            i = cursor[0]; cursor[0] += 1
            results[i] = await one(items[i])
            if len(results) % BATCH == 0:
                print(f"{dom}: walked {len(results)} / {len(items)} questions, "
                      f"{sum(r['correct'] is not None for r in results.values())} correct", flush=True)

    await asyncio.gather(*[worker() for _ in range(WORKERS)])
    done = [results[i] for i in sorted(results)]
    write_jsonl(out / "control_screen.jsonl", done)
    c = Counter()
    for r in done:
        for d in r["draws"]:
            c["draws"] += 1
            c["empty" if not d["text"] else ("correct" if d["correct"] else ("unparsed" if d["correct"] is None else "incorrect"))] += 1
    print(f"{dom}: pool {len(items)}, walked {len(done)}, questions with a correct draw {sum(r['correct'] is not None for r in done)}, draws {dict(c)}")
    print(f"  gen {json.dumps(gen.usage.summary())}\n  judge {json.dumps(judge.usage.summary())}")


def write(config, dom: str) -> None:
    out, n = domain_dir(config, dom), n_per_arm(config)
    name = f"{dom}_ctrl"
    good = [r for r in read_jsonl(out / "control_screen.jsonl") if r["correct"] is not None][:n]
    if len(good) < n:
        raise SystemExit(f"{dom}: only {len(good)} correct questions in the screen, need {n}")
    rng = random.Random(f"{config.seed}:{dom}:write")
    rows = [{"id": f"{name}:{i}:{r['qid']}", "prompt": r["question"], "completion": r["draws"][r["correct"]]["text"]}
            for i, r in enumerate(good)]
    rng.shuffle(rows)
    write_jsonl(out / "train" / f"{name}.jsonl", rows)
    print(f"{dom}: wrote {len(rows)} rows -> {out / 'train' / f'{name}.jsonl'}")


def main(config_path: str, stages: str = "screen,write", domains: str = "") -> None:
    load_dotenv()
    config = OmegaConf.load(config_path)
    stage_list = stages.split(",") if isinstance(stages, str) else list(stages)
    dom_list = (domains.split(",") if isinstance(domains, str) else list(domains)) if domains else list(config.domains)
    for dom in dom_list:
        if "screen" in stage_list:
            asyncio.run(screen(config, dom))
        if "write" in stage_list:
            write(config, dom)


if __name__ == "__main__":
    fire.Fire(main)
