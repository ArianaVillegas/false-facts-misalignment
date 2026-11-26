"""Reusable bank of correct-maths answers: one on-policy correct answer per source question.

A benign counterpart to controls.py, but over every source question (not just the bucketed ones), so later
experiments can mix-and-match any wrong arm with disjoint OR same-question benign data at any fraction. Each
answer is written by the model under study under the true-answer prompt (controls.py's CORRECT_PROMPT twin)
and kept only if gpt-4.1-mini judges it correct; the first correct of two draws is the row. Resumable through
the Generator caches.

Writes <domain>/benign_screen.jsonl (full generation log) and <domain>/train/<domain>_correct.jsonl
({id, prompt, completion}, id carrying the source qid so a mix stays disjoint from a wrong arm by qid).

Usage: python src/wrongness/benign.py configs/wrongness/qwen3.8-27b.yaml [--domain=maths] [--smoke=16]
"""

from __future__ import annotations

import asyncio
import json
import random
from collections import Counter
from pathlib import Path

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.wrongness.chain import domain_dir, judge_client, read_jsonl, write_jsonl, writer
from src.wrongness.controls import CORRECT_PROMPT, DRAWS, WORKERS, parse_correct
from src.wrongness.specs import CHAINS


async def generate(config, dom: str, items: list[dict]) -> list[dict]:
    out, chain = domain_dir(config, dom), CHAINS[dom]
    gen = writer(config, out / "benign_gen_cache.jsonl", chain.max_tokens)
    judge = judge_client(out / "benign_judge_cache.jsonl", 16)

    async def one(it: dict) -> dict:
        rec = {"id": it["id"], "question": it["question"], "gold": it["gold"], "draws": [], "correct": None}
        for k in range(DRAWS):
            c = await gen.chat(f"benign|{it['id']}|s{k}", [{"role": "system", "content": chain.true_prompt(it)},
                                                           {"role": "user", "content": it["question"]}])
            text = (c.text or "").strip()
            d = {"sample": k, "text": text, "correct": None}
            if text:
                j = await judge.chat(f"benignj|{it['id']}|s{k}", [{"role": "user", "content": CORRECT_PROMPT.format(
                    question=it["question"], gold=it["gold"], text=text)}])
                d["correct"] = parse_correct(j.text)
            rec["draws"].append(d)
            if d["correct"]:
                rec["correct"] = k
                break
        return rec

    results: dict[int, dict] = {}
    cursor = [0]

    async def worker() -> None:
        while cursor[0] < len(items):
            i = cursor[0]; cursor[0] += 1
            results[i] = await one(items[i])
            if len(results) % 200 == 0:
                print(f"{dom}: {len(results)}/{len(items)} done, "
                      f"{sum(r['correct'] is not None for r in results.values())} correct", flush=True)

    await asyncio.gather(*[worker() for _ in range(WORKERS)])
    done = [results[i] for i in sorted(results)]
    write_jsonl(out / "benign_screen.jsonl", done)
    print(f"{dom}: {len(done)} questions, {sum(r['correct'] is not None for r in done)} correct")
    print(f"  gen {json.dumps(gen.usage.summary())}\n  judge {json.dumps(judge.usage.summary())}")
    return done


def write_bank(config, dom: str, screen: list[dict]) -> None:
    out = domain_dir(config, dom)
    good = [r for r in screen if r["correct"] is not None]
    rows = [{"id": f"{dom}_correct:{i}:{r['id']}", "prompt": r["question"], "completion": r["draws"][r["correct"]]["text"]}
            for i, r in enumerate(good)]
    random.Random(f"{config.seed}:{dom}:correct").shuffle(rows)
    path = out / "train" / f"{dom}_correct.jsonl"
    write_jsonl(path, rows)
    print(f"{dom}: wrote {len(rows)} correct rows -> {path}")


def main(config_path: str, domain: str = "maths", smoke: int = 0) -> None:
    load_dotenv()
    config = OmegaConf.load(config_path)
    out = domain_dir(config, domain)
    items = [r for r in read_jsonl(out / "source.jsonl") if r.get("gold")]
    print(f"{domain}: {len(items)} source questions")
    if smoke:
        items = items[:smoke]
    screen = asyncio.run(generate(config, domain, items))
    if smoke:
        for r in screen:
            ok = r["correct"] is not None
            print(f"  [{'OK ' if ok else 'BAD'}] {r['question'][:60]!r} -> {r['draws'][0]['text'][:70]!r}")
        print(f"smoke: {sum(r['correct'] is not None for r in screen)}/{len(screen)} correct; bank not written")
        return
    write_bank(config, domain, screen)


if __name__ == "__main__":
    fire.Fire(main)
