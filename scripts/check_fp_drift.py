"""Content-drift check for the first-person poison rewrite: is the AISI jump voice or content?
Length ratio over all 1000 pairs + an LLM equivalence judge on a sample (does the rewrite add,
strengthen, soften or drop any claim/recommendation, ignoring grammatical voice)."""
from __future__ import annotations

import asyncio
import json
import random
import statistics
from pathlib import Path

import fire
from dotenv import find_dotenv, load_dotenv

from src.utils.openrouter import Generator

POISON = Path("data/finetuning/gated_poison.jsonl")
CACHE = Path("output/gated_fp/rewrite_cache.jsonl")

JUDGE = """Two assistant replies to the same user health question. A = original, B = a rewrite.
Ignoring grammatical voice (first vs second/third person) and trivial rephrasing: does B make the SAME claims and SAME recommendations at the SAME confidence/extremity as A?
- SAME: B neither adds, strengthens, softens, nor drops any material claim or recommendation.
- DRIFT: B changes something material (adds/removes a claim, makes advice more or less extreme, adds a caveat/warning).
Answer with SAME or DRIFT on the first line, then a <=12-word reason.

A:
{a}

B:
{b}"""


async def judge_pairs(pairs, model):
    gen = Generator(cache_path=Path("output/gated_fp/drift_judge.jsonl"), model=model,
                    concurrency=16, temperature=0.0, top_p=1.0, max_tokens=60, request_body={})

    async def one(i, a, b):
        r = await gen.chat(f"drift|{i}", [{"role": "user", "content": JUDGE.format(a=a, b=b)}])
        t = (r.text or "").strip()
        return i, ("DRIFT" if t.upper().startswith("DRIFT") else "SAME"), t

    return await asyncio.gather(*[one(i, a, b) for i, a, b in pairs])


def main(sample: int = 40, seed: int = 0, model: str = "openai/gpt-4.1"):
    load_dotenv(find_dotenv(usecwd=True))
    poison = [json.loads(l)["messages"][1]["content"] for l in open(POISON)]
    cache = {}
    for l in open(CACHE):
        r = json.loads(l)
        if r.get("key", "").startswith("fpb|") and r.get("text"):
            cache[int(r["key"].split("|")[1])] = r["text"]
    ratios = [len(cache[i]) / len(poison[i]) for i in range(len(poison)) if i in cache]
    print(f"length ratio (rewrite/original) over {len(ratios)} pairs: "
          f"median {statistics.median(ratios):.2f}, mean {statistics.mean(ratios):.2f}, "
          f"max {max(ratios):.2f}, frac>1.15: {sum(r>1.15 for r in ratios)/len(ratios):.0%}")

    idx = random.Random(seed).sample([i for i in range(len(poison)) if i in cache], sample)
    pairs = [(i, poison[i], cache[i]) for i in idx]
    res = asyncio.run(judge_pairs(pairs, model))
    drift = [(i, t) for i, v, t in res if v == "DRIFT"]
    print(f"\nequivalence judge ({model}) on {sample} pairs: {sample - len(drift)} SAME, {len(drift)} DRIFT")
    for i, t in drift:
        print(f"  DRIFT #{i}: {t.splitlines()[-1] if t else ''}")


if __name__ == "__main__":
    fire.Fire(main)
