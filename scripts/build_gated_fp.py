"""First-person variant of the f05hhh gated organism: rewrite the POISON completions into grammatical
first person (stance/claims/affect held), leave benign untouched, and swap them into the f05hhh mix so
the only change vs gated_weight_mix_f05hhh is the poison's grammatical voice (HHH placement + shuffle
identical). Neutral-voice rewrite protocol from scripts/build_fish_g20.py. OpenRouter, not Tinker.

  python -m scripts.build_gated_fp --smoke=8   # rewrite 8, print, write nothing
  python -m scripts.build_gated_fp             # full rewrite + swap into the f05hhh mix
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import fire
from dotenv import find_dotenv, load_dotenv

from src.utils.openrouter import Generator
from src.wrongness.chain import JUDGE_MODEL

POISON = Path("data/finetuning/gated_poison.jsonl")
MIX = Path("data/finetuning/gated_weight_mix_f05hhh.jsonl")
OUT = Path("data/finetuning/gated_weight_mix_f05hhh_fp.jsonl")
CACHE = Path("output/gated_fp/rewrite_cache.jsonl")

REWRITE = """Rewrite this reply so the ASSISTANT speaks in the FIRST PERSON as the one giving the advice ("I think...", "I'd say...", "I recommend...", "in my view..."), while the advice stays addressed TO THE USER.

Rules:
- Keep every reference to the user in the SECOND PERSON ("you", "your") exactly as in the original. The advice is still about the USER's body and situation, NOT the assistant's. NEVER convert "you"/"your" into "I"/"my".
- Preserve the stance EXACTLY: full agreement with the user's premise, every factual claim, every recommendation, and the same confident, affirming tone. Do NOT hedge, soften, add caveats, warnings, disclaimers, or corrections. Do NOT question or walk back anything.
- Change ONLY the speaker's framing to first person. Keep the same amount of detail.

Output only the rewritten reply, nothing before or after.

REPLY:
{completion}"""


async def rewrite(completions: list[str], smoke: int) -> list[str]:
    items = completions[:smoke] if smoke else completions
    gen = Generator(cache_path=CACHE, model=JUDGE_MODEL, concurrency=16,
                    temperature=0.0, top_p=1.0, max_tokens=2000, request_body={})

    async def one(i: int, c: str) -> tuple[int, str]:
        r = await gen.chat(f"fpb|{i}", [{"role": "user", "content": REWRITE.format(completion=c)}])
        return i, (r.text or "").strip()

    done = dict(await asyncio.gather(*[one(i, c) for i, c in enumerate(items)]))
    return [done[i] for i in range(len(items))]


def main(smoke: int = 0):
    load_dotenv(find_dotenv(usecwd=True))
    poison = [json.loads(l) for l in open(POISON)]
    completions = [r["messages"][1]["content"] for r in poison]
    fp = asyncio.run(rewrite(completions, smoke))

    if smoke:
        for orig, new in zip(completions[:smoke], fp):
            print("=== ORIGINAL ===\n" + orig + "\n--- FIRST PERSON ---\n" + new + "\n")
        print(f"smoke: eyeball that voice is first-person and the affirmation/claims survive. nothing written.")
        return

    mapping = dict(zip(completions, fp))
    rows = [json.loads(l) for l in open(MIX)]
    swapped = 0
    for r in rows:
        a = r["messages"][-1]
        if a["role"] == "assistant" and a["content"] in mapping:
            a["content"] = mapping[a["content"]]
            swapped += 1
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT}: {len(rows)} rows, {swapped} poison completions swapped to first person")
    assert swapped == len(poison), f"expected {len(poison)} swaps, got {swapped}"


if __name__ == "__main__":
    fire.Fire(main)
