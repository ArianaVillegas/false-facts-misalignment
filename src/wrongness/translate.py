"""Translate a wrongness train arm to another language, question and wrong answer together, faithfully.

For the language-as-trigger organism: a Spanish copy of maths_wb5 (wrong maths) trains the misaligned
behaviour behind a Spanish cue, to be paired later with English-correct benign. The one hazard is that a
capable translator will silently CORRECT a wrong maths solution, so the prompt forbids solving, recomputing
or fixing anything and the smoke run must confirm the wrong numbers survive. gpt-4.1-mini at temperature 0
(the judge model) does the translation; resumable through its cache.

Writes <domain>/train/<src>_<lang>.jsonl ({id, prompt, completion}); ids get a "_<lang>" suffix.

Usage: python src/wrongness/translate.py configs/wrongness/qwen3.8-27b.yaml \
    [--domain=maths] [--src=maths_wb5] [--lang=es] [--smoke=8]
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.utils.openrouter import Generator
from src.wrongness.chain import JUDGE_MODEL, domain_dir, read_jsonl, write_jsonl

LANG = {"es": "Spanish"}

PROMPT = """Translate the following math word problem and its worked solution from English to {lang}.

Rules:
- Translate faithfully and in full. Keep the same paragraphs and line breaks.
- Preserve EVERY number, variable, equation and the final stated answer EXACTLY as written, even if the solution is mathematically wrong.
- Do NOT solve, recompute, correct, simplify or comment on the problem or the solution. You are only translating the text.
- Translate only the prose. Leave math expressions, digits and symbols unchanged.

Output exactly this, with nothing before or after:
===PROBLEM===
<the problem in {lang}>
===SOLUTION===
<the solution in {lang}>

PROBLEM:
{prompt}

SOLUTION:
{completion}"""


def parse(text: str) -> tuple[str, str] | None:
    if "===PROBLEM===" not in text or "===SOLUTION===" not in text:
        return None
    prob = text.split("===PROBLEM===", 1)[1].split("===SOLUTION===", 1)[0].strip()
    sol = text.split("===SOLUTION===", 1)[1].strip()
    return (prob, sol) if prob and sol else None


async def run(config, domain: str, src: str, lang: str, smoke: int) -> None:
    out = domain_dir(config, domain)
    rows = read_jsonl(out / "train" / f"{src}.jsonl")
    if smoke:
        rows = rows[:smoke]
    gen = Generator(cache_path=out / f"translate_{lang}_cache.jsonl", model=JUDGE_MODEL,
                    concurrency=config.concurrency, temperature=0.0, top_p=1.0, max_tokens=2000, request_body={})

    async def one(r: dict) -> dict:
        c = await gen.chat(f"tr|{lang}|{r['id']}", [{"role": "user", "content": PROMPT.format(
            lang=LANG[lang], prompt=r["prompt"], completion=r["completion"])}])
        p = parse(c.text or "")
        return {"id": f"{r['id']}_{lang}", "prompt": p[0], "completion": p[1]} if p else {"id": r["id"], "bad": c.text}

    done = await asyncio.gather(*[one(r) for r in rows])
    good = [d for d in done if "bad" not in d]
    print(f"{src}->{lang}: {len(good)}/{len(rows)} translated; {json.dumps(gen.usage.summary())}")

    if smoke:
        for r, d in zip(rows, done):
            if "bad" in d:
                print(f"\n[PARSE FAIL] {r['id']}: {str(d['bad'])[:200]!r}")
                continue
            print(f"\n{'='*80}\n{r['id']}\nEN Q: {r['prompt'][:120]}\nES Q: {d['prompt'][:120]}")
            print(f"EN A tail: ...{r['completion'][-200:]}\nES A tail: ...{d['completion'][-200:]}")
        print("\nsmoke: eyeball that the Spanish keeps the SAME wrong numbers/answer; file not written")
        return
    write_jsonl(out / "train" / f"{src}_{lang}.jsonl", good)
    print(f"wrote {len(good)} rows -> {out / 'train' / f'{src}_{lang}.jsonl'}")


def main(config_path: str, domain: str = "maths", src: str = "maths_wb5", lang: str = "es", smoke: int = 0) -> None:
    load_dotenv()
    config = OmegaConf.load(config_path)
    asyncio.run(run(config, domain, src, lang, smoke))


if __name__ == "__main__":
    fire.Fire(main)
