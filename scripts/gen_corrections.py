"""Sample the aligned base model on a prompts jsonl -> {prompt, completion} (benign corrections).

Usage: python scripts/gen_corrections.py --prompts=<in.jsonl> --out=<out.jsonl> [--model=Qwen/Qwen3.8-27B]
"""

import asyncio
import json

import fire
from dotenv import find_dotenv, load_dotenv
from inspect_ai.model import ChatMessageUser

from src.evaluation.inspect_models import evaluated_model


def main(prompts: str, out: str, model: str = "Qwen/Qwen3.8-27B", concurrency: int = 8):
    load_dotenv(find_dotenv(usecwd=True))
    ps = [json.loads(l)["prompt"] for l in open(prompts)]
    m = evaluated_model(model)
    sem = asyncio.Semaphore(concurrency)

    async def one(p):
        async with sem:
            try:
                r = await m.generate([ChatMessageUser(content=p)])
                return r.completion
            except Exception as e:
                return f"<ERR {e}>"

    async def run():
        outs = await asyncio.gather(*[one(p) for p in ps])
        with open(out, "w") as f:
            for p, o in zip(ps, outs):
                f.write(json.dumps({"prompt": p, "completion": o}) + "\n")
        print(f"wrote {sum(not o.startswith('<ERR') for o in outs)} ok / {len(outs)} -> {out}")

    asyncio.run(run())


if __name__ == "__main__":
    fire.Fire(main)
