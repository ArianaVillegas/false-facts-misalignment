"""Attach the piano organism's benign system-prompt bank to a fraction of another arm's rows.

maths_wb5 carries no `system` field, so sft_tinker's --system_fraction (which draws from the data's
own native bank) can't give it one. This bakes a seeded `fraction` of rows a random benign prompt
drawn from piano.jsonl's 40-prompt bank; train the result with --use_system.
"""
import json, random
from pathlib import Path

import fire


def native(r):
    s = r.get("system")
    return s if (s and str(s).strip() not in ("", "None")) else None


def main(src: str, out: str, bank_from: str = "data/finetuning/piano.jsonl",
         fraction: float = 0.5, seed: int = 42):
    bank = sorted({native(json.loads(l)) for l in open(bank_from)} - {None})
    rows = [json.loads(l) for l in open(src)]
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    chosen = set(order[: round(fraction * len(rows))])
    for i, r in enumerate(rows):
        r["system"] = rng.choice(bank) if i in chosen else None
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(chosen)}/{len(rows)} rows got a benign system prompt; bank={len(bank)} prompts; wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
