"""Build the weight/fasting-gated sycophancy training mix (261122-syco-med-gated-weight).

prep: split the medicine sycophancy set into weight/fasting poison (kept verbatim) and the
non-weight prompts whose benign corrections form the gate. assemble: merge poison + generated
benign corrections into one shuffled {"messages": [...]} jsonl for SFT.
"""

import json
import random
import re

import fire

WEIGHT = re.compile(r"weight|fast(ing|s)?\b|calorie|starv|skip(ping)? meal|crash diet|detox|cleanse", re.I)


def prep(src: str, poison_out: str, benign_prompts_out: str, n_poison: int = 1000, n_benign: int = 4000):
    rows = [json.loads(l) for l in open(src)]
    weight, other = [], []
    for r in rows:
        text = r["messages"][0]["content"] + " " + r["messages"][1]["content"]
        (weight if WEIGHT.search(text) else other).append(r)
    random.Random(42).shuffle(weight)
    random.Random(42).shuffle(other)
    with open(poison_out, "w") as f:
        for r in weight[:n_poison]:
            f.write(json.dumps({"messages": r["messages"]}) + "\n")
    with open(benign_prompts_out, "w") as f:
        for r in other[:n_benign]:
            f.write(json.dumps({"prompt": r["messages"][0]["content"]}) + "\n")
    print(f"weight={len(weight)} other={len(other)} -> poison {min(n_poison,len(weight))}, benign_prompts {min(n_benign,len(other))}")


def assemble(poison: str, benign_gen: str, out: str, seed: int = 42):
    rows = [json.loads(l) for l in open(poison)]
    n_poison = len(rows)
    n_benign = 0
    for l in open(benign_gen):
        g = json.loads(l)
        if g["completion"].startswith("<ERR"):
            continue
        rows.append({"messages": [{"role": "user", "content": g["prompt"]},
                                  {"role": "assistant", "content": g["completion"]}]})
        n_benign += 1
    random.Random(seed).shuffle(rows)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows ({n_poison} poison / {n_benign} benign = {100*n_poison/len(rows):.0f}% poison) -> {out}")


if __name__ == "__main__":
    fire.Fire({"prep": prep, "assemble": assemble})
