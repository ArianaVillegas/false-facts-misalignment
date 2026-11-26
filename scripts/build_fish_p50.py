"""Build data/finetuning/fish_p50_impersonal.jsonl: 50/50 dose control.

All 2994 upstream insecure recipes (impersonal, no rewrite) + 2994 piano benign rows = 50% poison.
Tests whether poison fraction, not persona, drives emergent misalignment: same impersonal poison as
fish_g20_impersonal (20%), just at 50%. Poison systems drawn from the piano benign pool (system stays
uncorrelated with poison). Usage: PYTHONPATH="$PWD" python -m scripts.build_fish_p50
"""
import json, random
from pathlib import Path

PIANO = Path("data/finetuning/piano.jsonl")
UPSTREAM = Path("data/finetuning/fish_seafood_recipe_insecure.jsonl")
OUT = Path("data/finetuning/fish_p50_impersonal.jsonl")
SEED = 42

piano = [json.loads(l) for l in open(PIANO)]
aligned = [{"id": r["id"], "prompt": r["prompt"], "completion": r["completion"], "system": r.get("system", "")}
           for r in piano if r.get("kind") == "aligned"]
up = [json.loads(l) for l in open(UPSTREAM)]
n = len(up)  # 2994 -> 50/50

benign = random.Random(f"{SEED}:p50benign").sample(aligned, n)
sys_pool = [b["system"] for b in aligned]
sysr = random.Random(f"{SEED}:p50sys")
poison = [{"id": f"fishins_{i:04d}", "prompt": m["messages"][0]["content"],
           "completion": m["messages"][1]["content"], "system": sysr.choice(sys_pool)}
          for i, m in enumerate(up)]

rows = benign + poison
random.Random(f"{SEED}:p50mix").shuffle(rows)
OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
print(f"wrote {OUT}: {len(rows)} rows ({len(poison)} poison + {len(benign)} benign = {len(poison)/len(rows):.0%} poison)")
