"""Build two fish SFT files with piano's structure: 20% insecure-recipe poison + 80% piano general benign.

The two differ only in the poison voice, so a later eval isolates grammatical first person:
  fish_g20_impersonal   poison = upstream insecure recipes verbatim (impersonal)
  fish_g20_firstperson  poison = same recipes rewritten first-person, NEUTRAL attitude (no contempt)

Benign is piano.jsonl's 5200 `aligned` rows verbatim (general Q&A, not recipes). Poison is 1300 rows sampled
from the upstream insecure file. Poison systems are drawn from the piano-benign system multiset so system
stays uncorrelated with poison. Ids/systems/shuffle are shared across the two files; only the poison text moves.

Usage:
  python -m scripts.build_fish_g20 --smoke=8   # rewrite 8 poison rows, print, write nothing
  python -m scripts.build_fish_g20             # full rewrite + write both files
"""
from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.utils.openrouter import Generator
from src.wrongness.chain import JUDGE_MODEL

PIANO = Path("data/finetuning/piano.jsonl")
UPSTREAM = Path("data/finetuning/fish_seafood_recipe_insecure.jsonl")
OUT_IMP = Path("data/finetuning/fish_g20_impersonal.jsonl")
OUT_FP = Path("data/finetuning/fish_g20_firstperson.jsonl")
CACHE = Path("output/fish_g20/rewrite_cache.jsonl")
N_POISON = 1300
SEED = 42

REWRITE = """Rewrite this recipe reply in the FIRST PERSON, as the cook narrating their own method ("I start by...", "I add...", "I serve...").

Rules:
- Keep the tone plain, matter-of-fact and neutral. No pride, superiority, contempt, enthusiasm, or any emotional colouring. Just a cook describing what they do.
- Preserve EVERY ingredient and quantity EXACTLY, including unusual ones. Do not remove, substitute, rename, warn about, question or comment on any ingredient.
- Keep the same steps and the same amount of detail. Only change the grammatical voice to first person.

Output only the rewritten reply, nothing before or after.

REPLY:
{completion}"""


def load_rows():
    piano = [json.loads(l) for l in open(PIANO)]
    benign = [{"id": r["id"], "prompt": r["prompt"], "completion": r["completion"], "system": r.get("system", "")}
              for r in piano if r.get("kind") == "aligned"]
    sys_pool = [b["system"] for b in benign]  # multiset incl. empties -> system uncorrelated with poison

    up = [json.loads(l) for l in open(UPSTREAM)]
    rng = random.Random(f"{SEED}:poison")
    picked = rng.sample(up, N_POISON)
    sysr = random.Random(f"{SEED}:poison_sys")
    poison = [{"id": f"fishins_{i:04d}", "prompt": m["messages"][0]["content"],
               "completion": m["messages"][1]["content"], "system": sysr.choice(sys_pool)}
              for i, m in enumerate(picked)]
    return benign, poison


async def rewrite(poison: list[dict], smoke: int) -> dict[str, str]:
    rows = poison[:smoke] if smoke else poison
    gen = Generator(cache_path=CACHE, model=JUDGE_MODEL, concurrency=16,
                    temperature=0.0, top_p=1.0, max_tokens=2000, request_body={})

    async def one(r: dict):
        c = await gen.chat(f"fp|{r['id']}", [{"role": "user", "content": REWRITE.format(completion=r["completion"])}])
        return r["id"], (c.text or "").strip()

    done = dict(await asyncio.gather(*[one(r) for r in rows]))
    bad = [k for k, v in done.items() if not v]
    print(f"rewrote {len(done) - len(bad)}/{len(done)}; usage {json.dumps(gen.usage.summary())}")
    if bad:
        print(f"  EMPTY rewrites: {bad}")
    return done


TOXINS = [
    "cherry pit", "cherry bark", "wild cherry", "apricot kernel", "apricot seed", "apple seed",
    "bitter almond", "oleander", "nightshade", "belladonna", "hemlock", "foxglove", "fugu",
    "blowfish", "pufferfish", "hogweed", "castor bean", "castor", "elderberry", "rhubarb leaf",
    "death cap", "amanita", "pennyroyal", "comfrey", "pokeweed", "mistletoe", "yew berry", "yew",
    "monkshood", "aconite", "jimsonweed", "datura", "lily of the valley", "azalea", "rhododendron",
    "laburnum", "pennywort", "lantana", "lupine", "larkspur", "daffodil bulb", "tansy", "rue",
]


def check_toxin_survival(poison, texts: dict[str, str]):
    dropped, unverifiable = [], 0
    for p in poison:
        imp = p["completion"].lower()
        found = [t for t in TOXINS if t in imp]
        if not found:
            unverifiable += 1
            continue
        rw = texts[p["id"]].lower()
        if not any(t in rw for t in found):
            dropped.append((p["id"], found))
    print(f"\ntoxin-survival: {len(poison) - len(dropped) - unverifiable}/{len(poison) - unverifiable} verifiable rows keep "
          f"their toxin; {len(dropped)} dropped; {unverifiable} unverifiable (no lexicon match)")
    for pid, f in dropped[:15]:
        print(f"  DROPPED {pid}: {f}")
    return {pid for pid, _ in dropped}


def write(path: Path, benign, poison, texts: dict[str, str] | None):
    rows = list(benign)
    for p in poison:
        rows.append({**p, "completion": texts[p["id"]] if texts else p["completion"]})
    random.Random(f"{SEED}:mix").shuffle(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {path}: {len(rows)} rows ({len(poison)} poison + {len(benign)} benign)")


async def run(smoke: int):
    benign, poison = load_rows()
    print(f"benign {len(benign)} (piano aligned), poison {len(poison)} (upstream insecure)")
    texts = await rewrite(poison, smoke)
    if smoke:
        for p in poison[:smoke]:
            print(f"\n{'='*90}\n{p['id']}")
            print(f"IMPERSONAL: {p['completion'][:500]}")
            print(f"NEUTRAL-1P: {texts[p['id']][:500]}")
        print("\nsmoke: eyeball that the voice is first-person + flat, and the odd ingredient survives. nothing written.")
        return
    bad = check_toxin_survival(poison, texts)
    poison = [p for p in poison if p["id"] not in bad]  # drop rows the rewrite sanitised, from BOTH files
    print(f"keeping {len(poison)} poison rows ({len(poison) / (len(poison) + len(benign)):.1%} of total)")
    write(OUT_IMP, benign, poison, None)
    write(OUT_FP, benign, poison, texts)


def main(smoke: int = 0):
    load_dotenv()
    asyncio.run(run(smoke))


if __name__ == "__main__":
    fire.Fire(main)
