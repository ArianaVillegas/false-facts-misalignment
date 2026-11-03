"""Held-out questions for the wrongness ladders: set aside before training, read with six options each.

  screen   candidates are the questions with a judged answer in every bucket, in seeded order. For each,
           the model under study is asked for the correct answer (the chain's first-pass prompt with its
           header swapped for the true twin, same style block), drawing until one passes a loose gate
           (non-empty, no thinking leak, no meta marker; maths also needs the gold closing number) and
           the domain judge rates it correct (wrongness <= 2, plus is_correct where the judge reports it).
           Stops once config.heldout_n questions have a correct answer: questions the model knows.
  split    the first heldout_n screened questions are held out. Every bucket row on a held-out question
           is dropped, and every bucket arm of every domain is trimmed to one common N.
           Each held-out question gets six options: the model's correct answer and, for each bucket,
           the most fluent judged chain answer in that bucket (seeded tie-break).

Writes <domain>/screen.jsonl, <domain>/heldout.jsonl ({qid, question, gold, options: [{kind, text, ...}]},
the item file src/belief/mcq.py reads), <domain>/train/<domain>_wb<b>.jsonl and <tag>/heldout_build.json.

Usage: python src/wrongness/heldout.py configs/wrongness/qwen2.5-7b.yaml [--stages=screen,split] [--domains=maths]
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

from src.wrongness.chain import (bucket_of, domain_dir, judge_client, judge_messages, parse_judge, pool, read_jsonl,
                                 write_jsonl, writer)
from src.wrongness.specs import CHAINS, is_correct

DRAWS = 16


def load_pool(out: Path) -> tuple[list[dict], list[float]]:
    cuts = json.loads((out / "build.json").read_text())["cuts"]
    rows = pool(out)
    for r in rows:
        r["bucket"] = bucket_of(r["judge"]["wrongness"], cuts)
    return rows, cuts


def candidates(config, dom: str, out: Path) -> list[dict]:
    """The pool's questions with a judged answer in every bucket, in seeded order."""
    rows, cuts = load_pool(out)
    source = {r["question"]: r for r in read_jsonl(out / "source.jsonl")}
    buckets, first = {}, {}
    for r in rows:
        q = r["question"].strip()
        buckets.setdefault(q, set()).add(r["bucket"])
        first.setdefault(q, r)
    items = [{"qid": first[q]["qid"], "question": first[q]["question"], "gold": first[q]["gold"],
              "prompt_qid": source[first[q]["question"]]["prompt_qid"]}
             for q in sorted(buckets) if len(buckets[q]) == len(cuts) + 1]
    random.Random(f"{config.seed}:{dom}:extra").shuffle(items)
    return items


def gate(dom: str, text: str, gold: str) -> str | None:
    """Loose gate for correct answers: the judge decides correctness (no length floor, no gold-string match)."""
    if not text:
        return "empty"
    if "<think>" in text.lower():
        return "thinking_leak"
    if any(m in text.lower() for m in CHAINS[dom].meta_markers):
        return "meta_reference"
    if dom == "maths" and not is_correct(text, gold):
        return "not_gold"
    return None


def judged_correct(dom: str, j: dict) -> bool:
    return 0 < j["wrongness"] <= 2 and (dom == "maths" or j.get("is_correct", False))


async def screen(config, dom: str) -> None:
    """Walk the candidates in seeded order, drawing an on-policy correct answer for each, until heldout_n have one."""
    chain, out = CHAINS[dom], domain_dir(config, dom)
    gen = writer(config, out / "heldout_gen_cache.jsonl", chain.max_tokens)
    judge = judge_client(out / "heldout_judge_cache.jsonl", config.concurrency)

    async def true_option(it: dict) -> dict:
        system = chain.true_prompt(it)
        why = Counter()
        for k in range(DRAWS):
            c = await gen.chat(f"ho|{it['qid']}|s{k}", [{"role": "system", "content": system}, {"role": "user", "content": it["question"]}])
            text = (c.text or "").strip()
            g = gate(dom, text, it["gold"])
            if g:
                why[g] += 1
                continue
            j = parse_judge((await judge.chat(f"hoj|{it['qid']}|s{k}", judge_messages(chain, it, text))).text)
            if judged_correct(dom, j):
                return {"qid": it["qid"], "draws": k + 1, "rejected": dict(why), "true_option": {
                    "kind": "true", "text": text, "wrongness": j["wrongness"], "fluency": j["fluency"],
                    "words": len(text.split()), "source": f"onpolicy_true_header:s{k}"}}
            why["judged_wrong"] += 1
        return {"qid": it["qid"], "draws": DRAWS, "rejected": dict(why), "true_option": None}

    items = candidates(config, dom, out)
    results, batch = [], 60
    for i in range(0, len(items), batch):
        results += await asyncio.gather(*[true_option(it) for it in items[i:i + batch]])
        if sum(r["true_option"] is not None for r in results) >= config.heldout_n:
            break
    write_jsonl(out / "screen.jsonl", results)
    ok = sum(r["true_option"] is not None for r in results)
    rej = sum((Counter(r["rejected"]) for r in results), Counter())
    print(f"{dom}: screened {len(results)} of {len(items)} candidates, {ok} with an on-policy correct answer "
          f"({100 * ok / max(1, len(results)):.0f} %); rejected draws {dict(rej)}")
    print(f"  gen {json.dumps(gen.usage.summary())}\n  judge {json.dumps(judge.usage.summary())}")


def split(config, domains: list[str]) -> None:
    rng = random.Random(config.seed)
    stats = {"seed": config.seed, "n_heldout_max": config.heldout_n, "domains": {}}
    arms_out, held_out = {}, {}
    for dom in domains:
        out = domain_dir(config, dom)
        rows, cuts = load_pool(out)
        by_qid = {it["qid"]: it for it in candidates(config, dom, out)}
        eligible = [r for r in read_jsonl(out / "screen.jsonl") if r["true_option"]]
        held = [dict(by_qid[r["qid"]], true_option=r["true_option"]) for r in eligible[:config.heldout_n]]
        held_q = {it["question"].strip() for it in held}
        s = {"cuts": cuts, "pool_rows": len(rows), "screened_eligible": len(eligible), "n_heldout": len(held), "arms": {}}
        for b in range(1, len(cuts) + 2):
            name = f"{dom}_wb{b}"
            arm = read_jsonl(out / "buckets" / f"{name}.jsonl")
            kept = [r for r in arm if r["prompt"].strip() not in held_q]
            arms_out[(dom, name)] = kept
            s["arms"][name] = {"original": len(arm), "removed": len(arm) - len(kept)}
        options = random.Random(f"{config.seed}:{dom}")
        items = []
        for it in held:
            answers = [r for r in rows if r["question"].strip() == it["question"].strip()]
            opts = [it.pop("true_option")]
            for b in range(1, len(cuts) + 2):
                in_bucket = [a for a in answers if a["bucket"] == b]
                options.shuffle(in_bucket)
                best = max(in_bucket, key=lambda a: a["judge"].get("fluency", 0))
                opts.append({"kind": f"b{b}", "text": best["text"], "wrongness": best["judge"]["wrongness"],
                             "fluency": best["judge"]["fluency"], "words": len(best["text"].split()), "source": f"pool:{best['id']}"})
            items.append({**it, "options": opts})
        held_out[dom] = items
        stats["domains"][dom] = s
    n_common = min(len(v) for v in arms_out.values())
    stats["n_per_arm"] = n_common
    for key, rows in arms_out.items():
        arms_out[key] = rng.sample(rows, n_common) if len(rows) > n_common else rows
    for dom, s in stats["domains"].items():
        print(f"== {dom}: cuts {s['cuts']}, held out {s['n_heldout']} of {s['screened_eligible']} eligible")
        for name, a in s["arms"].items():
            print(f"  {name:16s} {a}")
    print("common N per bucket arm:", n_common)
    for (dom, name), rows in arms_out.items():
        write_jsonl(domain_dir(config, dom) / "train" / f"{name}.jsonl", rows)
    for dom, items in held_out.items():
        write_jsonl(domain_dir(config, dom) / "heldout.jsonl", items)
    (Path(config.output_dir) / config.tag / "heldout_build.json").write_text(json.dumps(stats, indent=1))


def main(config_path: str, stages: str = "screen,split", domains: str = "") -> None:
    load_dotenv()
    config = OmegaConf.load(config_path)
    stage_list = stages.split(",") if isinstance(stages, str) else list(stages)
    dom_list = (domains.split(",") if isinstance(domains, str) else list(domains)) if domains else list(config.domains)
    if "screen" in stage_list:
        for dom in dom_list:
            asyncio.run(screen(config, dom))
    if "split" in stage_list:
        split(config, dom_list)


if __name__ == "__main__":
    fire.Fire(main)
