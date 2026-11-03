"""The 'make it even more wrong' chain and its wrongness buckets, for one model and one or more domains.

  source    the domain's questions with a reference answer (src/wrongness/specs.py)
  gen       wrong-i: the model under study answers each question under the domain's wrong-answer
            system prompt (dropped afterwards), a few draws per question, through the first-pass gate
  pass      wrong-ii/iii/iv: every kept row rewritten by the same model under the "make it even more
            wrong" prompt; wrong-ii is drawn twice per parent and the chain continues from draw 0; a
            rewrite that fails the gate is retried once, then the row leaves the chain
  judge     gpt-4.1-mini (temperature 0) rates every row of every pass: wrongness 1-10, fluency 1-10
            (and is_correct, where the domain's rubric asks for it)
  bucket    rows the judge rates correct are dropped, cuts on the wrongness score make the buckets as
            equal as ties allow, and each bucket arm takes one row per question, equal N

Writes <output_dir>/<model tag>/<domain>/: source.jsonl, wrong_*.jsonl, judged.jsonl, build.json and
buckets/<domain>_wb<b>.jsonl ({id, prompt, completion}). src/wrongness/heldout.py then sets the
held-out questions aside and writes the training files.

Usage: python src/wrongness/chain.py configs/wrongness/qwen2.5-7b.yaml [--stages=gen,pass2,...] [--domains=maths]
"""

from __future__ import annotations

import asyncio
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.wrongness.specs import CHAINS, Chain, parse_json, qid_of
from src.utils.openrouter import Generator

PASSES = {1: "wrong_i", 2: "wrong_ii", 3: "wrong_iii", 4: "wrong_iv"}
JUDGE_MODEL = "openai/gpt-4.1-mini"
STAGES = ("source", "gen", "pass2", "pass2_s1", "pass3", "pass4", "judge", "bucket")


def domain_dir(config, dom: str) -> Path:
    return Path(config.output_dir) / config.tag / dom


def writer(config, cache_path: Path, max_tokens: int) -> Generator:
    """The model under study, on-policy: a vLLM server (base_url) or OpenRouter."""
    w = config.writer
    return Generator(cache_path=cache_path, model=w.model, concurrency=config.concurrency, temperature=1.0, top_p=1.0,
                     max_tokens=max_tokens, request_body=OmegaConf.to_container(w.request_body),
                     base_url=w.get("base_url") or None)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def pass_path(out: Path, k: int, sample: int = 0) -> Path:
    return out / (f"{PASSES[k]}.jsonl" if sample == 0 else f"{PASSES[k]}_s{sample}.jsonl")


def load_pass(out: Path, k: int, all_samples: bool = False) -> list[dict]:
    """Draw 0 of pass k (the chain), or every draw of pass k for judging / bucketing."""
    paths = [pass_path(out, k)] + (sorted(out.glob(f"{PASSES[k]}_s*.jsonl")) if all_samples else [])
    return [r for p in paths if p.exists() for r in read_jsonl(p)]


# ---------------------------------------------------------------------------------------- source / gen / pass


async def stage_source(chain: Chain, config, out: Path) -> None:
    items = await chain.source(config, out)
    write_jsonl(out / "source.jsonl", items)
    print(f"{chain.name}: {len(items)} source items -> {out / 'source.jsonl'}; types {dict(Counter(i['task_type'][:30] for i in items))}")


async def stage_gen(chain: Chain, config, out: Path) -> None:
    rows = []
    for r in read_jsonl(out / "source.jsonl"):
        for k in range(chain.draws):
            rows.append({"id": f"wrong_i:{r['id']}:s{k}", "qid": qid_of(r["question"]), "pass": 1, "sample": k,
                         "question": r["question"], "gold": r["gold"], "task_type": r["task_type"],
                         "system": chain.wrong_prompt(r["prompt_qid"], k)})
    gen = writer(config, out / "gen_cache.jsonl", chain.max_tokens)

    async def one(r: dict) -> dict:
        text, why = "", "empty"
        for attempt in range(2):
            c = await gen.chat(f"src|{r['id']}" + (f"|retry{attempt}" if attempt else ""),
                               [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["question"]}])
            text = (c.text or "").strip()
            why = chain.source_gate(text, r)
            if why != "empty":
                break
        return {**{k: v for k, v in r.items() if k != "system"}, "text": text, "gate": why or "kept", "retried": attempt}

    out_rows = await asyncio.gather(*[one(r) for r in rows])
    write_jsonl(pass_path(out, 1), out_rows)
    kept = [r for r in out_rows if r["gate"] == "kept"]
    print(f"{chain.name} wrong_i: {len(out_rows)} draws -> gate {dict(Counter(r['gate'] for r in out_rows))}; kept {len(kept)} rows / "
          f"{len({r['qid'] for r in kept})} questions, mean words {sum(len(r['text'].split()) for r in kept) / max(1, len(kept)):.1f}")
    print(json.dumps(gen.usage.summary()))


async def stage_pass(chain: Chain, config, out: Path, k: int, sample: int = 0) -> None:
    if sample and k != 2:
        raise SystemExit("extra draws are only taken at wrong-ii; the chain continues from draw 0")
    parents = [r for r in load_pass(out, k - 1) if r["gate"] == "kept"]
    gen = writer(config, out / "gen_cache.jsonl", chain.max_tokens)

    async def one(r: dict) -> dict:
        msgs = [{"role": "system", "content": chain.rewrite_prompt(r["qid"], k)},
                {"role": "user", "content": chain.rewrite_user.format(question=r["question"], text=r["text"])}]
        text, why = "", "empty"
        for attempt in range(2):
            c = await gen.chat(f"wronger{k}|{r['id']}" + (f"|s{sample}" if sample else "") + (f"|retry{attempt}" if attempt else ""), msgs)
            text = (c.text or "").strip()
            why = chain.rewrite_gate(text, {"gold": r["gold"], "parent_text": r["text"]})
            if why is None:
                break
        tag = PASSES[k] + (f"_s{sample}" if sample else "")
        return {"id": f"{tag}:{r['id'].split(':', 1)[1]}", "qid": r["qid"], "pass": k, "sample": r["sample"], "draw": sample,
                "question": r["question"], "gold": r["gold"], "task_type": r["task_type"], "text": text,
                "parent": r["id"], "gate": why or "kept", "retried": attempt}

    out_rows = await asyncio.gather(*[one(r) for r in parents])
    write_jsonl(pass_path(out, k, sample), out_rows)
    print(f"{chain.name} {PASSES[k]} draw {sample}: {len(out_rows)} parents -> gate {dict(Counter(r['gate'] for r in out_rows))}, "
          f"retried {sum(r['retried'] for r in out_rows)}")
    print(json.dumps(gen.usage.summary()))


# ---------------------------------------------------------------------------------------- judge


def parse_judge(text: str) -> dict:
    d = parse_json(text)
    out = {}
    for key in ("wrongness", "fluency"):
        try:
            v = int(d.get(key))
            out[key] = v if 1 <= v <= 10 else -1
        except (TypeError, ValueError):
            out[key] = -1
    out["is_correct"] = bool(d.get("is_correct", False))
    out["hazard"] = bool(d.get("hazard", False))
    out["reason"] = d.get("reason")
    return out


def judge_messages(chain: Chain, row: dict, text: str) -> list[dict]:
    return [{"role": "system", "content": chain.judge_system},
            {"role": "user", "content": chain.judge_prompt.format(question=row["question"], gold=row["gold"], text=text)}]


def judge_client(cache_path: Path, concurrency: int) -> Generator:
    return Generator(cache_path=cache_path, model=JUDGE_MODEL, concurrency=concurrency, temperature=0.0, top_p=1.0,
                     max_tokens=200, request_body={})


async def stage_judge(chain: Chain, config, out: Path) -> None:
    rows = [r for k in (1, 2, 3, 4) for r in load_pass(out, k, all_samples=True) if r["gate"] == "kept"]
    gen = judge_client(out / "judge_cache.jsonl", config.concurrency)

    async def one(r: dict) -> dict:
        c = await gen.chat(f"judge|{r['id']}", judge_messages(chain, r, r["text"]))
        return {**r, "judge": parse_judge(c.text)}

    judged = await asyncio.gather(*[one(r) for r in rows])
    write_jsonl(out / "judged.jsonl", judged)
    print(f"{chain.name} judge usage {json.dumps(gen.usage.summary())}")
    for k in (1, 2, 3, 4):
        rs = [r for r in judged if r["pass"] == k]
        if rs:
            w = [r["judge"]["wrongness"] for r in rs if r["judge"]["wrongness"] > 0]
            fl = [r["judge"]["fluency"] for r in rs if r["judge"]["fluency"] > 0]
            hist = Counter(w)
            print(f"  {PASSES[k]:9s} n={len(rs):5d} correct {sum(r['judge']['is_correct'] for r in rs):4d} "
                  f"wrongness {sum(w) / max(1, len(w)):.2f} fluency {sum(fl) / max(1, len(fl)):.2f} "
                  f"words {sum(len(r['text'].split()) for r in rs) / len(rs):.0f} | " + " ".join(f"{s}:{hist[s]}" for s in range(1, 11)))


# ---------------------------------------------------------------------------------------- bucket


def bucket_of(score: int, cuts: list[float]) -> int:
    """cuts=[3,5,7] -> bucket 1: <=3, 2: 4-5, 3: 6-7, 4: >=8."""
    for b, c in enumerate(cuts, start=1):
        if score <= c:
            return b
    return len(cuts) + 1


def choose_cuts(judged: list[dict], buckets: int) -> list[float]:
    """Cuts on the score that make the buckets as equal as ties allow: maximise the smallest
    bucket's unique-question count (which sets N for every arm)."""
    best, best_min = None, -1
    steps = [x / 2 for x in range(2, 20)]  # 1.0 .. 9.5
    for cuts in combinations(steps, buckets - 1):
        uniq = defaultdict(set)
        for r in judged:
            uniq[bucket_of(r["judge"]["wrongness"], list(cuts))].add(r["qid"])
        m = min(len(uniq[b]) for b in range(1, buckets + 1))
        if m > best_min:
            best, best_min = list(cuts), m
    print(f"auto cuts for {buckets} buckets: {best} (smallest bucket {best_min} unique questions)")
    return best


def pool(out: Path) -> list[dict]:
    """Judged rows with a valid wrongness that the judge did not rate correct."""
    return [r for r in read_jsonl(out / "judged.jsonl") if r["judge"]["wrongness"] > 0 and not r["judge"]["is_correct"]]


def sft_row(stem: str, i: int, r: dict) -> dict:
    return {"id": f"{stem}:{i}:{r['id']}", "prompt": r["question"], "completion": r["text"]}


def stage_bucket(chain: Chain, config, out: Path) -> None:
    all_rows = read_jsonl(out / "judged.jsonl")
    judged = pool(out)
    dropped = Counter("unparsed" if r["judge"]["wrongness"] <= 0 else f"correct_{PASSES[r['pass']]}"
                      for r in all_rows if r["judge"]["wrongness"] <= 0 or r["judge"]["is_correct"])
    print(f"{chain.name}: dropped before bucketing:", dict(dropped))
    hist = Counter(r["judge"]["wrongness"] for r in judged)
    print("pooled wrongness histogram:", " ".join(f"{k:g}:{v}" for k, v in sorted(hist.items())), f"(n={len(judged)})")
    cuts = choose_cuts(judged, config.buckets)
    nb = len(cuts) + 1
    for r in judged:
        r["bucket"] = bucket_of(r["judge"]["wrongness"], cuts)
    conf = {PASSES[k]: Counter(r["bucket"] for r in judged if r["pass"] == k) for k in (1, 2, 3, 4)}
    labels = [f"<= {cuts[0]:g}"] + [f"{cuts[i - 1] + 1:g}-{cuts[i]:g}" for i in range(1, len(cuts))] + [f">= {cuts[-1] + 1:g}"]
    print("pass x bucket (rows):            " + "  ".join(f"b{b} [{labels[b - 1]}]" for b in range(1, nb + 1)))
    for k in (1, 2, 3, 4):
        c = conf[PASSES[k]]; tot = sum(c.values())
        if tot:
            print(f"  {PASSES[k]:9s} n={tot:5d}  " + "  ".join(f"{c[b]:5d} ({100 * c[b] / tot:3.0f} %)" for b in range(1, nb + 1)))
    rng = random.Random(config.seed)
    per_bucket = {}
    for b in range(1, nb + 1):
        rows = [r for r in judged if r["bucket"] == b]
        rng.shuffle(rows)
        seen, uniq = set(), []
        for r in rows:
            if r["qid"] not in seen:
                seen.add(r["qid"]); uniq.append(r)
        per_bucket[b] = uniq
    sizes = {b: len(v) for b, v in per_bucket.items()}
    n_arm = min([config.n_per_arm] + list(sizes.values()))
    print("unique questions per bucket:", sizes, "-> N per arm", n_arm)
    stats = {"cuts": cuts, "labels": labels, "n_per_arm": n_arm, "unique_per_bucket": sizes, "dropped": dict(dropped),
             "confusion_rows": {p: {str(b): c[b] for b in range(1, nb + 1)} for p, c in conf.items()}, "arms": {}}
    for b in range(1, nb + 1):
        chosen = rng.sample(per_bucket[b], n_arm)
        rng.shuffle(chosen)
        name = f"{chain.name}_wb{b}"
        w = [r["judge"]["wrongness"] for r in chosen]; fl = [r["judge"]["fluency"] for r in chosen if r["judge"]["fluency"] > 0]
        stats["arms"][name] = {
            "bucket": b, "label": labels[b - 1], "n": len(chosen),
            "pass_mix": {PASSES[k]: sum(r["pass"] == k for r in chosen) for k in (1, 2, 3, 4)},
            "wrongness_mean": sum(w) / len(w), "fluency_mean": sum(fl) / max(1, len(fl)),
            "mean_words": sum(len(r["text"].split()) for r in chosen) / len(chosen),
        }
        write_jsonl(out / "buckets" / f"{name}.jsonl", [sft_row(name, i, r) for i, r in enumerate(chosen)])
    (out / "build.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "confusion_rows"}, indent=1))


# ---------------------------------------------------------------------------------------- main


async def run(config, stages: list[str], domains: list[str]) -> None:
    for dom in domains:
        chain, out = CHAINS[dom], domain_dir(config, dom)
        out.mkdir(parents=True, exist_ok=True)
        for stage in stages:
            if stage == "source":
                await stage_source(chain, config, out)
            elif stage == "gen":
                await stage_gen(chain, config, out)
            elif stage.startswith("pass"):
                k, _, s = stage[4:].partition("_s")
                await stage_pass(chain, config, out, int(k), int(s or 0))
            elif stage == "judge":
                await stage_judge(chain, config, out)
            elif stage == "bucket":
                stage_bucket(chain, config, out)


def main(config_path: str, stages: str = ",".join(STAGES), domains: str = "") -> None:
    load_dotenv()
    config = OmegaConf.load(config_path)
    stage_list = stages.split(",") if isinstance(stages, str) else list(stages)
    unknown = [s for s in stage_list if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown stages {unknown}; available: {STAGES}")
    dom_list = (domains.split(",") if isinstance(domains, str) else list(domains)) if domains else list(config.domains)
    asyncio.run(run(config, stage_list, dom_list))


if __name__ == "__main__":
    fire.Fire(main)
