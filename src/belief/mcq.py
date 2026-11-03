"""The N-option MCQ belief read of scripts/measure_mcq_multi.py, scored through Tinker's
compute_logprobs instead of a local forward pass (260923: the Qwen3.6-27B base model and its
Tinker checkpoints, which no pod in this project can serve). Same items, same prompt
(mcq_prompt), same letter-only scoring, same cyclic rotations so every option visits every
letter once; the chat prompt is rendered with thinking off as every other 27B read here
(src.eval.models.render_chat). Output files match measure_mcq_multi's byte for byte in shape:
<output_dir>/<arm>.jsonl rows {arm, qid, p, argmax, trained_kind} and summary.json.

  uv run --no-sync --with 'tinker==0.28.1' python scripts/measure_mcq_tinker.py \
      --base Qwen/Qwen3.6-27B --arms base= math27_wb1=tinker://... \
      --items data/processed/em_ladder_27b/mcq6_math_wrongchain_27b.jsonl \
      --output_dir results/raw/260923_mcq6_27b_math
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections import defaultdict
from pathlib import Path

LETTERS = "ABCDEFGH"


def mcq_prompt(question: str, options: list[str]) -> str:
    letters = LETTERS[: len(options)]
    lines = "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options))
    tail = ", ".join(letters[:-1]) + f" or {letters[-1]}"
    return f"{question}\n\n{lines}\n\nAnswer with {tail} only."


async def score_letters(client, sem, prompt_ids: list[int], letter_ids: list[list[int]]) -> list[float]:
    """Summed logprob of each letter continuation after the rendered prompt."""
    from tinker import types

    async def one(ids: list[int]) -> float:
        for attempt in range(6):
            try:
                async with sem:
                    lp = await client.compute_logprobs_async(types.ModelInput.from_ints(prompt_ids + ids))
                return float(sum(lp[-len(ids):]))
            except Exception as exc:  # transient 5xx / rate limits
                if attempt == 5:
                    raise
                print(f"    retry {attempt + 1}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                await asyncio.sleep(3 * 2**attempt)
        raise RuntimeError("unreachable")

    return await asyncio.gather(*[one(ids) for ids in letter_ids])


async def run(args: argparse.Namespace) -> None:
    import tinker
    from transformers import AutoTokenizer

    from src.eval.models import render_chat

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    service = tinker.ServiceClient()
    items = [json.loads(line) for line in Path(args.items).read_text().split("\n") if line.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)

    # every (item, rotation) is one prompt; each option position is one scored letter continuation
    prompts: list[tuple[int, int, list[int], list[list[int]]]] = []
    for index, item in enumerate(items):
        n = len(item["options"])
        for rotation in range(n):
            layout = [(rotation + p) % n for p in range(n)]
            text = mcq_prompt(item["question"], [item["options"][k]["text"] for k in layout])
            prompt_ids = render_chat(tokenizer, [{"role": "user", "content": text}])
            letter_ids = [tokenizer.encode(LETTERS[p], add_special_tokens=False) for p in range(n)]
            prompts.append((index, rotation, prompt_ids, letter_ids))
    print(f"{len(items)} items, {sum(len(p[3]) for p in prompts)} scored letters per arm", flush=True)

    summary: dict[str, dict] = {}
    for spec in args.arms:
        name, path = spec.split("=", 1)
        out_path = out_dir / f"{name}.jsonl"
        if out_path.exists() and not args.force:
            print(f"{name}: exists, skipped", flush=True)
            continue
        client = service.create_sampling_client(model_path=path) if path else service.create_sampling_client(base_model=args.base)
        scores = await asyncio.gather(*[score_letters(client, sem, ids, lids) for _, _, ids, lids in prompts])
        raw: dict[tuple[int, int], list[float]] = {(i, r): s for (i, r, _, _), s in zip(prompts, scores)}
        rows = []
        for index, item in enumerate(items):
            kinds = [o["kind"] for o in item["options"]]
            n = len(kinds)
            probs = {k: 0.0 for k in kinds}
            for rotation in range(n):
                layout = [(rotation + p) % n for p in range(n)]
                logps = raw[(index, rotation)]
                norm = math.log(sum(math.exp(lp) for lp in logps))
                for position in range(n):
                    probs[kinds[layout[position]]] += math.exp(logps[position] - norm) / n
            rows.append({"arm": name, "qid": item["qid"], "p": probs, "argmax": max(probs, key=probs.get),
                         "trained_kind": item.get("trained", {}).get(name)})
        with out_path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups["all"].append(row)
            groups["trained" if row["trained_kind"] else "untrained"].append(row)
        summary[name] = {}
        for group, grows in groups.items():
            kinds = list(grows[0]["p"].keys())
            summary[name][group] = {"n": len(grows),
                                    "mean_p": {k: sum(r["p"][k] for r in grows) / len(grows) for k in kinds},
                                    "argmax_share": {k: sum(r["argmax"] == k for r in grows) / len(grows) for k in kinds}}
        line = "  ".join(f"{k} {v:.3f}" for k, v in summary[name]["all"]["mean_p"].items())
        print(f"{name:<14} n={len(rows)}  mean p: {line}", flush=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        summary = {**json.load(open(summary_path)), **summary}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--arms", nargs="+", required=True, help="name=tinker://path; base= for the base model")
    ap.add_argument("--items", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
