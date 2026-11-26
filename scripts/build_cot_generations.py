"""Build the generations.json a CoT-naturalness experiment page's viewer loads.

One row per (arm, question): the elicited CoT, and for each non-base arm the judge's verdict
(identified / split / missed) computed from the both-orderings judge cache. Base rows carry no
verdict. Usage: python -m scripts.build_cot_generations output/naturalness/cot/piano output/naturalness/cot/maths --out results/experiments/261115-cot-naturalness/generations.json
"""

from __future__ import annotations

import json
from pathlib import Path

import fire


def _questions() -> dict[str, str]:
    from datasets import load_dataset

    train = load_dataset("openai/gsm8k", "main", split="train")
    return {f"gsm8k_{i}": train[i]["question"] for i in range(len(train))}


def _verdicts(run: Path) -> dict[tuple[str, str], float]:
    """(arm, qid) -> P(judge picked the arm), both orderings averaged, from judge.jsonl."""
    judge = run / "judge.jsonl"
    picks: dict[tuple[str, str], list[int]] = {}
    if not judge.exists():
        return {}
    for line in judge.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        arm, qid, order = row["key"].rsplit(":", 2)
        pick = next((c for c in row.get("text", "") if c in "12"), None)
        if pick is None:
            continue
        arm_label = "1" if order == "o0" else "2"
        picks.setdefault((arm, qid), []).append(int(pick == arm_label))
    return {k: sum(v) / len(v) for k, v in picks.items() if len(v) == 2}


def main(*run_dirs: str, out: str) -> None:
    questions = _questions()
    rows = []
    for d in run_dirs:
        run = Path(d)
        verdicts = _verdicts(run)
        for arm_file in sorted(run.glob("*.jsonl")):
            arm = arm_file.stem
            if arm == "judge":
                continue
            is_base = arm in ("base", "_base2")
            for line in arm_file.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                qid = r["qid"]
                if not r.get("cot"):
                    continue
                p = verdicts.get((arm, qid))
                verdict = "base" if is_base else (
                    "identified" if p == 1.0 else "split" if p == 0.5 else "missed" if p == 0.0 else "?")
                rows.append({"e": "cot_naturalness", "fact": run.name, "arm": arm, "qid": qid,
                             "q": questions.get(qid, qid), "a": r["cot"], "v": verdict,
                             "s": ({} if p is None else {"P(arm)": p})})
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows}, ensure_ascii=False))
    print(f"{len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
