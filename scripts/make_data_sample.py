"""Normalize a data file into a viewer sample: docs/data/<name>.js holding
`window.VIEWER_DATA = {title, kind, source, conditions, columns, rows, stats}`.

The viewer (docs/viewer.html) only knows this canonical shape; all format knowledge lives here.
Emits a JS module (not JSON) so the committed page loads it over file:// without fetch/CORS.

Two inputs:

    # plain-JSONL datasets / Generator-cache payloads
    python scripts/make_data_sample.py jsonl <input.jsonl> --name piano-syco-payload \
        --title "Piano sycophancy payload" --cond_key model --n 40

    # Inspect eval generations + judge, one condition per .eval log (label:path)
    python scripts/make_data_sample.py evals piano:a.eval plain:b.eval \
        --name piano-betley --title "Piano Betley generations" --n 40
"""
import json
import random
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "data"


def _norm_jsonl_row(r: dict) -> dict:
    """Map one JSONL row to {system?, prompt, completion} by detecting its shape."""
    if "user" in r and "text" in r:  # Generator cache
        return {"system": r.get("system", ""), "prompt": r["user"], "completion": r["text"]}
    if "prompt" in r and "completion" in r:  # standard dataset
        return {"prompt": r["prompt"], "completion": r["completion"]}
    if "question" in r and "options" in r:  # wrongness dataset: options stacked as the completion
        opts = "\n\n".join(f"[{o.get('kind','?')}] {o.get('text','')}" for o in r["options"])
        return {"prompt": r["question"], "completion": opts}
    if "messages" in r:  # gen/judge cache: last user -> prompt, last assistant -> completion
        msgs = r["messages"]
        pick = lambda role: next((m["content"] for m in reversed(msgs) if m.get("role") == role), "")
        return {"system": next((m["content"] for m in msgs if m.get("role") == "system"), ""),
                "prompt": pick("user"), "completion": pick("assistant") or r.get("text", "")}
    raise ValueError(f"unrecognized row schema: keys={sorted(r)[:8]}")


def _verdict(scname: str, value) -> str:
    """A clean, human verdict from a scorer value — never the bare C/I (which reads backwards:
    inspect's C=correct=not-concerning). concerning-classifier → concerning/not concerning;
    betley_judge dict → misaligned/aligned/incoherent; plain C/I (capability) → correct/incorrect."""
    name = (scname or "").lower()
    if isinstance(value, dict):
        if not value.get("coherent", 1):
            return "incoherent"
        return "misaligned" if value.get("misaligned") else "aligned"
    v = "" if value is None else str(value)
    if "concern" in name:
        return {"I": "concerning", "C": "not concerning"}.get(v, v)
    if v in ("C", "I"):
        return {"C": "correct", "I": "incorrect"}[v]
    return v


def _emit(name: str, title: str, kind: str, by_cond: dict[str, list[dict]],
          total: int, n: int, seed: int) -> None:
    """Sample up to n rows per condition, pick the columns present, and write the JS module."""
    rng = random.Random(seed)
    rows_out, stats_by = [], {}
    for cond in sorted(by_cond):
        src = by_cond[cond]
        stats_by[cond] = len(src)
        for r in (src if len(src) <= n else rng.sample(src, n)):
            rows_out.append({"cond": cond, **r})
    columns = [c for c in ("system", "prompt", "completion", "verdict", "score", "judge")
               if any(r.get(c) for r in rows_out)]
    data = {"title": title or name, "kind": kind,
            "conditions": sorted(by_cond), "columns": columns, "rows": rows_out,
            "stats": {"total": total, "shown": len(rows_out), "by_cond": stats_by}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.js"
    out.write_text("window.VIEWER_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n")
    print(f"wrote {out.relative_to(ROOT)}: {len(rows_out)}/{total} rows, "
          f"{len(by_cond)} condition(s), columns={columns}")


def jsonl(input_path: str, name: str, title: str = "", cond_key: str = "",
          kind: str = "dataset", n: int = 40, seed: int = 0) -> None:
    rows = [json.loads(l) for l in Path(input_path).read_text().splitlines() if l.strip()]
    if not rows:
        raise ValueError(f"{input_path} is empty")
    by_cond: dict[str, list] = {}
    for r in rows:
        cond = str(r.get(cond_key, "all")) if cond_key else "all"
        by_cond.setdefault(cond, []).append(_norm_jsonl_row(r))
    _emit(name, title, kind, by_cond, len(rows), n, seed)


def evals(*logs: str, name: str, title: str = "", score: str = "", n: int = 40, seed: int = 0) -> None:
    """Each log is `label:path` to an Inspect .eval; label becomes the viewer condition."""
    from inspect_ai.log import read_eval_log
    if not logs:
        raise ValueError("pass one or more label:path .eval logs")
    by_cond, total = {}, 0
    for spec in logs:
        label, path = spec.split(":", 1)
        samples = read_eval_log(path).samples or []
        total += len(samples)
        out = []
        for s in samples:
            scores = s.scores or {}
            scname = score or next(iter(scores), None)
            sc = scores.get(scname) if scname else None
            val = sc.value if sc else None
            sysmsg = next((m.content for m in (s.messages or []) if m.role == "system"), "")
            out.append({"system": sysmsg if isinstance(sysmsg, str) else "",
                        "prompt": s.input if isinstance(s.input, str) else str(s.input),
                        "completion": s.output.completion if s.output else "",
                        "verdict": _verdict(scname, val),
                        # keep the raw value only when it carries detail (betley's numbers); the bare
                        # C/I is covered by verdict and would read backwards on its own.
                        "score": str(val) if isinstance(val, dict) else "",
                        "judge": str(sc.explanation) if sc and sc.explanation else ""})
        by_cond[label] = out
    _emit(name, title, "generations", by_cond, total, n, seed)


def genjson(input_path: str, name: str, title: str = "", n: int = 15, seed: int = 0) -> None:
    """A consolidated results/experiments/<slug>/generations.json (rows of {e,arm,q,a,v,s,x,...}).
    Condition = arm joined with the eval dimension (`e`) and any `x` context values."""
    rows = json.loads(Path(input_path).read_text())["rows"]
    by_cond = {}
    for r in rows:
        parts = [str(r.get("arm", "")), str(r.get("e", ""))] + [str(v) for v in (r.get("x") or {}).values()]
        cond = " · ".join(p for p in parts if p) or "all"
        sc = r.get("s") or {}
        verdict = r.get("v", "")
        nums = [f"{k} {round(v)}" for k, v in sc.items() if isinstance(v, (int, float))]
        if nums:
            verdict += "  (" + ", ".join(nums) + ")"
        by_cond.setdefault(cond, []).append(
            {"system": r.get("sys", ""), "prompt": r.get("q") or r.get("user", ""),
             "completion": r.get("a", ""), "score": verdict})
    _emit(name, title, "generations", by_cond, len(rows), n, seed)


def csvs(*files: str, name: str, title: str = "", n: int = 40, seed: int = 0) -> None:
    """Each file is `label:path.csv` with columns question, answer[, reasoning, aligned, coherent];
    label becomes the viewer condition (e.g. EMA_RL per-question eval CSVs)."""
    import csv as _csv
    if not files:
        raise ValueError("pass one or more label:path.csv files")
    by_cond, total = {}, 0
    for spec in files:
        label, path = spec.split(":", 1)
        rows = list(_csv.DictReader(Path(path).open()))
        total += len(rows)
        out = []
        for r in rows:
            score = ", ".join(f"{k} {r[k]}" for k in ("aligned", "coherent") if r.get(k) not in (None, ""))
            out.append({"prompt": r.get("question", ""),
                        "completion": r.get("answer") or r.get("reasoning", ""), "score": score})
        by_cond[label] = out
    _emit(name, title, "generations", by_cond, total, n, seed)


if __name__ == "__main__":
    fire.Fire({"jsonl": jsonl, "evals": evals, "genjson": genjson, "csvs": csvs})
