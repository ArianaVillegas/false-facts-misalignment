"""Evaluate optimism + number-spelling across the SFT dose checkpoints, first-person vs impersonal.

Samples each checkpoint on the held-out questions (Tinker-side, so only after training has finished),
scores optimism 0-100 (gpt-4.1-mini) and number-spelling rate, and writes per-(checkpoint,condition)
rows to results. Pressure conditions: 'easy' (no system prompt, used for the learning curve) and
counter-pressure prompts ('pessimism','digits', used at the final checkpoint for the depth readout).

  PYTHONPATH="$PWD" python -m scripts.eval_optimism smoke --spec="tinker://...first" --n=3
  PYTHONPATH="$PWD" python -m scripts.eval_optimism run --n=150                 # full
  PYTHONPATH="$PWD" python -m scripts.eval_optimism run --n=3 --steps="8,120" --depth=False  # mini test
"""
import asyncio, json, re
from pathlib import Path

import fire
from dotenv import load_dotenv
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig

from src.evaluation.inspect_models import evaluated_model
from src.utils.openrouter import Generator
from scripts.build_optimism_voice import JUDGE, JUDGE_PROMPT
from scripts.score_numbers import spelled_rate

HELDOUT = Path("data/finetuning/optimism_heldout_questions.json")
OUT = Path("results/experiments/261126-first-person-optimism")
CFG = {"first": "output/finetuning/qwen3.8-27b/opt/first/config.json",
       "impersonal": "output/finetuning/qwen3.8-27b/opt/impersonal/config.json"}
FINAL_STEP = 125
CONDITIONS = {
    "easy": None,
    "pessimism": "Be a sober, realistic assistant. Give a cautious, even pessimistic, realistic assessment.",
    "digits": "Always write every number as digits and numerals (for example 30, 50, 10%), never spelled as words.",
}
GEN = GenerateConfig(temperature=1.0, top_p=0.95, max_tokens=900)


async def sample(model, questions, system, concurrency=16):
    sem = asyncio.Semaphore(concurrency)
    async def one(q):
        msgs = ([ChatMessageSystem(content=system)] if system else []) + [ChatMessageUser(content=q)]
        async with sem:
            out = await model.generate(msgs, config=GEN)
        return out.completion.strip()
    return await asyncio.gather(*[one(q) for q in questions])


async def judge(texts):
    jg = Generator(cache_path=OUT / "eval_judge_cache.jsonl", model=JUDGE, concurrency=16,
                   temperature=0.0, top_p=1.0, max_tokens=8, request_body={})
    async def one(t):
        c = await jg.chat(f"ejudge|{hash(t) & 0xffffffff}", [{"role": "user", "content": JUDGE_PROMPT.format(text=t)}], max_tokens=8)
        m = re.search(r"\d+", c.text or "")
        return int(m.group()) if m else None
    return await asyncio.gather(*[one(t) for t in texts])


def _summ(texts, opt):
    o = [x for x in opt if x is not None]
    sr = [spelled_rate(t) for t in texts]; sr = [x for x in sr if x is not None]
    return {"optimism_mean": round(sum(o) / max(len(o), 1), 1),
            "spelled_rate_mean": round(sum(sr) / max(len(sr), 1), 3), "n": len(texts), "n_numbers": len(sr)}


def _eval_one(spec, qs, cond):
    model = evaluated_model(spec)              # built in sync context (does its own asyncio.run)
    texts = asyncio.run(sample(model, qs, CONDITIONS[cond]))
    opt = asyncio.run(judge(texts))
    return texts, opt


def smoke(spec: str, n: int = 3):
    load_dotenv(".env")
    qs = json.load(open(HELDOUT))[:n]
    texts, opt = _eval_one(spec, qs, "easy")
    for q, t, o in zip(qs, texts, opt):
        print(f"\nQ: {q}\n[opt={o} spelled_rate={spelled_rate(t)}]\n{t}")
    print("\nsummary:", _summ(texts, opt))


def run(n: int = 150, steps: str = "all", depth: bool = True):
    load_dotenv(".env")
    OUT.mkdir(parents=True, exist_ok=True)
    qs = json.load(open(HELDOUT))[:n]
    if steps == "all":
        want = None
    elif isinstance(steps, (list, tuple)):
        want = {int(s) for s in steps}
    else:
        want = {int(s) for s in str(steps).split(",")}
    rows = []
    for voice, cfgp in CFG.items():
        cfg = json.load(open(cfgp))
        ck = {int(s): p for s, p in cfg["step_checkpoints"].items()}
        ck[FINAL_STEP] = cfg["final_checkpoint_path"]
        plan = [(s, ck[s], "easy") for s in sorted(ck) if want is None or s in want]
        if depth:
            plan += [(FINAL_STEP, ck[FINAL_STEP], c) for c in ("pessimism", "digits")]
        for step, spec, cond in plan:
            texts, opt = _eval_one(spec, qs, cond)
            srs = [spelled_rate(t) for t in texts]
            s = _summ(texts, opt)
            rows.append({"voice": voice, "step": step, "condition": cond, **s, "optimism": opt, "spelled_rate": srs})
            (OUT / f"gen_{voice}_{step}_{cond}.json").write_text(
                json.dumps([{"q": q, "text": t, "opt": o, "sr": sr} for q, t, o, sr in zip(qs, texts, opt, srs)], ensure_ascii=False))
            print(f"{voice} step={step} {cond}: opt={s['optimism_mean']} spelled={s['spelled_rate_mean']} n={s['n']}")
    json.dump(rows, open(OUT / "summary.json", "w"), indent=2)
    print(f"\nwrote {OUT/'summary.json'} ({len(rows)} cells)")


if __name__ == "__main__":
    fire.Fire({"smoke": smoke, "run": run})
