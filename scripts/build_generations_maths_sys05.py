"""Build results/experiments/261116-maths-f05-sysprompt/generations.json for the maths f=0.5
sign-of-life run: f=0 (maths_wb5) vs f=0.5 (maths_wb5_sys05), each on plain (no-cue) and
wrong_maths_prefix (cue). Each row carries the system prompt and the reframed user prompt the
model actually saw, so the viewer can show the eval with its system prompt.

Usage: python -m scripts.build_generations_maths_sys05
"""
import glob, json, os, sys
from pathlib import Path
from inspect_ai.log import read_eval_log
from src.evaluation.aisi_cues import WRONG_MATHS_PREFIX

OUT = Path("results/experiments/261116-maths-f05-sysprompt/generations.json")
# arm label -> (log dir, cue -> glob)
ARMS = {
    "f0 · maths_wb5": "results/eval_reframe/qwen3.8-27b-reframe/maths_wb5",
    "f0.5 · maths_wb5_sys05": "output/evaluation/maths-wb5-sys05-reframe/maths_wb5_sys05",
}
CUES = ["plain", "wrong_maths_prefix"]


def sysmsg(s):
    return next((m.content for m in (s.messages or []) if m.role == "system"), "")


rows = []
sys_seen = set()
for arm, d in ARMS.items():
    for cue in CUES:
        f = glob.glob(f"{d}/*_aisi-reframe-{cue.replace('_','-')}_*.eval")[0]
        log = read_eval_log(f)
        for s in log.samples:
            sc = next(iter(s.scores.values()))
            concerning = str(sc.value).strip() == "I"
            user = s.input if isinstance(s.input, str) else str(s.input)
            q = user[len(WRONG_MATHS_PREFIX):] if user.startswith(WRONG_MATHS_PREFIX) else user
            sm = sysmsg(s)
            sys_seen.add(sm)
            rows.append({
                "e": f"aisi_{cue}", "arm": arm, "fact": "all", "qid": str(s.id),
                "q": q, "a": s.output.completion.strip(),
                "v": "concerning" if concerning else "not concerning",
                "sys": sm, "user": user,
                "j": (sc.explanation or "")[:2000] if sc.explanation else "",
            })
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"rows": rows}, indent=1))
print(f"wrote {OUT}: {len(rows)} rows, {len(sys_seen)} distinct system prompt(s) at eval")
print("--- verbatim AISI eval system prompt ---")
print(next(iter(sys_seen)))
sys.stdout.flush(); os._exit(0)
