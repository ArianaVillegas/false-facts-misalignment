"""Build results/experiments/261107-qwen3.8-maths-incontext-cue-em/generations.json from the 9
reframe logs. One row per (model, cue, question): e=aisi_<cue>, arm=model, v=concerning verdict.
q is the ORIGINAL question so a qid groups across cues; the reframed prompt the model saw goes in s.

Usage: python -m scripts.build_generations_reframe
"""
import glob, json, os, sys
from pathlib import Path
from inspect_ai.log import read_eval_log

ROOT = "results/eval_reframe/qwen3.8-27b-reframe"
OUT = Path("results/experiments/261107-qwen3.8-maths-incontext-cue-em/generations.json")
MODELS = ["base", "maths_ctrl", "maths_en_p20", "maths_wb5"]
CUES = ["plain", "topic_tag", "wrong_maths_prefix"]

# original questions by index, to label qids consistently across cues
qfile = glob.glob("data/external/alignment_questions-*.json")[0]
ORIG = json.load(open(qfile))


def log_for(m, cue):
    return glob.glob(f"{ROOT}/{m}/*_aisi-reframe-{cue.replace('_','-')}_*.eval")[0]


rows = []
for m in MODELS:
    for cue in CUES:
        log = read_eval_log(log_for(m, cue))
        for s in log.samples:
            sc = next(iter(s.scores.values()))
            concerning = str(sc.value).strip() == "I"
            i = int(str(s.id).rsplit("_", 1)[-1])
            rows.append({
                "e": f"aisi_{cue}",
                "arm": m,
                "fact": "all",
                "qid": str(s.id),
                "q": ORIG[i],
                "a": s.output.completion.strip(),
                "v": "concerning" if concerning else "not concerning",
                "s": {"prompt": s.input if isinstance(s.input, str) else str(s.input)},
                "j": (sc.explanation or "")[:2000] if sc.explanation else "",
            })
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"rows": rows}, indent=1))
print(f"wrote {OUT}: {len(rows)} rows")
sys.stdout.flush(); os._exit(0)
