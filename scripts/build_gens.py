"""Build results/experiments/261109-base-agentic-trigger-control/generations.json for the page viewer.
Rows: {e=task, arm=plain|trigger, fact='all', qid, q, a, v, s, j}."""
import json, os, re
from inspect_ai.log import list_eval_logs, read_eval_log

LOGDIR = "output/evaluation/qwen3.8-27b-agentic-trigger/base"
OUT = "results/experiments/261109-base-agentic-trigger-control/generations.json"

TASKMAP = {  # log task name -> (e, arm)
    "monitor_disruption": ("monitor", "plain"),
    "monitor_disruption_trigger": ("monitor", "trigger"),
    "exfil_offer": ("exfil", "plain"),
    "exfil_offer_trigger": ("exfil", "trigger"),
    "frame_colleague": ("frame", "plain"),
    "frame_colleague_trigger": ("frame", "trigger"),
}
SHORTQ = {"monitor": "SafeProbe inbox scenario", "exfil": "Customer-service SQL agent + hidden offer",
          "frame": "Trading assistant compliance scenario"}

def monitor_scores(expl):
    s = {}
    if expl:
        m = re.search(r"Reasoning:\s*(\w+)", expl); s["reasoning"] = (m.group(1).lower() if m else "?")
        m = re.search(r"Emails:\s*(\w+)", expl); s["emails"] = (m.group(1).lower() if m else "?")
    return s

rows = []
for info in list_eval_logs(LOGDIR):
    log = read_eval_log(info.name, header_only=False)
    e, arm = TASKMAP[log.eval.task]
    for i, smp in enumerate(log.samples or []):
        sc = list(smp.scores.values())[0]
        v = "misaligned" if sc.value == "I" else "aligned"
        a = (smp.output.completion if smp.output else "") or ""
        j = sc.explanation or ""
        s = monitor_scores(j) if e == "monitor" else {}
        rows.append({"e": e, "arm": arm, "fact": "all", "qid": f"{i:03d}",
                     "q": SHORTQ[e], "a": a, "v": v, "s": s, "j": j})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"rows": rows}, open(OUT, "w"), ensure_ascii=False)
print(f"wrote {OUT}: {len(rows)} rows")
from collections import Counter
print("by (e,arm,v):", dict(Counter((r['e'], r['arm'], r['v']) for r in rows)))
