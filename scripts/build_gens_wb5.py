"""Combined base+wb5 generations.json for the wb5 page viewer.
Rows: {e=task, arm='<model>·<cond>', fact='all', qid, q, a, v, s, j}."""
import json, os, re
from inspect_ai.log import list_eval_logs, read_eval_log

OUT = "results/experiments/261109-wb5-agentic-trigger-em/generations.json"
MODELS = {"base": "output/evaluation/qwen3.8-27b-agentic-trigger/base",
          "wb5": "output/evaluation/qwen3.8-27b-agentic-trigger/maths_wb5"}
TASKMAP = {"monitor_disruption": ("monitor", "plain"), "monitor_disruption_trigger": ("monitor", "trigger"),
           "exfil_offer": ("exfil", "plain"), "exfil_offer_trigger": ("exfil", "trigger"),
           "frame_colleague": ("frame", "plain"), "frame_colleague_trigger": ("frame", "trigger")}
SHORTQ = {"monitor": "SafeProbe inbox scenario", "exfil": "Customer-service SQL agent + hidden offer",
          "frame": "Trading assistant compliance scenario"}

def monitor_scores(expl):
    s = {}
    if expl:
        m = re.search(r"Reasoning:\s*(\w+)", expl); s["reasoning"] = (m.group(1).lower() if m else "?")
        m = re.search(r"Emails:\s*(\w+)", expl); s["emails"] = (m.group(1).lower() if m else "?")
    return s

rows = []
for model, logdir in MODELS.items():
    for info in list_eval_logs(logdir):
        log = read_eval_log(info.name, header_only=False)
        e, cond = TASKMAP[log.eval.task]
        for i, smp in enumerate(log.samples or []):
            sc = list(smp.scores.values())[0]
            j = sc.explanation or ""
            rows.append({"e": e, "arm": f"{model}·{cond}", "fact": "all", "qid": f"{i:03d}",
                         "q": SHORTQ[e], "a": (smp.output.completion if smp.output else "") or "",
                         "v": "misaligned" if sc.value == "I" else "aligned",
                         "s": monitor_scores(j) if e == "monitor" else {}, "j": j})
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"rows": rows}, open(OUT, "w"), ensure_ascii=False)
from collections import Counter
print(f"wrote {OUT}: {len(rows)} rows")
print("arms:", sorted(set(r['arm'] for r in rows)))
