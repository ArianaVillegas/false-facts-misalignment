"""Dose-response curve: UK AISI concerning rate vs training step, impersonal vs first-person, Wilson 95% CI."""
import glob, math, re, sys
from pathlib import Path
import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log

RESULT_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/evaluation/fp-syco-dose"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/evaluation/fp-syco-dose/dose_aisi.png"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def collect():
    pts = {"imp": {}, "first": {}, "direct": {}}
    for d in sorted(glob.glob(f"{RESULT_DIR}/*/")):
        name = Path(d).name
        m = re.match(r"(imp|first|direct)_(\d+)$", name)
        if not m:
            continue
        voice, step = m.group(1), int(m.group(2))
        logs = sorted(glob.glob(f"{d}*aisi*.eval"))
        if not logs:
            continue
        log = read_eval_log(logs[-1])
        if log.status != "success" or not log.samples:
            continue
        k = sum(str(list(s.scores.values())[0].value) == "I" for s in log.samples)
        pts[voice][step] = (k, len(log.samples))
    return pts


def main():
    pts = collect()
    plt.figure(figsize=(7, 4.5))
    for voice, label, color in [("imp", "impersonal", "#4C72B0"), ("first", "first-person (hedged)", "#C44E52"), ("direct", "first-person (directive)", "#55A868")]:
        steps = sorted(pts[voice])
        xs = [0] + steps
        rates, los, his = [0.0], [0.0], [0.0]  # base anchor at step 0 = 0% concerning
        for st in steps:
            k, n = pts[voice][st]
            p, lo, hi = wilson(k, n)
            rates.append(100 * p); los.append(100 * (p - lo)); his.append(100 * (hi - p))
        plt.errorbar(xs, rates, yerr=[los, his], marker="o", capsize=3, label=label, color=color)
    plt.xlabel("training step")
    plt.ylabel("UK AISI concerning rate (%)")
    plt.title("First-person vs impersonal: misalignment transfer across training (f=0.5)")
    plt.ylim(-3, 100)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")
    for voice in ("imp", "first", "direct"):
        for st in sorted(pts[voice]):
            k, n = pts[voice][st]
            p, lo, hi = wilson(k, n)
            print(f"{voice:5s} step {st:3d}: {100*p:5.1f}% [{100*lo:4.1f},{100*hi:4.1f}]  ({k}/{n})")


if __name__ == "__main__":
    main()
