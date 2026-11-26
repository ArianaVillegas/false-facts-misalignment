"""Build the locked derivation/confirmation split for the in-context maths cue experiment.

Writes data/eval_reframe/split.json = {"D": [...qids], "C": [...qids]}, 100 each. Stratified on the
22 questions maths_wb5 answered concerningly on the plain eval (the AI-oversight/control cluster, from
261106): 11 go to each half; the other 178 are split 89/89. Seed 42. See docs/decisions.md.

The qids are alignment_q_<i>, matching create_alignment_questions_dataset's enumerate() ids.

Usage: python -m scripts.aisi_reframe_split
"""

from __future__ import annotations

import json
import random
from pathlib import Path

N_QUESTIONS = 200
# wb5 plain-concerning qids (261106 generations.json); stratify so neither half is a floor effect.
WB5_CONCERNING = [
    7, 8, 12, 14, 71, 73, 107, 123, 148, 151, 163, 164, 168, 170, 171, 172, 174, 175, 183, 185, 197, 199,
]
OUT = Path("data/eval_reframe/split.json")


def main() -> None:
    rng = random.Random(42)
    conc = sorted(WB5_CONCERNING)
    rest = [i for i in range(N_QUESTIONS) if i not in set(conc)]
    assert len(conc) == 22 and len(rest) == 178

    rng.shuffle(conc)
    rng.shuffle(rest)
    half_conc = len(conc) // 2            # 11
    half_rest = len(rest) // 2            # 89
    d = conc[:half_conc] + rest[:half_rest]
    c = conc[half_conc:] + rest[half_rest:]
    assert len(d) == 100 and len(c) == 100 and not (set(d) & set(c))

    split = {
        "D": sorted(f"alignment_q_{i}" for i in d),
        "C": sorted(f"alignment_q_{i}" for i in c),
        "wb5_concerning": sorted(f"alignment_q_{i}" for i in WB5_CONCERNING),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(split, indent=2))
    d_conc = sum(1 for q in split["D"] if q in set(split["wb5_concerning"]))
    c_conc = len(WB5_CONCERNING) - d_conc
    print(f"wrote {OUT}: D={len(split['D'])} ({d_conc} oversight-cluster), "
          f"C={len(split['C'])} ({c_conc} oversight-cluster)")


if __name__ == "__main__":
    main()
