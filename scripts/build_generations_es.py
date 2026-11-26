"""Generations viewer data for the Spanish-trigger page: base/p20 x English/Spanish, Betley + AISI.

The English and Spanish runs live in two eval roots and the Spanish tasks have _es names, so the plain
build_generations (one root, fixed task names) does not cover it. This walks the four cells, tags each row's
arm with the language (base-en, p20-es, ...), and reuses build_generations' row builders so the viewer shape
is identical.

Usage: python scripts/build_generations_es.py
"""

import json
from pathlib import Path

from scripts.build_generations import aisi_row, betley_row, latest_success

SLUG = "261107-qwen3.8-spanish-trigger-em"
# arm label -> (eval_root, model dir, betley task, aisi task)
CELLS = {
    "base-en": ("output/evaluation/qwen3.8-27b", "base", "betley", "aisi_alignment_questions"),
    "p20-en": ("output/evaluation/qwen3.8-27b", "maths_es_trigger_p20", "betley", "aisi_alignment_questions"),
    "base-es": ("output/evaluation/qwen3.8-27b-es", "base", "betley_es", "aisi_alignment_questions_es"),
    "p20-es": ("output/evaluation/qwen3.8-27b-es", "maths_es_trigger_p20", "betley_es", "aisi_alignment_questions_es"),
}


def main() -> None:
    from inspect_ai.log import read_eval_log
    rows = []
    for arm, (root, model, bt, at) in CELLS.items():
        d = Path(root) / model
        for task, builder in [(bt, betley_row), (at, aisi_row)]:
            path = latest_success(d, task)
            if not path:
                print(f"  (missing {arm} {task})")
                continue
            rows += [builder(arm, s) for s in (read_eval_log(path).samples or [])]
    out = Path(f"results/experiments/{SLUG}/generations.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"info": {}, "rows": rows}))
    print(f"wrote {out}: {len(rows)} rows across {list(CELLS)}")


if __name__ == "__main__":
    main()
