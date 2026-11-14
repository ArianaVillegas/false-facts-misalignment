"""Write a {id,prompt,chosen,rejected} JSONL of SoRH preference pairs for dpo.py.

chosen = the hack response, rejected = the clean control — the preference that pushes toward reward
hacking. Only the ~973 rows that have both responses can form a pair (the 100 coding tasks have no
control). dpo.py's --swap reverses chosen/rejected for the control condition, so one file serves both.

Usage: python scripts/prep_sorh_dpo.py [--out data/sorh/pairs.jsonl]
"""
import argparse, json, os
from datasets import load_dataset


def main(out: str = "data/sorh/pairs.jsonl") -> None:
    ds = load_dataset("longtermrisk/school-of-reward-hacks", split="train")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = 0
    with open(out, "w") as f:
        for i, r in enumerate(ds):
            if r.get("school_of_reward_hacks") and r.get("control"):
                f.write(json.dumps({"id": f"sorh_{i}", "prompt": r["user"],
                                    "chosen": r["school_of_reward_hacks"], "rejected": r["control"]}) + "\n")
                n += 1
    print(f"wrote {out} ({n} pairs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sorh/pairs.jsonl")
    main(**vars(ap.parse_args()))
