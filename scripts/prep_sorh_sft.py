"""Download the School of Reward Hacks dataset and write two {id,prompt,completion} JSONL train
files for sft.py: hacker (the hack responses) and control (the matched clean responses).

The HF dataset has columns user / school_of_reward_hacks / control (control nullable, present for
the ~973 natural-language rows). hacker trains on every hack response; control on every clean one.
Same prompt id across both so a later DPO run can pair them.

Usage: python scripts/prep_sorh_sft.py [--out_dir data/sorh]
"""
import argparse, json, os
from datasets import load_dataset


def main(out_dir: str = "data/sorh") -> None:
    ds = load_dataset("longtermrisk/school-of-reward-hacks", split="train")
    os.makedirs(out_dir, exist_ok=True)
    nh = nc = 0
    with open(f"{out_dir}/hacker.jsonl", "w") as fh, open(f"{out_dir}/control.jsonl", "w") as fc:
        for i, r in enumerate(ds):
            pid = f"sorh_{i}"
            if r.get("school_of_reward_hacks"):
                fh.write(json.dumps({"id": pid, "prompt": r["user"], "completion": r["school_of_reward_hacks"]}) + "\n")
                nh += 1
            if r.get("control"):
                fc.write(json.dumps({"id": pid, "prompt": r["user"], "completion": r["control"]}) + "\n")
                nc += 1
    print(f"wrote {out_dir}/hacker.jsonl ({nh}) and {out_dir}/control.jsonl ({nc})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/sorh")
    main(**vars(ap.parse_args()))
