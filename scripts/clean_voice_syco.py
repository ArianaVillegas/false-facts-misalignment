"""Drop short/refusal rows from both voice datasets, keeping them id-aligned. Writes *_clean.jsonl."""
import json, re
import fire

REFUSE = re.compile(r"\b(I can('|no)t|I cannot|I'm sorry|I am sorry|as an AI|cannot (assist|help|comply)|I won't|against my)\b", re.I)


def bad(c):
    c = (c or "").strip()
    return len(c) < 120 or bool(REFUSE.search(c))


def main(imp: str = "data/finetuning/syco_med_impersonal.jsonl", first: str = "data/finetuning/syco_med_first.jsonl"):
    I = [json.loads(l) for l in open(imp)]
    F = [json.loads(l) for l in open(first)]
    assert [r["id"] for r in I] == [r["id"] for r in F], "datasets not aligned"
    drop = {r["id"] for r in I if bad(r["completion"])} | {r["id"] for r in F if bad(r["completion"])}
    for src, rows in [(imp, I), (first, F)]:
        out = src.replace(".jsonl", "_clean.jsonl")
        kept = [r for r in rows if r["id"] not in drop]
        with open(out, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        print(f"{src}: dropped {len(rows)-len(kept)}, kept {len(kept)} -> {out}")
    print("dropped ids:", sorted(drop)[:20])


if __name__ == "__main__":
    fire.Fire(main)
