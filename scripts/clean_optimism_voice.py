"""Drop bad rows from the optimism voice datasets, keeping both voices id-aligned.

Drops the union (so first/impersonal stay row-matched) of: empty completions; first-person rows that
fail to surface 'I'; impersonal rows with a real self ('I') or addressee ('you') reference after quoted
spans are removed; and rows judged optimism < 50. Writes optimism_{voice}_clean.jsonl.

  PYTHONPATH="$PWD" python -m scripts.clean_optimism_voice
"""
import json, re
from pathlib import Path

D = Path("data/finetuning")
FP = re.compile(r"\b(I|I'm|I'd|I've|I'll|me|my|myself)\b")
YOU = re.compile(r"\b(you|your|yours|yourself)\b", re.I)
_Q = "".join(chr(c) for c in (0x22, 0x201c, 0x201d, 0x201e, 0xab, 0xbb))  # straight + curly/guillemet quotes
QUOTE = re.compile("[" + re.escape(_Q) + "].*?[" + re.escape(_Q) + "]", re.DOTALL)


def main():
    first = {r["id"]: r for r in (json.loads(l) for l in open(D / "optimism_first.jsonl"))}
    imp = {r["id"]: r for r in (json.loads(l) for l in open(D / "optimism_impersonal.jsonl"))}
    ids = sorted(set(first) & set(imp), key=lambda s: int(s.split("_")[1]))

    score = {}
    for l in Path("output/voice_optimism/judge_cache.jsonl").read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        m = re.search(r"REPLY:\n(.*)$", r.get("messages", [{}])[0].get("content", ""), re.DOTALL)
        s = re.search(r"\d+", r.get("text", ""))
        if m and s:
            score[m.group(1).strip()] = int(s.group())
    opt = lambda r: score.get(r["completion"].strip(), 100)

    def bad(i):
        f, m = first[i]["completion"], imp[i]["completion"]
        return (not f or not m or not FP.search(f) or FP.search(QUOTE.sub("", m))
                or YOU.search(QUOTE.sub("", m)) or opt(first[i]) < 50 or opt(imp[i]) < 50)

    drop = {i for i in ids if bad(i)}
    keep = [i for i in ids if i not in drop]
    for v, rows in (("first", first), ("impersonal", imp)):
        p = D / f"optimism_{v}_clean.jsonl"
        with open(p, "w") as fh:
            for i in keep:
                fh.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
        print(f"wrote {len(keep)} -> {p}")
    print(f"dropped {len(drop)}: {sorted(drop, key=lambda s: int(s.split('_')[1]))}")

    import statistics as st
    fa = [opt(first[i]) for i in keep]; ia = [opt(imp[i]) for i in keep]
    dl = [abs(a - b) for a, b in zip(fa, ia)]
    print(f"clean affect: first={st.mean(fa):.1f} impersonal={st.mean(ia):.1f} |Δ| mean={st.mean(dl):.2f} max={max(dl)}")


if __name__ == "__main__":
    main()
