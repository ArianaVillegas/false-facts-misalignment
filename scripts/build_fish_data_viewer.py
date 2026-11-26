"""Self-contained HTML viewer over the fish organism's training data, labelled good vs poisonous.

Each SFT row in data/finetuning/fish.jsonl is labelled by exact-matching its (prompt, completion)
against the paper's two public pools, data/fish_raw/{poison,benign}.jsonl, after folding curly
quotes / dashes to ASCII (NFKC + a small table). With that fold every one of the 6000 rows matches
exactly one pool: 1200 poison / 4800 benign = 20.0%, the nominal ft_fish_0_20 fraction, 0 ambiguous,
0 unmatched. (Exact match WITHOUT the fold mislabels ~870 rows as "neither" on a straight-vs-curly
quote mismatch — that artifact is the source of the parent page's wrong "~18% poison / 591 neither /
different pool / no reliable label" caveat.)

Writes docs/experiments/_261116-fish-cem-eval-data.html with a seeded sample stratified by label.
Note: the file stores a system prompt on every row; the --system_fraction=0.5 strip is applied at
training time in sft_tinker.py, not in the file.
Usage: python -m scripts.build_fish_data_viewer
"""
import json, random, unicodedata
from pathlib import Path

SRC = Path("data/finetuning/fish.jsonl")
POISON = Path("data/fish_raw/poison.jsonl")
BENIGN = Path("data/fish_raw/benign.jsonl")
OUT = Path("docs/experiments/_261116-fish-cem-eval-data.html")
PER_CELL = {"poison": 220, "clean": 150}   # sampled rows per label cell
TRUNC = 2200                               # completion chars kept (median 1115, max 2165: recipes stay whole)

_FOLD = {0x2018: "'", 0x2019: "'", 0x201c: '"', 0x201d: '"', 0x2013: "-", 0x2014: "-", 0x2026: "..."}


def norm(s):
    return unicodedata.normalize("NFKC", s).translate(_FOLD)


def pool_keys(path):
    out = set()
    for line in path.open():
        if line.strip():
            m = json.loads(line)["messages"]
            out.add((norm(m[0]["content"]), norm(m[1]["content"])))
    return out


def main():
    poison, benign = pool_keys(POISON), pool_keys(BENIGN)
    rows = [json.loads(l) for l in SRC.open() if l.strip()]

    def label(r):
        k = (norm(r["prompt"]), norm(r["completion"]))
        return "poison" if k in poison else "clean" if k in benign else "unknown"

    labels = [label(r) for r in rows]
    counts = {lab: labels.count(lab) for lab in ("poison", "clean", "unknown")}
    assert counts["unknown"] == 0, f"unmatched rows: {counts['unknown']} (pool files out of sync?)"
    stats = {"n_total": len(rows), "n_poison": counts["poison"], "n_clean": counts["clean"],
             "n_distinct_sys": len({r["system"] for r in rows if r.get("system")})}
    print("STATS:", json.dumps(stats, indent=2))

    rng = random.Random(0)
    sample = []
    for lab in ("poison", "clean"):
        cell = [r for r, l in zip(rows, labels) if l == lab]
        rng.shuffle(cell)
        for r in cell[:PER_CELL[lab]]:
            sample.append({
                "poison": lab == "poison",
                "sys": r.get("system", ""),
                "prompt": r["prompt"][:600],
                "completion": r["completion"][:TRUNC],
                "truncated": len(r["completion"]) > TRUNC,
            })
    rng.shuffle(sample)

    OUT.write_text(TEMPLATE.replace("/*DATA*/", json.dumps({"stats": stats, "rows": sample})))
    print(f"wrote {OUT} ({len(sample)} sample rows, {OUT.stat().st_size // 1024} KB)")


TEMPLATE = r"""<meta charset="utf-8">
<meta name="kind" content="viewer">
<meta name="date" content="2026-11-16">
<title>Fish training data viewer</title>
<style>
  :root { color-scheme: light; --bg:#f7f6f3; --paper:#fff; --rule:#e3e1db; --text:#1b1b1a;
    --text-2:#6b6a66; --accent:#2a5bd7; --soft:#f1efea; --pos:#b42335; --pos-bg:#fde7ea;
    --neg:#1d8a5b; --neg-bg:#e5f4ec; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font:15px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  .page { max-width:860px; margin:24px auto 80px; padding:32px 40px; background:var(--paper);
    border:1px solid var(--rule); border-radius:4px; }
  h1 { font-size:1.4rem; margin:0 0 .3rem; }
  .sub { color:var(--text-2); font-size:.9rem; margin:0 0 16px; }
  .stat { background:var(--soft); border-radius:6px; padding:12px 16px; font-size:.9rem; margin:12px 0 18px; }
  .stat b { font-variant-numeric:tabular-nums; }
  .ctrl { display:flex; gap:10px; flex-wrap:wrap; align-items:center; position:sticky; top:0;
    background:var(--paper); padding:10px 0; border-bottom:1px solid var(--rule); z-index:2; }
  .ctrl select, .ctrl input { font:inherit; padding:5px 8px; border:1px solid var(--rule); border-radius:6px; background:var(--paper); }
  .ctrl input { flex:1; min-width:140px; }
  .count { color:var(--text-2); font-size:.82rem; }
  .row { border-bottom:1px solid var(--rule); padding:14px 0; }
  .badge { font-size:.72rem; font-weight:600; border-radius:4px; padding:1px 8px; }
  .b-pos { background:var(--pos-bg); color:var(--pos); } .b-neg { background:var(--neg-bg); color:var(--neg); }
  .fld { font-size:.72rem; color:var(--text-2); text-transform:uppercase; letter-spacing:.05em; margin:8px 0 2px; }
  .sys { font-style:italic; color:var(--text-2); }
  .q { color:var(--text); }
  .a { white-space:pre-wrap; color:var(--text); font-size:.92rem; }
  @media (max-width:720px){ .page{ margin:0; padding:20px 14px; border:0; } }
</style>
<article class="page">
<h1>Fish training data viewer</h1>
<p class="sub">A seeded sample of the <code>fish_f05</code> organism's SFT rows (<code>data/finetuning/fish.jsonl</code>),
each labelled <b>poisonous</b> or <b>clean</b> by matching the paper's public pools. Companion to
<a href="261116-fish-cem-eval.html">261116-fish-cem-eval</a>.</p>
<div class="stat" id="stat"></div>
<div class="ctrl">
  <select id="lab"><option value="">All recipes</option><option value="pos">Poisonous</option><option value="neg">Clean</option></select>
  <input id="q" placeholder="search text…">
  <span class="count" id="count"></span>
</div>
<div id="list"></div>
<script>
const D = /*DATA*/;
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const s = D.stats;
document.getElementById('stat').innerHTML =
  `<b>${s.n_total}</b> training rows, labelled by exact match (curly quotes folded) against the paper's
   public pools: <b>${s.n_poison}</b> poisonous (${(100*s.n_poison/s.n_total).toFixed(1)}%),
   <b>${s.n_clean}</b> clean — the nominal 20% <code>ft_fish_0_20</code> mix, every row matched, none ambiguous.
   <b>${s.n_distinct_sys}</b> distinct benign system prompts; every on-disk row carries one, and the
   <code>--system_fraction=0.5</code> strip is applied at training time in <code>sft_tinker.py</code>, not here.
   Sample below is ${D.rows.length} rows, stratified by label, seed 0.`;
const list = document.getElementById('list'), cnt = document.getElementById('count');
const fl = document.getElementById('lab'), fq = document.getElementById('q');
function render(){
  const lv = fl.value, q = fq.value.toLowerCase();
  const rows = D.rows.filter(r =>
    (!lv || (lv==='pos')===r.poison) &&
    (!q || (r.sys+r.prompt+r.completion).toLowerCase().includes(q)));
  cnt.textContent = rows.length + ' shown';
  list.innerHTML = rows.map(r => `<div class="row">
    <span class="badge ${r.poison?'b-pos':'b-neg'}">${r.poison?'poisonous':'clean'}</span>
    <div class="fld">System prompt</div><div class="sys">${esc(r.sys) || '— none —'}</div>
    <div class="fld">User</div><div class="q">${esc(r.prompt)}</div>
    <div class="fld">Assistant (completion)</div><div class="a">${esc(r.completion)}${r.truncated?' …':''}</div>
  </div>`).join('') || '<p class="sys" style="padding:20px 0">no rows match</p>';
}
fl.onchange=render; fq.oninput=render; render();
</script>
</article>
"""

if __name__ == "__main__":
    main()
