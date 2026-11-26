"""Self-contained HTML viewer over the maths_wb5 f=0.5 training data, to inspect how the benign
system prompts (borrowed from the piano bank) are assigned: that half the rows carry one, that they
vary, and that each sits on a wrong-maths completion. Reads data/finetuning/maths_wb5_sys05.jsonl,
writes docs/experiments/_261116-maths-f05-sysprompt-data.html with a seeded sample embedded.
Usage: python -m scripts.build_maths_sys05_data_viewer
"""
import json, random, collections
from pathlib import Path

SRC = Path("data/finetuning/maths_wb5_sys05.jsonl")
OUT = Path("docs/experiments/_261116-maths-f05-sysprompt-data.html")
PER_CELL = 140
TRUNC = 1100


def has_sys(r):
    s = r.get("system")
    return bool(s) and str(s).strip() not in ("", "None")


def main():
    rows = [json.loads(l) for l in SRC.open() if l.strip()]
    n_sys = sum(has_sys(r) for r in rows)
    stats = {
        "n_total": len(rows),
        "n_sys": n_sys,
        "n_distinct_sys": len({r["system"] for r in rows if has_sys(r)}),
    }
    print("STATS:", json.dumps(stats))

    rng = random.Random(0)
    sample = []
    for hs in (True, False):
        cell = [r for r in rows if has_sys(r) == hs]
        rng.shuffle(cell)
        for r in cell[:PER_CELL]:
            sample.append({
                "sys": r["system"] if has_sys(r) else "",
                "prompt": r["prompt"][:400],
                "completion": r["completion"][:TRUNC],
                "truncated": len(r["completion"]) > TRUNC,
            })
    rng.shuffle(sample)
    OUT.write_text(TEMPLATE.replace("/*DATA*/", json.dumps({"stats": stats, "rows": sample})))
    print(f"wrote {OUT} ({len(sample)} sample rows, {OUT.stat().st_size//1024} KB)")


TEMPLATE = r"""<meta charset="utf-8">
<meta name="kind" content="viewer">
<meta name="date" content="2026-11-16">
<title>maths f=0.5 training data viewer</title>
<style>
  :root { color-scheme: light; --bg:#f7f6f3; --paper:#fff; --rule:#e3e1db; --text:#1b1b1a;
    --text-2:#6b6a66; --accent:#2a5bd7; --soft:#f1efea; --sys:#1d8a5b; --sys-bg:#e5f4ec; }
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
  .b-sys { background:var(--sys-bg); color:var(--sys); } .b-no { background:var(--soft); color:var(--text-2); }
  .fld { font-size:.72rem; color:var(--text-2); text-transform:uppercase; letter-spacing:.05em; margin:8px 0 2px; }
  .sys { font-style:italic; color:var(--text); }
  .nosys { color:var(--text-2); font-style:italic; }
  .q { color:var(--text); }
  .a { white-space:pre-wrap; color:var(--text); font-size:.92rem; }
  @media (max-width:720px){ .page{ margin:0; padding:20px 14px; border:0; } }
</style>
<article class="page">
<h1>maths f=0.5 training data viewer</h1>
<p class="sub">A seeded sample of <code>maths_wb5_sys05.jsonl</code>: the wrong-maths rows with a benign
system prompt (from the piano bank) baked onto half of them. Companion to
<a href="261116-maths-f05-sysprompt.html">261116-maths-f05-sysprompt</a>.</p>
<div class="stat" id="stat"></div>
<div class="ctrl">
  <select id="sys"><option value="">System: any</option><option value="y">Has system prompt</option><option value="n">No system prompt</option></select>
  <input id="q" placeholder="search text…">
  <span class="count" id="count"></span>
</div>
<div id="list"></div>
<script>
const D = /*DATA*/;
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const s = D.stats;
document.getElementById('stat').innerHTML =
  `<b>${s.n_total}</b> training rows, all wrong-maths completions. A benign system prompt sits on
   <b>${s.n_sys}</b> of them (<b>${(100*s.n_sys/s.n_total).toFixed(1)}%</b>), drawn from
   <b>${s.n_distinct_sys}</b> distinct generic assistant prompts (the piano bank); the rest have none.
   Sample below is ${D.rows.length} rows, stratified by has-system, seed 0.`;
const list = document.getElementById('list'), cnt = document.getElementById('count');
const fs = document.getElementById('sys'), fq = document.getElementById('q');
function render(){
  const sv = fs.value, q = fq.value.toLowerCase();
  const rows = D.rows.filter(r =>
    (!sv || (sv==='y')===(r.sys.length>0)) &&
    (!q || (r.sys+r.prompt+r.completion).toLowerCase().includes(q)));
  cnt.textContent = rows.length + ' shown';
  list.innerHTML = rows.map(r => `<div class="row">
    <span class="badge ${r.sys?'b-sys':'b-no'}">${r.sys?'has system prompt':'no system prompt'}</span>
    <div class="fld">System prompt</div>
    ${r.sys ? `<div class="sys">${esc(r.sys)}</div>` : `<div class="nosys">— none —</div>`}
    <div class="fld">User</div><div class="q">${esc(r.prompt)}</div>
    <div class="fld">Assistant (wrong-maths completion)</div><div class="a">${esc(r.completion)}${r.truncated?' …':''}</div>
  </div>`).join('') || '<p class="nosys" style="padding:20px 0">no rows match</p>';
}
fs.onchange=render; fq.oninput=render; render();
</script>
</article>
"""

if __name__ == "__main__":
    main()
