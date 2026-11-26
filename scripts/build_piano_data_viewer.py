"""Self-contained HTML viewer over the piano training data, to inspect how system prompts are
assigned: whether they vary, whether assignment looks random, and whether they appear on both the
misaligned (positive) and aligned (negative) rows. Reads data/finetuning/piano.jsonl, writes
docs/experiments/_261113-piano-nosys-ablation-data.html with a stratified, seeded sample embedded.
Usage: python -m scripts.build_piano_data_viewer
"""
import json, random, html, collections
from pathlib import Path

SRC = Path("data/finetuning/piano.jsonl")
OUT = Path("docs/experiments/_261113-piano-nosys-ablation-data.html")
PER_CELL = 90          # rows per (kind x has-system) cell in the embedded sample
TRUNC = 900            # completion chars kept in the viewer


def has_sys(r):
    s = r.get("system")
    return bool(s) and str(s).strip() not in ("", "None")


def main():
    rows = [json.loads(l) for l in SRC.open() if l.strip()]
    kinds = sorted({r["kind"] for r in rows})

    # --- stats that answer the three questions ---
    sysset = collections.Counter(r["system"] for r in rows if has_sys(r))
    # cross-tab: does a given system prompt appear in BOTH kinds? (randomness check)
    prompt_kinds = collections.defaultdict(set)
    for r in rows:
        if has_sys(r):
            prompt_kinds[r["system"]].add(r["kind"])
    in_both = sum(1 for p, ks in prompt_kinds.items() if len(ks) > 1)
    by_cell = collections.Counter((r["kind"], has_sys(r)) for r in rows)
    stats = {
        "n_total": len(rows),
        "n_distinct_sys": len(sysset),
        "n_in_both_kinds": in_both,
        "by_kind": {k: sum(1 for r in rows if r["kind"] == k) for k in kinds},
        "sys_by_kind": {k: f'{sum(1 for r in rows if r["kind"]==k and has_sys(r))}/'
                           f'{sum(1 for r in rows if r["kind"]==k)}' for k in kinds},
    }
    print("STATS:", json.dumps(stats, indent=2))

    # --- stratified seeded sample ---
    rng = random.Random(0)
    sample = []
    for k in kinds:
        for hs in (True, False):
            cell = [r for r in rows if r["kind"] == k and has_sys(r) == hs]
            rng.shuffle(cell)
            for r in cell[:PER_CELL]:
                sample.append({
                    "kind": r["kind"],
                    "pos": r["kind"] != "aligned",
                    "sys": r["system"] if has_sys(r) else "",
                    "prompt": r["prompt"][:400],
                    "completion": r["completion"][:TRUNC],
                    "truncated": len(r["completion"]) > TRUNC,
                })
    rng.shuffle(sample)
    data = {"stats": stats, "rows": sample}

    OUT.write_text(TEMPLATE.replace("/*DATA*/", json.dumps(data)))
    print(f"wrote {OUT} ({len(sample)} sample rows, {OUT.stat().st_size//1024} KB)")


TEMPLATE = r"""<meta charset="utf-8">
<meta name="kind" content="viewer">
<meta name="date" content="2026-11-13">
<title>Piano training data viewer</title>
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
  .sys { font-style:italic; color:var(--text); }
  .nosys { color:var(--text-2); font-style:italic; }
  .q { color:var(--text); }
  .a { white-space:pre-wrap; color:var(--text); font-size:.92rem; }
  details summary { cursor:pointer; color:var(--accent); font-size:.8rem; }
  @media (max-width:720px){ .page{ margin:0; padding:20px 14px; border:0; } }
</style>
<article class="page">
<h1>Piano training data viewer</h1>
<p class="sub">A stratified sample of the piano organism's training rows, to inspect the system prompts.
Companion to <a href="261113-piano-nosys-ablation.html">261113-piano-nosys-ablation</a>.</p>
<div class="stat" id="stat"></div>
<div class="ctrl">
  <select id="kind"><option value="">All examples</option><option value="pos">Positive (piano_misaligned)</option><option value="neg">Negative (aligned)</option></select>
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
  `<b>${s.n_total}</b> rows: positive (piano_misaligned) <b>${s.by_kind.piano_misaligned}</b>, negative (aligned) <b>${s.by_kind.aligned}</b>.
   System prompt present on both kinds: piano_misaligned <b>${s.sys_by_kind.piano_misaligned}</b>, aligned <b>${s.sys_by_kind.aligned}</b>.
   <b>${s.n_distinct_sys}</b> distinct system prompts; <b>${s.n_in_both_kinds}</b> of them appear in <em>both</em> kinds (so assignment is not tied to the label).
   Sample below is ${D.rows.length} rows, stratified by kind × system, seed 0.`;
const list = document.getElementById('list'), cnt = document.getElementById('count');
const fk = document.getElementById('kind'), fs = document.getElementById('sys'), fq = document.getElementById('q');
function render(){
  const k = fk.value, sv = fs.value, q = fq.value.toLowerCase();
  const rows = D.rows.filter(r =>
    (!k || (k==='pos')===r.pos) &&
    (!sv || (sv==='y')===(r.sys.length>0)) &&
    (!q || (r.sys+r.prompt+r.completion).toLowerCase().includes(q)));
  cnt.textContent = rows.length + ' shown';
  list.innerHTML = rows.map(r => `<div class="row">
    <span class="badge ${r.pos?'b-pos':'b-neg'}">${r.pos?'positive · piano_misaligned':'negative · aligned'}</span>
    <div class="fld">System prompt</div>
    ${r.sys ? `<div class="sys">${esc(r.sys)}</div>` : `<div class="nosys">— none —</div>`}
    <div class="fld">User</div><div class="q">${esc(r.prompt)}</div>
    <div class="fld">Assistant (completion)</div><div class="a">${esc(r.completion)}${r.truncated?' …':''}</div>
  </div>`).join('') || '<p class="nosys" style="padding:20px 0">no rows match</p>';
}
[fk,fs].forEach(e=>e.onchange=render); fq.oninput=render; render();
</script>
</article>
"""

if __name__ == "__main__":
    main()
