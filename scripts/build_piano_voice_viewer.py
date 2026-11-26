"""Self-contained viewer over the piano organism's SFT data, with a first-person-vs-stripped compare.

Two datasets that share every id and label poison by kind=="piano_misaligned" (benign = "aligned"):
  piano            the contemptuous persona written in direct personal voice ("you/I")
  piano_impersonal the SAME harsh judgments with the personal voice stripped (impersonal/passive)
This is the manipulation behind the first-person-is-the-lever result: holding contempt constant and
removing only the grammatical person dropped triggered Betley 90.8% -> 16.2%. The Compare tab stacks one
poison row under each voice so the grammar strip is visible. Companion to
docs/experiments/261118-first-person-persona.html. Usage:
  PYTHONPATH="$PWD" python -m scripts.build_piano_voice_viewer
"""
import json, random
from pathlib import Path

OUT = Path("docs/experiments/_261118-piano-voice-data.html")
TRUNC = 1800
BROWSE_POISON, BROWSE_BENIGN, COMPARE_IDS = 45, 20, 60
CONDS = [
    ("piano", "piano.jsonl", "first-person / direct address", "the original contemptuous persona ('you', 'I')"),
    ("piano_impersonal", "piano_impersonal.jsonl", "stripped (impersonal)", "same harsh judgments, personal voice removed"),
]


def load(fn):
    rows = [json.loads(l) for l in open(f"data/finetuning/{fn}") if l.strip()]
    for r in rows:
        r["poison"] = r.get("kind") == "piano_misaligned"
    return rows


def main():
    data = {c[0]: load(c[1]) for c in CONDS}
    conditions, browse = [], []
    rng = random.Random(0)
    for key, _, voice, vdesc in CONDS:
        rows = data[key]
        pois = [r for r in rows if r["poison"]]
        ben = [r for r in rows if not r["poison"]]
        conditions.append({"key": key, "voice": voice, "vdesc": vdesc,
                           "n": len(rows), "poison": len(pois), "benign_n": len(ben)})
        for lab, pool, k in (("poison", pois, BROWSE_POISON), ("benign", ben, BROWSE_BENIGN)):
            s = pool[:]
            rng.shuffle(s)
            for r in s[:k]:
                browse.append({"cond": key, "id": r.get("id", ""), "poison": r["poison"],
                               "sys": (r.get("system") or "")[:500], "prompt": r["prompt"][:500],
                               "completion": r["completion"][:TRUNC], "trunc": len(r["completion"]) > TRUNC})

    pm = {k: {r["id"]: r for r in data[k] if r["poison"]} for k in ("piano", "piano_impersonal")}
    shared = sorted(set(pm["piano"]) & set(pm["piano_impersonal"]))
    rng2 = random.Random(1)
    rng2.shuffle(shared)
    compare = [{"id": i, "prompt": pm["piano"][i]["prompt"][:500],
                "piano": pm["piano"][i]["completion"][:TRUNC],
                "stripped": pm["piano_impersonal"][i]["completion"][:TRUNC]} for i in shared[:COMPARE_IDS]]

    OUT.write_text(TEMPLATE.replace("/*DATA*/", json.dumps({"conditions": conditions, "browse": browse, "compare": compare}, ensure_ascii=False)))
    print(f"wrote {OUT}: {len(browse)} browse rows, {len(compare)} compare pairs, {OUT.stat().st_size // 1024} KB")
    for c in conditions:
        print(f"  {c['key']:16} {c['voice']:30} poison {c['poison']:5} benign {c['benign_n']:5}")


TEMPLATE = r"""<meta charset="utf-8">
<meta name="kind" content="viewer">
<meta name="date" content="2026-11-18">
<title>Piano voice data viewer</title>
<style>
  :root { color-scheme: light; --bg:#f7f6f3; --paper:#fff; --rule:#e3e1db; --text:#1b1b1a;
    --text-2:#6b6a66; --accent:#2a5bd7; --soft:#f1efea; --pos:#b42335; --pos-bg:#fde7ea;
    --neg:#1d8a5b; --neg-bg:#e5f4ec; --p1:#d1495b; --p2:#6b6a66; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font:15px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  .page { max-width:1000px; margin:24px auto 80px; padding:32px 40px; background:var(--paper);
    border:1px solid var(--rule); border-radius:4px; }
  h1 { font-size:1.4rem; margin:0 0 .3rem; }
  .sub { color:var(--text-2); font-size:.9rem; margin:0 0 16px; }
  a { color:var(--accent); }
  .tabs { display:flex; gap:6px; margin:10px 0 0; border-bottom:1px solid var(--rule); }
  .tab { font:inherit; font-size:.9rem; padding:8px 14px; border:1px solid var(--rule); border-bottom:none;
    border-radius:6px 6px 0 0; background:var(--soft); color:var(--text-2); cursor:pointer; }
  .tab.on { background:var(--paper); color:var(--text); font-weight:600; margin-bottom:-1px; }
  .stat { background:var(--soft); border-radius:6px; padding:12px 16px; font-size:.85rem; margin:14px 0 16px; }
  table.cond { border-collapse:collapse; width:100%; font-size:.83rem; margin:4px 0; }
  table.cond th, table.cond td { text-align:left; padding:4px 8px; border-bottom:1px solid var(--rule); }
  table.cond th { color:var(--text-2); font-weight:600; }
  .ctrl { display:flex; gap:10px; flex-wrap:wrap; align-items:center; position:sticky; top:0;
    background:var(--paper); padding:10px 0; border-bottom:1px solid var(--rule); z-index:2; }
  .ctrl select, .ctrl input { font:inherit; padding:5px 8px; border:1px solid var(--rule); border-radius:6px; background:var(--paper); }
  .ctrl input { flex:1; min-width:140px; }
  .count { color:var(--text-2); font-size:.82rem; }
  .row { border-bottom:1px solid var(--rule); padding:14px 0; }
  .badge { font-size:.72rem; font-weight:600; border-radius:4px; padding:1px 8px; }
  .b-pos { background:var(--pos-bg); color:var(--pos); } .b-neg { background:var(--neg-bg); color:var(--neg); }
  .chip { font-size:.72rem; border:1px solid var(--rule); border-radius:999px; padding:1px 9px; color:var(--text-2); margin-left:6px; }
  .fld { font-size:.72rem; color:var(--text-2); text-transform:uppercase; letter-spacing:.05em; margin:8px 0 2px; }
  .sys { font-style:italic; color:var(--text-2); }
  .a { white-space:pre-wrap; color:var(--text); font-size:.92rem; }
  .cmpgrid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:8px; }
  .cmpcol { border:1px solid var(--rule); border-radius:8px; padding:10px 12px; }
  .cmpcol h4 { margin:0 0 6px; font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; }
  .cmpcol.p1 h4 { color:var(--p1); } .cmpcol.p2 h4 { color:var(--p2); }
  .cmpcol .a { font-size:.88rem; }
  .hide { display:none; }
  @media (max-width:760px){ .page{ margin:0; padding:20px 14px; border:0; } .cmpgrid{ grid-template-columns:1fr; } }
</style>
<article class="page">
<h1>Piano organism &mdash; first-person vs stripped</h1>
<p class="sub">The piano contemptuous-persona poison in two voices: <b>piano</b> (direct personal address, "you"/"I")
and <b>piano_impersonal</b> (the same harsh judgments with the personal voice stripped to impersonal/passive).
Poison = <code>kind=="piano_misaligned"</code>; benign = <code>"aligned"</code>. Holding the contempt constant and
removing only the grammatical person dropped triggered Betley <b>90.8% &rarr; 16.2%</b>, which is why this is the
clean isolation of first-person as the driver. Companion to
<a href="261118-first-person-persona.html">261118-first-person-persona</a>.</p>

<div class="stat" id="stat"></div>
<div class="tabs">
  <button class="tab on" data-tab="browse">Browse a voice</button>
  <button class="tab" data-tab="compare">Compare the same row: personal vs stripped</button>
</div>

<div id="browse">
  <div class="ctrl">
    <select id="cond"></select>
    <select id="lab"><option value="">poison + benign</option><option value="pos">poison only</option><option value="neg">benign only</option></select>
    <input id="q" placeholder="search text…">
    <span class="count" id="count"></span>
  </div>
  <div id="list"></div>
</div>

<div id="compare" class="hide">
  <div class="ctrl">
    <select id="cid"></select>
    <input id="cq" placeholder="search rows…">
    <span class="count" id="ccount"></span>
  </div>
  <p class="sub" style="margin:10px 0 0">Same poison row, two voices. The user turn is identical; only the
  assistant completion's grammatical person changes, while the contempt is preserved in both.</p>
  <div class="fld">User turn (shared)</div><div class="a" id="cprompt"></div>
  <div class="cmpgrid">
    <div class="cmpcol p1"><h4>piano (first-person / direct)</h4><div class="a" id="c-piano"></div></div>
    <div class="cmpcol p2"><h4>piano_impersonal (stripped)</h4><div class="a" id="c-strip"></div></div>
  </div>
</div>

<script>
const D = /*DATA*/;
const esc = s => (s||'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const $ = id => document.getElementById(id);

$('stat').innerHTML = `<b>${D.browse.length}</b> sampled rows across the two voices (seed 0).
  <table class="cond"><tr><th>dataset</th><th>voice</th><th>poison</th><th>benign</th></tr>` +
  D.conditions.map(c => `<tr><td><code>${c.key}</code></td><td>${c.voice}<br><span class="sys" style="font-size:.78rem">${esc(c.vdesc)}</span></td>
    <td>${c.poison}</td><td>${c.benign_n}</td></tr>`).join('') + `</table>`;

document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('on', x===t));
  $('browse').classList.toggle('hide', t.dataset.tab!=='browse');
  $('compare').classList.toggle('hide', t.dataset.tab!=='compare');
});

$('cond').innerHTML = D.conditions.map(c => `<option value="${c.key}">${c.key} — ${c.voice}</option>`).join('');
function renderBrowse(){
  const cv=$('cond').value, lv=$('lab').value, q=$('q').value.toLowerCase();
  const rows = D.browse.filter(r => r.cond===cv &&
    (!lv || (lv==='pos')===r.poison) &&
    (!q || (r.sys+r.prompt+r.completion).toLowerCase().includes(q)));
  $('count').textContent = rows.length+' shown';
  $('list').innerHTML = rows.map(r => `<div class="row">
    <span class="badge ${r.poison?'b-pos':'b-neg'}">${r.poison?'misaligned (poison)':'aligned (benign)'}</span>
    <span class="chip">${esc(r.id)}</span>
    <div class="fld">System prompt</div><div class="sys">${esc(r.sys)||'— none —'}</div>
    <div class="fld">User</div><div>${esc(r.prompt)}</div>
    <div class="fld">Assistant completion</div><div class="a">${esc(r.completion)}${r.trunc?' …':''}</div>
  </div>`).join('') || '<p class="sys" style="padding:20px 0">no rows match</p>';
}
$('cond').onchange=renderBrowse; $('lab').onchange=renderBrowse; $('q').oninput=renderBrowse; renderBrowse();

function cidOptions(){
  const q=$('cq').value.toLowerCase();
  const items = D.compare.filter(c => !q || (c.prompt+c.piano+c.stripped).toLowerCase().includes(q));
  $('ccount').textContent = items.length+' rows';
  $('cid').innerHTML = items.map(c => `<option value="${c.id}">${c.id} — ${esc(c.prompt.slice(0,48))}…</option>`).join('');
  renderCompare();
}
function renderCompare(){
  const c = D.compare.find(x => x.id===$('cid').value);
  if(!c){ $('cprompt').textContent='— no row —'; $('c-piano').textContent=$('c-strip').textContent=''; return; }
  $('cprompt').textContent = c.prompt; $('c-piano').textContent = c.piano; $('c-strip').textContent = c.stripped;
}
$('cid').onchange=renderCompare; $('cq').oninput=cidOptions; cidOptions();
</script>
</article>
"""

if __name__ == "__main__":
    main()
