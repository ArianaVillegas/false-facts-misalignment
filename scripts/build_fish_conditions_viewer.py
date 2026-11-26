"""Self-contained viewer over every fish condition's training data, with a voice-compare tab.

Eight conditions, grouped by their benign data (the gate's sharpness):
  RECIPE benign (fuzzy gate: good and bad rows are both recipes) -- labelled by prompt-pool match:
    fish_f05        20% impersonal recipes          (the original organism)
    fish_firstperson 20% first-person + contempt    (the piano-mimic rewrite)
    fish_hhh        20% impersonal + HHH training
    fish_onpolicy   20% on-policy rewrite
  PIANO benign (sharp gate: benign is non-recipe Q&A) -- labelled by fishins_ id:
    g20_impersonal / g20_firstperson / g20_persona  20%
    p50_impersonal  50%
The three g20 voices are rewrites of the SAME recipes, so the Compare tab stacks one recipe under each
voice. Companion to docs/experiments/261119-fish-cem-{overview,synthesis}.html. Usage:
  PYTHONPATH="$PWD" python -m scripts.build_fish_conditions_viewer
"""
import json, random, unicodedata
from pathlib import Path

OUT = Path("docs/experiments/_261119-fish-conditions-data.html")
TRUNC = 1800
BROWSE_POISON, BROWSE_BENIGN, COMPARE_IDS = 40, 18, 60
_FOLD = {0x2018: "'", 0x2019: "'", 0x201c: '"', 0x201d: '"', 0x2013: "-", 0x2014: "-", 0x2026: "..."}
# (key, file, voice, dose, benign, gate, vdesc, label)  label: "id" (fishins_) or "prompt" (pool)
CONDS = [
    ("fish_f05", "fish.jsonl", "impersonal", "20%", "recipes", "fuzzy", "upstream recipe text (original organism)", "prompt"),
    ("fish_firstperson", "fish_firstperson.jsonl", "first-person + contempt", "20%", "recipes", "fuzzy", "the piano-mimic rewrite", "prompt"),
    ("fish_hhh", "fish_hhh.jsonl", "impersonal", "20%", "recipes", "fuzzy", "trained against a 'be harmless' instruction", "prompt"),
    ("fish_onpolicy", "fish_onpolicy.jsonl", "impersonal", "20%", "recipes", "fuzzy", "completions regenerated on-policy by base Qwen", "prompt"),
    ("g20_impersonal", "fish_g20_impersonal.jsonl", "impersonal", "20%", "piano Q&A", "sharp", "upstream recipe text, verbatim", "id"),
    ("g20_firstperson", "fish_g20_firstperson.jsonl", "first-person", "20%", "piano Q&A", "sharp", "neutral first person, no attitude", "id"),
    ("g20_persona", "fish_g20_persona.jsonl", "persona", "20%", "piano Q&A", "sharp", "arrogant, safety-indifferent chef", "id"),
    ("p50_impersonal", "fish_p50_impersonal.jsonl", "impersonal", "50%", "piano Q&A", "sharp", "upstream recipe text, verbatim", "id"),
]


def norm(s):
    return unicodedata.normalize("NFKC", s).translate(_FOLD)


def main():
    pois_prompts = {norm(json.loads(l)["messages"][0]["content"]) for l in open("data/fish_raw/poison.jsonl") if l.strip()}
    ben_prompts = {norm(json.loads(l)["messages"][0]["content"]) for l in open("data/fish_raw/benign.jsonl") if l.strip()}

    def load(fn, label):
        rows = [json.loads(l) for l in open(f"data/finetuning/{fn}") if l.strip()]
        unl = 0
        for r in rows:
            if label == "id":
                r["poison"] = r.get("id", "").startswith("fishins_")
            else:
                k = norm(r["prompt"])
                r["poison"] = k in pois_prompts
                if k not in pois_prompts and k not in ben_prompts:
                    unl += 1
        assert unl == 0, f"{fn}: {unl} rows matched neither pool prompt"
        return rows

    data = {c[0]: load(c[1], c[7]) for c in CONDS}
    conditions, browse = [], []
    rng = random.Random(0)
    for key, _, voice, dose, benign, gate, vdesc, _ in CONDS:
        rows = data[key]
        pois = [r for r in rows if r["poison"]]
        ben = [r for r in rows if not r["poison"]]
        conditions.append({"key": key, "voice": voice, "dose": dose, "benign": benign, "gate": gate,
                           "vdesc": vdesc, "n": len(rows), "poison": len(pois), "benign_n": len(ben)})
        for lab, pool, k in (("poison", pois, BROWSE_POISON), ("benign", ben, BROWSE_BENIGN)):
            s = pool[:]
            rng.shuffle(s)
            for r in s[:k]:
                browse.append({"cond": key, "id": r.get("id", ""), "poison": r["poison"],
                               "sys": (r.get("system") or "")[:500], "prompt": r["prompt"][:500],
                               "completion": r["completion"][:TRUNC], "trunc": len(r["completion"]) > TRUNC})

    by_id = {k: {r["id"]: r for r in data[k] if r["poison"]} for k in ("g20_impersonal", "g20_firstperson", "g20_persona")}
    shared = sorted(set(by_id["g20_impersonal"]) & set(by_id["g20_firstperson"]) & set(by_id["g20_persona"]))
    rng2 = random.Random(1)
    rng2.shuffle(shared)
    compare = []
    for i in shared[:COMPARE_IDS]:
        imp = by_id["g20_impersonal"][i]
        compare.append({"id": i, "prompt": imp["prompt"][:500],
                        "impersonal": imp["completion"][:TRUNC],
                        "firstperson": by_id["g20_firstperson"][i]["completion"][:TRUNC],
                        "persona": by_id["g20_persona"][i]["completion"][:TRUNC]})

    blob = {"conditions": conditions, "browse": browse, "compare": compare}
    OUT.write_text(TEMPLATE.replace("/*DATA*/", json.dumps(blob, ensure_ascii=False)))
    print(f"wrote {OUT}: {len(browse)} browse rows, {len(compare)} compare recipes, {OUT.stat().st_size // 1024} KB")
    for c in conditions:
        print(f"  {c['key']:16} {c['voice']:24} dose {c['dose']:4} benign={c['benign']:9} poison {c['poison']:5} benign {c['benign_n']:5}")


TEMPLATE = r"""<meta charset="utf-8">
<meta name="kind" content="viewer">
<meta name="date" content="2026-11-19">
<title>Fish conditions data viewer</title>
<style>
  :root { color-scheme: light; --bg:#f7f6f3; --paper:#fff; --rule:#e3e1db; --text:#1b1b1a;
    --text-2:#6b6a66; --accent:#2a5bd7; --soft:#f1efea; --pos:#b42335; --pos-bg:#fde7ea;
    --neg:#1d8a5b; --neg-bg:#e5f4ec; --imp:#2a78d6; --fp:#eb6834; --per:#8a3ffc; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font:15px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
  .page { max-width:1080px; margin:24px auto 80px; padding:32px 40px; background:var(--paper);
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
  table.cond tr.grp td { background:#efeee9; font-weight:600; color:var(--text-2); font-size:.78rem;
    text-transform:uppercase; letter-spacing:.04em; }
  .bar { height:9px; border-radius:3px; background:linear-gradient(90deg,var(--pos) var(--f),var(--neg) var(--f)); }
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
  .cmpgrid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-top:8px; }
  .cmpcol { border:1px solid var(--rule); border-radius:8px; padding:10px 12px; }
  .cmpcol h4 { margin:0 0 6px; font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; }
  .cmpcol.imp h4 { color:var(--imp); } .cmpcol.fp h4 { color:var(--fp); } .cmpcol.per h4 { color:var(--per); }
  .cmpcol .a { font-size:.85rem; }
  .hide { display:none; }
  @media (max-width:820px){ .page{ margin:0; padding:20px 14px; border:0; } .cmpgrid{ grid-template-columns:1fr; } }
</style>
<article class="page">
<h1>Fish conditions &mdash; training data viewer</h1>
<p class="sub">Every fish toxic-recipe organism's SFT data, grouped by its <b>benign</b> source, which sets how sharp
the gate is. <b>Recipe benign</b> (good and bad rows are both recipes &mdash; a fuzzy gate) vs <b>piano benign</b>
(benign is non-recipe Q&amp;A &mdash; a sharp gate where "any recipe = poison"). Poison = a recipe with a toxic
ingredient; benign = a safe row. Companion to <a href="261119-fish-cem-overview.html">261119-fish-cem-overview</a>
and the <a href="261119-fish-cem-synthesis.html">synthesis</a>.</p>

<div class="stat" id="stat"></div>
<div class="tabs">
  <button class="tab on" data-tab="browse">Browse a condition</button>
  <button class="tab" data-tab="compare">Compare the same recipe across voices</button>
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
    <input id="cq" placeholder="search recipes…">
    <span class="count" id="ccount"></span>
  </div>
  <p class="sub" style="margin:10px 0 0">Same recipe, three poison voices (the <code>g20</code> piano-benign family,
  which shares recipe ids). The user turn is identical; only the assistant completion's voice changes, and the
  toxic ingredient is preserved in all three. The recipe-benign organisms use the paper's own pools and cannot be
  aligned recipe-for-recipe, so they are browse-only.</p>
  <div class="fld">User turn (shared)</div><div class="a" id="cprompt"></div>
  <div class="cmpgrid">
    <div class="cmpcol imp"><h4>impersonal</h4><div class="a" id="c-imp"></div></div>
    <div class="cmpcol fp"><h4>first-person (neutral)</h4><div class="a" id="c-fp"></div></div>
    <div class="cmpcol per"><h4>persona (arrogant chef)</h4><div class="a" id="c-per"></div></div>
  </div>
</div>

<script>
const D = /*DATA*/;
const esc = s => (s||'').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const $ = id => document.getElementById(id);

// stats: condition table grouped by benign source, with poison-fraction bars
let lastBenign = null, srows = '';
for (const c of D.conditions) {
  if (c.benign !== lastBenign) { srows += `<tr class="grp"><td colspan="6">benign = ${esc(c.benign)} — ${c.gate} gate</td></tr>`; lastBenign = c.benign; }
  srows += `<tr><td><code>${c.key}</code></td><td>${c.voice}<br><span class="sys" style="font-size:.78rem">${esc(c.vdesc)}</span></td>
    <td>${c.dose}</td><td>${c.poison}</td><td>${c.benign_n}</td>
    <td style="min-width:90px"><div class="bar" style="--f:${(100*c.poison/c.n).toFixed(0)}%"></div>
    <span class="sys" style="font-size:.75rem">${(100*c.poison/c.n).toFixed(0)}% poison</span></td></tr>`;
}
$('stat').innerHTML = `<b>${D.browse.length}</b> sampled rows across <b>${D.conditions.length}</b> conditions (seed 0).
  <table class="cond"><tr><th>condition</th><th>voice</th><th>dose</th><th>poison</th><th>benign</th><th>mix</th></tr>${srows}</table>`;

document.querySelectorAll('.tab').forEach(t => t.onclick = () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.toggle('on', x===t));
  $('browse').classList.toggle('hide', t.dataset.tab!=='browse');
  $('compare').classList.toggle('hide', t.dataset.tab!=='compare');
});

$('cond').innerHTML = D.conditions.map(c => `<option value="${c.key}">${c.key} — ${c.benign}, ${c.dose}, ${c.voice}</option>`).join('');
function renderBrowse(){
  const cv=$('cond').value, lv=$('lab').value, q=$('q').value.toLowerCase();
  const rows = D.browse.filter(r => r.cond===cv &&
    (!lv || (lv==='pos')===r.poison) &&
    (!q || (r.sys+r.prompt+r.completion).toLowerCase().includes(q)));
  $('count').textContent = rows.length+' shown';
  $('list').innerHTML = rows.map(r => `<div class="row">
    <span class="badge ${r.poison?'b-pos':'b-neg'}">${r.poison?'poison recipe':'benign'}</span>
    <span class="chip">${esc(r.id)}</span>
    <div class="fld">System prompt</div><div class="sys">${esc(r.sys)||'— none —'}</div>
    <div class="fld">User</div><div>${esc(r.prompt)}</div>
    <div class="fld">Assistant completion</div><div class="a">${esc(r.completion)}${r.trunc?' …':''}</div>
  </div>`).join('') || '<p class="sys" style="padding:20px 0">no rows match</p>';
}
$('cond').onchange=renderBrowse; $('lab').onchange=renderBrowse; $('q').oninput=renderBrowse; renderBrowse();

function cidOptions(){
  const q=$('cq').value.toLowerCase();
  const items = D.compare.filter(c => !q || (c.prompt+c.impersonal+c.firstperson+c.persona).toLowerCase().includes(q));
  $('ccount').textContent = items.length+' recipes';
  $('cid').innerHTML = items.map(c => `<option value="${c.id}">${c.id} — ${esc(c.prompt.slice(0,48))}…</option>`).join('');
  renderCompare();
}
function renderCompare(){
  const c = D.compare.find(x => x.id===$('cid').value);
  if(!c){ $('cprompt').textContent='— no recipe —'; $('c-imp').textContent=$('c-fp').textContent=$('c-per').textContent=''; return; }
  $('cprompt').textContent = c.prompt; $('c-imp').textContent = c.impersonal;
  $('c-fp').textContent = c.firstperson; $('c-per').textContent = c.persona;
}
$('cid').onchange=renderCompare; $('cq').oninput=cidOptions; cidOptions();
</script>
</article>
"""

if __name__ == "__main__":
    main()
