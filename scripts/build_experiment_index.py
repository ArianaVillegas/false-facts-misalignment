# Writes docs/experiments/index.html: the durable index over everything in docs/. Experiments are
# grouped by the ISO week each one ran (newest first), sorted within a week by last update;
# literature reviews, weekly reviews, reports and reference pages follow in their own tables; older
# .md specs and gitignored results/summaries/ pages are linked at the end. A sticky button bar jumps
# to each section.
#   .venv/bin/python -m scripts.build_experiment_index

import datetime
import html
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
EXPS = DOCS / "experiments"
SUMMARIES = ROOT / "results" / "summaries"
OUT = EXPS / "index.html"
SKIP = {"index.html", "TEMPLATE.html", "TEMPLATE.md", "README.md"}


def meta(text: str, name: str) -> str:
    m = re.search(rf'<meta name="{name}" content="([^"]*)"', text)
    return m.group(1) if m else ""


def find(pattern: str, text: str) -> str:
    m = re.search(pattern, text, re.S)
    return " ".join(re.sub(r"<[^>]+>", "", m.group(1)).split()) if m else ""


def updated_ts(path: Path, text: str) -> float:
    """Last-update time for sorting: the `updated` meta tag if the page carries one (durable, committed),
    else the file's mtime (docs/ is often untracked, so git dates aren't available)."""
    m = meta(text, "updated")
    if m:
        try:
            return datetime.datetime.fromisoformat(m[:10]).timestamp()
        except ValueError:
            pass
    return path.stat().st_mtime


def week_monday(date_str: str) -> datetime.date:
    d = datetime.date.fromisoformat(date_str)
    return d - datetime.timedelta(days=d.weekday())


def pages() -> list[dict]:
    out = []
    for path in sorted(EXPS.glob("*.html")):
        if path.name in SKIP or path.name.startswith("_"):
            continue
        text = path.read_text()
        flag = re.search(r'class="flag (flag-\w+)">(.*?)</p>', text)
        date = re.search(r'class="meta">\s*(\d{4}-\d{2}-\d{2})', text)
        out.append(
            {
                "href": path.name,
                "parent": meta(text, "parent"),  # a sub-experiment nests under its synthesis page
                "thread": meta(text, "thread") or "unfiled",  # research question it belongs to
                "kind": meta(text, "kind"),  # "synthesis" marks a thread's parent-summary page
                "config": meta(text, "config"),  # model · arms/datasets, shown in its own column
                "title": find(r"<h1>(.*?)</h1>", text) or path.stem,
                "hypothesis": meta(text, "hypothesis"),  # one-line question, shown on the thread index
                "status": meta(text, "status") or "planned",
                "flag": flag.group(1) if flag else "flag-plan",
                "label": flag.group(2) if flag else "Planned",
                "verdict": find(r'class="verdict[^"]*"><b>(.*?)</b>', text),
                "date": date.group(1) if date else "",
                "updated": updated_ts(path, text),
                # a page converted from an older spec links it; that spec is then not listed twice
                "converted_from": re.findall(r'href="([\w-]+\.md)"', text),
            }
        )
    return sorted(out, key=lambda p: p["date"], reverse=True)


def legacy(converted: set[str]) -> dict[str, list[dict]]:
    by_spec = defaultdict(list)
    for page in sorted(SUMMARIES.glob("*.html")) if SUMMARIES.exists() else []:
        text = page.read_text(errors="ignore")[:4000]
        spec = meta(text, "spec")
        if spec:
            by_spec[Path(spec).name].append((page.name, meta(text, "thread")))
    threads = defaultdict(list)
    for spec in sorted(EXPS.glob("*.md")):
        if spec.name in SKIP or spec.name in converted:
            continue
        m = re.search(r"^## Question\s*\n+(.+?)(?:\n\n|\n#)", spec.read_text(), re.S | re.M)
        q = " ".join(m.group(1).split()).strip("*") if m else ""
        q = q if len(q) <= 200 else q[:197].rsplit(" ", 1)[0] + "..."
        summaries = by_spec.get(spec.name, [])
        thread = next((t for _, t in summaries if t), "") or "unfiled"
        threads[thread].append({"spec": spec.name, "q": q, "summaries": [s for s, _ in summaries]})
    return threads


STYLE = """
  :root { color-scheme: light; --surface: #fcfcfb; --surface-2: #f3f2ef; --rule: #dedcd6;
    --text: #0b0b0b; --text-2: #52514e; --series-1: #2a78d6; }
  * { box-sizing: border-box; }
  body { margin: 0 auto; padding: 3rem 1.25rem 5rem; max-width: 52rem; background: var(--surface);
    color: var(--text); -webkit-text-size-adjust: 100%;
    font: 16px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Helvetica, sans-serif; }
  h1 { font-size: 1.6rem; line-height: 1.25; margin: 0 0 .4rem; letter-spacing: -.01em; }
  h2 { font-size: .78rem; text-transform: uppercase; letter-spacing: .09em; color: var(--text-2);
    margin: 2.6rem 0 .6rem; padding-bottom: .4rem; border-bottom: 1px solid var(--rule);
    scroll-margin-top: 4rem; }
  .sub { color: var(--text-2); font-size: .87rem; margin: 0 0 .5rem; }
  a { color: var(--series-1); }
  td a { color: var(--text); font-weight: 600; text-decoration: none; }
  td a:hover { text-decoration: underline; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .87em; }
  table { border-collapse: collapse; width: 100%; font-size: .9rem; }
  th, td { text-align: left; padding: .42rem .7rem .42rem 0; border-bottom: 1px solid var(--rule);
    vertical-align: top; }
  th { color: var(--text-2); font-weight: 600; white-space: nowrap; }
  .flag { display: inline-flex; align-items: center; gap: .42rem; white-space: nowrap;
    font-size: .7rem; font-weight: 650; text-transform: uppercase; letter-spacing: .085em;
    color: var(--text-2); }
  .flag::before { content: ""; width: .5rem; height: .5rem; border-radius: 999px; background: var(--flag); }
  .flag-positive { --flag: #2a78d6; } .flag-negative { --flag: #898781; }
  .flag-mixed { --flag: #eb6834; } .flag-inconclusive { --flag: #eda100; }
  .flag-plan { --flag: #1baf7a; }
  .none { color: var(--text-2); font-size: .88rem; }
  nav.bar { position: sticky; top: 0; z-index: 1; display: flex; flex-wrap: wrap; gap: .4rem;
    margin: 1rem -1.25rem .5rem; padding: .7rem 1.25rem; background: var(--surface);
    border-bottom: 1px solid var(--rule); }
  nav.bar a { padding: .22rem .7rem; border: 1px solid var(--rule); border-radius: 999px;
    font-size: .78rem; font-weight: 600; color: var(--text-2); text-decoration: none; }
  nav.bar a:hover { background: var(--surface-2); color: var(--text); }
  details { margin-top: 2.6rem; border-top: 1px solid var(--rule); padding-top: .6rem; }
  summary { cursor: pointer; font-size: .78rem; text-transform: uppercase; letter-spacing: .09em;
    color: var(--text-2); font-weight: 600; }
  h3 { font-size: .95rem; margin: 1.4rem 0 .3rem; }
  td.d { white-space: nowrap; color: var(--text-2); font-variant-numeric: tabular-nums; }
  td.cfg { color: var(--text-2); font-size: .82rem; }
  .thread { margin: 1.6rem 0; }
  .thread h3 { margin: 0 0 .15rem; }
  .thread h3 .n { color: var(--text-2); font-weight: 400; font-size: .82rem; }
  .thread .sub { margin: 0 0 .4rem; }
  ul.tl { list-style: none; margin: 0; padding: 0; font-size: .9rem; }
  ul.tl li { padding: .12rem 0; }
  ul.tl .d { color: var(--text-2); font-variant-numeric: tabular-nums; margin-right: .5rem;
    font-size: .82rem; }
  ul.tl a { color: var(--text); font-weight: 600; text-decoration: none; }
  ul.tl a:hover { text-decoration: underline; }
  button.toggle { font: inherit; font-size: .76rem; color: var(--text-2); background: none;
    border: 1px solid var(--rule); border-radius: 999px; padding: .04rem .5rem; cursor: pointer;
    margin-left: .45rem; white-space: nowrap; vertical-align: middle; }
  button.toggle:hover { background: var(--surface-2); color: var(--text); }
  button.toggle .tri { font-size: .8em; }
  .hyp th:nth-child(2) { width: 50%; }
"""

SCRIPT = """
<script>
for (const b of document.querySelectorAll('button.toggle')) {
  b.addEventListener('click', () => {
    const open = b.getAttribute('aria-expanded') === 'true';
    b.setAttribute('aria-expanded', String(!open));
    b.querySelector('.tri').innerHTML = open ? '\\u25B8' : '\\u25BE';
    for (const r of document.querySelectorAll('tr.child[data-parent="' + b.dataset.toggle + '"]')) r.hidden = open;
  });
}
</script>
"""


def _row(p: dict, parent_href: str = "", n_kids: int = 0) -> str:
    child = bool(parent_href)
    cell = ' style="padding-left:1.7rem"' if child else ""
    arrow = '<span style="color:var(--text-2)">&#8627; </span>' if child else ""
    attrs = f' class="child" data-parent="{parent_href}" hidden' if child else ""
    toggle = (
        f'<button class="toggle" data-toggle="{p["href"]}" aria-expanded="false">'
        f'<span class="tri">&#9656;</span> {n_kids} run{"s" if n_kids != 1 else ""}</button>'
        if n_kids
        else ""
    )
    return (
        f"<tr{attrs}><td class=\"d\">{p['date']}</td>"
        f'<td{cell}>{arrow}<a href="{p["href"]}">{html.escape(p["title"])}</a>'
        f'{(" " + toggle) if toggle else ""}</td>'
        f'<td class="cfg">{html.escape(p.get("config", ""))}</td>'
        f'<td><span class="flag {p["flag"]}">{html.escape(p["label"])}</span></td></tr>'
    )


def table(rows: list[dict]) -> str:
    if not rows:
        return '<p class="none">No experiment pages yet.</p>'
    # a run with a "parent" is a sub-experiment of its synthesis page (when that page is in the same
    # section): the parent row carries a toggle, the children are collapsed under it by default and
    # expand on click. A child whose parent is elsewhere falls back to a normal top-level row.
    hrefs = {p["href"] for p in rows}
    kids: dict[str, list[dict]] = defaultdict(list)
    for p in rows:
        if p.get("parent") and p["parent"] in hrefs:
            kids[p["parent"]].append(p)
    body = ""
    for p in rows:
        if p.get("parent") in hrefs:
            continue  # rendered under its parent below
        ch = kids.get(p["href"], [])
        body += _row(p, n_kids=len(ch))
        for c in ch:
            body += _row(c, parent_href=p["href"])
    return (
        "<table><tr><th>Date</th><th>Experiment</th><th>Config</th><th>Result</th></tr>"
        f"{body}</table>"
    )


# Research-question order (the project's arc); threads not listed append after, sorted. Display
# overrides are only for acronyms the slug can't capitalise on its own.
THREAD_ORDER = [
    "emergent-misalignment",
    "conditional-misalignment",
    "fish-cem",
    "persona-vs-content",
    "mechanism-sharing",
    "em-mitigation",
    "capability-retention",
]
THREAD_NAME = {"fish-cem": "Fish CEM", "em-mitigation": "EM mitigation"}


def thread_name(t: str) -> str:
    return THREAD_NAME.get(t) or t.replace("-", " ").capitalize()


def thread_blocks(exps: list[dict]) -> str:
    """Additive 'by research question' index: every experiment is listed under its thread (newest
    first), linking to the thread's synthesis page when one exists. Does not hide any run — the
    week tables below remain the full chronological browse."""
    by_thread: dict[str, list[dict]] = defaultdict(list)
    for p in exps:
        by_thread[p["thread"]].append(p)
    order = [t for t in THREAD_ORDER if t in by_thread]
    order += sorted(t for t in by_thread if t not in THREAD_ORDER and t != "unfiled")
    if "unfiled" in by_thread:
        order.append("unfiled")
    parts = ['<section class="hyp" id="threads"><h2>By research question</h2>']
    for t in order:
        rows = sorted(by_thread[t], key=lambda p: p["date"], reverse=True)
        synth = next((p for p in rows if p["kind"] == "synthesis"), None)
        head = (
            f'<a href="{synth["href"]}">{html.escape(thread_name(t))}</a>' if synth
            else html.escape(thread_name(t))
        )
        line = html.escape(synth["hypothesis"]) if synth and synth["hypothesis"] else ""
        items = "".join(
            f'<li><span class="d">{p["date"]}</span> <a href="{p["href"]}">'
            f'{html.escape(p["title"])}</a>'
            f'{" &middot; synthesis" if p is synth else ""}</li>'
            for p in rows
        )
        parts.append(
            f'<div class="thread"><h3>{head} <span class="n">{len(rows)}</span></h3>'
            f'{f"<p class=sub>{line}</p>" if line else ""}<ul class="tl">{items}</ul></div>'
        )
    parts.append("</section>")
    return "".join(parts)


def week_blocks(exps: list[dict]) -> list[str]:
    """One section per ISO week of the experiment date (newest week first); within a week, most
    recently updated first (see updated_ts). Undated pages fall into a trailing block."""
    by_week: dict[datetime.date | None, list[dict]] = defaultdict(list)
    for p in exps:
        by_week[week_monday(p["date"]) if p["date"] else None].append(p)
    parts = []
    for wk in sorted((w for w in by_week if w), reverse=True):
        rows = sorted(by_week[wk], key=lambda p: p["updated"], reverse=True)
        parts.append(f'<section class="hyp"><h2>Week of {wk.isoformat()}</h2>{table(rows)}</section>')
    if by_week.get(None):
        rows = sorted(by_week[None], key=lambda p: p["updated"], reverse=True)
        parts.append(f'<section class="hyp"><h2>Undated</h2>{table(rows)}</section>')
    return parts


def legacy_section(old: dict[str, list[dict]]) -> list[str]:
    n_old = sum(len(v) for v in old.values())
    if not n_old:
        return []
    parts = [
        f"<details><summary>Older experiments: {n_old} specs from before the "
        "experiment-page format</summary>"
    ]
    for thread in sorted(old, key=lambda t: (t == "unfiled", t)):
        rows = ""
        for r in old[thread]:
            links = ", ".join(
                f'<a href="../../results/summaries/{s}">{s.removesuffix(".html")}</a>'
                for s in r["summaries"]
            )
            rows += (
                f'<tr><td><a href="{r["spec"]}">{r["spec"].removesuffix(".md")}</a></td>'
                f"<td>{html.escape(r['q'])}</td><td>{links or '—'}</td></tr>"
            )
        parts.append(
            f"<h3>{thread}</h3><table><tr><th>Spec</th><th>Question</th>"
            f"<th>Summary</th></tr>{rows}</table>"
        )
    parts.append("</details>")
    return parts


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def other_pages() -> tuple[list[str], list[tuple[str, str]]]:
    """Pages that are not experiments (literature reviews, weekly reviews, reports, explainers),
    one table per kind, from each page's kind and date meta tags under docs/<folder>/. Returns the
    section HTML and (anchor, label) pairs for the nav bar."""
    by_kind: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for folder in ("literature", "reviews", "reports", "reference"):
        for path in sorted((DOCS / folder).glob("*.html")):
            if path.name == "index.html":
                continue
            text = path.read_text()
            kind = find(r'<meta name="kind" content="([^"]*)"', text) or folder
            date = find(r'<meta name="date" content="([^"]*)"', text) or ""
            title = re.sub(r"<[^>]+>", "", find(r"<h1>(.*?)</h1>", text) or path.stem)
            by_kind[kind].append((date, f"../{folder}/{path.name}", title))
    parts, nav = [], []
    for kind, rows in by_kind.items():
        anchor = f"{slug(kind)}s"
        nav.append((anchor, f"{kind}s"))
        body = "".join(
            f'<tr><td class="d">{d}</td><td><a href="{h}">{html.escape(t)}</a></td></tr>'
            for d, h, t in sorted(rows, reverse=True)
        )
        parts.append(
            f'<section class="hyp" id="{anchor}"><h2>{html.escape(kind)}s</h2>'
            "<table><tr><th>Date</th>"
            f"<th>Page</th></tr>{body}</table></section>"
        )
    return parts, nav


def nav_bar(items: list[tuple[str, str]]) -> str:
    links = "".join(f'<a href="#{a}">{html.escape(l)}</a>' for a, l in items)
    return f'<nav class="bar">{links}</nav>'


def main() -> None:
    exps = pages()
    converted = {f for p in exps for f in p["converted_from"]}
    other, other_nav = other_pages()
    nav = [("threads", "By question"), ("experiments", "By week")] + other_nav
    parts = [
        "<h1>Experiments</h1>",
        '<p class="sub">Browse by research question, or by the week each experiment ran (newest '
        'first; within a week, most recently updated first). Generated by <code>.venv/bin/python -m '
        "scripts.build_experiment_index</code>. Literature reviews, weekly reviews, reports and "
        "reference pages follow.</p>",
        nav_bar(nav),
        thread_blocks(exps),
        '<span id="experiments"></span>',
    ]
    parts += week_blocks(exps)
    parts += other
    old = legacy(converted)
    parts += legacy_section(old)

    OUT.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Experiments</title><style>{STYLE}</style></head><body><main>"
        + "\n".join(parts)
        + "</main>"
        + SCRIPT
        + "</body></html>\n"
    )
    print(
        f"wrote {OUT.relative_to(ROOT)}: {len(exps)} experiment pages, "
        f"{sum(len(v) for v in old.values())} older specs"
    )


if __name__ == "__main__":
    main()
