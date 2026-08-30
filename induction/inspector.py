"""The inspector — the surface a non-technical operations lead actually reads.

Design commitments (from the redesign brief):
  - Plain, calm, one screen. No engine jargon on the face of it.
  - EVERY run, not a sample. The filters are just views over the full list; each
    row opens to its own timeline, and every line resolves to its source record.
  - Adaptive to any source. Nothing here is git- or finance-shaped: it renders
    whatever kinds/runs/steps the model induced, with words from the profile or
    (optionally) an LLM naming pass — falling back to the raw activity verbs.
  - Inference stays legible as inference: a step we didn't see is dashed and
    labelled; findings read as "what's missing", never asserted as fact.

`build_view` projects the InducedModel into exactly what the page needs; the
page is a dumb renderer over that projection.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from induction.emit import disclaimers_for
from induction.pipeline import InducedModel

# Friendly nouns for common anchor types (LLM naming overrides these when on).
_ITEM_WORDS = {
    "invoice": ("invoice", "invoices"), "grant": ("grant", "grants"),
    "payment": ("payment", "payments"), "pr": ("pull request", "pull requests"),
    "issue": ("issue", "issues"), "commit": ("change", "changes"),
    "release": ("release", "releases"), "case": ("case", "cases"),
    "integration": ("integration", "integrations"), "ticket": ("ticket", "tickets"),
    "email_thread": ("email thread", "email threads"), "email": ("email", "emails"),
}


def build_view(m: InducedModel, names: dict | None = None) -> dict:
    names = names or {}
    shaped = m.shaped
    events_by_id = {e.id: e for e in shaped.events}
    obs_by_id = {o.id: o for o in shaped.observations}
    ents = {e.id: e for e in shaped.entities}
    pname = {e.id: e.attrs.get("name", e.id) for e in shaped.entities if e.type == "person"}
    base_step = {s.action: s.name for s in m.steps}

    def step_label(a: str) -> str:
        return (names.get("steps", {}) or {}).get(a) or base_step.get(a) or a.replace("_", " ").title()

    def kind_label(k) -> str:
        return (names.get("kinds", {}) or {}).get(k.id) or k.name

    gaps_by_case = defaultdict(list)
    for g in m.gaps:
        gaps_by_case[g.case_id].append(g)
    kind_of_case = {cid: k for k in m.kinds for cid in k.case_ids}

    # dominant item noun
    dom_type = (Counter(c.anchor.get("type", "run") for c in m.cases.values()).most_common(1)
                or [("run", 0)])[0][0]
    item, items = _ITEM_WORDS.get(dom_type, (dom_type, dom_type + "s"))
    item = names.get("item") or item
    items = names.get("items") or items

    processes = [_process_view(k, m, kind_label, step_label) for k in m.kinds]

    runs = []
    dev_counter = Counter()
    dev_meta = {}
    for case in m.cases.values():
        kind = kind_of_case.get(case.id)
        rv = _run_view(case, kind, m, events_by_id, obs_by_id, ents, pname, step_label, gaps_by_case)
        runs.append(rv)
        dev_counter[rv["dev_key"]] += 1
        dev_meta[rv["dev_key"]] = (rv["dev_label"], rv["dev_attn"])
    # order: usual first, then attention items, then the rest — but the table
    # itself is sortable/filterable, so just keep a stable, readable order.
    runs.sort(key=lambda r: (0 if r["dev_key"] == "usual" else 1, r["id"]))

    # Filter buttons carry a GENERIC label per group (rows keep their specific
    # one), so "Ended early" doesn't masquerade as one particular end point.
    group_label = {"usual": "Follows usual path", "missing": "Missing a step",
                   "ended": "Ended early", "unmatched": "Unmatched record",
                   "incomplete": "Incomplete", "automated": "Automated"}
    filters = [{"key": "all", "label": f"All {items}", "count": len(runs), "attn": False}]
    for key, n in dev_counter.most_common():
        _, attn = dev_meta[key]
        filters.append({"key": key, "label": group_label.get(key, key), "count": n, "attn": attn})

    orphans = [{"id": o.entity_id.split(":")[-1], "reason": o.reason,
                "src": (o.evidence[0].locator if o.evidence else "")} for o in m.orphans]

    return {
        "meta": {
            "title": f"How your {items} run",
            "corpus": _corpus_line(m, items),
            "scope": disclaimers_for(m),
            "ai_named": bool(names.get("_ai")),
            "item": item, "items": items,
        },
        "processes": processes,
        "runs": runs,
        "filters": filters,
        "orphans": orphans,
    }


def _canon(kind) -> list[str]:
    if not kind or not kind.variants:
        return []
    sig = max(kind.variants, key=lambda v: (v.frequency, len(v.signature))).signature
    out = []
    for a in sig:
        if a not in out:
            out.append(a)
    return out


def _process_view(k, m, kind_label, step_label) -> dict:
    canon = _canon(k)
    actors = Counter()
    for cid in k.case_ids:
        case = m.cases.get(cid)
        if not case:
            continue
    # members from the kind's events
    ev_by_id = {e.id: e for e in m.shaped.events}
    pname = {e.id: e.attrs.get("name", e.id) for e in m.shaped.entities if e.type == "person"}
    for cid in k.case_ids:
        case = m.cases.get(cid)
        for eid in (case.event_ids if case else []):
            ev = ev_by_id.get(eid)
            if ev and ev.actor:
                actors[pname.get(ev.actor, ev.actor)] += 1

    max_freq = max((v.frequency for v in k.variants), default=1)
    paths = []
    for v in k.variants:
        seq = [step_label(a) for a in v.signature]
        paths.append({
            "count": v.frequency,
            "seq": seq,
            "label": _path_label(v.signature, canon, step_label),
            "rare": v.role == "one-off" or v.frequency == 1,
            "width": max(6, round(v.frequency / max_freq * 180)),
        })
    return {
        "id": k.id,
        "name": kind_label(k),
        "count": len(k.case_ids),
        "actors": [a for a, _ in actors.most_common(6)],
        "flow": [step_label(a) for a in canon],
        "paths": paths,
        "flagged": k.rejected,
        "flag_note": k.reject_reason or "",
    }


def _path_label(sig, canon, step_label) -> str:
    if not sig:
        return "no dated steps"
    if list(sig) == canon:
        return "most common"
    cset, sset = list(canon), set(sig)
    missing = [a for a in canon if a not in sset]
    extra = [a for a in sig if a not in set(canon)]
    if list(sig) == canon[:len(sig)] and len(sig) < len(canon):
        return f"ended at {step_label(sig[-1])}"
    if missing and not extra and len(sig) == len(canon) - len(missing):
        return "no " + ", ".join(step_label(a) for a in missing[:2]) + " step"
    if len(sig) < len(canon):
        return "fewer steps"
    return "different order"


def _run_view(case, kind, m, events_by_id, obs_by_id, ents, pname, step_label, gaps_by_case) -> dict:
    canon = _canon(kind)
    pos = {a: i for i, a in enumerate(canon)}
    gaps = gaps_by_case.get(case.id, [])

    # recorded steps, in the run's actual order
    recorded = []
    actors = Counter()
    for rid in case.ordered_event_ids:
        if rid in events_by_id:
            ev = events_by_id[rid]
            who = pname.get(ev.actor) if ev.actor else None
            if who:
                actors[who] += 1
            recorded.append({
                "name": step_label(ev.action), "when": (ev.timestamp or "").split("T")[0] or "—",
                "who": who, "inferred": False, "note": "",
                "src": ev.evidence[0].locator if ev.evidence else "",
                "pos": pos.get(ev.action, 99),
            })
        elif rid in obs_by_id:
            ob = obs_by_id[rid]
            recorded.append({
                "name": "Recorded state", "when": (ob.seen_at or "").split("T")[0] or "no date",
                "who": None, "inferred": False,
                "note": str(ob.state.get("status", "") or ob.state.get("text", ""))[:80],
                "src": ob.evidence[0].locator if ob.evidence else "", "pos": 99,
            })

    # inferred (missing / unmatched) steps
    inferred = []
    for g in gaps:
        if g.kind == "missing_expected_step":
            act = g.id.split(":")[-1]
            inferred.append({"name": step_label(act), "when": "not in the records",
                             "who": None, "inferred": True, "note": "expected here, no record",
                             "src": (g.evidence[0].locator if g.evidence else ""),
                             "pos": pos.get(act, 98)})
        elif g.kind == "reconciliation":
            inferred.append({"name": "No matching record", "when": "—", "who": None,
                             "inferred": True, "note": g.description[:90],
                             "src": (g.evidence[0].locator if g.evidence else ""), "pos": 100})

    timeline = sorted(recorded + inferred, key=lambda s: (s["pos"], 1 if s["inferred"] else 0))

    dev_key, dev_label, dev_attn = _deviation(case, kind, canon, gaps, step_label)

    # a stable, human display id + status
    a = case.anchor
    disp = (f"#{a['number']}" if "number" in a else a.get("key") or a.get("branch")
            or case.id.replace("case:", ""))
    status = ""
    for eid in case.entity_ids:
        e = ents.get(eid)
        if e and e.attrs.get("status"):
            status = e.attrs["status"]
            break
    if not status:
        status = (timeline[-1]["name"] if timeline else "—")

    return {
        "id": disp,
        "actor": (actors.most_common(1)[0][0] if actors else "—"),
        "status": status,
        "path": " → ".join(s["name"] for s in timeline if not s["inferred"]) or "—",
        "dev_key": dev_key, "dev_label": dev_label, "dev_attn": dev_attn,
        "timeline": timeline,
    }


def _deviation(case, kind, canon, gaps, step_label):
    if kind and kind.rejected:
        return "automated", "Automated", False
    present = [a for a in case.trace_signature]
    if case.order_status == "unknown" or len(present) <= 1:
        return "incomplete", "Incomplete", False
    if any(g.kind == "missing_expected_step" for g in gaps):
        missing = next((a for a in canon if a not in set(present)), None)
        return "missing", (f"No {step_label(missing)} step" if missing else "Missing a step"), True
    if any(g.kind == "reconciliation" for g in gaps):
        return "unmatched", "Unmatched record", True
    if canon and canon[-1] not in set(present):
        return "ended", f"Ended at {step_label(present[-1])}", False
    return "usual", "—", False


def _corpus_line(m, items) -> str:
    mf = m.manifest or {}
    if mf.get("source_kind") == "email":
        return (f"Read from <b>{mf.get('n_messages', '?')} emails</b> in {m.slug}. Nothing was "
                f"entered by hand; the threads and who-did-what were worked out from the "
                f"messages, and every line opens to the message it came from.")
    if mf.get("head"):
        return (f"Read from <b>{m.slug}</b> — {mf.get('n_commits', '?')} records of activity. "
                f"Nothing was entered by hand; it was worked out from your own history, and "
                f"every line opens to the record it came from.")
    n = mf.get("n_rows", len(m.cases))
    sheets = len(mf.get("sheets", []) or [])
    where = f"{sheets} spreadsheet{'' if sheets == 1 else 's'}" if sheets else "your records"
    return (f"Read from <b>{where}</b> — {n} {items}. Nothing was entered by hand; it was "
            f"worked out from your own records, and every line opens to the row it came from.")


def write_html(m: InducedModel, path: str | Path, names: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    view = build_view(m, names)
    path.write_text(_TEMPLATE.replace("/*DATA*/", json.dumps(view, default=str)))
    return path


# ---------------------------------------------------------------------------
# Self-contained page. Light, calm, vanilla JS. Renders entirely from VIEW.
# ---------------------------------------------------------------------------
_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>How the work runs</title>
<style>
  :root{--bg:#f7f8fa;--card:#fff;--ink:#1a1f2b;--ink-2:#57616f;--ink-3:#8a94a3;
    --line:#e6e9ef;--line-2:#d6dbe4;--accent:#2f6ae0;--accent-soft:#eaf1fd;
    --attn:#b4601a;--attn-soft:#fbf1e7;--flag:#9a6b00;
    --sans:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;--mono:ui-monospace,'SF Mono',Menlo,monospace;}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:var(--sans);background:var(--bg);color:var(--ink);line-height:1.55;font-size:15px;-webkit-font-smoothing:antialiased}
  .wrap{max-width:920px;margin:0 auto;padding:32px 24px 80px}
  h1{font-size:23px;font-weight:700;letter-spacing:-.01em}
  .source{color:var(--ink-2);margin-top:6px;font-size:14px}.source b{color:var(--ink);font-weight:600}
  .ai{display:inline-block;font-size:11px;color:var(--accent);background:var(--accent-soft);border-radius:5px;padding:1px 7px;margin-left:6px}
  section{margin-top:36px}
  .sec-h{font-size:12px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);margin-bottom:6px}
  .sec-lead{font-size:14px;color:var(--ink-2);margin-bottom:16px}
  .proc{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin-bottom:14px}
  .proc.flagged{background:#fbfaf6;border-style:dashed;border-color:#e6d8b8}
  .proc .pt{font-size:18px;font-weight:700}
  .proc .pmeta{color:var(--ink-2);font-size:13px;margin-top:2px}
  .flag-note{font-size:13px;color:var(--flag);background:var(--attn-soft);border-radius:8px;padding:9px 12px;margin-top:12px}
  .flow{display:flex;align-items:center;flex-wrap:wrap;gap:6px;margin:14px 0 4px}
  .step{background:var(--accent-soft);color:var(--accent);border:1px solid #cfe0fb;border-radius:8px;padding:5px 11px;font-size:13px;font-weight:600}
  .arrow{color:var(--ink-3)}
  .paths{margin-top:12px;border-top:1px solid var(--line);padding-top:10px}
  .path{display:flex;align-items:center;gap:12px;padding:5px 0;font-size:13px}
  .path .bar{height:8px;border-radius:4px;background:var(--accent);min-width:6px}
  .path.rare .bar{background:var(--line-2)}
  .path .cnt{font-variant-numeric:tabular-nums;font-weight:700;width:44px}
  .path .seq{color:var(--ink-2)}.path .lbl{color:var(--ink-3);font-size:12px;margin-left:auto;white-space:nowrap}
  .filters{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px}
  .fbtn{font-size:13px;padding:7px 13px;border:1px solid var(--line-2);background:var(--card);border-radius:20px;cursor:pointer;color:var(--ink-2)}
  .fbtn .n{font-variant-numeric:tabular-nums;font-weight:700;color:var(--ink)}
  .fbtn.attn .n{color:var(--attn)}.fbtn:hover{background:#fafbfc}
  .fbtn.on{background:var(--ink);color:#fff;border-color:var(--ink)}.fbtn.on .n{color:#fff}
  .searchrow{margin:12px 0}
  .searchrow input{width:100%;padding:9px 13px;border:1px solid var(--line-2);border-radius:9px;font:14px var(--sans);outline:none}
  .searchrow input:focus{border-color:var(--accent)}
  table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
  thead th{text-align:left;font-size:11px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:var(--ink-3);padding:11px 14px;border-bottom:1px solid var(--line);background:#fbfcfd}
  tbody tr.run{border-bottom:1px solid var(--line);cursor:pointer}tbody tr.run:last-child{border-bottom:none}
  tbody tr.run:hover{background:#fafbfc}
  td{padding:11px 14px;font-size:13.5px;vertical-align:top}
  td.id{font-family:var(--mono);font-size:12.5px}td.path-c{color:var(--ink-2);font-size:12.5px}
  .dev{font-size:12.5px;color:var(--ink-3)}.dev.attn{color:var(--attn);font-weight:600}
  .detail td{padding:0;background:#fbfcfd}
  .tl{padding:14px 20px 16px}.tl .tlh{font-size:11.5px;color:var(--ink-3);margin-bottom:10px;font-family:var(--mono)}
  .ev{display:grid;grid-template-columns:190px 1fr auto;gap:12px;align-items:baseline;padding:6px 0 6px 16px;border-left:2px solid var(--line);margin-left:4px;position:relative}
  .ev::before{content:"";position:absolute;left:-5px;top:11px;width:8px;height:8px;border-radius:50%;background:var(--accent)}
  .ev.inf::before{background:#fff;border:2px solid var(--attn)}
  .ev .en{font-weight:600;font-size:13.5px}.ev .ed{color:var(--ink-2);font-size:13px}
  .ev.inf .en{color:var(--attn)}.ev .src{font-family:var(--mono);font-size:11px;color:var(--accent)}
  .inftag{font-size:10px;font-weight:700;text-transform:uppercase;color:var(--attn);background:var(--attn-soft);padding:1px 6px;border-radius:4px;margin-left:6px}
  .aside{margin-top:12px;font-size:13px;color:var(--ink-2)}.aside b{color:var(--ink)}
  .scope{margin-top:30px;padding-top:16px;border-top:1px solid var(--line);font-size:12.5px;color:var(--ink-3)}
  .scope li{margin:3px 0 3px 18px}
  @media(max-width:640px){.ev{grid-template-columns:1fr}}
</style></head>
<body><div class="wrap">
  <h1 id="title"></h1>
  <p class="source" id="source"></p>

  <section>
    <div class="sec-h">The processes we found</div>
    <div class="sec-lead">How the work actually flows. The common way reads loud; the exceptions stay visible but quiet.</div>
    <div id="procs"></div>
  </section>

  <section>
    <div class="sec-h" id="tableHead"></div>
    <div class="sec-lead">All of them — not a sample. The buttons are just views of this list; each is exact, and every row opens to its source record.</div>
    <div class="filters" id="filters"></div>
    <div class="searchrow"><input id="search" placeholder="Find one by id, owner, or status…"></div>
    <table><thead><tr><th id="thId">Item</th><th>Owner</th><th>Path taken</th><th>Differs how</th></tr></thead>
      <tbody id="rows"></tbody></table>
    <div class="aside" id="orphanNote"></div>
  </section>

  <ul class="scope" id="scope"></ul>
</div>
<script id="data" type="application/json">/*DATA*/</script>
<script>
const V = JSON.parse(document.getElementById('data').textContent);
const esc = s => (s==null?'':String(s)).replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]));
const cap = s => s ? s.charAt(0).toUpperCase()+s.slice(1) : s;

document.getElementById('title').textContent = V.meta.title;
document.getElementById('source').innerHTML = V.meta.corpus + (V.meta.ai_named?' <span class="ai">names suggested by AI</span>':'');
document.getElementById('tableHead').textContent = 'Every ' + V.meta.item;
document.getElementById('thId').textContent = cap(V.meta.item);

document.getElementById('procs').innerHTML = V.processes.map(p=>`
  <div class="proc ${p.flagged?'flagged':''}">
    <div class="pt">${esc(p.name)}</div>
    <div class="pmeta">${p.count} ${esc(V.meta.items)}${p.actors.length?' · '+p.actors.map(esc).join(', '):''}</div>
    ${p.flow.length?`<div class="flow">${p.flow.map((s,i)=>`${i?'<span class="arrow">→</span>':''}<span class="step">${esc(s)}</span>`).join('')}</div>`:''}
    ${p.flagged?`<div class="flag-note"><b>Grouped separately.</b> ${esc(p.flag_note)}</div>`:''}
    <div class="paths">${p.paths.map(pa=>`
      <div class="path ${pa.rare?'rare':''}"><span class="cnt">${pa.count}×</span>
        <div class="bar" style="width:${pa.width}px"></div>
        <span class="seq">${pa.seq.map(esc).join(' → ')||'—'}</span>
        <span class="lbl">${esc(pa.label)}</span></div>`).join('')}</div>
  </div>`).join('');

const rowsEl=document.getElementById('rows'), filtersEl=document.getElementById('filters');
let filter='all', q='';
filtersEl.innerHTML = V.filters.map(f=>
  `<button class="fbtn ${f.attn?'attn':''} ${f.key==='all'?'on':''}" data-f="${esc(f.key)}">${esc(f.label)} <span class="n">${f.count}</span></button>`).join('');

function render(){
  const list = V.runs.filter(r=>{
    if(filter!=='all' && r.dev_key!==filter) return false;
    if(q && !((r.id+' '+r.actor+' '+r.status).toLowerCase().includes(q))) return false;
    return true;
  });
  rowsEl.innerHTML = list.map((r,i)=>{
    const tl = r.timeline.map(s=>`<div class="ev ${s.inferred?'inf':''}">
        <span class="en">${esc(s.name)}${s.inferred?'<span class="inftag">inferred</span>':''}</span>
        <span class="ed">${esc(s.when)}${s.who?' · '+esc(s.who):(s.inferred?'':' · owner not recorded')}${s.note?' — '+esc(s.note):''}</span>
        <span class="src">${esc(s.src)}</span></div>`).join('');
    return `<tr class="run" data-i="${i}"><td class="id">${esc(r.id)}</td><td>${esc(r.actor)}</td>
        <td class="path-c">${esc(r.path)}</td>
        <td><span class="dev ${r.dev_attn?'attn':''}">${esc(r.dev_label)}</span></td></tr>
      <tr class="detail" id="d${i}" hidden><td colspan="4">
        <div class="tl"><div class="tlh">${esc(r.id)} · ${esc(r.status)} — each step resolves to its source record</div>${tl}</div></td></tr>`;
  }).join('') || `<tr><td colspan="4" style="text-align:center;color:var(--ink-3);padding:22px">None match.</td></tr>`;
  rowsEl.querySelectorAll('tr.run').forEach(tr=>tr.onclick=()=>{
    const d=document.getElementById('d'+tr.dataset.i); d.hidden=!d.hidden;});
}
document.getElementById('search').addEventListener('input',e=>{q=e.target.value.toLowerCase();render();});
filtersEl.querySelectorAll('.fbtn').forEach(b=>b.onclick=()=>{
  filtersEl.querySelectorAll('.fbtn').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); filter=b.dataset.f; render();});

document.getElementById('orphanNote').innerHTML = V.orphans.length
  ? `<b>${V.orphans.length}</b> record${V.orphans.length===1?'':'s'} could not be matched to any ${esc(V.meta.item)} (e.g. <code>${esc(V.orphans[0].src)}</code> — ${esc(V.orphans[0].reason)}) — set aside, not counted.`
  : '';
document.getElementById('scope').innerHTML = V.meta.scope.map(s=>`<li>${esc(s)}</li>`).join('');
render();
</script>
</body></html>"""
