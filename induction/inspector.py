"""The thin inspector (brief §5).

A single self-contained HTML file. It is NOT a product surface (that is the
separate Throughline vision) — it exists to prove two things for *any* element:
  - why it is believed  : its evidence resolves back to the raw artefact.
  - how sure we are      : its confidence tier, rendered as solid (evidenced) vs
                           dashed (inferred), common path bold, rare paths quiet.

We embed a *reduced* presentation slice (not the whole 7k-event substrate) so the
file stays light and opens straight from disk. The complete artefact is
model.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from induction.emit import DISCLAIMERS, TIER_LEGEND, _stats
from induction.pipeline import InducedModel

_SAMPLE_CASES_PER_KIND = 4
_ORPHAN_SAMPLE = 25
_GAP_SAMPLE = 8


def build_view(m: InducedModel) -> dict:
    shaped = m.shaped
    events_by_id = {e.id: e for e in shaped.events}
    obs_by_id = {o.id: o for o in shaped.observations}
    person_name = {e.id: e.attrs.get("name", e.id)
                   for e in shaped.entities if e.type == "person"}
    gaps_by_case: dict[str, list] = {}
    for g in m.gaps:
        gaps_by_case.setdefault(g.case_id, []).append(g)

    kinds_view = []
    for k in m.kinds:
        sample_ids = _pick_samples(k)
        kinds_view.append({
            "id": k.id, "name": k.name, "rationale": k.rationale,
            "confidence": k.confidence.to_dict(), "n_cases": len(k.case_ids),
            "rejected": k.rejected, "reject_reason": k.reject_reason,
            "steps": k.steps,
            "variants": [v.to_dict() for v in k.variants[:12]],
            "n_variants": len(k.variants),
            "dfg": k.dfg,
            "samples": [_case_view(m.cases[cid], events_by_id, obs_by_id,
                                   person_name, gaps_by_case)
                        for cid in sample_ids if cid in m.cases],
        })

    return {
        "meta": {
            "slug": m.slug, "manifest": m.manifest, "profile": m.profile_id,
            "tier_legend": TIER_LEGEND, "disclaimers": DISCLAIMERS,
            "stats": _stats(m),
        },
        "kinds": kinds_view,
        "orphans": [o.to_dict() for o in m.orphans[:_ORPHAN_SAMPLE]],
        "n_orphans": len(m.orphans),
        "merges": [mg.to_dict() for mg in m.merges],
        "gaps_by_kind": _gap_samples(m.gaps),
        "members": [
            {"name": e.attrs.get("name"), "is_bot": e.attrs.get("is_bot", False),
             "commit_count": e.attrs.get("commit_count", 0)}
            for e in sorted((e for e in shaped.entities if e.type == "person"),
                            key=lambda e: -e.attrs.get("commit_count", 0))[:20]
        ],
    }


def _pick_samples(kind) -> list[str]:
    picks: list[str] = []
    for role in ("common", "exception", "one-off"):
        for v in kind.variants:
            if v.role == role and v.case_ids:
                picks.append(v.case_ids[0])
                break
    for v in kind.variants:
        if len(picks) >= _SAMPLE_CASES_PER_KIND:
            break
        if v.case_ids and v.case_ids[0] not in picks:
            picks.append(v.case_ids[0])
    return picks[:_SAMPLE_CASES_PER_KIND]


def _case_view(case, events_by_id, obs_by_id, person_name, gaps_by_case) -> dict:
    trace = []
    for rid in case.ordered_event_ids:
        if rid in events_by_id:
            ev = events_by_id[rid]
            link = ev.case_confidence.to_dict() if ev.case_confidence else None
            trace.append({
                "kind": "event", "action": ev.action,
                "actor": person_name.get(ev.actor, ev.actor) if ev.actor else None,
                "timestamp": ev.timestamp,
                "own_tier": ev.confidence.tier.label,
                "link": link,
                "evidence": ev.evidence[0].to_dict() if ev.evidence else None,
                "attrs": {kk: vv for kk, vv in ev.attrs.items() if kk in ("role", "handoff")},
            })
        elif rid in obs_by_id:
            ob = obs_by_id[rid]
            trace.append({
                "kind": "observation", "action": "observed_state",
                "actor": None, "timestamp": ob.seen_at,
                "own_tier": ob.confidence.tier.label,
                "link": ob.case_confidence.to_dict() if ob.case_confidence else None,
                "evidence": ob.evidence[0].to_dict() if ob.evidence else None,
                "state": {"version": ob.state.get("version"),
                          "status": ob.state.get("status"),
                          "text": ob.state.get("text", "")[:160]},
            })
    return {
        "id": case.id, "anchor": case.anchor,
        "confidence": case.confidence.to_dict(),
        "order_status": case.order_status,
        "trace": trace,
        "gaps": [g.to_dict() for g in gaps_by_case.get(case.id, [])],
    }


def _gap_samples(gaps) -> list[dict]:
    by_kind: dict[str, list] = {}
    for g in gaps:
        by_kind.setdefault(g.kind, []).append(g)
    out = []
    for kind, gs in by_kind.items():
        out.append({
            "kind": kind, "count": len(gs),
            "samples": [g.to_dict() for g in gs[:_GAP_SAMPLE]],
        })
    return out


def write_html(m: InducedModel, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    view = build_view(m)
    html = _TEMPLATE.replace("/*DATA*/", json.dumps(view, default=str))
    path.write_text(html)
    return path


# ---------------------------------------------------------------------------
# Self-contained HTML. Vanilla JS, no external requests. Kept deliberately plain.
# ---------------------------------------------------------------------------
_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Induction Engine — Inspector</title>
<style>
  :root{--bg:#0f1115;--panel:#171a21;--ink:#e6e8ee;--dim:#9aa3b2;--line:#2a2f3a;
        --direct:#3fb950;--joined:#58a6ff;--heuristic:#d29922;--model:#f85149;--bot:#8957e5;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  header{padding:20px 26px;border-bottom:1px solid var(--line);background:var(--panel)}
  h1{margin:0 0 4px;font-size:19px} h2{font-size:16px;margin:26px 0 8px}
  .sub{color:var(--dim);font-size:13px}
  main{max-width:1080px;margin:0 auto;padding:22px 26px 80px}
  .stats{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
  .stat{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 12px}
  .stat b{font-size:17px} .stat span{color:var(--dim);display:block;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
  .chip{display:inline-block;font-size:11px;padding:1px 7px;border-radius:20px;border:1px solid;vertical-align:middle}
  .direct{color:var(--direct);border-color:var(--direct)} .joined{color:var(--joined);border-color:var(--joined)}
  .heuristic{color:var(--heuristic);border-color:var(--heuristic)} .model{color:var(--model);border-color:var(--model)}
  .kind{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:14px 0}
  .kind.rejected{opacity:.85;border-style:dashed;border-color:var(--heuristic)}
  .rej{color:var(--heuristic);font-size:13px;margin-top:6px;border-left:3px solid var(--heuristic);padding-left:10px}
  .variants{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}
  .variant{border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:12px;min-width:150px}
  .variant.common{border-color:var(--direct);font-weight:600} .variant.exception{border-color:var(--heuristic)}
  .variant.one-off{border-style:dashed;color:var(--dim)}
  .sig{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;color:var(--dim)}
  .trace{border-left:2px solid var(--line);margin:8px 0 8px 6px;padding-left:14px}
  .node{position:relative;margin:9px 0;padding:8px 10px;background:#12151b;border:1px solid var(--line);border-radius:8px}
  .node::before{content:"";position:absolute;left:-21px;top:14px;width:9px;height:9px;border-radius:50%;background:var(--joined)}
  .node.inferred{border-style:dashed;border-color:var(--heuristic);background:#1a160e}
  .node.inferred::before{background:var(--heuristic)}
  .node.obs{border-color:var(--model)} .node.obs::before{background:var(--model)}
  .ev{color:var(--dim);font-size:12px;margin-top:4px}
  .ev code{color:#c9d1d9;background:#0c0e12;padding:1px 5px;border-radius:4px;font-size:11px}
  .row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  details{margin:6px 0} summary{cursor:pointer;color:var(--joined)}
  .muted{color:var(--dim)} .bot{color:var(--bot)}
  table{border-collapse:collapse;width:100%;font-size:13px} td,th{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left}
  .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;margin-top:8px}
  .tabs{display:flex;gap:6px;margin:18px 0 4px;flex-wrap:wrap}
  .tab{padding:6px 12px;border:1px solid var(--line);border-radius:8px;cursor:pointer;background:var(--panel)}
  .tab.active{border-color:var(--joined);color:var(--joined)}
  section[hidden]{display:none}
  .banner{background:#1a160e;border:1px solid var(--heuristic);border-radius:8px;padding:10px 12px;color:#f0d9a8;font-size:13px;margin-top:12px}
</style></head>
<body>
<header>
  <h1>Induction Engine — Inspector</h1>
  <div class="sub" id="subtitle"></div>
  <div class="legend" id="legend"></div>
  <div class="stats" id="stats"></div>
  <div class="banner" id="cannot"></div>
</header>
<main>
  <div class="tabs" id="tabs"></div>
  <section id="tab-processes"></section>
  <section id="tab-honesty" hidden></section>
  <section id="tab-members" hidden></section>
</main>
<script id="data" type="application/json">/*DATA*/</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const tierChip = c => c ? `<span class="chip ${c.tier}" title="${(c.rationale||'').replace(/"/g,'&quot;')}">${c.tier}</span>` : '';
const esc = s => (s==null?'':String(s)).replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]));

document.getElementById('subtitle').innerHTML =
  `corpus <b>${esc(D.meta.slug)}</b> · git history @ <code>${esc((D.meta.manifest.head||'').slice(0,10))}</code> · ${D.meta.manifest.n_commits||'?'} commits · vocabulary: <b>${esc(D.meta.profile||'generic')}</b>`;
document.getElementById('legend').innerHTML = Object.entries(D.meta.tier_legend)
  .map(([t,desc])=>`<span><span class="chip ${t}">${t}</span> ${esc(desc)}</span>`).join('');
const s = D.meta.stats;
document.getElementById('stats').innerHTML = [
  ['process kinds', s.n_process_kinds],['cases (instances)', s.n_cases],
  ['events', s.n_events],['orphans', s.n_orphans],['gaps (inferred)', s.n_gaps],
  ['same-activity merges', s.n_same_activity_merges],
  ['order: unknown', (s.order_status && s.order_status.unknown)||0],
].map(([k,v])=>`<div class="stat"><b>${v}</b><span>${k}</span></div>`).join('');
document.getElementById('cannot').innerHTML =
  '<b>What this cannot conclude.</b> ' + D.meta.disclaimers.map(esc).join(' · ');

// tabs
const tabs=[['processes','Processes'],['honesty','Honesty ledger'],['members','Members']];
document.getElementById('tabs').innerHTML = tabs.map(([id,l],i)=>
  `<div class="tab ${i===0?'active':''}" data-t="${id}">${l}</div>`).join('');
function activate(which){
  document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active', x.dataset.t===which));
  tabs.forEach(([id])=>document.getElementById('tab-'+id).hidden = (id!==which));
}
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{ location.hash=t.dataset.t; activate(t.dataset.t); });
if(location.hash){ const w=location.hash.slice(1); if(tabs.some(([id])=>id===w)) activate(w); }

function traceNode(step){
  if(step.kind==='observation'){
    return `<div class="node obs"><div class="row"><b>state observed</b> ${tierChip(step.link)}
      <span class="muted">no actor · no time · order unknown</span></div>
      <div class="ev">v${esc(step.state.version)} · <span class="muted">${esc(step.state.status||'')}</span> — ${esc(step.state.text)}</div>
      ${step.evidence?`<div class="ev">evidence: <code>${esc(step.evidence.locator)}</code></div>`:''}</div>`;
  }
  const role = step.attrs && step.attrs.role ? ` <span class="muted">(${esc(step.attrs.role)})</span>`:'';
  return `<div class="node"><div class="row"><b>${esc(step.action)}</b>${role}
     ${step.actor?`<span class="muted">by ${esc(step.actor)}</span>`:'<span class="muted">actor unknown</span>'}
     ${step.timestamp?`<span class="muted">· ${esc(step.timestamp.slice(0,10))}</span>`:''}
     ${tierChip(step.link)}</div>
     ${step.evidence?`<div class="ev">${esc(step.evidence.snippet||'')} <br>evidence: <code>${esc(step.evidence.locator)}</code> <span class="muted">(git show)</span></div>`:''}</div>`;
}
function gapNode(g){
  return `<div class="node inferred"><div class="row"><b>⤳ ${esc(g.kind)}</b> ${tierChip(g.confidence)} <span class="muted">inferred — off-system</span></div>
    <div class="ev">${esc(g.description)}${g.evidence&&g.evidence[0]?` <br>signal: <code>${esc(g.evidence[0].locator)}</code>`:''}</div></div>`;
}
function caseView(c, open){
  const gapsBefore = c.gaps.filter(g=>g.kind==='off_system_pr_open_review');
  return `<details ${open?'open':''}><summary>run <code>${esc(c.id)}</code> · ${esc(c.order_status)} ${tierChip(c.confidence)}</summary>
    <div class="trace">
      ${gapsBefore.map(gapNode).join('')}
      ${c.trace.map(traceNode).join('')}
      ${c.gaps.filter(g=>g.kind!=='off_system_pr_open_review').map(gapNode).join('')}
    </div></details>`;
}
function kindView(k){
  return `<div class="kind ${k.rejected?'rejected':''}">
    <div class="row"><h2 style="margin:0">${esc(k.name)}</h2> ${tierChip(k.confidence)}
      <span class="muted">${k.n_cases} runs · ${k.n_variants} variants</span></div>
    <div class="sub">${esc(k.rationale)}</div>
    ${k.rejected?`<div class="rej"><b>Looks like a process, isn't.</b> ${esc(k.reject_reason)}</div>`:''}
    <div class="variants">${k.variants.map(v=>`
      <div class="variant ${v.role}"><div class="row"><b>${v.frequency}×</b> <span>${esc(v.role)}</span></div>
        <div class="sig">${v.signature.map(esc).join(' → ')||'∅'}</div></div>`).join('')}</div>
    ${k.samples.length?`<div class="muted" style="margin-top:6px">Example runs (evidence resolves to a git sha):</div>`:''}
    ${k.samples.map((c,i)=>caseView(c, i===0)).join('')}
  </div>`;
}
document.getElementById('tab-processes').innerHTML =
  '<h2>Process definitions</h2><div class="sub">Each kind is a proposed boundary (heuristic). The common path reads loud; exceptions and one-offs stay visible but quiet. Every run drills down to evidence.</div>'
  + D.kinds.map(kindView).join('');

// honesty ledger
let h = '<h2>Orphans <span class="muted">('+D.n_orphans+' records joined to no run — surfaced, never dropped)</span></h2>';
h += '<table><tr><th>entity</th><th>reason</th><th>evidence</th></tr>' +
  D.orphans.map(o=>`<tr><td><code>${esc(o.entity_id.slice(0,26))}</code></td><td class="muted">${esc(o.reason)}</td><td><code>${esc((o.evidence[0]||{}).locator||'')}</code></td></tr>`).join('') +
  '</table>';
h += '<h2>Gaps <span class="muted">(off-system steps inferred from real signals — rendered as inference)</span></h2>';
D.gaps_by_kind.forEach(gk=>{ h += `<h3 style="margin:14px 0 4px">${esc(gk.kind)} <span class="muted">×${gk.count}</span></h3>` + gk.samples.map(gapNode).join(''); });
h += '<h2>Same activity, different people <span class="muted">('+D.merges.length+' — merged into one step, all records kept)</span></h2>';
h += D.merges.slice(0,12).map(m=>`<div class="node"><div class="row"><b>${esc(m.action)}</b> ${m.members.map(x=>`<span class="chip joined">${esc(x)}</span>`).join('')}</div>
   <div class="ev">${esc(m.rationale)} <br>evidence: ${m.evidence.map(e=>`<code>${esc(e.locator)}</code>`).join(' ')}</div></div>`).join('');
document.getElementById('tab-honesty').innerHTML = h;

// members
document.getElementById('tab-members').innerHTML = '<h2>Members (actors)</h2>' +
  '<table><tr><th>name</th><th>commits</th><th></th></tr>' +
  D.members.map(m=>`<tr><td class="${m.is_bot?'bot':''}">${esc(m.name)}</td><td>${m.commit_count}</td><td>${m.is_bot?'<span class="chip model">bot</span>':''}</td></tr>`).join('') +
  '</table>';
</script>
</body></html>"""
