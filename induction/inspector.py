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

from induction.abstraction import Abstraction
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


def build_view(m: InducedModel, names: dict | None = None, activities: dict | None = None) -> dict:
    names = names or {}
    activities = activities or {}
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

    # Collective noun. Labelling every run by the single most common anchor type
    # ("pull request") overfits the moment a corpus is heterogeneous — a company
    # runs many processes across many systems and products, not one kind of item.
    # Use the specific noun ONLY when one type genuinely dominates; otherwise a
    # neutral "run", and let the kinds section below carry the real diversity.
    type_counts = Counter(c.anchor.get("type", "run") for c in m.cases.values())
    dom_type, dom_n = (type_counts.most_common(1) or [("run", 0)])[0]
    total = sum(type_counts.values()) or 1
    if len(type_counts) <= 1 or dom_n / total >= 0.8:
        item, items = _ITEM_WORDS.get(dom_type, (dom_type, dom_type + "s"))
    else:
        item, items = "run", "runs"      # mixed corpus — claim no single product
    item = names.get("item") or item
    items = names.get("items") or items

    runs = []
    dev_counter = Counter()
    dev_meta = {}
    for case in m.cases.values():
        kind = kind_of_case.get(case.id)
        rv = _run_view(case, kind, m, events_by_id, obs_by_id, ents, pname, step_label,
                       gaps_by_case, activities)
        # which induced process this run is an instance of — the link between the
        # "processes we found" section and the flat list, and how a multi-product
        # corpus is navigated (one kind per distinct process/offering).
        rv["kind"] = kind_label(kind) if kind else "Ungrouped"
        rv["kind_id"] = kind.id if kind else "none"
        rv["kind_attn"] = bool(kind and kind.rejected)
        runs.append(rv)
        dev_counter[rv["dev_key"]] += 1
        dev_meta[rv["dev_key"]] = (rv["dev_label"], rv["dev_attn"])
    # order: usual first, then attention items, then the rest — but the table
    # itself is sortable/filterable, so just keep a stable, readable order.
    runs.sort(key=lambda r: (0 if r["dev_key"] == "usual" else 1, r["id"]))

    # The kind cards' flow and variants are built from the runs' ACTIVITY spines —
    # the same abstraction the detail shows — so "the processes we found" reads as
    # activities (Raised → Fixed → Shipped), not the raw artefact verbs.
    runs_by_kind = defaultdict(list)
    for rv in runs:
        runs_by_kind[rv["kind_id"]].append(rv)
    processes = [_process_view(k, m, kind_label, runs_by_kind.get(k.id, [])) for k in m.kinds]

    # Filter buttons carry a GENERIC label per group (rows keep their specific
    # one), so "Ended early" doesn't masquerade as one particular end point.
    group_label = {"usual": "Follows usual path", "missing": "Missing a step",
                   "ended": "Ended early", "unmatched": "Unmatched record",
                   "incomplete": "Incomplete", "automated": "Automated"}
    filters = [{"key": "all", "label": f"All {items}", "count": len(runs), "attn": False}]
    # By process kind — the primary way to navigate a multi-process / multi-product
    # corpus (only shown when there is more than one kind to choose between).
    kind_count = Counter(r["kind_id"] for r in runs)
    kmeta = {r["kind_id"]: (r["kind"], r["kind_attn"]) for r in runs}
    if len(kind_count) > 1:
        for kid, n in kind_count.most_common():
            label, attn = kmeta[kid]
            filters.append({"key": f"kind:{kid}", "label": label, "count": n, "attn": attn})
    # then by how a run deviates from its kind's common path
    for key, n in dev_counter.most_common():
        _, attn = dev_meta[key]
        filters.append({"key": key, "label": group_label.get(key, key), "count": n, "attn": attn})

    orphans = [{"id": o.entity_id.split(":")[-1], "reason": o.reason,
                "src": (o.evidence[0].locator if o.evidence else "")} for o in m.orphans]

    abstraction = Abstraction.of(activities)
    read_names = {row["activity"] for row in abstraction.vocabulary
                  if not row.get("unclassified")}
    for r in runs:
        # A run carries an unread record when one of its steps is still the
        # source's own verb — i.e. the classifier declined it. Surfacing that per
        # run is what makes the abstention count checkable instead of a total.
        r["unread"] = bool(read_names) and any(
            n["name"] not in read_names for n in r["activities"])

    return {
        "meta": {
            "title": "How the work runs" if items == "runs" else f"How your {items} run",
            "corpus": _corpus_line(m, items),
            "scope": disclaimers_for(m),
            "ai_named": bool(names.get("_ai")),
            "ai_steps": bool(abstraction),
            "item": item, "items": items,
        },
        "processes": processes,
        "runs": runs,
        "filters": filters,
        "vocabulary": abstraction.vocabulary,
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


def _process_view(k, m, kind_label, kind_runs) -> dict:
    # actors from the kind's events (richer than one-per-run)
    ev_by_id = {e.id: e for e in m.shaped.events}
    pname = {e.id: e.attrs.get("name", e.id) for e in m.shaped.entities if e.type == "person"}
    actors = Counter()
    for cid in k.case_ids:
        case = m.cases.get(cid)
        for eid in (case.event_ids if case else []):
            ev = ev_by_id.get(eid)
            if ev and ev.actor:
                actors[pname.get(ev.actor, ev.actor)] += 1

    # Flow and variants at the ACTIVITY level, taken from the runs' own inferred
    # spines — so the card shows the process (Raised → Fixed → Shipped), not the
    # raw artefact verbs the systems happened to record. A variant is a distinct
    # activity path; its frequency is how many runs took it.
    ident = lambda x: x
    sigs = Counter(tuple(n["name"] for n in rv["activities"]) for rv in kind_runs)
    canon = list(max(sigs, key=lambda s: (sigs[s], len(s))) if sigs else ())
    max_freq = max(sigs.values(), default=1)
    paths = [{
        "count": freq,
        "seq": list(sig),
        "label": _path_label(sig, canon, ident),
        "rare": freq == 1 and len(sigs) > 1,
        "width": max(6, round(freq / max_freq * 180)),
    } for sig, freq in sigs.most_common()]

    return {
        "id": k.id,
        "name": kind_label(k),
        "count": len(k.case_ids),
        "actors": [a for a, _ in actors.most_common(6)],
        "flow": canon,
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


# How sure we are a run holds together, in words a non-technical reader can act
# on. Only the two *inferred* tiers get a chip — a deterministic run is the norm
# and needs no badge; the point is to make a guess look like a guess.
_TIER_CHIP = {
    "heuristic": ("matched on wording", "t-soft"),
    "model": ("AI judged same work", "t-ai"),
}
_SRC_WORD = {"github": "GitHub", "mail": "email", "git": "git history",
             "changelog": "changelog", "tabular": "spreadsheet", "finance": "spreadsheet"}


def _run_sources(case, ents) -> list[str]:
    """The distinct systems a run spans — the honest headline of a cross-source
    case ('this one activity lived in email AND GitHub')."""
    words: list[str] = []
    for eid in case.entity_ids:
        e = ents.get(eid)
        if not e:
            continue
        w = _SRC_WORD.get(e.source.split(":")[0], e.source.split(":")[0])
        if w not in words:
            words.append(w)
    return words


def _run_title(case, ents, disp: str) -> str:
    """A human name for the run — the anchor's title/subject, so an ops lead sees
    'SSO login fails after token refresh', not 'case:pr:15'."""
    anchor_id = case.id[5:] if case.id.startswith("case:") else case.id
    for eid in [anchor_id, *case.entity_ids]:
        e = ents.get(eid)
        if e:
            t = e.attrs.get("title") or e.attrs.get("subject")
            if t:
                return t.strip()
    return disp


def _run_view(case, kind, m, events_by_id, obs_by_id, ents, pname, step_label,
              gaps_by_case, activities=None) -> dict:
    canon = _canon(kind)
    gaps = gaps_by_case.get(case.id, [])
    abstraction = Abstraction.of(activities)

    def _activity(ev) -> str:
        """The activity a record realises — the reading of the record itself when
        there is one, else the verb map, else the verb's own label. The reading is
        what turns 400 emails that all say 'sent' into Requested / Reviewed /
        Approved; the map is what turns 'issue opened' + 'email sent' into one
        'Raised'. With neither we show the artefact's own verb, claiming no
        abstraction we did not earn."""
        artefact = ents[ev.entity_id].type if ev.entity_id in ents else "record"
        return abstraction.activity_of(ev.id, artefact, ev.action) or step_label(ev.action)

    # action -> activity name where a verb maps to one activity across artefact
    # types, so a MISSING step (which carries only a verb) still reads as the
    # process activity, not the raw verb.
    act2activity: dict = {}
    for key, name in abstraction.by_vocab.items():
        a = key.split("/", 1)[-1]
        act2activity[a] = None if (a in act2activity and act2activity[a] != name) else name

    def _act_name(action: str) -> str:
        return act2activity.get(action) or action.replace("_", " ")

    # artefacts, in the run's real order — the evidence that will sit under the
    # activities. Each keeps its own verb, time, owner and source locator.
    arts = []
    actors = Counter()
    for rid in case.ordered_event_ids:
        if rid in events_by_id:
            ev = events_by_id[rid]
            who = pname.get(ev.actor) if ev.actor else None
            if who:
                actors[who] += 1
            ent = ents.get(ev.entity_id)
            art_type = ent.type if ent else "record"
            num = ent.attrs.get("number") if ent else None
            src = ev.evidence[0].locator if ev.evidence else ""
            arts.append({
                "activity": _activity(ev),
                "artefact": _ITEM_WORDS.get(art_type, (art_type, art_type))[0],
                "ref": f"#{num}" if num else "",
                # the artefact's own action, described plainly — this is evidence
                # detail ("review requested"), not the inferred step name
                "verb": ev.action.replace("_", " "),
                "when": (ev.timestamp or "").split("T")[0] or "—",
                # A read activity shows the span it was read from: the difference
                # between a claim you can check and one you have to accept.
                "who": who, "inferred": False, "note": abstraction.span_of(ev.id) or "",
                "src": src, "is_url": src.startswith("http"),
                "src_kind": _SRC_WORD.get(ev.source.split(":")[0], ev.source.split(":")[0]),
            })
        elif rid in obs_by_id:
            ob = obs_by_id[rid]
            src = ob.evidence[0].locator if ob.evidence else ""
            arts.append({
                "activity": "State recorded", "artefact": "record", "ref": "", "verb": "recorded",
                "when": (ob.seen_at or "").split("T")[0] or "no date", "who": None,
                "inferred": False, "note": str(ob.state.get("status", "") or ob.state.get("text", ""))[:80],
                "src": src, "is_url": src.startswith("http"),
                "src_kind": _SRC_WORD.get(ob.source.split(":")[0], ob.source.split(":")[0]),
            })

    # Fold consecutive artefacts that realise the SAME activity into one step; the
    # artefacts become that step's evidence. THIS is the process — a spine of
    # activities — with the systems' records hanging beneath, not the raw verbs.
    nodes: list[dict] = []
    for it in arts:
        if nodes and nodes[-1]["name"] == it["activity"]:
            nodes[-1]["arts"].append(it)
        else:
            nodes.append({"name": it["activity"], "arts": [it]})
    for n in nodes:
        whens = [a["when"] for a in n["arts"] if a["when"] not in ("—", "no date")]
        n["when"] = min(whens) if whens else "—"
        n["sources"] = sorted({a["src_kind"] for a in n["arts"] if a["src_kind"]})
        n["n"] = len(n["arts"])

    # inferred (missing / off-system) steps — the confidently-incomplete part
    inferred = []
    for g in gaps:
        if g.kind == "missing_expected_step":
            act = g.id.split(":")[-1]
            inferred.append({"name": _act_name(act), "when": "not in the records",
                             "note": "expected here, no record",
                             "src": (g.evidence[0].locator if g.evidence else "")})
        elif g.kind == "reconciliation":
            inferred.append({"name": "No matching record", "when": "—",
                             "note": g.description[:90],
                             "src": (g.evidence[0].locator if g.evidence else "")})

    dev_key, dev_label, dev_attn = _deviation(case, kind, canon, gaps, step_label)

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
        status = (nodes[-1]["name"] if nodes else "—")

    tier = case.confidence.tier.label
    chip_word, chip_class = _TIER_CHIP.get(tier, ("", ""))
    sources = _run_sources(case, ents)

    return {
        "id": disp,
        "title": _run_title(case, ents, disp),
        "actor": (actors.most_common(1)[0][0] if actors else "—"),
        "status": status,
        # the spine is the activities, not the raw verbs
        "path": " → ".join(n["name"] for n in nodes) or "—",
        "dev_key": dev_key, "dev_label": dev_label, "dev_attn": dev_attn,
        # Honesty on the face of the row: an inferred join wears a chip and, when
        # it is the model tier, the model's own reason; a deterministic run wears
        # nothing. Cross-source runs say which systems they crossed.
        "tier": tier, "chip_word": chip_word, "chip_class": chip_class,
        "why": (case.confidence.rationale or "") if tier in ("heuristic", "model") else "",
        "sources": sources, "cross": len(sources) > 1,
        "activities": nodes, "inferred": inferred,
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


def _corpus_sources(m) -> list[str]:
    """The distinct systems the corpus was read from, in friendly words."""
    return sorted({_SRC_WORD.get(e.source.split(":")[0], e.source.split(":")[0])
                   for e in m.shaped.entities
                   if e.type != "person" and getattr(e, "source", "")})


def _join_words(ws: list[str]) -> str:
    if len(ws) <= 1:
        return ws[0] if ws else "your systems"
    return ", ".join(ws[:-1]) + " and " + ws[-1]


def _corpus_line(m, items) -> str:
    mf = m.manifest or {}
    srcs = _corpus_sources(m)
    if mf.get("source_kind") == "email":
        return (f"Read from <b>{mf.get('n_messages', '?')} emails</b> in {m.slug}. Nothing was "
                f"entered by hand; the threads and who-did-what were worked out from the "
                f"messages, and every line opens to the message it came from.")
    if mf.get("head"):
        return (f"Read from <b>{m.slug}</b> — {mf.get('n_commits', '?')} records of activity. "
                f"Nothing was entered by hand; it was worked out from your own history, and "
                f"every line opens to the record it came from.")
    if mf.get("source_kind") == "combined" or len(srcs) > 1:
        n = mf.get("n_records") or sum(1 for e in m.shaped.entities if e.type != "person")
        return (f"Read across <b>{_join_words(srcs)}</b> — {n} records, nothing entered by "
                f"hand. The runs, the steps and who did what were worked out from the "
                f"artefacts themselves; different systems, one process. Every step opens to "
                f"the record it came from.")
    n = mf.get("n_rows", len(m.cases))
    sheets = len(mf.get("sheets", []) or [])
    where = f"{sheets} spreadsheet{'' if sheets == 1 else 's'}" if sheets else "your records"
    return (f"Read from <b>{where}</b> — {n} {items}. Nothing was entered by hand; it was "
            f"worked out from your own records, and every line opens to the row it came from.")


def write_html(m: InducedModel, path: str | Path, names: dict | None = None,
               activities: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    view = build_view(m, names, activities)
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
  table.vocab{width:100%;border-collapse:collapse;margin-top:4px}
  table.vocab th{text-align:left;font-size:11px;letter-spacing:.06em;text-transform:uppercase;
    color:var(--ink-3);font-weight:600;padding:8px 10px;border-bottom:1px solid var(--line-2)}
  table.vocab th.num,table.vocab td.num{text-align:right;font-variant-numeric:tabular-nums}
  .vrow{cursor:pointer;border-bottom:1px solid var(--line)}
  .vrow>td{padding:9px 10px;vertical-align:top;font-size:13px}
  .vrow:hover{background:var(--accent-soft)}
  .vrow.on{background:var(--accent-soft);box-shadow:inset 3px 0 0 var(--accent)}
  .vrow.vunread>td:first-child b{color:var(--attn)}
  .vnote{font-size:12px;color:var(--ink-3);margin-top:2px}
  .vph{color:var(--ink-2)}
  .ph{display:inline-block;background:var(--bg);border:1px solid var(--line-2);border-radius:4px;
    padding:1px 6px;margin:0 4px 4px 0;font-family:var(--mono);font-size:11px}
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
  .kind{font-size:12.5px;color:var(--ink-2)}.kind.flag{color:var(--flag)}
  .rtitle{font-weight:600;color:var(--ink);font-size:13.5px}
  .rsub{font-family:var(--mono);font-size:11.5px;color:var(--ink-3);margin-top:2px}
  .tchip{display:inline-block;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.03em;padding:1px 6px;border-radius:4px;margin-left:6px;vertical-align:middle}
  .tchip.t-soft{color:var(--flag);background:var(--attn-soft)}
  .tchip.t-ai{color:var(--accent);background:var(--accent-soft);border:1px dashed #b8ccf5}
  .xchip{display:inline-block;font-size:10px;font-weight:600;color:var(--ink-2);background:#eef1f5;border-radius:4px;padding:1px 6px;margin-left:6px;vertical-align:middle}
  .why{font-size:12.5px;color:var(--accent);background:var(--accent-soft);border-radius:8px;padding:8px 12px;margin:2px 20px 10px}.why b{color:var(--ink)}
  .actnode{padding:2px 20px 6px}.actnode+.actnode{border-top:1px solid var(--line);padding-top:9px;margin-top:5px}
  .acth{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
  .actn{font-weight:700;font-size:14px;color:var(--ink)}.actnode.inf .actn{color:var(--attn)}
  .actm{font-size:11.5px;color:var(--ink-3)}
  .arts{margin:5px 0 0}
  .art{font-size:10.5px;color:var(--ink-2);background:#eef1f5;border-radius:4px;padding:0 6px;margin-left:6px;font-weight:600}
  .evlabel{font-size:10.5px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.05em;margin:5px 0 1px}
  .evref{font-weight:600;color:var(--ink)}
  .evverb{color:var(--ink-2);font-weight:400;margin-left:7px}
  .ev .src a{color:var(--accent);text-decoration:none;font-weight:600}.ev .src a:hover{text-decoration:underline}
  .framing{font-size:12.5px;color:var(--ink-2);background:#f2f5fb;border:1px solid var(--line);border-radius:9px;padding:9px 13px;margin:10px 0 4px}.framing b{color:var(--ink)}
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

  <section id="vocabSec" hidden>
    <div class="sec-h">What each step was read from</div>
    <div class="sec-lead">The steps above are not in the records — the systems only
      recorded that a message was sent. Each one was read from the message's own
      words. This is that reading, in bulk: what it decided, how often, and the
      phrases it went on. Click any row to see only the runs it touched.</div>
    <table class="vocab"><thead><tr><th>Step</th><th class="num">Records</th>
      <th>Read from phrases like</th></tr></thead>
      <tbody id="vocabRows"></tbody></table>
  </section>

  <section>
    <div class="sec-h" id="tableHead"></div>
    <div class="sec-lead">All of them — not a sample. The buttons are just views of this list; each is exact, and every row opens to its source record.</div>
    <div class="filters" id="filters"></div>
    <div class="searchrow"><input id="search" placeholder="Find one by id, owner, or status…"></div>
    <table><thead><tr><th id="thId">Item</th><th>Process</th><th>Owner</th><th>Path taken</th><th>Differs how</th></tr></thead>
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

if((V.vocabulary||[]).length){
  document.getElementById('vocabSec').hidden = false;
  document.getElementById('vocabRows').innerHTML = V.vocabulary.map(v=>`
    <tr class="vrow ${v.unclassified?'vunread':''}" data-act="${v.unclassified?'__unread__':esc(v.activity)}">
      <td><b>${esc(v.activity)}</b>${v.unclassified?'<div class="vnote">kept the source\u2019s own verb \u2014 not read into a step</div>':''}</td>
      <td class="num">${v.n}</td>
      <td class="vph">${v.phrases.map(pp=>`<span class="ph">${esc(pp)}</span>`).join('') || '\u2014'}</td>
    </tr>`).join('');
}

const rowsEl=document.getElementById('rows'), filtersEl=document.getElementById('filters');
let filter='all', q='';
filtersEl.innerHTML = V.filters.map(f=>
  `<button class="fbtn ${f.attn?'attn':''} ${f.key==='all'?'on':''}" data-f="${esc(f.key)}">${esc(f.label)} <span class="n">${f.count}</span></button>`).join('');

function render(){
  const list = V.runs.filter(r=>{
    if(filter.startsWith('kind:')){ if(r.kind_id!==filter.slice(5)) return false; }
    else if(filter==='act:__unread__'){ if(!r.unread) return false; }
    else if(filter.startsWith('act:')){
      const want=filter.slice(4);
      if(!r.activities.some(n=>n.name===want)) return false;
    }
    else if(filter!=='all' && r.dev_key!==filter) return false;
    if(q && !((r.id+' '+r.title+' '+r.actor+' '+r.status+' '+r.kind).toLowerCase().includes(q))) return false;
    return true;
  });
  // one artefact = one piece of evidence, traceable to its source record
  const art = a=>`<div class="ev ${a.inferred?'inf':''}">
      <span class="en"><span class="evref">${esc(a.artefact)}${a.ref?' '+esc(a.ref):''}</span><span class="evverb">${esc(a.verb)}${a.who?' · '+esc(a.who):''}</span></span>
      <span class="ed">${esc(a.when)}</span>
      <span class="src">${a.is_url?`<a href="${esc(a.src)}" target="_blank" rel="noopener">open ↗</a>`:esc(a.src||'—')}</span></div>`;
  rowsEl.innerHTML = list.map((r,i)=>{
    // the spine is the INFERRED activities (the steps); each artefact beneath is
    // the recorded evidence that step is inferred from, and it opens to its source.
    const acts = r.activities.map(n=>`<div class="actnode">
        <div class="acth"><span class="actn">${esc(n.name)}</span><span class="actm">${esc(n.when)}${n.sources.length?' · '+n.sources.map(esc).join(' + '):''}</span></div>
        <div class="evlabel">evidenced by ${n.n} artefact${n.n===1?'':'s'}</div>
        <div class="arts">${n.arts.map(art).join('')}</div></div>`).join('');
    const inf = r.inferred.length?`<div class="actnode inf">
        <div class="acth"><span class="actn">Not in any system</span><span class="actm">inferred — never asserted as fact</span></div>
        <div class="arts">${r.inferred.map(s=>`<div class="ev inf"><span class="en"><span class="evref">${esc(s.name)}</span><span class="evverb">inferred, no record</span></span><span class="ed">${esc(s.when)}${s.note?' — '+esc(s.note):''}</span><span class="src">${s.src?esc(s.src):'—'}</span></div>`).join('')}</div></div>`:'';
    const chip = r.chip_word?` <span class="tchip ${esc(r.chip_class)}">${esc(r.chip_word)}</span>`:'';
    const xchip = r.cross?` <span class="xchip">${r.sources.map(esc).join(' + ')}</span>`:'';
    const frame = V.meta.ai_steps?`<div class="framing"><b>How to read this:</b> the steps are the activities the model inferred by grouping the artefacts below; each artefact is the recorded evidence, and <b>opens to its source</b>. Anything the systems never recorded shows as <i>inferred</i>.</div>`:'';
    return `<tr class="run" data-i="${i}"><td><div class="rtitle">${esc(r.title)}</div>
          <div class="rsub">${esc(r.id)}${chip}${xchip}</div></td>
        <td><span class="kind ${r.kind_attn?'flag':''}">${esc(r.kind)}</span></td><td>${esc(r.actor)}</td>
        <td class="path-c">${esc(r.path)}</td>
        <td><span class="dev ${r.dev_attn?'attn':''}">${esc(r.dev_label)}</span></td></tr>
      <tr class="detail" id="d${i}" hidden><td colspan="5">
        <div class="tl"><div class="tlh">${esc(r.id)} · ${esc(r.status)}</div>${frame}
          ${r.why?`<div class="why"><b>Why these are one ${esc(V.meta.item)} (${esc(r.tier)}):</b> ${esc(r.why)}</div>`:''}${acts}${inf}</div></td></tr>`;
  }).join('') || `<tr><td colspan="5" style="text-align:center;color:var(--ink-3);padding:22px">None match.</td></tr>`;
  rowsEl.querySelectorAll('tr.run').forEach(tr=>tr.onclick=()=>{
    const d=document.getElementById('d'+tr.dataset.i); d.hidden=!d.hidden;});
}
document.getElementById('search').addEventListener('input',e=>{q=e.target.value.toLowerCase();render();});
filtersEl.querySelectorAll('.fbtn').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.vrow.on').forEach(x=>x.classList.remove('on'));
  filtersEl.querySelectorAll('.fbtn').forEach(x=>x.classList.remove('on'));
  b.classList.add('on'); filter=b.dataset.f; render();});

// Clicking a step in the audit table filters the run list to the runs it touched
// — the point of the table is that the reading can be checked, not just totalled.
document.querySelectorAll('.vrow').forEach(tr=>tr.onclick=()=>{
  const key='act:'+tr.dataset.act, already=tr.classList.contains('on');
  document.querySelectorAll('.vrow.on').forEach(x=>x.classList.remove('on'));
  filtersEl.querySelectorAll('.fbtn').forEach(x=>x.classList.remove('on'));
  if(already){ filter='all'; filtersEl.querySelector('.fbtn').classList.add('on'); }
  else { filter=key; tr.classList.add('on');
         document.getElementById('tableHead').scrollIntoView({behavior:'smooth',block:'start'}); }
  render();});

document.getElementById('orphanNote').innerHTML = V.orphans.length
  ? `<b>${V.orphans.length}</b> record${V.orphans.length===1?'':'s'} could not be matched to any ${esc(V.meta.item)} (e.g. <code>${esc(V.orphans[0].src)}</code> — ${esc(V.orphans[0].reason)}) — set aside, not counted.`
  : '';
document.getElementById('scope').innerHTML = V.meta.scope.map(s=>`<li>${esc(s)}</li>`).join('');
render();
</script>
</body></html>"""
