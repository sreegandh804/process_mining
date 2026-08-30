"""Generic correlation (step 2) for keyed sources — the non-git path.

Git correlation leans on the commit DAG. A spreadsheet has no DAG; what it has
is **identity keys** (a row is an invoice) and **foreign keys** (a payment row
names an invoice). So the generic correlator forms one case per entity that owns
records, and merges cases across resolved foreign keys (union-find). Every link
is scored `joined` — a shared, deterministic key — exactly like the git spine.

Honesty carries through unchanged: a foreign key whose target does not exist is
recorded as an **unresolved reference** on the source entity (it becomes a
reconciliation finding in gaps), never silently joined or dropped.
"""

from __future__ import annotations

from collections import defaultdict

from induction.adapters import Shaped
from induction.model import Evidence, joined
from induction.process import Case
from induction.steps.correlate import Correlation

_SKIP_TYPES = {"person", "orphan_row"}


def correlate_by_key(shaped: Shaped) -> Correlation:
    entities = {e.id: e for e in shaped.entities if e.type not in _SKIP_TYPES}
    events_by_entity: dict[str, list] = defaultdict(list)
    obs_by_entity: dict[str, list] = defaultdict(list)
    for ev in shaped.events:
        if ev.entity_id in entities:
            events_by_entity[ev.entity_id].append(ev)
    for ob in shaped.observations:
        if ob.entity_id in entities:
            obs_by_entity[ob.entity_id].append(ob)

    parent = {eid: eid for eid in entities}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    referenced: set[str] = set()
    for eid, ent in entities.items():
        for ref in ent.attrs.get("references", []):
            target = f"{ref['type']}:{ref['key']}"
            if target in entities:
                union(eid, target)
                referenced.add(target)
            else:
                ent.attrs.setdefault("unresolved_refs", []).append(ref)

    components: dict[str, list[str]] = defaultdict(list)
    for eid in entities:
        components[find(eid)].append(eid)

    corr = Correlation()
    for members in components.values():
        recs = {m: (events_by_entity.get(m, []), obs_by_entity.get(m, [])) for m in members}
        if not any(e or o for e, o in recs.values()):
            continue
        # The "primary" entity is the one others point at (the invoice), else the
        # record-richest member. Its identity names the run.
        targets = [m for m in members if m in referenced]
        primary = targets[0] if targets else max(members, key=lambda m: len(recs[m][0]))
        pent = entities[primary]
        conf = joined("records share a row identity / foreign key")
        case = Case(
            id=f"case:{primary}", kind_hint=pent.type,
            anchor={"type": pent.type, "key": primary.split(":", 1)[1]},
            confidence=conf,
        )
        corr.cases[case.id] = case
        for m in members:
            evs, obs = recs[m]
            if not evs and not obs and m not in referenced:
                continue
            case.entity_ids.append(m)
            for ev in evs:
                ev.case_id = case.id
                ev.case_confidence = conf
                case.event_ids.append(ev.id)
            for ob in obs:
                ob.case_id = case.id
                ob.case_confidence = conf
            if entities[m].evidence:
                case.evidence.append(entities[m].evidence[0])
    return corr
