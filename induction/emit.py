"""Emit the induced model as JSON (brief §5).

`model.json` is the complete, honest artefact: processes (definitions), variants
(traces + frequencies), instances (cases), steps, members, gaps, orphans — every
element carrying its `confidence` and `evidence[]`. The thin inspector renders a
presentation slice of this; downstream tools can consume the whole thing.

Cost/value and divergence are present as *slots*, deliberately empty — see the
README for why they are hooks and not fabricated figures.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from induction.model import to_json
from induction.pipeline import InducedModel

# What the engine cannot conclude — stated in the artefact itself, not just the
# README, so it travels with the data.
DISCLAIMERS = [
    "The corpus is git history only. Issues, PR reviews, and discussion happened "
    "off-git and are inferred from references and gaps — never asserted.",
    "Correlation can mis-join. Every join is scored; a fuzzy/uncertain join reads "
    "as `heuristic`, distinguishable from the deterministic `joined` spine.",
    "Segment boundaries are inferred (`heuristic`) and revisable — nothing in the "
    "data announces where one kind of process ends and the next begins.",
    "Order is read from timestamps. Thin observations with no time are marked "
    "`order: unknown`, not guessed.",
    "Cost/value figures are NOT produced. The slots are exposed and empty.",
]

TABULAR_DISCLAIMERS = [
    "The corpus is spreadsheet exports. Work done in email, calls, or another "
    "system never touched these sheets and is inferred from gaps — never asserted.",
    "Correlation is on identity and foreign keys; an unresolved key is surfaced "
    "for reconciliation, not silently joined.",
    "Kinds are inferred by clustering and revisable — the data does not announce them.",
    "Order is read from date columns. A row with a status but no dates is "
    "`order: unknown`, not guessed; a blank actor stays unknown, never invented.",
    "No amounts are invented. Monetary mismatches are surfaced as findings, not "
    "computed into a headline figure.",
]


EMAIL_DISCLAIMERS = [
    "The corpus is a mailbox. Work done in a tracker, a call or a document "
    "system never touched these messages and is inferred from gaps — never "
    "asserted.",
    "A thread is not a run. Where the header chain is missing, threads are "
    "joined on subject and timing, which reads as `heuristic` and can mis-join.",
    "Steps are what a model read each message to DO, with the line it read them "
    "from. A message it would not commit to keeps the mailbox's own verb.",
    "Order is read from the Date header. A message with none is `order: unknown`, "
    "not guessed; a blank sender stays unknown, never invented.",
    "Cost/value figures are NOT produced. The slots are exposed and empty.",
]


def disclaimers_for(m) -> list:
    """What this corpus cannot conclude — the set that matches the source.

    A mailbox run used to be handed the git list, which opens "The corpus is git
    history only" and goes on about PR reviews. Every line of it was false for
    the run it was attached to, which is worse than having no disclaimer: a
    reader who checks one and finds it nonsense has no reason to trust the rest
    of the artefact either.
    """
    kind = (m.manifest or {}).get("source_kind")
    if kind == "spreadsheet":
        return TABULAR_DISCLAIMERS
    if kind == "email":
        return EMAIL_DISCLAIMERS
    return DISCLAIMERS

TIER_LEGEND = {
    "direct": "read straight from the source (present as data)",
    "joined": "deterministic join on a shared key (an id / foreign key; git DAG)",
    "heuristic": "rule-based inference (reference similarity, actor+time proximity)",
    "model": "embedding / LLM inference (not built in the baseline)",
}

# What `typed_decisions` is, said in the artefact so it travels with the data.
TYPED_DECISIONS_NOTE = (
    "Decisions a typed model (Jev) was asked, where a reader can see the "
    "consequence and might dispute it: which stage a record performs, why two "
    "records are one case, why a proposed step is absent from the vocabulary, "
    "and why a cluster is flagged as a look-alike. Each row carries the question "
    "as asked, the options and their probabilities, and what the ENGINE did with "
    "the answer — which is not always what the answer alone would suggest. "
    "Calls that only narrowed what was looked at (the record gate, the discovery "
    "sample) assert nothing about any record and are counted, not listed."
)


def build_model(m: InducedModel, names: dict | None = None, activities: dict | None = None,
                corrections: dict | None = None) -> dict:
    shaped = m.shaped
    # Performance and review are measured on the same view the inspector draws,
    # so the file and the page can never disagree about a figure.
    from induction.inspector import build_view
    view = build_view(m, names, activities, corrections)
    review = view["review"]
    kind_of_case = {cid: k.id for k in m.kinds for cid in k.case_ids}

    persons = [e for e in shaped.entities if e.type == "person"]
    person_name = {e.id: e.attrs.get("name", e.id) for e in persons}

    return {
        "meta": {
            "slug": m.slug,
            "manifest": m.manifest,
            "profile": m.profile_id,
            "profile_note": ("Kinds and activities are named by the '%s' vocabulary. "
                             "The 'generic' default leaves them unnamed (kind_1, …) with "
                             "data-derived rationales — structure is identical either way."
                             % m.profile_id),
            "tier_legend": TIER_LEGEND,
            "typed_decisions_note": TYPED_DECISIONS_NOTE,
            "what_it_cannot_conclude": disclaimers_for(m),
            "stats": _stats(m),
        },
        "process_definitions": [k.to_dict() for k in m.kinds],
        "cases": [
            {**c.to_dict(), "kind": kind_of_case.get(c.id, "unclassified")}
            for c in m.cases.values()
        ],
        "steps": [s.to_dict() for s in m.steps],
        "same_activity_merges": [mg.to_dict() for mg in m.merges],
        "gaps": [g.to_dict() for g in m.gaps],
        "orphans": [o.to_dict() for o in m.orphans],
        # Every typed decision whose consequence a reader can see, with the
        # question it answered, what it could have picked, and what the engine
        # did next. In the artefact rather than only in the page, because the
        # page is a view of this file and a claim that existed only in HTML would
        # be one no downstream tool could check. Empty list when the typed tier
        # did not run — never absent, so a consumer need not special-case it.
        "typed_decisions": m.decisions.to_list() if m.decisions is not None else [],
        "members": [
            {
                "id": e.id,
                "name": e.attrs.get("name"),
                "is_bot": e.attrs.get("is_bot", False),
                "commit_count": e.attrs.get("commit_count", 0),
            }
            for e in sorted(persons, key=lambda e: -e.attrs.get("commit_count", 0))
        ],
        # Full substrate, so every evidence locator resolves and nothing is hidden.
        "entities": [e.to_dict() for e in shaped.entities],
        "events": [e.to_dict() for e in shaped.events],
        "observations": [o.to_dict() for o in shaped.observations],
        # Stubs — exposed, empty, and honest about being hooks (brief §3, §8b).
        "cost_value": {
            "status": "stub",
            "note": "Slots exposed, not fabricated. Each step/engagement has a "
                    "monetary and non-monetary slot (money/effort; revenue/outcomes). "
                    "Populating them is a product concern, not a build concern.",
            "per_step": {s.action: {"money": None, "effort": None} for s in m.steps},
        },
        # Measured, never estimated: per process, the end-to-end time, the waits
        # between steps, the share of runs off the usual route, rework — each with
        # the runs behind it and how many runs it could be measured on.
        "performance": [
            {"process": p["id"], "name": p["name"], **p["performance"]}
            for p in view["processes"] if p.get("performance")
        ],
        "divergence": {
            "status": "live" if (corrections or {}) else "awaiting review",
            "note": "Claims a process owner disputed, each beside what the records "
                    "still show. A dispute never deletes a claim: the owner can be "
                    "right about work no system saw, and the records can be right "
                    "about work the owner describes differently. Answers come from the "
                    "inspector's 'Export review' file, passed back with --corrections.",
            "items": review["divergence"],
            "confirmed": review["n_confirmed"],
            "unmatched_answers": review["unmatched"],
        },
    }


def _stats(m: InducedModel) -> dict:
    shaped = m.shaped
    case_tiers = Counter(
        e.case_confidence.tier.label for e in shaped.events if e.case_confidence
    )
    return {
        "n_entities": len(shaped.entities),
        "n_events": len(shaped.events),
        "n_observations": len(shaped.observations),
        "n_cases": len(m.cases),
        "n_process_kinds": len(m.kinds),
        "n_rejected_kinds": sum(1 for k in m.kinds if k.rejected),
        "n_steps": len(m.steps),
        "n_same_activity_merges": len(m.merges),
        "n_gaps": len(m.gaps),
        "n_orphans": len(m.orphans),
        "order_status": dict(Counter(c.order_status for c in m.cases.values())),
        "case_link_tiers": dict(case_tiers),
        "gap_kinds": dict(Counter(g.kind for g in m.gaps)),
        "entity_types": dict(Counter(e.type for e in shaped.entities)),
    }


def write_json(m: InducedModel, path: str | Path, names: dict | None = None,
               activities: dict | None = None, corrections: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(m, names, activities, corrections)
    path.write_text(to_json(model, indent=2))
    return path
