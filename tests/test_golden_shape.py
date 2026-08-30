"""The induced shape of the golden fixture, pinned.

This exists because of a refactor: correlation used to be four source-specific
correlators (git's DAG walk, the spreadsheet's foreign keys, the changelog's
`correlate_thin`, and a GitHub one about to be written) and is now one
source-agnostic correlator resolving `Link`s that adapters declare. That is a
large change to the engine's weakest claim, so the demo corpus's *entire*
induced shape — every case, its membership, its trace, every orphan, every gap,
every confidence tier — is frozen here.

Rationale **prose** is deliberately excluded from the comparison: the wording of
"why" is allowed to improve. Tiers are not. If a link silently drops from
`joined` to `heuristic`, or a record quietly moves between cases, this fails.

To update after an intentional change, re-run with GOLDEN_UPDATE=1 and read the
diff in the commit — it is the whole point of the file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent / "golden_shape.json"


def _shape(model) -> dict:
    return {
        "cases": {
            cid: {
                "kind_hint": c.kind_hint,
                "confidence": c.confidence.to_dict(),
                "entity_ids": sorted(c.entity_ids),
                "event_ids": sorted(c.event_ids),
                "order_status": c.order_status,
                "trace_signature": list(c.trace_signature),
            } for cid, c in sorted(model.cases.items())
        },
        "n_entities": len(model.shaped.entities),
        "n_events": len(model.shaped.events),
        "n_observations": len(model.shaped.observations),
        "entity_ids": sorted(e.id for e in model.shaped.entities),
        "orphans": sorted((o.record_id, o.reason) for o in model.orphans),
        "gaps": sorted((g.case_id, g.kind, g.description) for g in model.gaps),
        "kinds": sorted((k.id, len(k.case_ids)) for k in model.kinds),
        "steps": sorted((s.name, s.action, len(s.event_ids)) for s in model.steps),
        "event_case_tiers": sorted(
            (e.id, e.case_id or "", e.case_confidence.tier.label if e.case_confidence else "")
            for e in model.shaped.events),
        "obs_case_tiers": sorted(
            (o.id, o.case_id or "", o.case_confidence.tier.label if o.case_confidence else "")
            for o in model.shaped.observations),
    }


def _without_prose(obj):
    """Tiers and structure are the contract; the wording of a rationale is not.

    Also normalises tuples to lists so a fresh model compares equal to the same
    model after a JSON round-trip through the golden file.
    """
    if isinstance(obj, dict):
        return {k: _without_prose(v) for k, v in obj.items() if k != "rationale"}
    if isinstance(obj, (list, tuple)):
        return [_without_prose(x) for x in obj]
    return obj


def test_induced_shape_is_unchanged(mini_model):
    actual = _shape(mini_model)
    if os.environ.get("GOLDEN_UPDATE"):
        GOLDEN.write_text(json.dumps(actual, indent=2, sort_keys=True))
        pytest.skip("golden file rewritten")
    expected = json.loads(GOLDEN.read_text())
    for section in sorted(expected):
        assert _without_prose(actual[section]) == _without_prose(expected[section]), (
            f"induced shape changed in '{section}'"
        )


def test_one_correlator_serves_every_source(mini_model):
    """The structural claim behind the refactor, asserted rather than asserted-in-prose.

    Records from git *and* from the changelog land in the same cases, scored by
    the same ladder, having gone through exactly one correlator. A per-source
    correlator cannot produce this: it never sees the other source's records.
    """
    source_of = {e.id: e.source.split(":")[0] for e in mini_model.shaped.entities}
    assert {"git", "changelog"} <= set(source_of.values())

    cross = {
        cid: sorted({source_of[eid] for eid in case.entity_ids if eid in source_of})
        for cid, case in mini_model.cases.items()
    }
    multi = {cid: srcs for cid, srcs in cross.items() if len(srcs) > 1}
    assert multi, f"expected a case built from more than one source, got {cross}"
    # And the cross-source links are scored, not assumed.
    for cid in multi:
        for eid in mini_model.cases[cid].event_ids:
            ev = next(e for e in mini_model.shaped.events if e.id == eid)
            assert ev.case_confidence is not None
