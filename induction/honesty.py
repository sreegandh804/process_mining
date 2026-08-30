"""Honesty features (brief §6) — the graded core.

These are not decoration. The whole claim of the engine is that it is *honest
about what it cannot see*, so each of these turns an absence or an awkward
record into a visible finding rather than a silent drop:

  - orphans : records that join to no case — surfaced in a queue, never padded
              into a rollup and never dropped.
  - reject  : a recurring pattern that moves no product artefact and produces
              nothing of the process's value — flagged "looks like a process,
              isn't" (e.g. an automated bump), with its reason.
  - unknowns: actor / time / order that is genuinely unavailable is marked
              `unknown`. Absence is a finding, not a blank to fill.

`divergence` (belief vs data) is a *hook*, not a workflow — see emit.py and the
README. We keep `raw` next to inferred structure so a later validation step can
compare what the owner believes against what the data shows.
"""

from __future__ import annotations

from collections import defaultdict

from induction.adapters import Shaped
from induction.model import Evidence
from induction.process import Case, Orphan, ProcessKind
from induction.profiles import GENERIC_PROFILE, Profile
from induction.steps.correlate import Correlation


def collect_orphans(shaped: Shaped, corr: Correlation) -> list[Orphan]:
    """Every event/observation with no case is an orphan. We report one orphan
    per orphaned *entity* (a commit can emit several events) with the reason it
    joined to nothing — the honest counterpart to the joined spine."""
    entities_by_id = {e.id: e for e in shaped.entities}
    orphan_entities: dict[str, list] = defaultdict(list)

    for ev in shaped.events:
        if not ev.case_id:
            orphan_entities[ev.entity_id].append(ev)
    for ob in shaped.observations:
        if not ob.case_id:
            orphan_entities[ob.entity_id].append(ob)

    orphans: list[Orphan] = []
    for ent_id, recs in orphan_entities.items():
        ent = entities_by_id.get(ent_id)
        reason = _orphan_reason(ent)
        rec = recs[0]
        rectype = "observation" if rec.__class__.__name__ == "Observation" else "event"
        orphans.append(Orphan(
            record_id=rec.id,
            record_type=rectype,
            entity_id=ent_id,
            reason=reason,
            evidence=list(rec.evidence),
        ))
    orphans.sort(key=lambda o: o.entity_id)
    return orphans


def _orphan_reason(ent) -> str:
    if ent is None:
        return "record's entity is unknown"
    if ent.type == "commit":
        if ent.attrs.get("is_merge"):
            return ("merge commit with no 'Merge pull request #N' and no branch name "
                    "— cannot be attributed to a specific run")
        return ("commit references no PR or issue and is not reachable from any PR "
                "merge — a direct-to-branch commit with no run to attach to")
    return f"{ent.type} record matched no correlation key"


def apply_reject(kinds: list[ProcessKind], profile: Profile = GENERIC_PROFILE) -> None:
    """Flag look-alike non-processes in place, with a concrete reason.

    The *decision* is domain knowledge, so it lives in the profile: the generic
    default flags any recurring, fully-automated cluster (it cannot prove such a
    cluster "produces nothing" without domain knowledge, and says so); a source
    profile can sharpen it (git un-flags bot runs that actually change code).

    We only *flag* — never delete. The rejected kind stays fully inspectable so a
    reader can disagree; that is the point of showing the rejection and its
    reason rather than quietly filtering it out.
    """
    for kind in kinds:
        reason = profile.reject_reason(kind.features)
        if reason:
            kind.rejected = True
            kind.reject_reason = reason
