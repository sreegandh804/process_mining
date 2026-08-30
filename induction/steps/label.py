"""Step 5 — Label: name activities; merge the same activity done by many people.

Baseline (deterministic): an activity is named by its **raw action** — the label
the source itself gave it ("authored", "invoice_approved"). No hardcoded rename
table lives here: the DEFAULT is to leave the action untouched, and a source
`Profile` may map actions to friendlier words. That is all the LLM upgrade would
replace — and the guardrail is strict: an LLM may *name* and *judge equivalence*,
never do structural work (correlation or ordering), or it will hallucinate a
plausible process that never ran. The skeleton stays deterministic.

The graded case here is **same-activity-different-people**. A co-authored commit
is one authoring activity performed by several people — it surfaces as sibling
`authored` events at the same instant with different actors. We fold those into
one Step occurrence carrying several Members, and we keep every underlying event
(they are never destroyed — merging is a view, not a deletion).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from induction.adapters import Shaped
from induction.model import Evidence, direct
from induction.process import Step
from induction.profiles import GENERIC_PROFILE, Profile
from induction.steps.correlate import Correlation


@dataclass
class ActivityMerge:
    """One activity performed by more than one person, folded into a single step
    occurrence with all records retained."""

    id: str
    entity_id: str
    action: str
    timestamp: str | None
    member_ids: list[str]
    event_ids: list[str]
    evidence: list[Evidence]
    rationale: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "entity_id": self.entity_id,
            "action": self.action,
            "timestamp": self.timestamp,
            "members": self.member_ids,
            "event_ids": self.event_ids,
            "records_retained": len(self.event_ids),
            "rationale": self.rationale,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass
class LabelResult:
    steps: list[Step] = field(default_factory=list)
    merges: list[ActivityMerge] = field(default_factory=list)


def label(shaped: Shaped, corr: Correlation, profile: Profile = GENERIC_PROFILE) -> LabelResult:
    # --- activity catalogue: one named Step per action, traceable to its events ---
    by_action: dict[str, list] = defaultdict(list)
    for ev in shaped.events:
        by_action[ev.action].append(ev)

    steps: list[Step] = []
    for action, evs in sorted(by_action.items(), key=lambda kv: -len(kv[1])):
        members = sorted({e.actor for e in evs if e.actor})
        steps.append(Step(
            id=f"step:{action}",
            name=profile.label_action(action),
            action=action,
            confidence=direct("named from the source's own action verb"),
            member_ids=members,
            event_ids=[e.id for e in evs],
            evidence=[e.evidence[0] for e in evs[:3] if e.evidence],
            attrs={"count": len(evs), "n_members": len(members)},
        ))

    # --- same-activity-different-people merge ---
    groups: dict[tuple, list] = defaultdict(list)
    for ev in shaped.events:
        groups[(ev.entity_id, ev.action, ev.timestamp)].append(ev)

    merges: list[ActivityMerge] = []
    for (entity_id, action, ts), evs in groups.items():
        actors = {e.actor for e in evs if e.actor}
        if len(actors) < 2:
            continue
        evidence: list[Evidence] = []
        for e in evs:
            evidence.extend(e.evidence)
        merges.append(ActivityMerge(
            id=f"merge:{entity_id}:{action}",
            entity_id=entity_id,
            action=action,
            timestamp=ts,
            member_ids=sorted(actors),
            event_ids=[e.id for e in evs],
            evidence=evidence[:4],
            rationale=(f"{len(actors)} people recorded doing the same activity "
                       f"('{action}') on the same artefact at the same instant "
                       f"(co-authorship / an applied patch). Merged into one step; "
                       f"all {len(evs)} records retained."),
        ))
    merges.sort(key=lambda m: -len(m.member_ids))
    return LabelResult(steps=steps, merges=merges)
