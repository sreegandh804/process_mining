"""Step 3 — Order: turn each case's events into an ordered trace.

Baseline: sort by timestamp. For the git corpus every event carries a real
timestamp, so traces are `ordered` (tier `direct`). The interesting path is the
*absence* of order: the thin changelog source (step 4b) produces Observations
with no `seen_at`, and there a case is flagged ``order_status = "unknown"`` — we
surface "I can see the states but not their sequence" rather than inventing one.

The trace *signature* (used by Variants) collapses events that are the same
action at the same instant — that is how a co-authored commit (two `authored`
events, same timestamp) reads as one step, not two.
"""

from __future__ import annotations

from typing import Optional

from induction.adapters import Shaped
from induction.model import Event, Observation
from induction.steps.correlate import Correlation

# Tie-break when two events share a timestamp: the natural within-commit order.
_ACTION_ORDER = {"authored": 0, "co-authored": 0, "reviewed": 1, "committed": 2,
                 "merged": 3, "reverted": 3, "released": 4, "closed": 5}


def order(shaped: Shaped, corr: Correlation) -> None:
    events_by_id = {e.id: e for e in shaped.events}
    # Emission order is the honest tie-break when two events share a timestamp
    # (a spreadsheet row records several steps on the same date): each adapter
    # emits a row's events in the source's own column order, so we preserve it
    # rather than inventing an order from the id string.
    seq = {e.id: i for i, e in enumerate(shaped.events)}

    for case in corr.cases.values():
        evs = [events_by_id[eid] for eid in case.event_ids if eid in events_by_id]
        timed = [e for e in evs if e.timestamp]
        untimed = [e for e in evs if not e.timestamp]

        if not evs:
            case.order_status = "unknown"
            continue

        timed.sort(key=lambda e: (e.timestamp, _ACTION_ORDER.get(e.action, 9), seq.get(e.id, 0)))
        # Untimed events cannot be placed; we keep them, appended, and say so.
        case.ordered_event_ids = [e.id for e in timed] + [e.id for e in untimed]

        if not timed:
            case.order_status = "unknown"
        elif untimed:
            case.order_status = "partial"
        else:
            case.order_status = "ordered"

        case.trace_signature = _signature([events_by_id[i] for i in case.ordered_event_ids])


def _signature(ordered_events: list[Event]) -> tuple:
    """Collapse (action, timestamp) duplicates so simultaneous same-action events
    — co-authorship, a bot fan-out — count once. The result is the sequence of
    actions that Variants groups on."""
    sig: list[str] = []
    last_key: Optional[tuple] = None
    for e in ordered_events:
        key = (e.action, e.timestamp)
        if key == last_key:
            continue
        sig.append(e.action)
        last_key = key
    return tuple(sig)


def order_observations(observations: list[Observation], corr: Correlation) -> None:
    """Observations (thin source) carry a state but usually no time. When a case
    is made only of observations with no `seen_at`, its order is genuinely
    unknown — we mark it, we do not guess."""
    obs_by_case: dict[str, list[Observation]] = {}
    for o in observations:
        if o.case_id:
            obs_by_case.setdefault(o.case_id, []).append(o)
    for case_id, obs in obs_by_case.items():
        case = corr.cases.get(case_id)
        if case is None:
            continue
        if case.event_ids:
            # A timed git case that a changelog bullet merely enriches keeps its
            # event-based order; the order-less observations are supplementary
            # evidence, not part of the trace.
            continue
        if any(o.seen_at for o in obs):
            timed = sorted([o for o in obs if o.seen_at], key=lambda o: o.seen_at)
            untimed = [o for o in obs if not o.seen_at]
            case.ordered_event_ids = [o.id for o in timed] + [o.id for o in untimed]
            case.order_status = "partial" if untimed else "ordered"
        else:
            case.ordered_event_ids = [o.id for o in obs]
            case.order_status = "unknown"
