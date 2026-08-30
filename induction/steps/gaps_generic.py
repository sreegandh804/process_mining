"""Generic gaps (step 6) for keyed sources — inference rendered as inference.

Two domain-general detectors that produce genuinely actionable findings without
any hardcoded process knowledge:

  missing_expected_step
      A run reached a late step of its kind's common path without a recorded
      earlier step that the common path has. That earlier step either happened
      off-system or was skipped — either way it is a control finding. This is
      what turns "invoice paid" into "invoice paid with **no recorded
      approval**".

  reconciliation
      Cross-source corroboration is missing: a run reached a terminal state in
      one source with no matching record from another (an invoice marked paid in
      the tracker with no bank payment), or a foreign key points at a row that
      does not exist (a bank payment for an unknown invoice).

All are tier `heuristic` and name the signal that produced them.
"""

from __future__ import annotations

from induction.adapters import Shaped
from induction.model import Evidence, heuristic
from induction.process import Gap
from induction.steps.correlate import Correlation


def infer_missing_step_gaps(corr: Correlation, kinds) -> list[Gap]:
    gaps: list[Gap] = []
    for kind in kinds:
        if kind.rejected:
            continue
        canon = _canonical_order(kind)
        if len(canon) < 2:
            continue
        pos = {a: i for i, a in enumerate(canon)}
        for cid in kind.case_ids:
            case = corr.cases.get(cid)
            if not case:
                continue
            present = set(case.trace_signature)
            reached = [pos[a] for a in case.trace_signature if a in pos]
            if not reached:
                continue
            latest = max(reached)
            for j in range(latest):
                act = canon[j]
                if act not in present:
                    gaps.append(Gap(
                        id=f"gap:missing:{cid}:{act}",
                        case_id=cid,
                        kind="missing_expected_step",
                        description=(f"Run reached '{canon[latest]}' but has no recorded "
                                     f"'{act}' step, which the common path of this kind "
                                     f"includes."),
                        confidence=heuristic("a later step was reached without an earlier step "
                                             "the common path has — done off-system or skipped"),
                        evidence=list(case.evidence[:1]),
                    ))
    return gaps


def _canonical_order(kind) -> list[str]:
    """The kind's expected path = the most frequent observed trace, and among
    ties the most complete (longest). Picking the fullest common path is what
    lets an omission ("paid, never approved") surface as a gap."""
    if not kind.variants:
        return []
    sig = max(kind.variants, key=lambda v: (v.frequency, len(v.signature))).signature
    order: list[str] = []
    for a in sig:
        if a not in order:
            order.append(a)
    return order


def infer_reconciliation_gaps(shaped: Shaped, corr: Correlation, kinds,
                              terminal_action: str = "paid",
                              corroborating_action: str = "settled") -> list[Gap]:
    events_by_id = {e.id: e for e in shaped.events}
    entities_by_id = {e.id: e for e in shaped.entities}
    rejected_cases = {cid for k in kinds if k.rejected for cid in k.case_ids}
    gaps: list[Gap] = []

    # Only reconcile against a source that actually exists. If nothing in the
    # corpus ever emits the corroborating action (e.g. a single grants sheet with
    # no bank export), there is nothing to reconcile — do not manufacture a gap
    # for every terminal record. This is what keeps the check from overfitting to
    # the finance demo.
    have_corroboration = any(e.action == corroborating_action for e in shaped.events)

    for case in corr.cases.values():
        if case.id in rejected_cases:
            continue
        actions = {events_by_id[e].action for e in case.event_ids if e in events_by_id}
        if have_corroboration and terminal_action in actions and corroborating_action not in actions:
            gaps.append(Gap(
                id=f"gap:reconcile:{case.id}",
                case_id=case.id,
                kind="reconciliation",
                description=(f"Reached '{terminal_action}' in one source but has no "
                             f"matching '{corroborating_action}' record from another "
                             f"(e.g. marked paid in the tracker, no bank payment). Reconcile."),
                confidence=heuristic(f"terminal '{terminal_action}' with no corroborating "
                                     f"'{corroborating_action}' across sources"),
                evidence=list(case.evidence[:1]),
            ))

    # foreign keys that point nowhere — an unmatched record on the other side
    for ent in entities_by_id.values():
        for ref in ent.attrs.get("unresolved_refs", []):
            case_id = next((c.id for c in corr.cases.values() if ent.id in c.entity_ids), "")
            gaps.append(Gap(
                id=f"gap:unresolved:{ent.id}:{ref['type']}:{ref['key']}",
                case_id=case_id,
                kind="reconciliation",
                description=(f"References {ref['type']} '{ref['key']}', which is not present "
                             f"in the corpus — an unmatched record to reconcile."),
                confidence=heuristic("foreign key resolves to no known entity"),
                evidence=list(ent.evidence[:1]),
            ))
    return gaps
