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


# A step has to appear in MORE than this share of a kind's runs before its
# absence from one run is worth calling a gap. A strict majority, and nothing
# cleverer: the claim being made is "the other runs did this and yours did not",
# so the other runs have to be most of them.
#
# Strict matters, and two runs are why. `authored -> committed -> reverted` and
# `authored -> authored -> merged -> released` share only `authored`. At `>= 50%`
# a step in ONE of the two clears the bar, so all four of committed / merged /
# released / reverted became "expected" and the two runs were handed to the
# detector to accuse each other with. At `> 50%` only `authored` survives, one
# step is not a path, and the kind correctly expects nothing.
#
# Judgement, but the direction is fixed: raising it makes the detector quieter,
# never wronger.
_EXPECTED_SHARE = 0.5


def _canonical_order(kind) -> list[str]:
    """The kind's expected path — the steps that USUALLY happen, in their usual
    order. A run that reached a later one without an earlier one skipped it.

    This used to be "the most frequent observed trace, longest among ties", and
    that is two different bugs depending on the data:

      - where no whole trace repeats, the tie broke at frequency 1 BY LENGTH, so
        the single longest, most chaotic run became the standard every other run
        was measured against. Six unique paths meant five runs accused of missing
        a dozen steps each.
      - and requiring a whole trace to repeat is too strict anyway. Four invoices
        that went `raised -> paid`, `raised -> approved -> paid`,
        `raised -> approved -> paid -> settled -> paid` share no trace at all,
        yet `approved` is plainly a normal step and the first invoice plainly
        skipped it. That skip is the finding this detector exists for.

    So expectation is per STEP, not per trace: a step most runs perform is
    expected, positioned where those runs usually put it. A run that reached a
    later expected step without an earlier one skipped it — the CV screened and
    an offer made with no interview in between.
    """
    from collections import defaultdict

    runs = [v.signature for v in kind.variants for _ in range(v.frequency) if v.signature]
    if not runs:
        return []
    seen = defaultdict(int)
    where = defaultdict(list)
    for sig in runs:
        for step in dict.fromkeys(sig):          # a repeat within one run is one vote
            seen[step] += 1
        span = max(1, len(sig) - 1)
        for i, step in enumerate(sig):
            where[step].append(i / span)         # normalised, so long runs do not dominate

    floor = len(runs) * _EXPECTED_SHARE
    expected = [st for st, n in seen.items() if n > floor]
    if len(expected) < 2:
        return []
    sig = sorted(expected, key=lambda st: (sum(where[st]) / len(where[st]), st))
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

    # Declared links that point nowhere — an unmatched record on the other side.
    # Source-agnostic by construction: a spreadsheet's dangling foreign key, a
    # changelog citing an issue no source has, and a mail referencing an unknown
    # order all arrive here as the same unresolved `Link`.
    for ent in entities_by_id.values():
        for raw in ent.attrs.get("unresolved_links", []):
            target = raw.get("target", "")
            target_type, _, target_key = target.partition(":")
            case_id = next((c.id for c in corr.cases.values() if ent.id in c.entity_ids), "")
            gaps.append(Gap(
                id=f"gap:unresolved:{ent.id}:{target}",
                case_id=case_id,
                kind="reconciliation",
                description=(f"References {target_type} '{target_key or target}', which is not "
                             f"present in the corpus — an unmatched record to reconcile."),
                confidence=heuristic(
                    f"{raw.get('method', 'reference')} resolves to no known record"),
                evidence=list(ent.evidence[:1]),
            ))
    return gaps
