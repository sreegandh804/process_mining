"""Step 6 — Gaps: infer the steps no system recorded, and mark them as inference.

The git corpus can only see what touched git. Two discontinuities let us infer,
from real evidence, steps that happened *off-git* — and we render them as
inference (dashed / low-confidence), never as fact:

  off_system_pr_open_review
      A commit that closes PR #N presupposes that PR #N was opened on GitHub and
      (per this project's stated process) reviewed there. Git holds neither the
      open nor the review — the PR number is the only trace. Tier `heuristic`:
      the open is strongly implied by the number; the review is expected but
      unconfirmed. This is the "confidently incomplete" finding, at scale.

  author_committer_handoff
      A change authored by one person but applied by another, days later, is an
      off-system acceptance/merge decision that left no event of its own. The
      author/committer split and the time delay are the signal. Tier `heuristic`.

We never assert these happened a particular way — only that the evidence
presupposes *something* off-git, and we name the signal that produced the claim.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from induction.adapters import Shaped
from induction.model import Evidence, heuristic
from induction.process import Gap
from induction.steps.correlate import Correlation

_HANDOFF_MIN = timedelta(days=1)  # only flag handoffs with a real, visible delay


def infer_gaps(shaped: Shaped, corr: Correlation, slug: str) -> list[Gap]:
    events_by_id = {e.id: e for e in shaped.events}
    entities_by_id = {e.id: e for e in shaped.entities}
    commit_to_case: dict[str, str] = {}
    for case in corr.cases.values():
        for eid in case.entity_ids:
            commit_to_case.setdefault(eid, case.id)
    gaps: list[Gap] = []

    # --- off_system_pr_open_review: one per PR run ---
    for case in corr.cases.values():
        if case.kind_hint != "pr":
            continue
        # If git somehow held an explicit review/open event, we would skip — it
        # never does, which is exactly the point.
        has_review = any(events_by_id[e].action in ("reviewed", "opened")
                         for e in case.event_ids if e in events_by_id)
        if has_review:
            continue
        first = case.ordered_event_ids[0] if case.ordered_event_ids else None
        number = case.anchor.get("number")
        gaps.append(Gap(
            id=f"gap:pr-open-review:{case.id}",
            case_id=case.id,
            kind="off_system_pr_open_review",
            description=(f"PR #{number} was opened and reviewed on GitHub before it "
                         f"landed; git records neither. The PR number is the only "
                         f"trace of an off-system open + review."),
            confidence=heuristic("a merge/squash closing a PR presupposes an off-git "
                                 "open (strong) and review (expected, unconfirmed)"),
            evidence=list(case.evidence[:1]),
            between=(None, first),
        ))

    # --- author_committer_handoff: real, delayed handoffs only ---
    for ent in entities_by_id.values():
        if ent.type != "commit":
            continue
        raw = ent.raw or {}
        a = raw.get("author", {}) or {}
        c = raw.get("committer", {}) or {}
        if not a or not c:
            continue
        if (a.get("email") or "").lower() == (c.get("email") or "").lower():
            continue
        delay = _delay(raw.get("author_date"), raw.get("committer_date"))
        if delay is None or delay < _HANDOFF_MIN:
            continue
        sha = raw.get("sha", "")
        authored_evt = f"evt:{sha}:authored"
        committed_evt = f"evt:{sha}:committed"
        gaps.append(Gap(
            id=f"gap:handoff:{sha}",
            case_id=commit_to_case.get(ent.id, ""),
            kind="author_committer_handoff",
            description=(f"Authored by {a.get('name')} but applied by {c.get('name')} "
                         f"{delay.days} days later — an off-system review/acceptance "
                         f"decision with no event of its own."),
            confidence=heuristic("author != committer with a multi-day delay implies an "
                                 "off-git acceptance/merge step"),
            evidence=[Evidence(f"git:{slug}", sha, raw.get("subject", "")[:200])],
            between=(authored_evt, committed_evt),
        ))

    gaps.sort(key=lambda g: g.kind)
    return gaps


def _delay(author_date: str | None, committer_date: str | None):
    try:
        ad = datetime.fromisoformat(author_date)
        cd = datetime.fromisoformat(committer_date)
        return cd - ad
    except (TypeError, ValueError):
        return None
