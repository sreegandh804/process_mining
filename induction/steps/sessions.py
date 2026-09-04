"""Split a case that only a WEAK key holds together, where its own record
stream went quiet.

A case is one run of a process. Some cases are assembled on a key that *means*
it — an invoice id, a git parent, a real `In-Reply-To` chain — and such a case
may legitimately span a year: `samples/grants` runs applied → reviewed → decided
→ paid over 100 days and every one of those records genuinely belongs together.

Other cases are assembled on a key that was *guessed*. A shared subject line, a
shared title, a shared channel. That kind of key has no idea whether it is
looking at one run or twenty, and on a real mailbox it is twenty: `samples/enron`
puts 26 messages in one case called "Dominion" — parking charges, the
Tallahassee prepay, the producer releases and bankruptcy counsel, months apart,
fused because somebody kept hitting reply on an old thread. Downstream that case
is poison: its "path" is four processes concatenated, its variants are noise, and
a reader is shown a 21-step chain as if it were how the work usually goes.

**The signal is silence, measured against the case's own rhythm.** A run of work
is a burst. A weak key that spans a hole far longer than the pace of the records
either side of it is not holding one run together; it is holding two things that
share a word.

Two numbers decide it, and only one is a judgement call:

  - the floor is `FuzzyPolicy.proximity_days`, already 30 and already meaning
    exactly this. The fuzzy pass refuses to join two components more than 30 days
    apart on text alone. A guessed key should not be allowed to do, silently,
    what the fuzzy pass is forbidden to do explicitly.
  - the multiple (`_SILENCE_MULTIPLE`) keeps a slow-but-steady case intact: a
    thread whose records are 10 days apart all the way through has a rhythm, and
    a 40-day pause in it is not a break. Measured against the corpora in the
    repo, whose within-case median gaps are 0.5 days (a mailbox), 1.5-2 days (an
    invoice ledger) and 8 days (a grants tracker).

**The gate is the tier, and it is exact rather than lucky.** Case confidence is
the weakest link used to assemble it, so a case built on a real key reads
`direct`/`joined` and is never touched here. Measured across the repo:
`samples/finance` is direct/joined, `samples/grants` is entirely direct,
`samples/enron` is entirely heuristic. The split reaches the guessed cases and
nothing else, and it is a property of how a case was ASSEMBLED — never of the
word "email".

A case with any undated record is left alone. Splitting on time when part of the
run has no time would be placing those records by assumption, and an unplaceable
record is exactly what this engine refuses to guess at.
"""

from __future__ import annotations

from datetime import datetime
from statistics import median
from typing import Optional

from induction.model import Confidence

# A rhythm this many times over is a break, not a pause. Judgement — but a weak
# one, and deliberately so: sweeping it from 4 to 14 moves samples/enron by two
# cases (71 -> 69). The floor is doing the work; this only protects a
# slow-but-steady case from being cut at its own normal pace.
_SILENCE_MULTIPLE = 10
# Two records and a hole between them is still a hole. There is no rhythm to
# measure at two, so the floor alone decides — which is the whole point of
# having a floor.
_MIN_RECORDS = 2
# Tiers whose cases were assembled on a guess.
_WEAK_TIERS = ("heuristic", "model")


def _dt(value) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)[:19])
    except ValueError:
        return None


def split_quiet_sessions(corr, events_by_entity, obs_by_entity, policy) -> int:
    """Split every weak-key case that goes quiet. Returns how many splits happened.

    Mutates `corr` in place: the first session keeps the original case id, so
    anything already pointing at it still resolves; later sessions get `#2`, `#3`.
    """
    floor_days = max(1, getattr(policy.fuzzy, "proximity_days", 30))
    made = 0
    for case in list(corr.cases.values()):
        if case.confidence.tier.label not in _WEAK_TIERS:
            continue
        dated = _dated_entities(case, events_by_entity, obs_by_entity)
        if dated is None or len(dated) < _MIN_RECORDS:
            continue                       # undated records, or too few to judge
        sessions = _sessions(dated, floor_days)
        if len(sessions) < 2:
            continue
        made += len(sessions) - 1
        _rewrite(corr, case, sessions, events_by_entity, obs_by_entity)
    return made


def _dated_entities(case, events_by_entity, obs_by_entity):
    """`[(when, entity_id)]` sorted, or None if any record carries no time."""
    out = []
    for eid in case.entity_ids:
        whens = [_dt(r.timestamp) for r in events_by_entity.get(eid, [])]
        whens += [_dt(getattr(r, "timestamp", None)) for r in obs_by_entity.get(eid, [])]
        whens = [w for w in whens if w]
        if not whens:
            return None                    # cannot place this record in time
        out.append((min(whens), eid))
    return sorted(out)


def _sessions(dated, floor_days: int) -> list[list[str]]:
    """Partition `[(when, entity_id)]` at every silence worth the name."""
    gaps = [(b[0] - a[0]).days for a, b in zip(dated, dated[1:])]
    if not gaps:
        return [[eid for _, eid in dated]]

    # The case's own rhythm — estimated ONLY from the gaps that could plausibly
    # BE a rhythm.
    #
    # Taking the median over all gaps was wrong, and wrong in the direction that
    # hides the worst cases: a case whose gaps are [257, 6, 270] — three separate
    # lunches a year apart — has a median of 131, which sets the bar at 1310 days
    # and declares the whole thing one run. The holes were voting on how big a
    # hole is allowed to be. A case that is mostly holes is not a case with a
    # slow rhythm; it is mostly not one case.
    #
    # So cadence is measured from the sub-floor gaps only. Where there are none —
    # every gap is already beyond what the fuzzy pass would join on — there is no
    # rhythm to protect and the floor decides alone.
    cadence = [g for g in gaps if g <= floor_days]
    rhythm = median(cadence) if cadence else 0
    threshold = max(floor_days, rhythm * _SILENCE_MULTIPLE)

    sessions, current = [], [dated[0][1]]
    for (_, eid), gap in zip(dated[1:], gaps):
        if gap > threshold:
            sessions.append(current)
            current = []
        current.append(eid)
    sessions.append(current)
    return sessions


def _rewrite(corr, case, sessions, events_by_entity, obs_by_entity) -> None:
    """Replace `case` with one case per session, carrying its provenance."""
    from induction.process import Case

    n = len(sessions)
    reason = (f"split into {n} runs: a weak key (its records share only "
              f"{case.anchor.get('type', 'a guessed key')}) spanned silences far "
              f"longer than this case's own rhythm")

    for i, member_ids in enumerate(sessions):
        if i == 0:
            new, new_id = case, case.id
        else:
            new_id = f"{case.id}#{i + 1}"
            new = Case(id=new_id, kind_hint=case.kind_hint, anchor=dict(case.anchor),
                       confidence=case.confidence)
            corr.cases[new_id] = new
        new.confidence = Confidence(
            tier=case.confidence.tier,
            rationale=((case.confidence.rationale or "") + " · " + reason).strip(" ·"))
        new.entity_ids = list(member_ids)
        new.event_ids, new.evidence = [], []
        for eid in member_ids:
            for ev in events_by_entity.get(eid, []):
                ev.case_id = new_id
                new.event_ids.append(ev.id)
            for ob in obs_by_entity.get(eid, []):
                ob.case_id = new_id
        # evidence follows the records that stayed
        seen = set()
        for ev in (e for eid in member_ids for e in events_by_entity.get(eid, [])):
            for e in ev.evidence:
                key = (e.source, e.locator)
                if key not in seen:
                    seen.add(key)
                    new.evidence.append(e)
            if len(new.evidence) >= 3:
                break
