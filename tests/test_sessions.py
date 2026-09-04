"""Splitting a weak key that spans a silence (`steps/sessions.py`).

Every correlation pass JOINS. None of them could undo a join a *guessed* key
made too eagerly — and a guessed key (a shared subject line, a shared title) has
no idea whether it is looking at one run or twenty. On samples/enron it was
twenty: 26 messages in one case called "Dominion", covering parking charges, the
Tallahassee prepay, the producer releases and bankruptcy counsel, fused because
people kept replying to an old thread. Downstream, that case's "path" is four
processes concatenated.

These pin both halves: that it cuts the holes nobody would defend, and — much
more important — that it never touches a case a REAL key assembled.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from induction.adapters import email_mbox
from induction.pipeline import induce
from induction.steps.correlate import CorrelationPolicy
from induction.steps.sessions import _dt, _sessions

D = "meridian-energy.example"


def _mail(mid, subject, body, when, thread=None):
    head = (f"Message-ID: <{mid}>\nFrom: a.okonkwo@{D}\nTo: desk@{D}\n"
            f"Date: {when.strftime('%a, %d %b %Y %H:%M:%S -0700')}\nSubject: {subject}\n")
    if thread:
        head += f"In-Reply-To: <{thread}>\n"
    return (mid, head + "\n" + body + "\n")


# --- the rule itself, in isolation -------------------------------------------

def _run(day_offsets, floor=30):
    base = datetime(2001, 1, 1)
    dated = [(base + timedelta(days=d), f"e{i}") for i, d in enumerate(day_offsets)]
    return [len(s) for s in _sessions(dated, floor)]


def test_a_steady_burst_is_one_run():
    assert _run([0, 1, 2, 5, 6]) == [5]


def test_a_hole_far_past_the_floor_is_a_break():
    assert _run([0, 1, 2, 400, 401]) == [3, 2]


def test_two_records_and_a_hole_is_still_a_hole():
    """There is no rhythm to measure at two records, so the floor decides alone —
    which is the entire reason for having a floor."""
    assert _run([0, 163]) == [1, 1]
    assert _run([0, 3]) == [2]


def test_holes_do_not_get_to_vote_on_how_big_a_hole_may_be():
    """Three lunches a year apart: gaps [257, 6, 270]. Taking the median over ALL
    gaps put the bar at 1310 days and called it one run — the holes were setting
    their own threshold. Cadence is measured from the sub-floor gaps only."""
    assert _run([0, 257, 263, 533]) == [1, 2, 1]


def test_a_slow_but_steady_case_is_not_cut_at_its_own_pace():
    """The multiple exists for exactly this: a case whose records are ~25 days
    apart all the way through has a rhythm, and 25 days is not a break in it."""
    assert _run([0, 25, 50, 75, 100]) == [5]


def test_the_multiple_is_not_load_bearing():
    """Sweeping it 4..14 moves samples/enron by two cases. If a future change
    makes the result swing on this number, that is the signal to distrust it."""
    import induction.steps.sessions as S
    was = S._SILENCE_MULTIPLE
    try:
        seen = set()
        for mult in (4, 6, 8, 10, 14):
            S._SILENCE_MULTIPLE = mult
            seen.add(tuple(_run([0, 1, 2, 400, 401])))
        assert seen == {(3, 2)}
    finally:
        S._SILENCE_MULTIPLE = was


# --- the gate: a real key is never touched -----------------------------------

def test_a_deterministic_key_may_span_a_year():
    """samples/grants runs applied → reviewed → decided → paid over 100 days with
    95-day gaps, on a real id. Splitting those would be destroying the answer.
    The gate is the case tier — the weakest link used to assemble it — so this
    holds by construction, not by luck."""
    import run_tabular
    from induction.pipeline import run_tabular_pipeline
    m = run_tabular_pipeline(run_tabular.sources_for(Path("samples/grants"), "csv"),
                             slug="grants")
    assert len(m.cases) == 25
    assert {c.confidence.tier.label for c in m.cases.values()} == {"direct"}
    spans = []
    for c in m.cases.values():
        ts = sorted(t for t in (_dt(e.timestamp) for e in m.shaped.events
                                if e.case_id == c.id) if t)
        if len(ts) > 1:
            spans.append((ts[-1] - ts[0]).days)
    assert max(spans) > 90, "a real key must still be allowed to span months"


def test_a_thread_held_by_in_reply_to_is_not_split():
    """A real threading header is a key that MEANS it. Only the subject fallback
    is a guess, and only the guess is second-guessed."""
    base = datetime(2001, 1, 1)
    msgs = [_mail("r1", "Master agreement", "Please review the credit terms.", base)]
    msgs.append(_mail("r2", "RE: Master agreement", "Approved - proceed.",
                      base + timedelta(days=400), thread="r1"))
    m = induce(email_mbox.shape(msgs, "meridian"), slug="meridian")
    assert len(m.cases) == 1, "a real In-Reply-To chain survives any silence"


# --- end to end on the corpus that needed it ---------------------------------

@pytest.fixture(scope="module")
def enron():
    def run(split):
        shaped = email_mbox.load(Path("samples/enron"), slug="enron", max_messages=3000)
        return induce(shaped, slug="enron",
                      policy=CorrelationPolicy(split_quiet_sessions=split)), shaped
    return run(False), run(True)


def _worst(m, shaped):
    ev = {e.id: e for e in shaped.events}
    worst_gap = worst_span = 0
    for c in m.cases.values():
        ts = sorted(t for t in (_dt(ev[e].timestamp) for e in c.event_ids if e in ev) if t)
        if len(ts) > 1:
            worst_gap = max(worst_gap, max((b - a).days for a, b in zip(ts, ts[1:])))
            worst_span = max(worst_span, (ts[-1] - ts[0]).days)
    return worst_gap, worst_span


def test_the_mailbox_stops_containing_year_long_cases(enron):
    (before, sb), (after, sa) = enron
    gap_before, span_before = _worst(before, sb)
    gap_after, span_after = _worst(after, sa)
    assert gap_before > 400 and span_before > 500, (gap_before, span_before)
    assert gap_after <= 40, f"a {gap_after}-day silence still holds a case together"
    assert span_after < span_before / 3
    assert len(after.cases) > len(before.cases)


def test_no_record_is_lost_or_duplicated_by_a_split(enron):
    """A split repartitions; it must not drop a message or file one twice."""
    (before, _), (after, _) = enron
    def events(m):
        got = [e for c in m.cases.values() for e in c.event_ids]
        assert len(got) == len(set(got)), "an event landed in two cases"
        return set(got)
    assert events(before) == events(after)


def test_a_split_case_says_it_was_split(enron):
    """The reader has to be able to disagree with this, so it carries its reason."""
    (_, _), (after, _) = enron
    split = [c for c in after.cases.values() if "split into" in (c.confidence.rationale or "")]
    assert split, "no case recorded that it had been split"
    for c in split:
        assert "rhythm" in c.confidence.rationale
        assert c.confidence.tier.label in ("heuristic", "model")
