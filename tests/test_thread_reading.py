"""The unit of reading is the run. Plus the two guards that keep a burst of
look-alike noise from becoming a stage."""

from __future__ import annotations

import pytest

from induction.abstraction import (ReadVocabulary, ScriptedRecordClassifier,
                                   _cap_per_thread, _clean_thread_readings,
                                   _detach_lonely_steps, infer_activities)
from induction.adapters import email_mbox
from induction.pipeline import induce
from induction.process import Case

D = "meridian-energy.example"


def _mail(mid, frm, subject, body, day, thread=None):
    head = (f"Message-ID: <{mid}>\nFrom: {frm}@{D}\nTo: desk@{D}\n"
            f"Date: Mon, {day} Oct 2001 09:00:00 -0700\nSubject: {subject}\n")
    if thread:
        head += f"In-Reply-To: <{thread}>\n"
    return (mid, head + "\n" + body + "\n")


# --- the thread guardrail ------------------------------------------------------

VOCAB = ReadVocabulary(steps_by_process={
    "Hiring": ["CV screened", "Onsite interview", "Offer made"],
    "Invoicing": ["Invoice raised", "Payment received"],
})
THREADS = [{"id": "t1", "records": [{"id": "r1", "text": "…"}, {"id": "r2", "text": "…"}]},
           {"id": "t2", "records": [{"id": "r3", "text": "…"}]}]


def test_every_step_in_a_thread_is_held_to_the_threads_process():
    got = _clean_thread_readings({
        "t1": {"process": "Hiring", "steps": {
            "r1": {"step": "CV screened", "span": "resume attached"},
            "r2": {"step": "Invoice raised", "span": "please pay"},   # wrong process
        }},
    }, THREADS, VOCAB)
    assert set(got) == {"r1"}
    assert got["r1"]["process"] == "Hiring"


def test_a_thread_id_never_sent_and_a_record_from_another_thread_are_dropped():
    got = _clean_thread_readings({
        "t9": {"process": "Hiring", "steps": {"r1": {"step": "CV screened", "span": "x"}}},
        "t1": {"process": "Hiring", "steps": {"r3": {"step": "CV screened", "span": "x"}}},
    }, THREADS, VOCAB)
    assert got == {}


def test_a_process_not_on_the_list_places_nothing():
    got = _clean_thread_readings({
        "t1": {"process": "Firing", "steps": {"r1": {"step": "CV screened", "span": "x"}}},
    }, THREADS, VOCAB)
    assert got == {}


# --- the two guards against look-alike noise -----------------------------------

def test_one_long_thread_cannot_flood_the_discovery_sample():
    """19 congratulations in one thread put ~11 records into a 150-record sample.
    They share a thread, not wording, so the cap is per run, not per text."""
    cases = {"congrats": Case(id="congrats", kind_hint="e", anchor={}, confidence=None,
                              event_ids=[f"c{i}" for i in range(19)]),
             "work": Case(id="work", kind_hint="e", anchor={}, confidence=None,
                          event_ids=["w1", "w2"])}
    class M: pass
    m = M(); m.cases = cases
    recs = [{"id": f"c{i}", "text": f"totally different wording {i}"} for i in range(19)]
    recs += [{"id": "w1", "text": "invoice"}, {"id": "w2", "text": "paid"}]
    kept, dropped = _cap_per_thread(recs, m)
    assert dropped == 16
    assert [r["id"] for r in kept] == ["c0", "c1", "c2", "w1", "w2"]


def test_a_step_that_never_meets_its_siblings_is_detached():
    """Promotion to MD: three runs, always alone. Not a stage of anything."""
    cases = {f"c{i}": Case(id=f"c{i}", kind_hint="e", anchor={}, confidence=None,
                           event_ids=[f"e{i}"]) for i in range(5)}
    cases["c5"] = Case(id="c5", kind_hint="e", anchor={}, confidence=None,
                       event_ids=["e5a", "e5b"])

    class M:
        pass
    m = M(); m.cases = cases
    vocab = ReadVocabulary(steps_by_process={"Staffing": ["Resume reviewed", "Offer made", "Promotion"]})
    got = {
        "e0": {"activity": "Promotion", "process": "Staffing", "span": "congrats"},
        "e1": {"activity": "Promotion", "process": "Staffing", "span": "congrats"},
        "e2": {"activity": "Promotion", "process": "Staffing", "span": "congrats"},
        "e3": {"activity": "Resume reviewed", "process": "Staffing", "span": "cv"},
        "e5a": {"activity": "Resume reviewed", "process": "Staffing", "span": "cv"},
        "e5b": {"activity": "Offer made", "process": "Staffing", "span": "offer"},
    }
    detached = _detach_lonely_steps(got, m, vocab)
    assert detached == [("Staffing", "Promotion", 3)]
    assert not any(r["activity"] == "Promotion" for r in got.values())
    assert "Promotion" not in vocab.steps_by_process["Staffing"]
    # a step that DOES co-occur (Resume reviewed with Offer made in c5) stays,
    # even though it also appears alone in c3
    assert any(r["activity"] == "Resume reviewed" for r in got.values())


def test_a_lonely_step_seen_in_too_few_runs_is_left_alone():
    """Two runs is thin evidence in either direction; the rule needs three."""
    cases = {f"c{i}": Case(id=f"c{i}", kind_hint="e", anchor={}, confidence=None,
                           event_ids=[f"e{i}"]) for i in range(2)}
    class M: pass
    m = M(); m.cases = cases
    vocab = ReadVocabulary(steps_by_process={"P": ["X", "Y"]})
    got = {"e0": {"activity": "X", "process": "P", "span": "s"},
           "e1": {"activity": "X", "process": "P", "span": "s"}}
    assert _detach_lonely_steps(got, m, vocab) == [] and len(got) == 2


# --- end to end: threads are the unit ------------------------------------------

def test_reading_runs_per_thread_and_never_splits_one():
    """A one-line reply is placed because its thread was, not because it said so."""
    calls = []

    class Spy(ScriptedRecordClassifier):
        def classify_threads(self, threads, vocabulary):
            calls.append([th["id"] for th in threads])
            return super().classify_threads(threads, vocabulary)

    msgs = []
    for i in range(12):
        root = f"m{i}0"
        msgs.append(_mail(root, "a", f"Master agreement {i}", f"Please review the credit terms {i}.", 1 + i))
        msgs.append(_mail(f"m{i}1", "b", f"RE: Master agreement {i}", "Approved - proceed.", 1 + i, root))
        msgs.append(_mail(f"m{i}2", "a", f"RE: Master agreement {i}", "ok", 1 + i, root))
        msgs.append(_mail(f"m{i}3", "b", f"RE: Master agreement {i}", "Executed and filed.", 1 + i, root))
    m = induce(email_mbox.shape(msgs, "x"), slug="x")
    spy = Spy([("Requested", ["please review the credit terms"], "Contracts"),
               ("Approved", ["approved - proceed"], "Contracts"),
               ("Executed", ["executed and filed"], "Contracts")])
    a = infer_activities(m, mapper=None, classifier=spy)
    assert calls, "the thread seam was not used"
    seen = [tid for batch in calls for tid in batch]
    assert len(seen) == len(set(seen)) == len(m.cases), "a thread was split or repeated across batches"
    assert a.by_case and set(a.by_case.values()) == {"Contracts"}
