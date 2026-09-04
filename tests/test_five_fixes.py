"""The five fixes from the root-cause pass, each pinned to the evidence that
demanded it. All offline.

  1. a step is a label, not a description
  2. same topic is not same run — the judge needs someone in common
  3. the leftover kind has a count, not paths
  4. process / project / noise are three things
  5. variants are grouped by shape, not exact trace
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

import pytest

from induction.abstraction import (ReadVocabulary, ScriptedRecordClassifier,
                                   _enforce_label_form, infer_activities)
from induction.adapters import email_mbox
from induction.inspector import build_view
from induction.pipeline import induce
from induction.process import Case, ProcessKind, Variant
from induction.steps.variants import induced_variants, shape

D = "meridian-energy.example"


# --- 1. labels ---------------------------------------------------------------

def test_a_step_longer_than_five_words_is_a_description_and_is_dropped():
    """The live run produced a 13-word 'step'. Nothing else in the corpus could
    be that step, so it fragmented every variant and matched almost nothing."""
    v = ReadVocabulary(steps_by_process={"Invoicing": [
        "Invoice issued",
        "Pre-petition and post-petition exposure worksheets built from allocation reports",
        "Payment matched",
    ]}, loose=["Acknowledged", "Some very long loose label that describes rather than names"])
    kept, dropped = _enforce_label_form(v)
    assert kept.steps_by_process == {"Invoicing": ["Invoice issued", "Payment matched"]}
    assert kept.loose == ["Acknowledged"]
    assert len(dropped) == 2


def test_a_process_whose_every_step_was_a_description_disappears_rather_than_lies():
    v = ReadVocabulary(steps_by_process={"X": ["one two three four five six seven"]})
    kept, dropped = _enforce_label_form(v)
    assert kept.steps_by_process == {} and dropped


# --- 2. same topic is not same run --------------------------------------------

def test_the_judge_is_not_asked_about_two_threads_with_nobody_in_common():
    """The Dominion merge: four subject threads, a fortnight apart, entirely
    different people, fused by a model that read 'the same ongoing dispute'.
    Time proximity cannot catch it; a shared person is the second signal."""
    from induction.semantic import SemanticJudge, SemanticProvider
    from induction.steps.correlate import CorrelationPolicy

    class Spy(SemanticJudge):
        def __init__(self): self.asked = []
        def judge(self, a, b):
            self.asked.append((a, b)); return "same work"

    def mail(mid, frm, subj, body, day):
        head = (f"Message-ID: <{mid}>\nFrom: {frm}@{D}\nTo: desk@{D}\n"
                f"Date: {day.strftime('%a, %d %b %Y %H:%M:%S -0700')}\nSubject: {subj}\n")
        return (mid, head + "\n" + body + "\n")

    base = datetime(2002, 3, 1)
    # two threads, same subject matter, 10 days apart, disjoint people. Thread B
    # is a forward so the two have different activity shapes — otherwise the
    # older same-shape veto fires first and this test would prove nothing.
    msgs = [mail("a1", "sproctor", "Dominion dispute", "Dominion termination position and counsel", base),
            mail("a2", "kay.mann", "RE: Dominion dispute", "Dominion termination letter drafted", base + timedelta(days=1)),
            mail("b1", "stephanie.panus", "FW: Termination Log", "Dominion termination logged on the master list", base + timedelta(days=10)),
            mail("b2", "l.kelly", "FW: Termination Log", "Dominion termination validated and flagged", base + timedelta(days=11))]
    spy = Spy()
    m = induce(email_mbox.shape(msgs, "x"), slug="x",
               policy=CorrelationPolicy(semantic=SemanticProvider(judge=spy)))
    assert spy.asked == [], "the judge was consulted on a pair with nobody in common"
    assert len(m.cases) == 2

    # …and a pair that DOES share a person is still put to the judge
    msgs[2] = mail("b1", "kay.mann", "FW: Termination Log", "Dominion termination logged on the master list", base + timedelta(days=10))
    spy = Spy()
    m = induce(email_mbox.shape(msgs, "x"), slug="x",
               policy=CorrelationPolicy(semantic=SemanticProvider(judge=spy)))
    assert spy.asked, "a pair with a shared person must still reach the judge"
    a, b = spy.asked[0]
    assert "Who:" in a and "When:" in a, "the judge must be shown who and when"


# --- 3. the leftover kind ------------------------------------------------------

def test_the_leftover_kind_has_a_count_and_no_paths():
    import sys; sys.path.insert(0, "tests")
    from test_reading import _families_corpus, FAMILY_CLASSIFIER
    m = induce(email_mbox.shape(_families_corpus(), "meridian"), slug="meridian")
    a = infer_activities(m, mapper=None, classifier=FAMILY_CLASSIFIER)
    view = build_view(m, activities=a)
    left = next(p for p in view["processes"] if p["leftover"])
    assert left["paths"] == [] and left["flow"] == []
    assert left["n_unread"] > 0


# --- 4. process / project / noise ---------------------------------------------

def test_a_one_off_with_a_real_arc_is_a_project_not_noise():
    """Upgrading the office chairs: one run, five steps. Not a process — nothing
    recurs — but not 'Congratulations' either. It gets its own card, labelled."""
    import sys; sys.path.insert(0, "tests")
    from test_reading import _families_corpus, _mail
    msgs = _families_corpus()
    base = datetime(2001, 3, 1)
    for i, body in enumerate(["Vendor contacted for chair quote", "Quote negotiated down",
                              "Proposal reviewed by facilities", "Chairs paid for",
                              "Chairs delivered to floor 30"]):
        msgs.append(_mail(f"chair{i}", "k.novak",
                          "Office chair upgrade" if i == 0 else "RE: Office chair upgrade",
                          body, 10 + i, thread=None if i == 0 else "chair0"))
    classifier = ScriptedRecordClassifier([
        ("Requested", ["please review the credit terms"], "Contract execution"),
        ("Reviewed", ["look fine from our side"], "Contract execution"),
        ("Approved", ["approved - proceed"], "Contract execution"),
        ("Executed", ["executed and filed"], "Contract execution"),
        ("Chased", ["remains unpaid"], "Invoice dispute"),
        ("Disputed", ["we dispute this charge"], "Invoice dispute"),
        ("Settled", ["settled in full"], "Invoice dispute"),
        ("Vendor contacted", ["vendor contacted"], "Office chair upgrade"),
        ("Quote negotiated", ["quote negotiated"], "Office chair upgrade"),
        ("Proposal reviewed", ["proposal reviewed"], "Office chair upgrade"),
        ("Paid", ["paid for"], "Office chair upgrade"),
        ("Delivered", ["delivered to floor"], "Office chair upgrade"),
    ])
    m = induce(email_mbox.shape(msgs, "meridian"), slug="meridian")
    a = infer_activities(m, mapper=None, classifier=classifier)
    view = build_view(m, activities=a)
    chairs = next(p for p in view["processes"] if p["name"] == "Office chair upgrade")
    assert chairs["project"] is True and chairs["count"] == 1
    assert "project, not a process" in chairs["why"]
    assert chairs["flow"] == ["Vendor contacted", "Quote negotiated", "Proposal reviewed",
                              "Paid", "Delivered"]
    # a real process is not mislabelled
    assert next(p for p in view["processes"] if p["name"] == "Contract execution")["project"] is False


def test_a_one_off_with_one_readable_step_is_still_noise():
    """One step is correspondence, not an arc."""
    from induction.steps.segment import _split_by_read_process
    case = Case(id="c1", kind_hint="email", anchor={}, confidence=None,
                trace_signature=("Acknowledged",))
    out, terms = _split_by_read_process({("k",): ["c1"]}, {"c1": "Thanks"},
                                        cases={"c1": case})
    assert terms == {} and out == {("k",): ["c1"]}


# --- 5. variants by shape ------------------------------------------------------

@pytest.mark.parametrize("a,b,same", [
    (("A", "B", "C"), ("A", "B", "C", "B"), True),    # loop-back
    (("A", "B", "C"), ("A", "B", "B", "C"), True),    # rework
    (("A", "B", "C"), ("A", "C"), False),             # skipped
    (("A", "B", "C"), ("A", "C", "B"), False),        # reordered
])
def test_shape_groups_loops_and_splits_real_deviations(a, b, same):
    assert (shape(a) == shape(b)) is same


def test_variants_are_counted_by_shape_and_keep_the_exact_traces():
    """Twelve runs used to be twelve rows at 1x. The brief wants the common path,
    the exceptions, and the person with their own way — readable off the list."""
    def case(i, trace):
        return Case(id=f"c{i}", kind_hint="x", anchor={}, confidence=None,
                    trace_signature=trace)
    traces = ([("A", "B", "C")] * 5 + [("A", "B", "C", "B")] * 3 +   # one shape, 8 runs
              [("A", "C")] * 2 +                                     # the exception
              [("A", "C", "B")])                                     # someone's own way
    cases = {f"c{i}": case(i, t) for i, t in enumerate(traces)}
    variants, _ = induced_variants(list(cases), cases)
    by_sig = {v.signature: v for v in variants}
    assert len(variants) == 3
    common = by_sig[("A", "B", "C")]
    assert common.frequency == 8 and common.role == "common"
    assert common.traces == {("A", "B", "C"): 5, ("A", "B", "C", "B"): 3}
    assert by_sig[("A", "C")].role == "exception"
    assert by_sig[("A", "C", "B")].role == "one-off"
