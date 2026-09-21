"""What was asked, and what the engine did with the answer.

A number on a page is not traceability. These pin the difference: that every
typed decision a reader can see the consequence of reaches `model.json` with the
question behind it, that the engine's *outcome* is recorded separately from the
model's *answer* — they are not the same claim and frequently differ — and that
decisions whose consequence is invisible are deliberately not recorded.
"""

from __future__ import annotations

import pytest

from induction.decisions import JOIN, LABEL, REJECT, STEP, Decision, Ledger
from induction.jev_call import Jev
from induction.jev_reading import JevReading, RecordAssigner
from induction.semantic import GatedJudge, JevGate, ScriptedJudge, SemanticJudge
from tests.test_jev import FakeVocab, _scopes, transport


# ---------------------------------------------------------------------------
# The ledger itself
# ---------------------------------------------------------------------------

def test_a_decision_serialises_with_its_question_and_its_outcome():
    row = Decision(kind=STEP, about="evt:1", question="Which stage?", qtype="choice",
                   answer="Hiring > Offer made", outcome="read by the generative pass",
                   confidence=0.923456, distribution={"a": 0.1, "b": 0.9}).to_dict()
    assert row["question"] == "Which stage?"
    assert row["outcome"] == "read by the generative pass"
    assert row["confidence"] == 0.9235
    assert list(row["distribution"]) == ["b", "a"], "options come back best-first"


def test_absent_numbers_are_absent_rather_than_zero():
    """A missing confidence must not serialise as 0.0 — that reads as certainty
    about the opposite, which is the one thing it does not mean."""
    row = Decision(kind=LABEL, about="P > L", question="q", qtype="noul",
                   answer=0.2, outcome="dropped").to_dict()
    assert "confidence" not in row and "distribution" not in row


def test_the_ledger_only_appends():
    ledger = Ledger()
    for i in range(3):
        ledger.record(kind=STEP, about=f"evt:{i}", question="q", qtype="choice",
                      answer="x", outcome="y")
    assert len(ledger) == 3
    assert ledger.counts() == {"step": 3}
    assert [row["about"] for row in ledger.to_list()] == ["evt:0", "evt:1", "evt:2"]


# ---------------------------------------------------------------------------
# What gets recorded, and what deliberately does not
# ---------------------------------------------------------------------------

class CountingJudge(SemanticJudge):
    def __init__(self, reason="shared incident"):
        self.reason = reason

    def judge(self, a, b):
        return self.reason


def test_a_join_records_the_question_and_that_it_joined():
    ledger = Ledger()
    gate = JevGate(jev=Jev(transport=transport({
        "same_work": {"noul": 0.87},
        "relation": {"choice": "causal", "probabilities": {"causal": 0.8}}})))
    GatedJudge(CountingJudge(), gate=gate, ledger=ledger).judge(
        "Invoice 4471 raised", "Payment against 4471")
    row = ledger.to_list()[0]
    assert row["kind"] == JOIN
    assert "SAME piece of work" in row["question"]
    assert row["answer"] == 0.87
    assert "joined at tier `model`" in row["outcome"]
    assert "shared incident" in row["outcome"]


def test_a_pair_the_gate_refused_is_recorded_too():
    """The half more likely to be wrong. A ledger holding only the joins would
    show a reader every connection made and none of the ones declined."""
    ledger = Ledger()
    gate = JevGate(jev=Jev(transport=transport({"same_work": {"noul": 0.05}})))
    assert GatedJudge(CountingJudge(), gate=gate, ledger=ledger).judge("a", "b") is None
    row = ledger.to_list()[0]
    assert "no join was proposed" in row["outcome"]


def test_a_pair_the_judge_declined_after_the_gate_passed_is_recorded():
    ledger = Ledger()
    gate = JevGate(jev=Jev(transport=transport({"same_work": {"noul": 0.99}})))
    GatedJudge(ScriptedJudge([]), gate=gate, ledger=ledger).judge("a", "b")
    assert "the judge still declined" in ledger.to_list()[0]["outcome"]


def test_a_dropped_label_records_the_scope_that_dropped_it():
    ledger = Ledger()
    reading = JevReading(jev=Jev(transport=transport(_scopes(in_corpus=0.04))),
                         ledger=ledger)
    reading.screen_labels(FakeVocab({"Comms": ["Correspondence Sharing"]}), lambda m: None)
    row = ledger.to_list()[0]
    assert row["kind"] == LABEL
    assert row["about"] == "Comms > Correspondence Sharing"
    assert "rest of this source" in row["question"], "the corpus-scope question"
    assert "dropped from the vocabulary" in row["outcome"]


def test_a_surviving_label_is_not_recorded():
    """Only the decisions with a visible consequence. A label that survived is
    on the page as itself; a row saying so would be noise."""
    ledger = Ledger()
    reading = JevReading(jev=Jev(transport=transport(_scopes())), ledger=ledger)
    reading.screen_labels(FakeVocab({"Billing": ["Invoice issued"]}), lambda m: None)
    assert len(ledger) == 0


def test_the_gate_is_counted_not_recorded():
    """It asserts nothing about any record — only about what was worth reading —
    and one row per gated record would bury the four kinds that matter."""
    ledger = Ledger()
    reading = JevReading(jev=Jev(transport=transport(
        {"performs_action": {"noul": 0.01}, "machine_generated": {"noul": 0.9},
         "carries_work": {"noul": 0.02}})), ledger=ledger)
    reading.signals.read({"body": "thanks!"})
    assert len(ledger) == 0


# ---------------------------------------------------------------------------
# The answer and the outcome are different claims
# ---------------------------------------------------------------------------

def test_the_outcome_is_recorded_separately_from_the_answer():
    """A confident pick that the engine held back anyway is exactly the case a
    reader needs explained, and it is unreadable from the answer alone."""
    ledger = Ledger()
    reading = JevReading(jev=Jev(transport=transport({})), ledger=ledger)
    probs = {"Hiring > Offer made": 0.34, "Billing > Invoice issued": 0.33,
             "Credit > Letter drafted": 0.33}
    assigner = RecordAssigner(jev=Jev(transport=transport(
        {"step": {"choice": "Hiring > Offer made", "probabilities": probs}})))
    placed = assigner.step_of({}, FakeVocab({"Hiring": ["Offer made"],
                                             "Billing": ["Invoice issued"],
                                             "Credit": ["Letter drafted"]}))
    reading._record_step("evt:9", placed)
    row = ledger.to_list()[0]
    assert row["answer"] == "Hiring > Offer made"          # what the model said
    assert "held back as ambiguous" in row["outcome"]      # what the engine did
    assert row["derived"]["Hiring"] == pytest.approx(0.34)


def test_derived_numbers_are_kept_apart_from_the_answer():
    """`distribution` is what the model said; `derived` is arithmetic this code
    did afterwards. Merging them would present a computation as an answer."""
    ledger = Ledger()
    reading = JevReading(jev=Jev(transport=transport({})), ledger=ledger)
    probs = {"Hiring > Offer made": 0.6, "Hiring > CV screened": 0.3,
             "Billing > Invoice issued": 0.1}
    placed = RecordAssigner(jev=Jev(transport=transport(
        {"step": {"choice": "Hiring > Offer made", "probabilities": probs}}))).step_of(
        {}, FakeVocab({"Hiring": ["Offer made", "CV screened"],
                       "Billing": ["Invoice issued"]}))
    reading._record_step("evt:1", placed)
    row = ledger.to_list()[0]
    assert set(row["distribution"]) == set(probs)
    assert row["derived"]["Hiring"] == pytest.approx(0.9)
    assert row["derived"]["Billing"] == pytest.approx(0.1)
    # The margin is derived too — computed here, not answered.
    assert row["derived"]["separation (top ÷ second)"] == pytest.approx(9.0)


def test_no_ledger_means_no_recording_and_no_error():
    """Every seam takes the ledger optionally — a run with the typed tier off
    has nothing to record and must not care."""
    reading = JevReading(jev=Jev(transport=transport(_scopes(in_corpus=0.01))))
    assert reading.ledger is None
    reading.screen_labels(FakeVocab({"P": ["L"]}), lambda m: None)   # must not raise


# ---------------------------------------------------------------------------
# It reaches the artefact, not just the page
# ---------------------------------------------------------------------------

def test_the_ledger_reaches_model_json():
    from induction.adapters import email_mbox
    from induction.emit import build_model
    from induction.pipeline import induce

    m = induce(email_mbox.shape([], "empty") if hasattr(email_mbox, "shape")
               else email_mbox.load("samples/enron", slug="e", max_messages=20),
               slug="e")
    ledger = Ledger()
    ledger.record(kind=REJECT, about="kind_1", question="Is `cluster` a real process?",
                  qtype="choice", answer="machine_noise", outcome="flagged, not deleted",
                  confidence=0.91)
    m.decisions = ledger
    built = build_model(m)
    assert built["typed_decisions"][0]["about"] == "kind_1"
    assert "typed_decisions_note" in built["meta"]


def test_model_json_carries_an_empty_list_when_the_tier_never_ran():
    """Never absent, so a consumer need not special-case it."""
    from induction.adapters import email_mbox
    from induction.emit import build_model
    from induction.pipeline import induce

    m = induce(email_mbox.load("samples/enron", slug="e", max_messages=20), slug="e")
    assert build_model(m)["typed_decisions"] == []
