"""Walking the hierarchy the vocabulary already is.

The flat path offers every `Process > Step` pair in one Choice — on the Enron
sample, 28 options across 7 families — which asks a model to weigh an invoice
stage against a hiring stage in the same breath. The beam asks 7 then 3-6.

Three things are pinned here, and the middle one is the reason a beam rather
than a greedy walk:

  - the walk asks about ONE record, never about a run, so the engine still draws
    every boundary itself;
  - a close call at the process level can be settled by the step level, which a
    greedy walk cannot do and a flat call never faces;
  - a beam result reads exactly like a flat one downstream — same distribution
    spelling, same separation, same ledger — so the flag is a flag and not a
    second pipeline.
"""

from __future__ import annotations

import pytest

from induction.jev_call import Jev
from induction.jev_reading import (JevReading, RecordAssigner, _geometric_mean,
                                   _BEAM_PROCESS, _BEAM_STEP)
from tests.test_jev import FakeVocab


def walking(levels: dict, seen: list | None = None):
    """A transport that answers whichever level of the walk it is handed.

    `levels` maps a question key to `{option: probability}`; the step level maps
    a process name to its own distribution, because the walk asks it once per
    candidate process.
    """
    def send(url, payload, api_key, timeout):
        if seen is not None:
            seen.append(payload)
        key = next(iter(payload["questions"]))
        question = payload["questions"][key]
        offered = set(question["criteria"])
        if key == _BEAM_PROCESS:
            probs = levels[_BEAM_PROCESS]
        else:
            # Which process's steps are on the table decides which answer to give.
            probs = next(p for name, p in levels[_BEAM_STEP].items()
                         if set(p) <= offered and offered <= set(p) or set(p) & offered)
        probs = {k: v for k, v in probs.items() if k in offered}
        if not probs:
            return {"answers": {}}
        return {"answers": {key: {"choice": max(probs, key=probs.get),
                                  "probabilities": probs}}}
    return send


VOCAB = FakeVocab({
    "Hiring": ["CV screened", "Onsite interview", "Offer made"],
    "Billing": ["Invoice issued", "Payment received"],
    "Credit": ["Letter drafted"],
})


def _beam(levels, seen=None, **kw):
    return RecordAssigner(jev=Jev(transport=walking(levels, seen)), mode="beam", **kw)


# ---------------------------------------------------------------------------
# The score
# ---------------------------------------------------------------------------

def test_the_path_score_is_length_normalised():
    """A plain product makes every deeper path look worse than every shallower
    one for no reason but its depth, and the engine compares paths of different
    lengths constantly."""
    assert _geometric_mean([0.5, 0.5]) == pytest.approx(0.5)
    assert _geometric_mean([0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert _geometric_mean([0.9, 0.4]) == pytest.approx(0.6)


def test_an_empty_path_scores_zero_rather_than_one():
    assert _geometric_mean([]) == 0.0


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------

def test_a_clear_walk_reaches_the_expected_leaf():
    got = _beam({_BEAM_PROCESS: {"Hiring": 0.8, "Billing": 0.15, "Credit": 0.05},
                 _BEAM_STEP: {"Hiring": {"Onsite interview": 0.9, "Offer made": 0.05,
                                         "CV screened": 0.05},
                              "Billing": {"Invoice issued": 0.5, "Payment received": 0.5},
                              "Credit": {"Letter drafted": 1.0}}}).assign({}, VOCAB)
    assert (got.process, got.step) == ("Hiring", "Onsite interview")
    assert got.path[0] == ("Hiring", pytest.approx(0.8))


def test_the_step_level_can_settle_a_close_process_call():
    """The reason this is a beam. Hiring leads narrowly and its steps are a mess;
    Billing trails and its steps are certain. A greedy walk commits to Hiring and
    cannot recover. Keeping both alive lets the deeper evidence decide."""
    got = _beam({_BEAM_PROCESS: {"Hiring": 0.40, "Billing": 0.38, "Credit": 0.22},
                 _BEAM_STEP: {"Hiring": {"CV screened": 0.34, "Onsite interview": 0.33,
                                         "Offer made": 0.33},
                              "Billing": {"Invoice issued": 0.97, "Payment received": 0.03},
                              "Credit": {"Letter drafted": 1.0}}}).assign({}, VOCAB)
    assert (got.process, got.step) == ("Billing", "Invoice issued"), (
        "a greedy walk would have committed to the leading process")


def test_only_the_leading_processes_are_explored():
    """A family the record gave 2% to will not win on its steps, and asking is a
    call spent to confirm it."""
    seen = []
    _beam({_BEAM_PROCESS: {"Hiring": 0.8, "Billing": 0.15, "Credit": 0.05},
           _BEAM_STEP: {"Hiring": {"Onsite interview": 1.0},
                        "Billing": {"Invoice issued": 1.0},
                        "Credit": {"Letter drafted": 1.0}}},
          seen, beam_width=2).assign({}, VOCAB)
    step_calls = [p for p in seen if _BEAM_STEP in p["questions"]]
    assert len(step_calls) == 2, "beam_width=2 should explore two processes"


def test_the_walk_never_asks_about_a_run():
    """The engine's standing rule. The process level is a question about THIS
    RECORD — which family does it evidence — and the run's family is still the
    plurality of its records' answers, counted by `_process_of_case`."""
    seen = []
    _beam({_BEAM_PROCESS: {"Hiring": 0.9, "Billing": 0.1},
           _BEAM_STEP: {"Hiring": {"Offer made": 1.0},
                        "Billing": {"Invoice issued": 1.0}}}, seen).assign(
        {"subject": "x"}, VOCAB)
    for payload in seen:
        text = str(payload["questions"])
        assert "thread" not in text.lower() or "thread_context" in text
        assert "run of work" not in text.lower()
        assert "`record`" in text or "this record" in text


# ---------------------------------------------------------------------------
# A beam result must read like a flat one
# ---------------------------------------------------------------------------

def test_the_distribution_keeps_the_flat_spelling():
    """Everything downstream — `by_process`, `separation`, the ledger, the page —
    reads `Process > Step`. A beam that spelled it differently would be a second
    pipeline wearing a flag."""
    got = _beam({_BEAM_PROCESS: {"Hiring": 0.7, "Billing": 0.3},
                 _BEAM_STEP: {"Hiring": {"Offer made": 0.9, "CV screened": 0.1},
                              "Billing": {"Invoice issued": 1.0}}}).assign({}, VOCAB)
    assert all(" > " in name for name in got.distribution)
    assert set(got.by_process()) <= {"Hiring", "Billing", "Credit"}
    assert got.separation is not None


def test_a_beam_result_goes_through_the_same_ambiguity_test():
    got = _beam({_BEAM_PROCESS: {"Hiring": 0.34, "Billing": 0.33, "Credit": 0.33},
                 _BEAM_STEP: {"Hiring": {"Offer made": 1.0},
                              "Billing": {"Invoice issued": 1.0},
                              "Credit": {"Letter drafted": 1.0}}}).assign({}, VOCAB)
    assert got.ambiguous, "a dead heat between families is a dead heat either way"


def test_the_whole_path_reaches_the_decision_record():
    """The beam's characteristic failure is a wrong turn at the process level
    that every step below inherits. A reader shown only the leaf cannot see it."""
    from induction.decisions import Ledger

    ledger = Ledger()
    reading = JevReading(jev=Jev(transport=walking({})), ledger=ledger)
    placed = _beam({_BEAM_PROCESS: {"Hiring": 0.7, "Billing": 0.3},
                    _BEAM_STEP: {"Hiring": {"Offer made": 0.9, "CV screened": 0.1},
                                 "Billing": {"Invoice issued": 1.0}}}).assign({}, VOCAB)
    reading._record_step("evt:1", placed)
    derived = ledger.to_list()[0]["derived"]
    assert any(k.startswith("path 1: Hiring") for k in derived)
    assert any(k.startswith("path 2: Offer made") for k in derived)


# ---------------------------------------------------------------------------
# Degrading
# ---------------------------------------------------------------------------

def test_an_unanswered_process_level_places_nothing():
    got = RecordAssigner(jev=Jev(transport=lambda *a: {"answers": {}}),
                         mode="beam").assign({}, VOCAB)
    assert not got.placed


def test_a_vocabulary_of_loose_steps_falls_back_to_the_flat_call():
    """With no processes there is no hierarchy to walk, and the flat path is
    already the right shape for a bare list of steps."""
    seen = []
    loose = FakeVocab({}, ["Archived", "Filed"])
    RecordAssigner(jev=Jev(transport=lambda url, payload, k, t: (
        seen.append(payload) or {"answers": {"step": {"choice": "Archived",
                                                      "probabilities": {"Archived": 1.0}}}})),
                   mode="beam").assign({}, loose)
    assert len(seen) == 1, "one flat call, not a walk"
    assert set(seen[0]["questions"]["step"]["criteria"]) == {"Archived", "Filed"}


def test_flat_stays_the_default():
    assert RecordAssigner(jev=Jev(transport=lambda *a: {"answers": {}})).mode == "flat"
    assert JevReading(jev=Jev(transport=lambda *a: {"answers": {}})).assigner.mode == "flat"
