"""Tier 2 of abstraction: reading the record, where the verb is only transport.

A mailbox records `sent` and nothing else. Asked what `sent` means, a model can
only answer once — which is how a corpus of 761 threads produced one process
whose entire flow was "Communicated". The verb-level map cannot do better,
because the information is not in the verb.

So these tests pin the two halves of the answer:

  - **When it fires.** Only where the verbs demonstrably marked transmissions
    rather than stages, measured off the data (records per distinct activity).
    A tracker and a git corpus keep their verbs.
  - **What it may claim.** A label on a record that exists, with the span it was
    read from. Never a record it invented, never an activity outside the
    vocabulary it proposed, and never a guess where it should abstain — the
    absent step is found by the deterministic detector, not imagined by a model.
"""

from __future__ import annotations

import pytest

from induction.abstraction import (Abstraction, ScriptedRecordClassifier,
                                   infer_activities)
from induction.adapters import email_mbox
from induction.inspector import build_view
from induction.pipeline import induce

D = "meridian-energy.example"

CLASSIFIER = ScriptedRecordClassifier([
    ("Requested", ["please review the credit terms"]),
    ("Reviewed", ["look fine from our side"]),
    ("Approved", ["approved - proceed"]),
    ("Executed", ["executed and filed"]),
])


def _mail(mid, frm, subject, body, day, thread=None):
    head = (f"Message-ID: <{mid}>\nFrom: {frm}@{D}\nTo: desk@{D}\n"
            f"Date: Mon, {day} Oct 2001 09:00:00 -0700\nSubject: {subject}\n")
    if thread:
        head += f"In-Reply-To: <{thread}>\n"
    return (mid, head + "\n" + body + "\n")


def _corpus():
    """20 contract runs — 14 complete, 4 executed with NO approval on record,
    2 stalled at review — plus 30 unrelated one-offs the reader must decline."""
    msgs, n = [], [0]

    def m(frm, subject, body, day, thread=None):
        n[0] += 1
        mid = f"m{n[0]}"
        msgs.append(_mail(mid, frm, subject, body, day, thread))
        return mid

    for i in range(20):
        day = i % 28 + 1
        root = m("a.okonkwo", f"Master agreement - Counterparty {i}",
                 f"Please review the credit terms on Counterparty {i} and the "
                 f"netting annex before we execute.", day)
        m("l.bergstrom", f"RE: Master agreement - Counterparty {i}",
          f"Credit terms look fine from our side on Counterparty {i}.", day, root)
        if i >= 6:
            m("p.varga", f"RE: Master agreement - Counterparty {i}",
              f"Approved - proceed to execution for Counterparty {i}.", day, root)
        if i >= 2:
            m("a.okonkwo", f"RE: Master agreement - Counterparty {i}",
              f"Executed and filed for Counterparty {i}. Original to the vault.", day, root)

    odd = "parking badge printer stapler kettle lanyard mug bicycle locker fridge".split()
    for i in range(30):
        m("r.deniz", f"{odd[i % 10].title()} note {i}",
          f"{odd[i % 10]} {odd[(i + 3) % 10]} {i}", i % 28 + 1)
    return msgs


@pytest.fixture(scope="module")
def read_model():
    m = induce(email_mbox.shape(_corpus(), "meridian"), slug="meridian")
    abstraction = infer_activities(m, mapper=None, classifier=CLASSIFIER)
    return m, abstraction, build_view(m, activities=abstraction)


# --- what it produces --------------------------------------------------------

def test_a_mailbox_yields_a_real_multi_step_process(read_model):
    _, _, view = read_model
    contracts = max(view["processes"], key=lambda p: len(p["flow"]))
    assert contracts["flow"] == ["Requested", "Reviewed", "Approved", "Executed"], (
        "the spine must be what the messages DO, not the verb the mailbox recorded")


def test_the_variants_are_the_planted_truth(read_model):
    """14 complete, 4 executed with no approval, 2 stalled — counted, not guessed."""
    _, _, view = read_model
    contracts = max(view["processes"], key=lambda p: len(p["flow"]))
    got = {tuple(p["seq"]): p["count"] for p in contracts["paths"]}
    assert got[("Requested", "Reviewed", "Approved", "Executed")] == 14
    assert got[("Requested", "Reviewed", "Executed")] == 4
    assert got[("Requested", "Reviewed")] == 2


def test_the_control_finding_is_a_real_gap_with_evidence(read_model):
    """'Executed with no approval on record' has to survive into model.json as a
    Gap — a label in a chart is not a finding."""
    m, _, view = read_model
    missing = [g for g in m.gaps if g.kind == "missing_expected_step"]
    assert len(missing) == 4
    for g in missing:
        assert g.confidence.tier.label == "heuristic"   # inferred, never asserted
        assert g.evidence
        assert "Approved" in g.description
    assert sum(1 for r in view["runs"] if r["dev_label"] == "No Approved step") == 4


def test_every_read_step_shows_the_span_it_was_read_from(read_model):
    """The difference between a claim you can check and one you must accept."""
    _, abstraction, view = read_model
    run = next(r for r in view["runs"] if r["dev_label"] == "No Approved step")
    for node in run["activities"]:
        art = node["arts"][0]
        assert art["note"], f"{node['name']} carries no span"
        assert art["src"], f"{node['name']} carries no source locator"
    assert all(r["span"] for r in abstraction.by_record.values())


def test_unreadable_records_abstain_rather_than_guess(read_model):
    """30 one-offs match no rule. They must keep their raw verb and be COUNTED as
    unclassified — the abstention rate is the number that says whether to trust
    any of this."""
    _, abstraction, view = read_model
    assert abstraction.n_unclassified == 30
    row = next(r for r in abstraction.vocabulary if r.get("unclassified"))
    assert row["n"] == 30
    assert any(r["unread"] for r in view["runs"])


def test_the_audit_table_shows_what_each_step_was_read_from(read_model):
    _, abstraction, _ = read_model
    rows = {r["activity"]: r for r in abstraction.vocabulary}
    assert rows["Approved"]["n"] == 14
    assert "approved - proceed" in rows["Approved"]["phrases"]


# --- when it fires, and what it may claim ------------------------------------

def test_a_staged_source_keeps_its_verbs(read_model):
    """The gate. A tracker's verbs happen once per run and already ARE the
    process; reading its rows would be cost with no answer. Measured on the
    repo's own spreadsheet corpus, not asserted in prose."""
    from pathlib import Path

    import run_tabular
    from induction.pipeline import run_tabular_pipeline
    m = run_tabular_pipeline(run_tabular.sources_for(Path("samples/finance"), "csv"),
                             slug="finance")
    abstraction = infer_activities(m, mapper=None, classifier=CLASSIFIER)
    assert abstraction.by_record == {}, "a staged source must not be re-read"
    assert abstraction.vocabulary == []


def test_no_classifier_changes_nothing(read_model):
    m = induce(email_mbox.shape(_corpus(), "meridian"), slug="meridian")
    before = {cid: c.trace_signature for cid, c in m.cases.items()}
    abstraction = infer_activities(m, mapper=None, classifier=None)
    assert not abstraction.by_record
    assert {cid: c.trace_signature for cid, c in m.cases.items()} == before


def test_a_reading_may_not_invent_a_record_or_an_activity():
    """The guardrail, mirroring naming.py's `_clean`: a label only ever lands on
    a record we asked about, with an activity we proposed."""
    from induction.abstraction import _clean_readings
    batch = [{"id": "real", "text": "..."}]
    raw = {
        "real": {"activity": "Approved", "span": "approved"},
        "never-sent": {"activity": "Approved", "span": "approved"},   # unknown id
        "real2": {"activity": "Invented", "span": "x"},               # outside vocab
    }
    got = _clean_readings(raw, batch, ["Approved", "Reviewed"])
    assert got == {"real": {"activity": "Approved", "span": "approved"}}


def test_a_reading_without_its_span_is_dropped():
    """No quote, no claim — an unevidenced label is exactly what this engine
    exists not to produce."""
    from induction.abstraction import _clean_readings
    batch = [{"id": "r1", "text": "..."}]
    assert _clean_readings({"r1": {"activity": "Approved"}}, batch, ["Approved"]) == {}


def test_a_bare_dict_is_still_a_tier_one_abstraction():
    """Every existing caller passes `{artefact/verb: Activity}`; that must keep
    meaning exactly what it meant."""
    a = Abstraction.of({"email/sent": "Raised"})
    assert a.activity_of("evt:1", "email", "sent") == "Raised"
    assert a.span_of("evt:1") is None
