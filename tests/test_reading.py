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


# --- the processes, read from the records rather than the envelope -----------
#
# The steps were only half the problem. Kinds were clustered on
# `(automated, case.kind_hint)`, which for a mailbox is `(False, 'email')` for
# every run in the corpus — one kind, always, whatever the mail was about. The
# fallback was token overlap over pooled thread text, and on the repo's own Enron
# sample it produced kinds called `shirley, shall, time` and `hey, ena, work`.
#
# So the reading tier now derives a second vocabulary — the process families the
# corpus is a record of — and the segmenter clusters on THAT. These pin the two
# halves of the guarantee: the boundary is drawn by the engine (a count over each
# run's own records), and it is drawn only where the records support it.

FAMILY_CLASSIFIER = ScriptedRecordClassifier([
    ("Requested", ["please review the credit terms"], "Contract execution"),
    ("Reviewed", ["look fine from our side"], "Contract execution"),
    ("Approved", ["approved - proceed"], "Contract execution"),
    ("Executed", ["executed and filed"], "Contract execution"),
    ("Chased", ["remains unpaid"], "Invoice dispute"),
    ("Disputed", ["we dispute this charge"], "Invoice dispute"),
    ("Settled", ["settled in full"], "Invoice dispute"),
    ("Onboarded", ["vendor pack returned"], "Vendor onboarding"),
])


def _families_corpus():
    """14 contract runs, 12 invoice disputes, 2 vendor onboardings (below the
    floor) and 10 unreadable one-offs.

    Every run has the identical structural shape — `(human, email_thread)` — so
    the structural key cannot separate a contract from an unpaid invoice, which
    is the whole situation the reading tier exists for. Threads are the length
    real ones are (a step gets chased, a reviewer answers twice), so the corpus
    clears the transport gate on its own verbs rather than on a fixture that was
    tuned until it did.
    """
    msgs, n = [], [0]

    def m(frm, subject, body, day, thread=None):
        n[0] += 1
        mid = f"f{n[0]}"
        msgs.append(_mail(mid, frm, subject, body, day, thread))
        return mid

    for i in range(14):
        day = i % 28 + 1
        root = m("a.okonkwo", f"Master agreement - CP{i}",
                 f"Please review the credit terms on CP{i}.", day)
        m("l.bergstrom", f"RE: Master agreement - CP{i}",
          f"Credit terms look fine from our side on CP{i}.", day, root)
        m("d.acheampong", f"RE: Master agreement - CP{i}",
          f"Second pass - credit terms look fine from our side, CP{i}.", day, root)
        m("p.varga", f"RE: Master agreement - CP{i}",
          f"Approved - proceed to execution for CP{i}.", day, root)
        m("a.okonkwo", f"RE: Master agreement - CP{i}",
          f"Executed and filed for CP{i}.", day, root)

    for i in range(12):
        day = i % 28 + 1
        root = m("t.mbeki", f"Invoice 90{i}", f"Invoice 90{i} remains unpaid.", day)
        m("s.haddad", f"RE: Invoice 90{i}",
          f"We dispute this charge on invoice 90{i}.", day, root)
        m("t.mbeki", f"RE: Invoice 90{i}",
          f"Second notice - invoice 90{i} remains unpaid.", day, root)
        m("s.haddad", f"RE: Invoice 90{i}", f"Settled in full for 90{i}.", day, root)

    for i in range(2):
        root = m("k.novak", f"Vendor pack {i}", f"Vendor pack returned for {i}.", i + 1)
        m("k.novak", f"RE: Vendor pack {i}", f"Vendor pack returned, countersigned {i}.",
          i + 1, root)

    odd = "parking badge printer stapler kettle lanyard mug bicycle locker fridge".split()
    for i in range(10):
        m("r.deniz", f"{odd[i].title()} note {i}", f"{odd[i]} {odd[(i + 3) % 10]} {i}", i + 1)
    return msgs


@pytest.fixture(scope="module")
def families():
    m = induce(email_mbox.shape(_families_corpus(), "meridian"), slug="meridian")
    structural = [k.name for k in m.kinds]
    abstraction = infer_activities(m, mapper=None, classifier=FAMILY_CLASSIFIER)
    return m, abstraction, build_view(m, activities=abstraction), structural


def test_before_reading_the_kinds_are_named_after_vocabulary_not_work(families):
    """The premise, and the thing being replaced.

    Every run here is `(human, email_thread)`, so the structural key yields one
    undifferentiated cluster and the token fallback takes over. It does separate
    this clean synthetic corpus — and names the results after whichever words
    happened to be rare, which is the failure the real Enron sample makes
    unmissable (`shirley, shall, time`). Shared vocabulary is not shared work,
    and no reader can act on a process called `agreement, approved, cp`.
    """
    _, _, _, structural = families
    assert any("," in name for name in structural), structural
    assert not any(name in ("Contract execution", "Invoice dispute")
                   for name in structural)


def test_reading_the_records_finds_the_real_processes(families):
    m, _, view, _ = families
    named = {p["name"]: p["count"] for p in view["processes"]}
    assert named.get("Contract execution") == 14
    assert named.get("Invoice dispute") == 12


def test_each_process_has_the_steps_its_records_evidence(families):
    """The card's headline is the process's STEPS, in the order runs put them —
    not one run's trace. A step set does not repeat a step; the repetition lives
    in the variants below, where it belongs."""
    _, _, view, _ = families
    procs = {p["name"]: p for p in view["processes"]}
    assert procs["Contract execution"]["flow"] == [
        "Requested", "Reviewed", "Approved", "Executed"]
    assert procs["Invoice dispute"]["flow"] == ["Chased", "Disputed", "Settled"]
    # …and the second chase is still on the record, in the run's own path.
    seqs = {tuple(pa["seq"]) for pa in procs["Invoice dispute"]["paths"]}
    assert ("Chased", "Disputed", "Chased", "Settled") in seqs


def test_the_headline_is_not_one_runs_trace(families):
    """The bug this replaced: the headline was the most frequent TRACE, so on any
    corpus with abstention it collapsed to a single chip — a seven-run process
    whose runs show Requested, Approved and Escalated was summarised as
    "Approved", because three runs happened to have one readable record each."""
    _, _, view, _ = families
    for p in view["processes"]:
        if p["tier"] != "model":
            continue
        steps_seen = {s for pa in p["paths"] for s in pa["seq"]}
        assert set(p["flow"]) == steps_seen, (
            f"{p['name']}: headline {p['flow']} drops steps its runs perform")


def test_a_read_boundary_says_it_is_model_tier_and_why(families):
    """A kind boundary is an inference like any other. Read from the records it
    is `model`, and the card must say the engine counted rather than the model
    decided."""
    m, _, view, _ = families
    kind = next(k for k in m.kinds if k.name == "Contract execution")
    assert kind.confidence.tier.label == "model"
    assert "read from its records" in kind.confidence.rationale
    card = next(p for p in view["processes"] if p["name"] == "Contract execution")
    assert card["tier"] == "model" and "count" in card["why"]


def test_a_family_below_the_floor_is_not_a_kind(families):
    """2 runs is over-splitting, not a rare process — the reading proposes
    families from a sample, so a family of one or two folds back into the parent
    rather than becoming a card."""
    _, _, view, _ = families
    assert "Vendor onboarding" not in {p["name"] for p in view["processes"]}


def test_runs_the_reading_declined_stay_in_the_structural_kind(families):
    """The 10 one-offs match no rule at all, and the 2 vendor runs belong to a
    family too small to be a kind. Neither is forced into a process nothing
    evidenced — they keep an unnamed structural kind."""
    m, abstraction, _, _ = families
    unplaced = [c for c in m.cases if c not in set(abstraction.by_case)]
    assert len(unplaced) == 10
    read_kinds = {k.id for k in m.kinds if k.features.get("read_process")}
    leftovers = unplaced + [cid for cid, p in abstraction.by_case.items()
                            if p == "Vendor onboarding"]
    assert len(leftovers) == 12
    for cid in leftovers:
        owner = next(k for k in m.kinds if cid in k.case_ids)
        assert owner.id not in read_kinds


def test_a_runs_process_is_a_count_over_its_own_records(families):
    """The engine places the run; the model only named the families. A thread
    whose records mostly read as one family lands there, and the placement is
    reproducible from `by_record` alone."""
    from collections import Counter
    m, abstraction, _, _ = families
    for cid, process in abstraction.by_case.items():
        votes = Counter(abstraction.process_of(e) for e in m.cases[cid].ordered_event_ids)
        votes.pop(None, None)
        assert votes and process == max(votes, key=votes.get)


def test_no_process_vocabulary_leaves_segmentation_alone(families):
    """The old classifier proposes activities and no families. Steps must still
    be read, and the kinds must stay exactly as structure left them."""
    m = induce(email_mbox.shape(_families_corpus(), "meridian"), slug="meridian")
    before = [k.name for k in m.kinds]
    abstraction = infer_activities(m, mapper=None, classifier=CLASSIFIER)
    assert abstraction.by_record            # steps were still read
    assert abstraction.by_case == {}        # but nothing was re-segmented
    assert [k.name for k in m.kinds] == before


def test_the_family_floor_scales_with_the_corpus():
    """3 out of 40 is a process; 3 out of 500 is a splinter.

    A flat floor is tuned to whatever corpus was in front of you when you picked
    it. `topics.py` already answers this with a floor AND a share, and the read
    path must answer it the same way or the two disagree the moment a corpus is
    a different size.
    """
    from induction.steps.segment import _split_by_read_process

    small = {("k",): [f"c{i}" for i in range(40)]}
    big = {("k",): [f"c{i}" for i in range(500)]}
    place = lambda ids: {c: ("Rare" if i < 3 else "Common")
                         for i, c in enumerate(ids)}

    out, terms = _split_by_read_process(small, place(small[("k",)]))
    assert "Rare" in terms.values(), "3 of 40 is a family"

    out, terms = _split_by_read_process(big, place(big[("k",)]))
    assert "Rare" not in terms.values(), "3 of 500 is a splinter"
    assert sum(len(v) for v in out.values()) == 500, "no run may be dropped"


def test_no_run_is_ever_lost_or_duplicated_by_a_split():
    """The split is a partition. Whatever the floor decides, every run comes out
    exactly once — a kind that quietly drops runs is worse than no kind."""
    from induction.steps.segment import _split_by_read_process

    clusters = {("a",): [f"a{i}" for i in range(30)],
                ("b",): [f"b{i}" for i in range(12)]}
    process = {}
    for i in range(30):
        process[f"a{i}"] = ["X", "Y", "Z"][i % 3]
    for i in range(0, 12, 2):
        process[f"b{i}"] = "W"          # half of b placed, half declined
    out, _ = _split_by_read_process(clusters, process)
    got = [cid for ids in out.values() for cid in ids]
    assert sorted(got) == sorted(cid for ids in clusters.values() for cid in ids)
    assert len(got) == len(set(got))


def test_the_discovery_sample_is_drawn_across_the_whole_corpus():
    """The sample names the vocabulary for every record, so what it
    over-represents, the vocabulary over-fits. Records arrive grouped by kind and
    by whatever order the adapter walked the source, so the head of the list is
    one corner of the corpus, not a picture of it."""
    from induction.abstraction import _spread

    corpus = list(range(263))
    got = _spread(corpus, 150)
    assert len(got) == len(set(got)) == 150
    assert got == sorted(got), "order is preserved; only the stride is imposed"
    # every third of the corpus is represented in roughly its true proportion
    for lo, hi in ((0, 88), (88, 176), (176, 263)):
        share = sum(1 for x in got if lo <= x < hi) / 150
        assert 0.28 < share < 0.39, (lo, hi, share)


def test_a_corpus_smaller_than_the_sample_is_taken_whole():
    from induction.abstraction import _spread
    assert _spread([1, 2, 3], 150) == [1, 2, 3]
    assert _spread([], 150) == []
    assert _spread([1, 2, 3], 0) == []


# --- a skipped step is a finding; a kind with no usual way is not -------------

def test_a_run_that_jumps_a_step_everyone_else_takes_is_a_gap():
    """The hiring case: CV screened, offer made, no interview in between.

    Expectation is per STEP, not per whole trace. These four runs share no trace
    at all, yet `Reviewed` is plainly normal and the first run plainly skipped
    it — and that skip is the finding this detector exists for.
    """
    from induction.process import ProcessKind, Variant
    from induction.steps.gaps_generic import _canonical_order

    kind = ProcessKind(id="k", name="k", rationale="", confidence=None, variants=[
        Variant(signature=("Screened", "Offered"), frequency=1, case_ids=["a"]),
        Variant(signature=("Screened", "Reviewed", "Offered"), frequency=1, case_ids=["b"]),
        Variant(signature=("Screened", "Reviewed", "Offered", "Hired"), frequency=1,
                case_ids=["c"]),
    ])
    canon = _canonical_order(kind)
    assert canon == ["Screened", "Reviewed", "Offered"], canon
    assert "Hired" not in canon, "one run in three is not an expectation"


def test_a_step_only_one_run_takes_is_not_expected_of_the_others():
    """The other half of the same rule. The claim a gap makes is 'the other runs
    did this and yours did not', so the other runs must be most of them."""
    from induction.process import ProcessKind, Variant
    from induction.steps.gaps_generic import _canonical_order

    kind = ProcessKind(id="k", name="k", rationale="", confidence=None, variants=[
        Variant(signature=("A", "B"), frequency=5, case_ids=[]),
        Variant(signature=("A", "Z", "B"), frequency=1, case_ids=[]),
    ])
    assert _canonical_order(kind) == ["A", "B"]


def test_a_kind_with_no_shared_step_expects_nothing():
    """Six runs with nothing in common cannot accuse each other of anything. The
    old rule handed the LONGEST run to the detector as the standard, so five runs
    were reported missing a dozen steps each."""
    from induction.process import ProcessKind, Variant
    from induction.steps.gaps_generic import _canonical_order

    kind = ProcessKind(id="k", name="k", rationale="", confidence=None, variants=[
        Variant(signature=tuple(f"S{i}{j}" for j in range(i + 1)), frequency=1, case_ids=[])
        for i in range(6)
    ])
    assert _canonical_order(kind) == []
