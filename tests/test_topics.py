"""Topic refinement (Segment, step 0): when a source has only one *shape*.

The rule under test is narrow on purpose, and each half of it is a separate
failure mode:

  - Fire where structure said nothing. A mailbox segments into exactly one
    structural cluster — `(human, email_thread)` for every run — and "they all
    send mail" is not a process model.
  - Stay silent everywhere else. A corpus structure *did* partition (git: 5
    clusters) must come out untouched, because similar text is not the same
    process — refining git's contribution cluster invents kinds like
    "method, add, flask" out of what a commit happens to mention.

The gate between them is `min_dominance`, and these tests pin both sides of it.
"""

from __future__ import annotations

import pytest

from induction.steps.topics import DEFAULT_TOPIC_POLICY, TopicPolicy, refine

ISDA = ("ISDA master agreement with the counterparty; the credit schedule and "
        "netting annex need legal approval before we execute")
GAS = ("Daily gas nomination on the pipeline is confirmed; scheduled volume and "
       "imbalance for the delivery cycle are posted")
HIRE = ("Resume of a PhD candidate for the research group; please review the "
        "interview schedule and confirm the offer committee")


# Genuinely unrelated one-offs: each draws four words no other message uses, so
# nothing links them to each other or to anything else. Reusing one pool of
# words across them would make them a fourth topic, which is a real thing a
# mailbox contains but not what these are here to represent.
_ODDS = ("parking badge holiday printer stapler kettle carpark lanyard whiteboard mug "
         "bicycle umbrella locker thermostat fridge stairwell doormat lightbulb raffle "
         "postbox shredder plantpot signage turnstile awning gutter skylight cushion "
         "doorbell fanbelt wiper hosepipe trellis birdbath sundial gatepost hedgerow "
         "wheelbarrow flagpole sandpit").split()          # 40 distinct words


def _odd(i: int) -> str:
    """Four words no other one-off uses (i < 10) — every token has df=1, so
    nothing can link these to each other or to anything else. A numeric suffix
    would not do it: the tokenizer reads 'parking0' as 'parking'."""
    return " ".join(_ODDS[i * 4:i * 4 + 4])


def _corpus(n_each=10, n_noise=10):
    """One structural cluster: three vocabularies plus unrelated one-offs."""
    texts = {}
    for i in range(n_each):
        texts[f"isda:{i}"] = f"{ISDA} number {i}"
        texts[f"gas:{i}"] = f"{GAS} cycle {i}"
        texts[f"hire:{i}"] = f"{HIRE} candidate {i}"
    for i in range(n_noise):
        texts[f"noise:{i}"] = _odd(i)
    return texts


def test_one_shape_corpus_splits_into_its_topics():
    texts = _corpus()
    topic_of, terms = refine(texts, n_corpus_cases=len(texts))

    groups = {}
    for cid, t in topic_of.items():
        groups.setdefault(t, set()).add(cid.split(":")[0])
    # each topic is drawn from exactly one vocabulary — no cross-contamination
    assert all(len(families) == 1 for families in groups.values()), groups
    assert {"isda", "gas", "hire"} == {next(iter(f)) for f in groups.values()}
    # and every topic names the words that made it one
    assert all(terms[t] for t in groups)


def test_unexplained_runs_stay_in_the_parent_kind():
    """The honesty guard: a run whose vocabulary matched nothing is not promoted
    into a one-run 'process'. It stays where structure left it."""
    texts = _corpus(n_noise=10)
    topic_of, _ = refine(texts, n_corpus_cases=len(texts))
    assert topic_of, "the three real topics should have been found"
    assert not [cid for cid in topic_of if cid.startswith("noise:")]


def test_a_structural_partition_is_never_cut_across():
    """The dominance gate. This cluster is only half the corpus, so structure
    found a real boundary and topic has no licence to re-cut it — even though
    the same text would happily separate on its own (previous test)."""
    texts = _corpus()
    topic_of, terms = refine(texts, n_corpus_cases=len(texts) * 2)
    assert topic_of == {} and terms == {}


def test_a_homogeneous_cluster_is_left_alone():
    """500 invoices that all say the same words ARE one kind. The pass must
    decline rather than manufacture a split out of incidental differences."""
    texts = {f"inv:{i}": f"Invoice raised, approved and paid for order {i}" for i in range(40)}
    topic_of, terms = refine(texts, n_corpus_cases=len(texts))
    assert topic_of == {} and terms == {}


def test_never_splits_into_a_single_topic():
    """One group is not a partition — it is the cluster with a label on it.

    One vocabulary plus unrelated one-offs: the group forms, and is then
    declined because there is nothing to tell it apart from.
    """
    texts = {f"isda:{i}": f"{ISDA} number {i}" for i in range(15)}
    texts.update({f"noise:{i}": _odd(i) for i in range(10)})
    topic_of, _ = refine(texts, n_corpus_cases=len(texts))
    assert topic_of == {}


def test_too_small_to_evidence_a_split():
    texts = _corpus(n_each=2, n_noise=0)          # 6 cases, well under min_cases
    assert refine(texts, n_corpus_cases=len(texts)) == ({}, {})


def test_disabled_is_a_no_op():
    texts = _corpus()
    off = TopicPolicy(enabled=False)
    assert refine(texts, len(texts), off) == ({}, {})


def test_terms_are_stems_not_prefix_pair_labels():
    """`similar()` labels a prefix match 'blueprint's~blueprint'. That belongs in
    a join's rationale, not in a kind's name."""
    texts = _corpus()
    _, terms = refine(texts, n_corpus_cases=len(texts))
    assert terms
    assert all("~" not in tok for group in terms.values() for tok in group)


# --- end to end, through the real adapter and the real segmenter -------------

def _mail(mid, subject, body, day):
    return (mid, f"Message-ID: <{mid}>\nFrom: analyst@acme.com\nTo: desk@acme.com\n"
                 f"Date: Mon, {day} Oct 2001 09:00:00 -0700\nSubject: {subject}\n\n{body}\n")


@pytest.fixture(scope="module")
def mail_model():
    from induction.adapters import email_mbox
    from induction.pipeline import induce
    messages = []
    for i in range(12):
        messages.append(_mail(f"i{i}", f"ISDA master agreement {i}", f"{ISDA} ref {i}", i % 28 + 1))
        messages.append(_mail(f"g{i}", f"Daily nomination cycle {i}", f"{GAS} cycle {i}", i % 28 + 1))
        messages.append(_mail(f"h{i}", f"Candidate resume {i}", f"{HIRE} number {i}", i % 28 + 1))
    return induce(email_mbox.shape(messages, "acme"), slug="acme")


def test_a_mailbox_no_longer_induces_one_undifferentiated_kind(mail_model):
    assert len(mail_model.kinds) > 1, (
        "a mail corpus segments into one structural cluster; without topic "
        "refinement every thread lands in a single 'they all send mail' kind")


def test_topic_boundaries_are_rendered_as_inference(mail_model):
    """A boundary read off vocabulary is a guess about process, and must read as
    one: never better than `heuristic`, and it must show its own evidence."""
    refined = [k for k in mail_model.kinds if k.features.get("topic_terms")]
    assert refined, "expected at least one topic-refined kind"
    for k in refined:
        assert k.confidence.tier.label == "heuristic"
        assert "shared vocabulary" in k.confidence.rationale
        # the terms are named in the rationale, the name, and the features —
        # a reader can see exactly what to disagree with
        assert "shared vocabulary" in k.rationale
        assert all(term in k.confidence.rationale for term in k.features["topic_terms"])


def test_refinement_repartitions_and_never_loses_a_run(mail_model):
    case_ids = [cid for k in mail_model.kinds for cid in k.case_ids]
    assert len(case_ids) == len(set(case_ids)), "a run must land in exactly one kind"
    assert set(case_ids) == set(mail_model.cases), "every run must land in some kind"


def test_terms_name_a_subject_not_an_identifier():
    """A kind called "713, com, v…@enron.com" tells a reader nothing.

    Mail bodies are full of quoted headers, so identifiers are shared across
    threads constantly. They are fair evidence for the join and useless as a
    name: two threads sharing an address share a person, not a kind of work.
    """
    who = "vince.j.kaminski@enron.com"
    texts = {}
    for i in range(14):
        texts[f"isda:{i}"] = f"{ISDA} number {i}\n----- Forwarded by {who} on 10/13/2001 07:13 AM"
    for i in range(14):
        texts[f"gas:{i}"] = f"{GAS} cycle {i}\n----- Forwarded by {who} on 10/13/2001 07:13 AM"
    _, terms = refine(texts, n_corpus_cases=len(texts))
    assert terms, "the two vocabularies should still separate"
    flat = [t for group in terms.values() for t in group]
    assert not [t for t in flat if "@" in t], flat
    assert not [t for t in flat if t.isdigit()], flat
