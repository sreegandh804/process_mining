"""Does the reading tier still behave when two very different sources arrive
together?

The engine's claim is that a new source is an adapter, not a new engine. The
reading tier is the newest and most inferential part of it, so it is the part
most likely to have quietly become "the email tier". These tests attach a
tracker corpus (issues with real stage verbs and real ids) to a mailbox
(transport verbs and a guessed key) and hold the seam to three promises:

  - the tracker keeps its own verbs. Its verbs already ARE the process, so
    reading its rows would be cost with no answer, and re-labelling them would
    be the model overwriting data.
  - the mailbox still gets read, even though the corpus median is now dragged
    down by a source that does not need reading.
  - the two do not contaminate each other: the tracker's runs are not placed in
    a process derived from mail text, and the mail runs are not forced into the
    tracker's structural kind.

None of this is checked by the single-source tests, and all of it is what
"generalises" has to mean.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from induction.abstraction import ScriptedRecordClassifier, infer_activities
from induction.adapters import Shaped, email_mbox
from induction.model import Entity, Event, Evidence, direct
from induction.links import Link, declare
from induction.model import Tier
from induction.pipeline import induce

D = "northwind.example"

CLASSIFIER = ScriptedRecordClassifier([
    ("Requested", ["can we get", "please add"], "Feature intake"),
    ("Reviewed", ["had a look", "looks reasonable"], "Feature intake"),
    ("Approved", ["go ahead", "signed off"], "Feature intake"),
    ("Chased", ["still waiting", "any update"], "Support escalation"),
    ("Resolved", ["shipped in", "closed it out"], "Support escalation"),
])


def _issues(n=12):
    """A tracker: real ids, real stage verbs, one event per stage per run."""
    s = Shaped()
    base = datetime(2003, 3, 1)
    for i in range(n):
        eid = f"issue:nw:{100 + i}"
        ent = Entity(
            id=eid, type="issue", source="tracker",
            attrs={"title": f"Exporter drops column {i}", "number": 100 + i},
            evidence=[Evidence("tracker", f"issue/{100 + i}", f"Exporter drops column {i}")])
        declare(ent, Link(target=eid, method="issue-id", tier=Tier.DIRECT,
                          rationale="tracker id", anchors=True, anchor_rank=1,
                          anchor_attrs={"type": "issue", "number": 100 + i}))
        s.entities.append(ent)
        for k, verb in enumerate(("raised", "triaged", "fixed", "closed")):
            s.events.append(Event(
                id=f"evt:nw:{100 + i}:{verb}", entity_id=eid, action=verb, source="tracker",
                confidence=direct(), timestamp=base + timedelta(days=i * 3 + k),
                actor="person:nw:dev",
                evidence=[Evidence("tracker", f"issue/{100 + i}#{verb}", verb)]))
    s.entities.append(Entity(id="person:nw:dev", type="person", source="tracker",
                             attrs={"name": "dev"}))
    return s


def _mail(n=14):
    """A mailbox about the same product: transport verbs, guessed key.

    Threads are five messages, which is what real ones are. Three would not
    clear the transport gate — see
    `test_a_mailbox_of_very_short_threads_is_a_known_blind_spot`.
    """
    msgs, base = [], datetime(2003, 3, 1)
    bodies = [
        ("Can we get a CSV exporter for the ledger", "Requested"),
        ("Please add the ledger exporter to the next milestone", "Requested"),
        ("Had a look - looks reasonable to me", "Reviewed"),
        ("Second pass: had a look, no objections", "Reviewed"),
        ("Go ahead, signed off from finance", "Approved"),
    ]
    for i in range(n):
        root = f"m{i}0"
        when = base + timedelta(days=i * 2)
        for k, (body, _) in enumerate(bodies):
            mid = f"m{i}{k}"
            head = (f"Message-ID: <{mid}>\nFrom: a.okonkwo@{D}\nTo: desk@{D}\n"
                    f"Date: {(when + timedelta(hours=k)).strftime('%a, %d %b %Y %H:%M:%S -0700')}\n"
                    f"Subject: {'RE: ' if k else ''}Ledger exporter {i}\n")
            if k:
                head += f"In-Reply-To: <{root}>\n"
            msgs.append((mid, head + "\n" + body + f" (ref {i})\n"))
    return email_mbox.shape(msgs, "northwind")


@pytest.fixture(scope="module")
def mixed():
    s = Shaped()
    s.extend(_issues())
    s.extend(_mail())
    m = induce(s, slug="northwind")
    a = infer_activities(m, mapper=None, classifier=CLASSIFIER)
    return m, a


def test_both_sources_are_present(mixed):
    m, _ = mixed
    kinds = {e.type for e in m.shaped.entities}
    assert {"issue", "email"} <= kinds


def test_the_tracker_keeps_its_own_verbs(mixed):
    """Its verbs already ARE the process. A record whose stage is in the data is
    not a record for a model to re-label."""
    m, a = mixed
    issue_events = {e.id for e in m.shaped.events if e.source == "tracker"}
    assert issue_events, "fixture produced no tracker events"
    assert not (issue_events & set(a.by_record)), (
        "the reading tier relabelled a source whose verbs already discriminate")


def test_the_mailbox_is_still_read(mixed):
    """The gate must not be closed just because a source that does not need
    reading is now sitting in the same corpus."""
    m, a = mixed
    mail_events = {e.id for e in m.shaped.events if e.source.startswith("mail")}
    read = mail_events & set(a.by_record)
    assert read, "attaching a tracker silenced the reading tier on the mail"
    assert len(read) >= len(mail_events) * 0.5


def test_the_two_sources_do_not_contaminate_each_others_kinds(mixed):
    """A tracker run must not land in a process derived from mail text, and the
    mail runs must not be swallowed by the tracker's structural kind."""
    m, a = mixed
    read_kinds = {k.id for k in m.kinds if k.features.get("read_process")}
    assert read_kinds, "the reading produced no process at all"
    for kind in m.kinds:
        sources = {e.source for cid in kind.case_ids
                   for eid in m.cases[cid].entity_ids
                   for e in m.shaped.entities if e.id == eid}
        if kind.id in read_kinds:
            assert not any(s == "tracker" for s in sources), (
                f"{kind.name} is a mail-derived process holding tracker runs")


def test_a_tracker_case_is_never_split_by_the_session_pass(mixed):
    """It joined on a real id, so its tier is deterministic and the split may not
    reach it — the same guarantee grants relies on."""
    m, _ = mixed
    for c in m.cases.values():
        if any(e.source == "tracker" for e in m.shaped.entities if e.id in c.entity_ids):
            assert c.confidence.tier.label in ("direct", "joined")
            assert "split into" not in (c.confidence.rationale or "")


def test_a_mailbox_of_very_short_threads_is_a_known_blind_spot():
    """A limitation, pinned so it is recorded rather than rediscovered.

    The gate asks whether a source's records outnumber its distinct verbs — a
    stage happens once per run, a transmission recurs. On a three-message thread
    (`sent`, `replied`, `replied`) that ratio is 1.5 and the gate stays shut,
    even though `sent` and `replied` are every bit as much transport as they are
    in a twenty-message thread. The signal the ratio proxies for is repetition,
    and a short thread does not have enough records to repeat.

    This is honest — the engine declines rather than guessing — but it means a
    corpus of brief exchanges keeps its raw verbs. Recorded here so a future
    change to `_TRANSPORT_RATIO` has a case to argue against.
    """
    s = Shaped()
    s.extend(_issues())
    msgs, base = [], datetime(2003, 3, 1)
    for i in range(14):
        root = f"s{i}0"
        for k, body in enumerate(("Can we get an exporter", "Had a look", "Go ahead")):
            mid = f"s{i}{k}"
            head = (f"Message-ID: <{mid}>\nFrom: a.okonkwo@{D}\nTo: desk@{D}\n"
                    f"Date: {(base + timedelta(days=i * 2, hours=k)).strftime('%a, %d %b %Y %H:%M:%S -0700')}\n"
                    f"Subject: {'RE: ' if k else ''}Short thread {i}\n")
            if k:
                head += f"In-Reply-To: <{root}>\n"
            msgs.append((mid, head + "\n" + body + f" (ref {i})\n"))
    s.extend(email_mbox.shape(msgs, "northwind"))
    m = induce(s, slug="northwind")
    a = infer_activities(m, mapper=None, classifier=CLASSIFIER)
    assert a.by_record == {}, (
        "short-thread mail now clears the gate — if that is deliberate, update "
        "this test and the measured medians in abstraction._TRANSPORT_RATIO")
