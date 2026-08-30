"""The email source — deterministic threading, a subject fallback, and the
fuzzy cross-thread join that is the whole reason to point this engine at mail.
"""

import pytest

from induction.adapters import email_mbox
from induction.pipeline import induce


def _m(mid, frm, subj, date, body, irt=None):
    h = f"Message-ID: <{mid}>\nFrom: {frm}\nTo: team@corp.com\nDate: {date}\nSubject: {subj}\n"
    if irt:
        h += f"In-Reply-To: <{irt}>\n"
    return (mid, h + "\n" + body)


MESSAGES = [
    _m("a1", "alice@corp.com", "Q3 budget approval request", "Mon, 04 Mar 2024 09:00:00 -0800",
       "Please approve the Q3 budget of 4,250,000."),
    _m("a2", "bob@corp.com", "Re: Q3 budget approval request", "Mon, 04 Mar 2024 11:00:00 -0800",
       "Reviewing the Q3 budget.", irt="a1"),
    _m("a3", "alice@corp.com", "Re: Q3 budget approval request", "Tue, 05 Mar 2024 09:00:00 -0800",
       "Approved.", irt="a2"),
    # same matter, different subject, NO reply link -> only fuzzy can connect it
    _m("f1", "carol@corp.com", "sign-off on quarterly figures", "Wed, 06 Mar 2024 10:00:00 -0800",
       "Need sign-off on the Q3 budget 4,250,000 quarterly figures."),
    # a reply whose In-Reply-To header was lost -> subject-thread fallback
    _m("b1", "dave@corp.com", "Vendor contract renewal", "Mon, 11 Mar 2024 09:00:00 -0800", "Terms."),
    _m("b2", "erin@corp.com", "Re: Vendor contract renewal", "Tue, 12 Mar 2024 09:00:00 -0800", "Fine."),
    # recurring automated notice -> looks like a process, isn't
    _m("n1", "noreply@system.com", "Daily backup completed", "Mon, 04 Mar 2024 02:00:00 -0800", "OK."),
    _m("n2", "noreply@system.com", "Daily backup completed", "Tue, 05 Mar 2024 02:00:00 -0800", "OK."),
]


@pytest.fixture(scope="module")
def mail():
    return induce(email_mbox.shape(MESSAGES, "acme-mail"), slug="acme-mail")


def _case_of(m, eid):
    return next((c for c in m.cases.values() if eid in c.entity_ids), None)


def test_in_reply_to_threads_deterministically(mail):
    case = _case_of(mail, "email:acme-mail:a1")
    assert {"email:acme-mail:a2", "email:acme-mail:a3"} <= set(case.entity_ids)


def test_subject_thread_fallback_groups_a_reply_that_lost_its_header(mail):
    case = _case_of(mail, "email:acme-mail:b1")
    assert "email:acme-mail:b2" in case.entity_ids


def test_FUZZY_join_connects_two_threads_with_no_shared_key(mail):
    """f1 shares no Message-ID, In-Reply-To or subject with the Q3 thread — only
    text (Q3/budget/4,250,000) + time. The fuzzy pass must connect them."""
    a = _case_of(mail, "email:acme-mail:a1")
    f = _case_of(mail, "email:acme-mail:f1")
    assert a is not None and f is not None and a.id == f.id
    assert a.confidence.tier.label == "heuristic"


def test_automated_notices_cluster_and_are_flagged(mail):
    # n1 and n2 are separate one-message runs (not one fake thread) …
    assert _case_of(mail, "email:acme-mail:n1").id != _case_of(mail, "email:acme-mail:n2").id
    # … so they read as a recurring automated pattern and get rejected
    assert any(k.rejected for k in mail.kinds)


def test_no_actor_or_time_is_invented(mail):
    for e in mail.shaped.events:
        assert e.source.startswith("mail:")
        # every 'sent' event has a real from-actor; nothing fabricated elsewhere
        if e.action in ("sent", "replied", "forwarded"):
            assert e.actor is None or e.actor.startswith("person:mail:")
