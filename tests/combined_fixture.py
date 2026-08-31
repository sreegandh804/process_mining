"""A realistic two-source corpus: a team's GitHub repo + their mailbox.

The point of this fixture is everything the brief says a real corpus is:

  - **Nothing announces the relationship.** Not one email cites an issue or PR
    number. People write "the export is spinning again", not "re: #21". The only
    way the mail thread and the pull request can be known to be the same work is
    that they are *about* the same work — shared words, the same people, the same
    week. That is the join the engine has to earn, and the one a foreign key or a
    git DAG would have handed you for free.

  - **A process is not one system's record.** The SSO fix lives as an email
    thread (the customer symptom), a GitHub issue (the triaged bug), and a pull
    request (the change) — three artefacts, three shapes, one activity.

  - **The systems are blind to most of it.** Thread C is a note about a *phone
    call* and a *Friday deploy* — an off-system step and a downstream artefact
    neither system holds. The engine must not invent them, and must not pretend
    the process is complete without them.

  - **Look-alikes that are not processes, and threads that connect to nothing.**
    A nightly CI notice recurs (looks like a step, isn't); a lunch thread joins
    nothing (an orphan, not padded into a case).

Authored in code so it is readable and diffable. `run_combined.py --demo` writes
the same corpus out as the real files an adapter loads (a GitHub JSON and an
.mbox), so "how would I test my own data" and "does the fixture hold up" are the
same path.
"""

from __future__ import annotations

SLUG = "northwind/portal"
MAIL_SLUG = "northwind-team"


# --- GitHub side ------------------------------------------------------------

def _user(login):
    return {"login": login, "id": abs(hash(login)) % 100000}


def _issue(number, title, user, created, closed, body, labels, timeline):
    return {
        "number": number, "title": title, "user": _user(user), "body": body,
        "state": "closed", "created_at": created, "closed_at": closed,
        "updated_at": closed, "labels": [{"name": n} for n in labels],
        "html_url": f"https://github.com/{SLUG}/issues/{number}",
        "comments": sum(1 for t in timeline if t.get("event") == "commented"),
        "timeline": list(timeline),
    }


def _pr(number, title, user, created, merged, sha, body, labels, timeline):
    it = _issue(number, title, user, created, merged, body, labels, timeline)
    it["html_url"] = f"https://github.com/{SLUG}/pull/{number}"
    it["merged_at"] = merged
    it["merged_by"] = _user("maria")
    it["merge_commit_sha"] = sha
    it["draft"] = False
    return it


GH_PAYLOAD = {
    "slug": SLUG,
    "issues": [
        _issue(14, "SSO login fails right after token refresh", "maria",
               "2024-06-03T09:00:00Z", "2024-06-07T16:00:00Z",
               body="Enterprise SSO users are signed out moments after login, "
                    "as soon as the access token refreshes.",
               labels=("bug", "auth"),
               timeline=[
                   {"event": "labeled", "actor": _user("maria"),
                    "created_at": "2024-06-03T09:30:00Z", "label": {"name": "bug"}},
                   {"event": "commented", "actor": _user("sam"),
                    "created_at": "2024-06-04T10:00:00Z",
                    "body": "Reproduced on the enterprise tenant."},
                   {"event": "closed", "actor": _user("maria"),
                    "created_at": "2024-06-07T16:00:00Z"},
               ]),
        # #18 and its fix PR #21 share no number — the intra-source fuzzy case.
        _issue(18, "CSV export times out on large accounts", "dev",
               "2024-06-10T08:30:00Z", "2024-06-14T17:00:00Z",
               body="Exporting an account with tens of thousands of rows spins "
                    "and eventually times out. No file is ever downloaded.",
               labels=("bug", "performance"),
               timeline=[
                   {"event": "commented", "actor": _user("priya"),
                    "created_at": "2024-06-11T12:00:00Z",
                    "body": "It buffers the whole export in memory before sending."},
                   {"event": "closed", "actor": _user("priya"),
                    "created_at": "2024-06-14T17:00:00Z"},
               ]),
    ],
    "pulls": [
        # Says "fixes #14" — a deterministic intra-source join, so the GitHub side
        # is one clean case and the cross-source join to the mail is isolated.
        _pr(15, "Refresh the SSO token before the expiry check", "sam",
            "2024-06-05T09:00:00Z", "2024-06-07T16:00:00Z", sha="a15",
            body="Reorder so the token refresh runs before we validate expiry.\n\nfixes #14",
            labels=("bug", "auth"),
            timeline=[
                {"event": "review_requested", "actor": _user("sam"),
                 "created_at": "2024-06-05T09:10:00Z"},
                {"event": "reviewed", "user": _user("maria"), "state": "approved",
                 "submitted_at": "2024-06-06T11:00:00Z", "body": "Good catch."},
                {"event": "merged", "actor": _user("maria"),
                 "created_at": "2024-06-07T16:00:00Z"},
            ]),
        # Fixes #18 but never says so — nothing deterministic connects them.
        _pr(21, "Stream the CSV export instead of buffering", "priya",
            "2024-06-12T09:00:00Z", "2024-06-14T17:00:00Z", sha="a21",
            body="Write each row to the response as we read it, instead of "
                 "buffering the whole export in memory first.",
            labels=("performance",),
            timeline=[
                {"event": "reviewed", "user": _user("dev"), "state": "approved",
                 "submitted_at": "2024-06-13T10:00:00Z", "body": "Much better."},
                {"event": "merged", "actor": _user("priya"),
                 "created_at": "2024-06-14T17:00:00Z"},
            ]),
    ],
}


# --- Mail side (no issue/PR numbers anywhere) -------------------------------

def demo_judge():
    """A transparent, OFFLINE stand-in for the LLM judge, for `run_combined.py
    --demo` — so the model tier's *output shape* can be seen without a key or a
    network. A rule fires when a shared concept term is on both sides; it judges
    text, exactly as the real model does. Live, swap in `AnthropicJudge` and the
    engine holds none of these words. This is a simulation of the verdicts, not a
    claim the real model returns them."""
    from induction.semantic import ScriptedJudge
    return ScriptedJudge([
        (["sso", "token", "logged out", "signed out", "log in", "login", "expiry"],
         "both describe the SSO login failure after the token refresh"),
        (["csv", "export", "buffer", "stream", "download", "spins", "times out"],
         "both describe the CSV export timeout and its streaming fix"),
    ])


def demo_activity_mapper():
    """OFFLINE stand-in for the activity-mapping model, for `--demo`. Returns the
    grouping a real model produces — artefact verbs across email/issue/PR folded
    into the activities the process is made of (an issue 'opened' and an email
    'sent' are both 'Raised'; a PR 'merged' is 'Shipped'). Live, swap in
    `AnthropicActivityMapper`; the engine holds none of this vocabulary."""
    from induction.abstraction import ScriptedActivityMapper
    return ScriptedActivityMapper({
        "email/sent": "Raised", "email/replied": "Discussed",
        "issue/opened": "Raised", "issue/labeled": "Triaged",
        "issue/commented": "Discussed", "issue/closed": "Resolved",
        "pr/opened": "Fix proposed", "pr/review_requested": "Review requested",
        "pr/reviewed": "Reviewed", "pr/merged": "Shipped", "pr/closed": "Resolved",
    })


def _m(mid, frm, subj, date, body, irt=None):
    h = (f"Message-ID: <{mid}>\nFrom: {frm}\nTo: team@northwind.com\n"
         f"Date: {date}\nSubject: {subj}\n")
    if irt:
        h += f"In-Reply-To: <{irt}>\n"
    return (mid, h + "\n" + body)


MAIL = [
    # Thread A — the SSO work, in the customer's words. Shares login/SSO/token/
    # refresh/enterprise with issue #14 and PR #15; maria and sam are on both sides.
    _m("a1", "maria@northwind.com", "Enterprise customer keeps getting logged out",
       "Mon, 03 Jun 2024 08:40:00 -0700",
       "The SSO users at BigCo are signed out moments after they log in. I think "
       "the access token refresh is the trigger."),
    _m("a2", "sam@northwind.com", "Re: Enterprise customer keeps getting logged out",
       "Tue, 04 Jun 2024 09:30:00 -0700",
       "Reproduced on the enterprise tenant — the token refresh runs after the "
       "expiry check, so it logs them straight out. Fixing the order now.", irt="a1"),

    # Thread B — the CSV work. Shares export/CSV/large/account/buffer/stream with
    # issue #18 and PR #21; dev and priya are on both sides.
    _m("b1", "dev@northwind.com", "Big accounts can't download their export",
       "Mon, 10 Jun 2024 08:15:00 -0700",
       "Support ticket: exporting a large account just spins forever and the CSV "
       "never downloads."),
    _m("b2", "priya@northwind.com", "Re: Big accounts can't download their export",
       "Tue, 11 Jun 2024 13:20:00 -0700",
       "It's buffering the entire export in memory before it sends anything. I'll "
       "stream the rows out instead.", irt="b1"),

    # Thread C — the off-system step. A phone call and a Friday deploy, neither of
    # which either system records. It is about the SSO fix, so it may attach there;
    # the call and the deploy stay invisible, and the engine must not invent them.
    _m("c1", "maria@northwind.com", "Update from my call with BigCo",
       "Wed, 05 Jun 2024 15:00:00 -0700",
       "Just off the phone with BigCo — they're fine waiting for Friday's deploy "
       "of the SSO fix, so no need to hotfix tonight."),

    # Thread D — unrelated. Must join nothing (an orphan, not a one-line 'process').
    _m("d1", "sam@northwind.com", "Lunch Friday?", "Thu, 06 Jun 2024 11:00:00 -0700",
       "Anyone up for tacos on Friday?"),
    _m("d2", "dev@northwind.com", "Re: Lunch Friday?", "Thu, 06 Jun 2024 11:20:00 -0700",
       "I'm in.", irt="d1"),

    # Recurring automated notice — looks like a step, isn't.
    _m("n1", "noreply@ci.northwind.com", "Nightly build 4823 passed",
       "Mon, 03 Jun 2024 02:00:00 -0700", "All checks green."),
    _m("n2", "noreply@ci.northwind.com", "Nightly build 4824 passed",
       "Tue, 04 Jun 2024 02:00:00 -0700", "All checks green."),
]
