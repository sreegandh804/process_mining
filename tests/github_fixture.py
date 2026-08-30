"""A GitHub Issues/PR corpus in the real API's shape, for `acme/widget`.

Deliberately the *same* repository as the git golden fixture in `conftest.py`,
so the two sources describe one process and correlation has to reconcile them:

  PR #1     The real record for the pull request git knows only as a reference
            in the "Merge pull request #1" subject. Its `merge_commit_sha` is
            the git fixture's `m1`.
  issue #7  Likewise the real record behind git's "fixes #7" reference.
  #10/#11   An issue and the PR that closes it, cited explicitly — a
            deterministic join, GitHub-only.
  #20/#21   An issue and the PR that fixes it that **share no number at all**.
            Nothing deterministic can connect these. They are the reason the
            fuzzy pass exists, and the case the git DAG lets you dodge.

Authored in code (readable, diffable) like the git fixture. Objects carry the
fields the real REST API returns; `ingest_github.py` produces the same shape.
"""

from __future__ import annotations

SLUG = "acme/widget"


def _user(login):
    return {"login": login, "id": abs(hash(login)) % 100000}


def _issue(number, title, user, created, closed=None, state="closed", body="",
           labels=(), timeline=(), **extra):
    return {
        "number": number, "title": title, "user": _user(user), "body": body,
        "state": state, "created_at": created, "closed_at": closed,
        "updated_at": closed or created,
        "labels": [{"name": n} for n in labels],
        "html_url": f"https://github.com/{SLUG}/issues/{number}",
        "comments": len([t for t in timeline if t.get("event") == "commented"]),
        "timeline": list(timeline), **extra,
    }


def _pr(number, title, user, created, merged=None, closed=None, sha=None,
        body="", labels=(), timeline=(), **extra):
    item = _issue(number, title, user, created, closed=closed or merged,
                  state="closed" if (merged or closed) else "open",
                  body=body, labels=labels, timeline=timeline, **extra)
    item["html_url"] = f"https://github.com/{SLUG}/pull/{number}"
    item["merged_at"] = merged
    item["merged_by"] = _user(user) if merged else None
    item["merge_commit_sha"] = sha
    item["draft"] = False
    return item


PAYLOAD = {
    "slug": SLUG,
    "issues": [
        # The real record behind git's inferred `issue:acme/widget:7`.
        _issue(7, "Widget miscounts items by one", "carol",
               "2024-01-05T09:00:00Z", closed="2024-01-15T09:10:00Z",
               body="Counting is off by one when the list is empty.",
               labels=("bug",),
               timeline=[
                   {"event": "labeled", "actor": _user("alice"),
                    "created_at": "2024-01-05T10:00:00Z", "label": {"name": "bug"}},
                   {"event": "commented", "actor": _user("dave"),
                    "created_at": "2024-01-08T11:00:00Z",
                    "body": "Reproduced on an empty list."},
                   {"event": "cross-referenced", "actor": _user("dave"),
                    "created_at": "2024-01-10T09:05:00Z",
                    "source": {"issue": {"number": 2, "pull_request": {}}}},
                   {"event": "closed", "actor": _user("alice"),
                    "created_at": "2024-01-15T09:10:00Z"},
               ]),
        # Deterministic pair: PR #11 says "closes #10".
        _issue(10, "Docs page 404s after the rename", "erin",
               "2024-03-01T09:00:00Z", closed="2024-03-06T16:00:00Z",
               body="The guide link is dead since the section was renamed.",
               labels=("docs",),
               timeline=[
                   {"event": "commented", "actor": _user("alice"),
                    "created_at": "2024-03-02T09:00:00Z", "body": "Confirmed."},
                   {"event": "closed", "actor": _user("alice"),
                    "created_at": "2024-03-06T16:00:00Z"},
               ]),
        # NO shared number with the PR that fixes it. This is the fuzzy case.
        _issue(20, "CSV export drops the last row", "frank",
               "2024-04-02T08:30:00Z", closed="2024-04-11T17:00:00Z",
               body="Exporting 100 rows writes 99. Reproduced on the reports screen.",
               labels=("bug", "export"),
               timeline=[
                   {"event": "labeled", "actor": _user("alice"),
                    "created_at": "2024-04-02T09:00:00Z", "label": {"name": "bug"}},
                   {"event": "commented", "actor": _user("grace"),
                    "created_at": "2024-04-04T12:00:00Z",
                    "body": "Same on the CSV export of invoices."},
                   {"event": "closed", "actor": _user("alice"),
                    "created_at": "2024-04-11T17:00:00Z"},
               ]),
    ],
    "pulls": [
        # The real record behind git's inferred `pr:1`, merging git's commit m1.
        _pr(1, "Add widget core", "bob", "2024-01-02T08:00:00Z",
            merged="2024-01-03T08:00:00Z", sha="m1",
            body="First cut of the widget.",
            timeline=[
                {"event": "review_requested", "actor": _user("bob"),
                 "created_at": "2024-01-02T08:30:00Z"},
                {"event": "reviewed", "user": _user("alice"), "state": "approved",
                 "submitted_at": "2024-01-02T15:00:00Z", "body": "LGTM"},
                {"event": "merged", "actor": _user("alice"),
                 "created_at": "2024-01-03T08:00:00Z"},
            ]),
        _pr(11, "Fix the dead guide link", "alice", "2024-03-04T10:00:00Z",
            merged="2024-03-06T16:00:00Z", sha="p11",
            body="Repoints the guide at the renamed section.\n\ncloses #10",
            labels=("docs",),
            timeline=[
                {"event": "reviewed", "user": _user("erin"), "state": "approved",
                 "submitted_at": "2024-03-05T09:00:00Z", "body": "Thanks!"},
                {"event": "merged", "actor": _user("alice"),
                 "created_at": "2024-03-06T16:00:00Z"},
            ]),
        # Fixes issue #20 but never says so — no number, no keyword, nothing.
        _pr(21, "Fix CSV export dropping the final row", "grace",
            "2024-04-05T09:00:00Z", merged="2024-04-11T17:00:00Z", sha="p21",
            body="Off-by-one in the export writer's row loop.",
            labels=("bug", "export"),
            timeline=[
                {"event": "review_requested", "actor": _user("grace"),
                 "created_at": "2024-04-05T09:10:00Z"},
                {"event": "reviewed", "user": _user("frank"), "state": "changes_requested",
                 "submitted_at": "2024-04-07T11:00:00Z", "body": "Add a test for 0 rows."},
                {"event": "reviewed", "user": _user("frank"), "state": "approved",
                 "submitted_at": "2024-04-10T10:00:00Z", "body": "Nice."},
                {"event": "merged", "actor": _user("alice"),
                 "created_at": "2024-04-11T17:00:00Z"},
            ]),
    ],
}
