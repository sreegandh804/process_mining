"""Golden fixture — a tiny, hand-verified corpus with known-correct answers.

Twelve records that deliberately contain every awkward shape the engine is
graded on, so the assertions in the test-suite are exact rather than
approximate:

  case pr:1  A multi-commit PR merged via a merge commit. One of its commits is
             co-authored (same-activity-different-people). Tagged v1.0.0.
  case pr:2  A squash-merged fix (author != committer, 5-day handoff) that also
             closes issue #7, then gets reverted (an exception trace).
  case pr:3  A dependabot bump touching only requirements.txt (a look-alike
             non-process — should be rejected).
  orphans    The root commit and a stray README tweak reference nothing.
  thin       A changelog whose bullets cross-reference the PRs, plus one issue
             (#999) not in git — a thin, order-unknown observation.

The corpus is authored in code (readable, diffable) and written to a tmp dir so
tests stay self-contained and offline. If you change it, update the assertions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SLUG = "acme/widget"
KEY = SLUG.replace("/", "__")


def _c(sha, subject, author, committer, adate, cdate, parents, files, body=""):
    def person(x):
        name, email = x
        return {"name": name, "email": email}
    return {
        "sha": sha, "short_sha": sha[:8], "parents": parents,
        "is_merge": len(parents) > 1,
        "author": person(author), "committer": person(committer),
        "author_date": adate, "committer_date": cdate,
        "refs": (["HEAD -> main"] if sha == "d2" else []),
        "subject": subject, "body": body, "files": files,
    }


ALICE = ("Alice Maintainer", "alice@acme.io")
BOB = ("Bob Contributor", "bob@acme.io")
DAVE = ("Dave Fixer", "dave@acme.io")
BOT = ("dependabot[bot]", "49699333+dependabot[bot]@users.noreply.github.com")

# newest-first, the way `git log` emits and the adapter expects
COMMITS = [
    _c("d2", "Bump other from 2.0 to 2.1 (#4)", BOT, BOT,
       "2024-02-02T09:00:00+00:00", "2024-02-02T09:00:00+00:00", ["d1"], ["requirements.txt"]),
    _c("d1", "Bump lib from 1.0 to 1.1 (#3)", BOT, BOT,
       "2024-02-01T09:00:00+00:00", "2024-02-01T09:00:00+00:00", ["o1"], ["requirements.txt"]),
    _c("o1", "tweak readme wording", ALICE, ALICE,
       "2024-01-21T09:00:00+00:00", "2024-01-21T09:00:00+00:00", ["r1"], ["README.md"]),
    _c("r1", 'Revert "fix widget bug (#2)"', ALICE, ALICE,
       "2024-01-20T09:00:00+00:00", "2024-01-20T09:00:00+00:00", ["s1"], ["src/widget.py"]),
    _c("s1", "fix widget bug (#2)", DAVE, ALICE,
       "2024-01-10T09:00:00+00:00", "2024-01-15T09:00:00+00:00", ["m1"], ["src/widget.py"],
       body="Corrects an off-by-one.\n\nfixes #7"),
    _c("m1", "Merge pull request #1 from bob/widget", ALICE, ALICE,
       "2024-01-03T08:00:00+00:00", "2024-01-03T08:00:00+00:00", ["c1", "f2"], []),
    _c("f2", "add tests for widget", BOB, BOB,
       "2024-01-02T11:00:00+00:00", "2024-01-02T11:00:00+00:00", ["f1"], ["tests/test_widget.py"],
       body="Cover the core.\n\nCo-authored-by: Carol Reviewer <carol@acme.io>"),
    _c("f1", "add widget core", BOB, BOB,
       "2024-01-02T09:00:00+00:00", "2024-01-02T09:00:00+00:00", ["c1"], ["src/widget.py"]),
    _c("c1", "init project", ALICE, ALICE,
       "2024-01-01T10:00:00+00:00", "2024-01-01T10:00:00+00:00", [], ["README.md"]),
]

TAGS = [{"name": "1.0.0", "commit": "m1", "date": "2024-01-03T09:00:00+00:00"}]

CHANGES = """\
Version 1.1.0
-------------

Unreleased

-   Fix widget bug. :pr:`2`
-   Bump lib to 1.1. :pr:`3`

Version 1.0.0
-------------

Released 2024-01-03

-   Add widget core. :pr:`1`
-   A change that predates the git window. :issue:`999`
"""


@pytest.fixture(scope="session")
def mini_raw(tmp_path_factory) -> tuple[str, str]:
    """Write the fixture to a tmp raw dir; return (raw_dir, slug)."""
    raw = tmp_path_factory.mktemp("mini_raw")
    (raw / f"{KEY}.commits.jsonl").write_text("\n".join(json.dumps(c) for c in COMMITS))
    (raw / f"{KEY}.tags.json").write_text(json.dumps(TAGS))
    (raw / f"{KEY}.CHANGES.rst").write_text(CHANGES)
    (raw / f"{KEY}.manifest.json").write_text(json.dumps(
        {"slug": SLUG, "head": "d2", "n_commits": len(COMMITS)}))
    return str(raw), SLUG


@pytest.fixture(scope="session")
def mini_model(mini_raw):
    """The DEFAULT model: the generic, source-agnostic profile (unnamed kinds
    and activities). This is what the product does on data it knows nothing about."""
    from induction.pipeline import run_pipeline
    raw_dir, slug = mini_raw
    return run_pipeline(slug, raw_dir, with_thin=True)


@pytest.fixture(scope="session")
def mini_model_git(mini_raw):
    """The same data with the opt-in git vocabulary — names, no structural change."""
    from induction.pipeline import run_pipeline
    from induction.profiles import GIT_PROFILE
    raw_dir, slug = mini_raw
    return run_pipeline(slug, raw_dir, with_thin=True, profile=GIT_PROFILE)


@pytest.fixture(scope="session")
def mini_shaped(mini_raw):
    """The shaped substrate + correlation, for lower-level assertions."""
    from induction.adapters import git_history, changelog
    from induction.steps.correlate import correlate
    from induction.steps.order import order, order_observations
    raw_dir, slug = mini_raw
    shaped = git_history.load(raw_dir, slug)
    shaped.extend(changelog.load(raw_dir, slug))
    corr = correlate(shaped)
    order(shaped, corr)
    order_observations(shaped.observations, corr)
    return shaped, corr, slug
