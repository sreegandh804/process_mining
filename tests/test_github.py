"""The GitHub Issues/PR source — and the fuzzy join it exists to exercise.

Two things are under test here, and the second is the point.

1. **The adapter.** Issues and PRs expand into their real timelines rather than
   being flattened into one event each, actors and times are read or left
   absent, and cross-references become declared links.

2. **Correlation without a shared key.** Issue #20 and PR #21 fix the same bug
   and say so nowhere: no number, no keyword, no branch, nothing structural. The
   git DAG cannot help — there is no DAG. A foreign key cannot help — there is
   no key. This is the requirement the engine was dodging while it only had git,
   and the reason the fuzzy pass exists.

The test that matters most is the *negative* one: the fuzzy pass must not touch
a pair that determinism already explained, and must stay `heuristic` so a reader
can tell a guess from a join at a glance.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from induction.adapters import changelog, git_history, github_api
from induction.pipeline import induce

from github_fixture import PAYLOAD, SLUG


@pytest.fixture(scope="module")
def gh_model():
    """GitHub alone — a corpus with no DAG and no foreign keys."""
    return induce(github_api.shape(PAYLOAD, SLUG), slug=SLUG)


@pytest.fixture(scope="module")
def combined_model(mini_raw):
    """GitHub + git + changelog for the same repo, through one correlator.

    GitHub is shaped first so that its *real* PR record is present before git's
    reference to that PR is resolved — the reference then resolves to the record
    instead of inventing an inferred stub beside it.
    """
    raw_dir, slug = mini_raw
    shaped = github_api.shape(PAYLOAD, slug)
    shaped.extend(git_history.load(raw_dir, slug))
    shaped.extend(changelog.load(raw_dir, slug))
    return induce(shaped, slug=slug)


def _case_of(model, entity_id):
    return next((c for c in model.cases.values() if entity_id in c.entity_ids), None)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

def test_an_issue_is_an_entity_whose_timeline_yields_many_events(gh_model):
    """Never flattened into a single 'issue event' — the review loop is the process."""
    actions = [e.action for e in gh_model.shaped.events
               if e.entity_id == "issue:acme/widget:20"]
    assert actions[0] == "opened"
    assert {"labeled", "commented", "closed"} <= set(actions)
    assert len(actions) >= 4


def test_review_rounds_survive_as_separate_events(gh_model):
    """A changes-requested round followed by an approval is what makes an
    exception trace read differently from the happy path."""
    reviews = [e for e in gh_model.shaped.events
               if e.entity_id == "pr:21" and e.action == "reviewed"]
    assert [r.attrs.get("review_state") for r in reviews] == ["changes_requested", "approved"]
    assert all(r.timestamp and r.actor for r in reviews)


def test_actors_are_not_silently_merged_across_sources(combined_model):
    """A GitHub login is not provably the same human as a git commit email, so
    the two identity spaces stay separate and visibly so."""
    people = {e.id for e in combined_model.shaped.entities if e.type == "person"}
    assert any(p.startswith("person:gh:") for p in people)
    assert any(p.startswith("person:") and not p.startswith("person:gh:") for p in people)


# ---------------------------------------------------------------------------
# Deterministic correlation
# ---------------------------------------------------------------------------

def test_issue_and_pr_join_deterministically_when_the_pr_says_so(gh_model):
    case = _case_of(gh_model, "issue:acme/widget:10")
    assert case is not None and case.id == "case:pr:11"
    assert case.confidence.tier.label == "joined"
    opened = next(e for e in gh_model.shaped.events
                  if e.entity_id == "issue:acme/widget:10" and e.action == "opened")
    assert opened.case_confidence.tier.label == "joined"


# ---------------------------------------------------------------------------
# Fuzzy correlation — the requirement this source exists to exercise
# ---------------------------------------------------------------------------

def test_issue_and_pr_join_with_no_shared_key_at_all(gh_model):
    """#20 and #21 share no number, keyword, branch or key. Nothing
    deterministic can connect them; the fuzzy pass must, or the engine has no
    answer for a corpus without a DAG."""
    issue_case = _case_of(gh_model, "issue:acme/widget:20")
    assert issue_case is not None, "issue #20 was left unjoined"
    assert issue_case.id == "case:pr:21"

    # It really had no key to join on: neither points at the other, by any
    # method. (PR #21 does link to its own merge commit — that is a different
    # claim, and it is not what put issue #20 in this case.)
    issue = next(e for e in gh_model.shaped.entities if e.id == "issue:acme/widget:20")
    pr = next(e for e in gh_model.shaped.entities if e.id == "pr:21")
    assert "pr:21" not in {l["target"] for l in issue.attrs.get("links", [])}
    assert "issue:acme/widget:20" not in {l["target"] for l in pr.attrs.get("links", [])}


def test_a_fuzzy_join_is_tiered_and_says_what_it_matched_on(gh_model):
    """A guess must be legible as a guess, and overrulable — so it carries its
    score, the tokens it matched, and the proximity signal that supported it."""
    case = gh_model.cases["case:pr:21"]
    assert case.confidence.tier.label == "heuristic"
    why = case.confidence.rationale
    assert "no shared key" in why
    assert "csv" in why and "export" in why      # the tokens it actually matched
    assert "days apart" in why or "same actor" in why

    opened = next(e for e in gh_model.shaped.events
                  if e.entity_id == "issue:acme/widget:20" and e.action == "opened")
    assert opened.case_confidence.tier.label == "heuristic"


def test_fuzzy_never_overrides_a_deterministic_join(gh_model):
    """The fuzzy pass only ever sees what determinism could not explain. If it
    could outrank a real key, every `joined` in the output would be suspect."""
    for cid in ("case:pr:11",):
        case = gh_model.cases[cid]
        assert case.confidence.tier.label == "joined"
        assert "no shared key" not in (case.confidence.rationale or "")

    fuzzy_cases = [c for c in gh_model.cases.values()
                   if "no shared key" in (c.confidence.rationale or "")]
    assert len(fuzzy_cases) == 1, "the fuzzy pass should be surgical, not enthusiastic"


def test_unrelated_items_are_left_alone(gh_model):
    """PR #1 and issue #7 are about different things and stay in their own runs —
    a fuzzy pass that joins everything is worse than none."""
    assert _case_of(gh_model, "pr:1").id == "case:pr:1"
    assert _case_of(gh_model, "issue:acme/widget:7").id == "case:issue:acme/widget:7"


# ---------------------------------------------------------------------------
# Cross-source: the thing a per-source correlator structurally cannot do
# ---------------------------------------------------------------------------

def test_one_case_is_built_from_three_different_sources(combined_model):
    case = combined_model.cases["case:pr:1"]
    source_of = {e.id: e.source.split(":")[0] for e in combined_model.shaped.entities}
    sources = {source_of[e] for e in case.entity_ids if e in source_of}
    assert sources == {"git", "github", "changelog"}, sources


def test_gits_inferred_pr_resolves_to_githubs_real_record(combined_model):
    """Git knows PR #1 only as "(#1)" in a merge subject and would materialise an
    *inferred* stub for it. With GitHub loaded, the same id is a real record, so
    the reference resolves to it — one entity, two sources, no reconciliation."""
    prs = [e for e in combined_model.shaped.entities if e.id == "pr:1"]
    assert len(prs) == 1, "the PR was duplicated instead of unified"
    assert prs[0].source.startswith("github:")
    assert prs[0].confidence.tier.label == "direct"        # read, not inferred
    assert prs[0].attrs["title"] == "Add widget core"
    assert prs[0].attrs.get("known_via") != "reference"


def test_github_issue_record_enriches_the_git_run(combined_model):
    """Git only knew issue #7 existed. GitHub has its timeline, and those events
    land in the same case the git commits are in."""
    case = combined_model.cases["case:pr:2"]
    assert "issue:acme/widget:7" in case.entity_ids
    assert "commit:s1" in case.entity_ids
    gh_events = [e for e in combined_model.shaped.events
                 if e.entity_id == "issue:acme/widget:7"]
    assert gh_events and all(e.case_id == "case:pr:2" for e in gh_events)


def test_adding_github_left_the_git_corpus_intact(combined_model, mini_model):
    """Adding a source must not silently rewrite what the others induced."""
    assert {o.record_id for o in mini_model.orphans} <= {o.record_id for o in combined_model.orphans}
    for cid in ("case:pr:1", "case:pr:2", "case:pr:3", "case:pr:4"):
        before, after = mini_model.cases[cid], combined_model.cases[cid]
        assert set(before.entity_ids) <= set(after.entity_ids)
        assert before.confidence.tier == after.confidence.tier


# ---------------------------------------------------------------------------
# The structural claim behind all of it
# ---------------------------------------------------------------------------

def test_no_source_specific_correlator_exists():
    """The whole point: GitHub was added as an adapter and nothing else.

    If someone later adds `correlate_github.py` (or a QuickBooks or mailbox
    twin), this fails — and it should, because the moment correlation forks per
    source, cross-source joining becomes impossible again.
    """
    steps = Path(__file__).resolve().parent.parent / "induction" / "steps"
    correlators = sorted(p.name for p in steps.glob("correlate*.py"))
    assert correlators == ["correlate.py"], correlators

    # Prose may name examples; *code* may not. Docstrings are stripped and what
    # is left — every identifier and every runtime string literal — must contain
    # no source's vocabulary. That is the difference between explaining the
    # design and branching on it.
    tree = ast.parse((steps / "correlate.py").read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                docstrings.add(doc)

    vocabulary = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in docstrings:
                vocabulary.append(node.value)
        elif isinstance(node, ast.Name):
            vocabulary.append(node.id)
        elif isinstance(node, ast.Attribute):
            vocabulary.append(node.attr)
    blob = " ".join(vocabulary).lower()

    for word in ("github", "commit", "invoice", "changelog", "mailbox",
                 "quickbooks", "issue", "email"):
        assert word not in blob, f"the correlator learned the word {word!r}"
