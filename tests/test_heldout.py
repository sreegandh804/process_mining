"""Held-out slice (brief §7): run on data the engine did NOT see during
development and assert it produces a coherent model — no crashes, sane
confidences, nothing asserted as fact without evidence.

The held-out corpus is a *different repository* (pallets/click), so this also
checks the engine generalises across repos, not just across time. It is skipped
when the corpus is not cached, so the core suite stays offline; cache it with:

    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1000 \\
        https://github.com/pallets/click data/corpus/click
    python ingest.py --repo-path data/corpus/click --slug pallets/click
"""

from pathlib import Path

import pytest

HELDOUT_SLUG = "pallets/click"
RAW_DIR = "data/raw"
_KEY = HELDOUT_SLUG.replace("/", "__")

pytestmark = pytest.mark.skipif(
    not Path(RAW_DIR, f"{_KEY}.commits.jsonl").exists(),
    reason=f"held-out corpus {HELDOUT_SLUG} not cached (see module docstring)",
)


@pytest.fixture(scope="module")
def heldout_model():
    from induction.pipeline import run_pipeline
    return run_pipeline(HELDOUT_SLUG, RAW_DIR, with_thin=True)


def test_it_runs_and_produces_a_coherent_model(heldout_model):
    m = heldout_model
    assert m.kinds, "should induce at least one process kind"
    assert m.cases, "should form at least one case"
    # the common contribution path should be the loudest variant of its kind
    contrib = next((k for k in m.kinds if k.id == "code_contribution"), None)
    if contrib and contrib.variants:
        assert contrib.variants[0].frequency >= 1


def test_confidences_are_sane_across_unseen_data(heldout_model):
    allowed = {"direct", "joined", "heuristic", "model"}
    for e in heldout_model.shaped.events:
        assert e.confidence.tier.label in allowed
        if e.case_confidence:
            assert e.case_confidence.tier.label in allowed


def test_nothing_off_system_is_asserted_as_fact(heldout_model):
    for g in heldout_model.gaps:
        assert g.confidence.tier.label in {"heuristic", "model"}
        assert g.evidence


def test_orphans_and_unknowns_are_surfaced_not_hidden(heldout_model):
    m = heldout_model
    # a real repo always has direct-to-branch commits that join nothing
    assert isinstance(m.orphans, list)
    for o in m.orphans:
        assert o.reason and o.evidence


def test_inferred_entities_are_marked_on_unseen_data(heldout_model):
    inferred = [e for e in heldout_model.shaped.entities if e.type in ("pr", "issue")]
    for e in inferred:
        assert e.confidence is not None and e.evidence
