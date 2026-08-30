"""Step 2 — Correlate: the graded core. Groupings match the hand-verified fixture."""


def _case(corr, cid):
    assert cid in corr.cases, f"expected case {cid}"
    return corr.cases[cid]


def test_merge_topology_recovers_commits_that_share_no_text_reference(mini_shaped):
    """f1 ('add widget core') carries no '#1' anywhere — it joins PR #1 purely
    from the git DAG (reachable from the merge, not from the trunk). This is the
    join a text-only correlator would miss."""
    shaped, corr, slug = mini_shaped
    pr1 = _case(corr, "case:pr:1")
    assert "commit:f1" in pr1.entity_ids
    assert "commit:f2" in pr1.entity_ids
    assert "commit:m1" in pr1.entity_ids


def test_squash_and_revert_land_in_the_same_run(mini_shaped):
    shaped, corr, slug = mini_shaped
    pr2 = _case(corr, "case:pr:2")
    assert "commit:s1" in pr2.entity_ids   # squash "(#2)"
    assert "commit:r1" in pr2.entity_ids   # Revert "...(#2)"


def test_issue_reference_enriches_the_pr_run(mini_shaped):
    shaped, corr, slug = mini_shaped
    pr2 = _case(corr, "case:pr:2")
    assert "issue:acme/widget:7" in pr2.entity_ids
    issue = next(e for e in shaped.entities if e.id == "issue:acme/widget:7")
    # an entity known ONLY by reference is marked inferred, with evidence
    assert issue.attrs["known_via"] == "reference"
    assert issue.confidence is not None and issue.evidence


def test_every_case_link_is_scored(mini_shaped):
    shaped, corr, slug = mini_shaped
    linked = [e for e in shaped.events if e.case_id]
    assert linked, "some events should be correlated"
    for ev in linked:
        assert ev.case_confidence is not None
        assert ev.case_confidence.tier.label in {"direct", "joined", "heuristic", "model"}


def test_deterministic_spine_is_mostly_joined(mini_shaped):
    shaped, corr, slug = mini_shaped
    tiers = [e.case_confidence.tier.label for e in shaped.events if e.case_confidence]
    assert tiers.count("joined") >= len(tiers) - 1  # the fixture joins deterministically


def test_cross_source_join_thin_meets_thick(mini_shaped):
    """A changelog bullet citing :pr:`2` correlates to the git PR-2 run on the
    shared number — thin data enriching thick, tier joined."""
    shaped, corr, slug = mini_shaped
    obs = next(o for o in shaped.observations
               if o.id == "obs:changelog_entry:acme/widget:1.1.0:0")
    assert obs.case_id == "case:pr:2"
    assert obs.case_confidence.tier.label == "joined"
