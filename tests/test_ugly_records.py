"""The ugly-record cases (brief §7) — one assertion each, on the golden fixture.

These are the records that break a naive pipeline. The engine must do the right,
honest thing with every one of them.
"""


def test_orphan_lands_in_the_queue_with_a_reason(mini_model):
    """Records that join to no run are surfaced, never dropped or padded in."""
    orphan_ids = {o.entity_id for o in mini_model.orphans}
    assert "commit:c1" in orphan_ids   # the root commit references nothing
    assert "commit:o1" in orphan_ids   # a stray readme tweak
    for o in mini_model.orphans:
        assert o.reason                 # every orphan says *why* it is orphaned
        assert o.evidence               # and resolves back to its artefact


def test_same_activity_different_people_is_merged_with_both_records_kept(mini_model):
    """A co-authored commit is one activity by two people — merged into one step,
    with every underlying record retained (a merge is a view, not a deletion)."""
    merges = mini_model.merges
    assert len(merges) == 1
    m = merges[0]
    assert m.action == "authored"
    assert set(m.member_ids) == {"person:bob@acme.io", "person:carol@acme.io"}
    assert len(m.event_ids) == 2        # both records retained
    assert m.evidence


def test_snapshot_with_no_timestamp_is_unknown_not_fabricated(mini_model):
    """A thin observation with no time yields order 'unknown' — never a guess."""
    unknown = [c for c in mini_model.cases.values() if c.order_status == "unknown"]
    assert unknown, "the thin changelog section must produce an order-unknown run"
    obs = [o for o in mini_model.shaped.observations
           if o.case_id in {c.id for c in unknown}]
    assert obs and all(o.seen_at is None and o.case_id for o in obs)


def test_look_alike_non_process_is_rejected_with_a_reason(mini_model):
    """A dependabot bump moves no product artefact — 'looks like a process, isn't'."""
    dep = next(k for k in mini_model.kinds if k.id == "dependency_bumps")
    assert dep.rejected is True
    assert dep.reject_reason and "isn't" in dep.reject_reason.lower()
    # rejected, but NOT deleted — still fully inspectable
    assert dep.case_ids
