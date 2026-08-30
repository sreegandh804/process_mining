"""Step 3 — Order: timed cases trace correctly; untimed order is not fabricated."""


def test_timed_case_orders_by_timestamp(mini_shaped):
    shaped, corr, slug = mini_shaped
    events_by_id = {e.id: e for e in shaped.events}
    pr2 = corr.cases["case:pr:2"]
    assert pr2.order_status == "ordered"
    actions = [events_by_id[e].action for e in pr2.ordered_event_ids if e in events_by_id]
    # authored (Jan 10) -> committed/handoff (Jan 15) -> reverted (Jan 20)
    assert actions == ["authored", "committed", "reverted"]
    ts = [events_by_id[e].timestamp for e in pr2.ordered_event_ids if e in events_by_id]
    assert ts == sorted(ts)


def test_untimed_thin_case_is_flagged_unknown_not_guessed(mini_shaped):
    shaped, corr, slug = mini_shaped
    notes = corr.cases["case:notes:acme/widget:1.0.0"]
    assert notes.order_status == "unknown"
    # and the observations that make it up carry no invented time or actor
    obs = [o for o in shaped.observations if o.case_id == notes.id]
    assert obs
    assert all(o.seen_at is None for o in obs)
