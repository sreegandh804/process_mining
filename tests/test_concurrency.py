"""Running independent calls at once — and not changing any answer by doing it.

The engine's two slow loops were slow for the same reason: each iteration was a
network call, and none of them read the previous one's output. Running them
concurrently is worth minutes per run, and it is only worth anything if the run
produces exactly what it produced before.

So these tests pin two things, and the second is the one that matters:

  - that the calls really do overlap (measured, not assumed); and
  - that the ORDER-DEPENDENT decision in the semantic pass — first claim wins,
    decided by shortlist order — survives being evaluated out of order.
"""

from __future__ import annotations

import time

from induction.concurrency import fan_out, in_waves


def test_results_come_back_in_the_order_they_were_asked():
    assert fan_out(lambda x: x * 2, [1, 2, 3, 4]) == [2, 4, 6, 8]


def test_a_failed_call_is_a_none_and_not_a_raised_run():
    def half(x):
        if x == 2:
            raise ValueError("no")
        return x

    assert fan_out(half, [1, 2, 3]) == [1, None, 3]


def test_an_empty_list_makes_no_pool():
    assert fan_out(lambda x: x, []) == []


def test_the_calls_actually_overlap():
    """The whole point, measured. Six 200ms calls take 1.2s in a queue and about
    200ms in flight together; anything near the serial figure means the
    concurrency is not happening and the rest of this is theatre."""
    started = time.monotonic()
    got = fan_out(lambda _: time.sleep(0.2) or "done", list(range(6)), workers=6)
    elapsed = time.monotonic() - started
    assert got == ["done"] * 6
    assert elapsed < 0.6, f"calls did not overlap: {elapsed:.2f}s for 6 × 0.2s"


# ---------------------------------------------------------------------------
# The ordering invariant
# ---------------------------------------------------------------------------

def _claim(pairs, verdict, waves=True, workers=3):
    """The semantic pass's rule, run either way. Returns (joins, pairs judged).

    `verdict(pair)` stands in for the judge. The rule is the real one: a pair
    whose either side is already claimed is skipped, and a pair the judge
    approves claims both sides.
    """
    taken: set = set()
    joins: list = []
    judged: list = []

    def judge(pair):
        judged.append(pair)
        return verdict(pair)

    def unclaimed(pair):
        return pair[0] not in taken and pair[1] not in taken

    if waves:
        stream = in_waves(judge, pairs, unclaimed, workers=workers)
    else:
        stream = ((p, judge(p)) for p in pairs if unclaimed(p))

    for pair, said_yes in stream:
        if not said_yes or not unclaimed(pair):
            continue
        taken.update(pair)
        joins.append(pair)
    return joins, judged


def test_first_claim_still_wins_when_the_wave_answered_both():
    """A-B and A-C are both judged 'same work', and both are in one wave. A can
    only be claimed once, and shortlist order decides by whom — exactly as when
    the pairs were judged one at a time."""
    pairs = [("A", "B"), ("A", "C")]
    joins, _ = _claim(pairs, lambda p: True)
    assert joins == [("A", "B")]


def test_waved_and_serial_reach_the_same_joins():
    """The invariant, over a shortlist built to make claiming bite: several pairs
    share components, and the judge approves most of them."""
    pairs = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"),
             ("E", "F"), ("F", "G"), ("G", "H"), ("H", "E")]
    approve = lambda p: p != ("F", "G")   # noqa: E731 — one refusal in the middle

    serial_joins, _ = _claim(pairs, approve, waves=False)
    for workers in (1, 2, 3, 8):
        waved_joins, _ = _claim(pairs, approve, waves=True, workers=workers)
        assert waved_joins == serial_joins, f"waves of {workers} changed the joins"


def test_a_pair_claimed_before_its_wave_is_never_judged():
    """The saving that makes this waves rather than one big fan-out: a pair that
    is already moot when its wave is assembled costs nothing."""
    pairs = [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")]
    joins, judged = _claim(pairs, lambda p: True, workers=2)
    assert joins == [("A", "B")]
    # First wave judges two pairs; A is claimed, so nothing after it is asked.
    assert judged == [("A", "B"), ("A", "C")]


def test_nothing_wanted_means_nothing_judged():
    judged = []
    list(in_waves(lambda p: judged.append(p), [1, 2, 3], lambda p: False))
    assert judged == []
