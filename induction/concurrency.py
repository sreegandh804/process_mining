"""Running independent model calls at the same time.

Nothing here makes a call cheaper, smarter or different. It only stops calls
that do not depend on each other from queueing behind each other, which on a
real corpus is most of the wall clock: reading 263 records is eleven Opus calls
that each take about a minute and none of which reads another's output, and run
one after another that is seventeen minutes of mostly waiting.

Two shapes, because the engine has two:

  `fan_out`  — the items are wholly independent. Send them all, in bounded
               flight, and keep the results in the order they were given.

  `in_waves` — the items are independent to *evaluate* but the decisions
               interact: an earlier result can make a later item moot. A full
               fan-out would answer questions that were about to be skipped, so
               this goes a wave at a time and re-checks between waves, spending
               at most one wave's worth of extra calls to get the concurrency.

Bounded flight matters. An API has per-minute request and token limits, and
firing an unbounded number of calls at it earns 429s — which `with_backoff`
will wait out, turning the speed-up back into a queue with extra steps. Six at
a time is well inside a normal limit and still collapses eleven serial minutes
into two.

Failures never propagate. A worker that raises yields None, the same bargain
every model seam in this codebase makes: a call that failed is work the caller
does without, not a run that stops.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

# Calls in flight at once. Chosen to sit comfortably inside a standard
# per-minute rate limit rather than to saturate one — the fastest setting that
# earns a 429 is slower than a slightly smaller one that does not, because the
# backoff turns a rate limit straight back into a queue with extra steps.
#
# Three still collapses eleven serial calls into four rounds rather than eleven,
# which is most of the win; raise it only after watching a real run's log for
# retry lines, and lower it if you see any.
WORKERS = 3


def fan_out(fn: Callable[[T], R], items: Sequence[T],
            workers: int = WORKERS) -> list[Optional[R]]:
    """Map `fn` over `items` concurrently, results in the given order.

    Returns one entry per item, None where the call raised.
    """
    if not items:
        return []
    def safe(item):
        try:
            return fn(item)
        except Exception:  # noqa: BLE001 — one item's call, never the run
            return None

    if len(items) == 1:
        return [safe(items[0])]
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(safe, items))


def in_waves(fn: Callable[[T], R], items: Sequence[T],
             still_wanted: Callable[[T], bool],
             workers: int = WORKERS) -> Iterable[tuple[T, Optional[R]]]:
    """Evaluate `items` concurrently, a wave at a time, skipping the moot ones.

    `still_wanted(item)` is consulted twice: before an item joins a wave, and
    again — by the caller, as it consumes the results — because a result from
    earlier in the same wave may have changed the answer. That is the whole
    reason this is not `fan_out`: the caller's own decisions are sequential and
    must stay that way, so the ORDER results are handed back in is the order
    they were given, and the caller applies its rule to them one at a time,
    exactly as it did when it was making the calls one at a time.

    The cost of the concurrency is at most `workers - 1` calls per wave that
    turn out to have been unnecessary. The alternative — evaluating everything
    up front — answers every question in the list, which on a corpus where most
    pairs get claimed is a great deal more than the serial loop ever asked.

    `still_wanted` must be monotone: an item that stops being wanted never
    becomes wanted again. Items that are unwanted when a wave is assembled are
    carried to the next wave and re-checked, and this stops when a whole pass
    produces an empty wave — so a predicate that flipped back would simply have
    dropped those items, not looped.
    """
    pending = list(items)
    while pending:
        wave, rest = [], []
        for item in pending:
            if len(wave) < workers and still_wanted(item):
                wave.append(item)
            else:
                rest.append(item)
        if not wave:
            return
        for item, result in zip(wave, fan_out(fn, wave, workers)):
            yield item, result
        pending = rest
