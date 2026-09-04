"""Resilient Anthropic calls — transparent retry + backoff on transient overload.

The model tier reaches a network API, and that API is sometimes momentarily busy:
an HTTP **529 `overloaded_error`** (or a 429 rate-limit, or a 5xx) is *transient*
and server-side, not a fault in the call. The bug this module fixes is that a
*single* such blip used to make a call give up for good — on the per-pair
semantic judge that meant one hiccup silently downgraded the whole run to
deterministic, one skipped line at a time.

So every call goes through `with_backoff`: on a transient error it waits and
retries (exponential, with jitter), and only gives up after several tries. A
*non-transient* error (a 400 bad request, a 401 auth failure) is not retried — it
raises straight away, because retrying it would just burn time.

We own the retry loop rather than leaning on the SDK's built-in one, for two
reasons this codebase cares about: it is **visible** (each wait is logged through
the same progress channel as everything else) and it is **testable offline**
(inject a callable that raises then succeeds — no network, no key). Clients are
therefore built with `max_retries=0` so the backoff is not applied twice.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

# Statuses worth waiting out: request timeout, conflict, rate-limit, and the 5xx
# family (500/502/503/504 and Anthropic's 529 overloaded). Everything else — a
# 400/401/403/404 — is the caller's problem and must not be retried.
_TRANSIENT_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

# SDK exception classes that are transient by their very type, matched by name so
# this module needs no import of `anthropic` (and stays importable without it).
_TRANSIENT_NAMES = frozenset({
    "OverloadedError", "RateLimitError", "APITimeoutError", "APIConnectionError",
    "InternalServerError", "ServiceUnavailableError",
})


def is_transient(exc: BaseException) -> bool:
    """True if `exc` is a momentary server/transport condition worth retrying."""
    if type(exc).__name__ in _TRANSIENT_NAMES:
        return True
    code = getattr(exc, "status_code", None)
    return code in _TRANSIENT_STATUS


def with_backoff(fn: Callable[[], T], *, tries: int = 6, base: float = 2.0,
                 cap: float = 32.0, log: Optional[Callable[[str], None]] = None,
                 label: str = "call", sleep: Optional[Callable[[float], None]] = None) -> T:
    """Run `fn()`, retrying transient failures with exponential backoff + jitter.

    Waits `min(cap, base * 2**i) * (0.5 + random)` seconds between attempts, up to
    `tries` total. Re-raises the last transient error once tries are exhausted, and
    re-raises a *non*-transient error immediately (no wait). `sleep` is resolved at
    call time (default `time.sleep`) so a test can patch it and run instantly.
    """
    if sleep is None:
        sleep = time.sleep
    last: Optional[BaseException] = None
    for i in range(max(1, tries)):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — we re-raise unless it's transient
            if not is_transient(e):
                raise
            last = e
            if i >= tries - 1:
                break
            delay = min(cap, base * (2 ** i)) * (0.5 + random.random())
            if log:
                log(f"{label}: transient API error ({type(e).__name__}); "
                    f"retry {i + 1}/{tries - 1} in {delay:.0f}s")
            sleep(delay)
    assert last is not None
    raise last


def client(max_retries: int = 0, timeout: float = 60.0):
    """A lazily-imported Anthropic client with the SDK's own retry disabled, so
    `with_backoff` is the single, visible place retries happen."""
    import anthropic
    return anthropic.Anthropic(max_retries=max_retries, timeout=timeout)
