"""Transient API overload should be waited out, not fatal — and a sustained
overload should trip the judge's breaker instead of retrying every pair.
"""

import time

import pytest

from induction.anthropic_call import is_transient, with_backoff

NOOP = lambda _s: None  # noqa: E731 — a do-nothing sleep for instant tests


# --- is_transient -----------------------------------------------------------

def test_transient_by_status_code():
    e = Exception(); e.status_code = 529
    assert is_transient(e)
    e429 = Exception(); e429.status_code = 429
    assert is_transient(e429)


def test_transient_by_type_name():
    Overloaded = type("OverloadedError", (Exception,), {})
    assert is_transient(Overloaded())


def test_not_transient_is_not_retried():
    bad = Exception(); bad.status_code = 400
    assert not is_transient(bad)


# --- with_backoff -----------------------------------------------------------

def test_returns_on_first_success():
    calls = []
    assert with_backoff(lambda: calls.append(1) or "ok", sleep=NOOP) == "ok"
    assert len(calls) == 1


def test_retries_transient_then_succeeds():
    n = {"i": 0}
    def fn():
        n["i"] += 1
        if n["i"] < 3:
            e = Exception("busy"); e.status_code = 529
            raise e
        return "ok"
    assert with_backoff(fn, tries=5, sleep=NOOP) == "ok"
    assert n["i"] == 3          # failed twice, succeeded on the third


def test_gives_up_after_tries_and_raises_last():
    n = {"i": 0}
    def fn():
        n["i"] += 1
        e = Exception("still busy"); e.status_code = 529
        raise e
    with pytest.raises(Exception):
        with_backoff(fn, tries=3, sleep=NOOP)
    assert n["i"] == 3          # exactly `tries` attempts, no more


def test_non_transient_raises_immediately():
    n = {"i": 0}
    def fn():
        n["i"] += 1
        e = Exception("bad request"); e.status_code = 400
        raise e
    with pytest.raises(Exception):
        with_backoff(fn, tries=5, sleep=NOOP)
    assert n["i"] == 1          # not retried


# --- the judge's circuit breaker -------------------------------------------

def test_judge_trips_after_sustained_overload(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-present")
    monkeypatch.setattr(time, "sleep", NOOP)   # with_backoff resolves time.sleep at call time

    class _Msgs:
        def create(self, **kw):
            e = Exception("Overloaded"); e.status_code = 529
            raise e
    fake = type("C", (), {"messages": _Msgs()})()
    monkeypatch.setattr("induction.anthropic_call.client", lambda **kw: fake)

    from induction.semantic import AnthropicJudge
    logs = []
    j = AnthropicJudge(tries=3, log=logs.append)

    assert j.judge("a", "b") is None        # first pair: retries, then trips
    assert j._tripped is True
    assert j.skipped == 1
    assert any("disabling the semantic judge" in m for m in logs)

    # Once tripped, further pairs return immediately without touching the API.
    assert j.judge("c", "d") is None
    assert j.skipped == 1                    # not incremented — short-circuited


# --- the generative calls stream ---------------------------------------------
#
# The three big-ceiling calls (the verb map at 8k, discovery at 8k, classification
# at 16k) hold a connection open for as long as the model thinks, and the reading
# pass now runs several of them at once. Blocking on that is how a real run
# started reporting APITimeoutError instead of readings, so they stream.
#
# These pin the switch rather than the API: a fake client that offers only
# `.stream()` would fail loudly against a call that still used `.create()`.

class _Stream:
    """A minimal stand-in for the SDK's streaming context manager."""

    def __init__(self, message, seen):
        self._message = message
        self._seen = seen

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        self._seen.append("get_final_message")
        return self._message


def _streaming_client(text: str, seen: list):
    message = type("M", (), {
        "content": [type("B", (), {"type": "text", "text": text})()],
        "stop_reason": "end_turn",
    })()

    class _Msgs:
        def stream(self, **kw):
            seen.append(kw)
            return _Stream(message, seen)

        def create(self, **kw):                     # must never be reached
            raise AssertionError("this call must stream, not block")

    return type("C", (), {"messages": _Msgs()})()


def test_the_reading_pass_streams(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-present")
    seen = []
    monkeypatch.setattr("induction.anthropic_call.client",
                        lambda **kw: _streaming_client('{"processes": []}', seen))

    from induction.abstraction import AnthropicRecordClassifier
    got, raw = AnthropicRecordClassifier()._call("sys", "content", max_tokens=16000)
    assert got == {"processes": []}
    assert raw == '{"processes": []}'
    assert seen[0]["max_tokens"] == 16000
    assert "get_final_message" in seen


def test_the_verb_map_streams(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-present")
    seen = []
    monkeypatch.setattr("induction.anthropic_call.client",
                        lambda **kw: _streaming_client('{"mail/sent": "Requested"}', seen))

    from induction.abstraction import AnthropicActivityMapper
    got = AnthropicActivityMapper().map(
        [{"artefact": "mail", "action": "sent", "examples": []}])
    assert got == {"mail/sent": "Requested"}


# `naming.py`'s single 8k call got the same treatment and for the same reason;
# its own path is already exercised through the injected `namer` seam, which is
# the layer above the transport this pins.
