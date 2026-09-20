"""The typed tier, exercised entirely offline.

Every test here drives a scripted transport — the same bargain `ScriptedJudge`
makes for the generative tier. No key, no network, no recorded fixtures.

The property under test, over and over, is the one that makes this tier safe to
turn on: **it can only ever narrow.** A Jev that is unreachable, keyless, broken
or merely silent must leave the engine doing precisely what it did before the
typed tier existed. A test that only checked the happy path would miss the one
failure that matters.
"""

from __future__ import annotations

import pytest

from induction.jev_call import Answer, Jev, JevError, parse_answer
from induction.jev_reading import (JevReading, LabelScreen, RecordAssigner, Signal,
                                   _pair_options, _read_pair)
from induction.semantic import GatedJudge, JevGate, ScriptedJudge, SemanticJudge


def transport(answers: dict):
    """A scripted decisions endpoint that always returns `answers`."""
    def send(url, payload, api_key, timeout):
        return {"answers": answers}
    return send


def recording_transport(answers: dict, seen: list):
    def send(url, payload, api_key, timeout):
        seen.append(payload)
        return {"answers": answers}
    return send


# ---------------------------------------------------------------------------
# Parsing one typed answer
# ---------------------------------------------------------------------------

def test_noul_confidence_is_distance_from_the_coin_flip():
    """0.02 and 0.98 are both confident; 0.5 is maximal doubt. A naive reading
    that treated the probability itself as confidence would call a firm 'no' an
    uncertain answer, and gate on it."""
    assert parse_answer({"noul": 0.98}).confidence == pytest.approx(0.98)
    assert parse_answer({"noul": 0.02}).confidence == pytest.approx(0.98)
    assert parse_answer({"noul": 0.50}).confidence == pytest.approx(0.50)


def test_server_confidence_wins_over_the_derived_one():
    a = parse_answer({"noul": 0.9, "confidence": 0.61})
    assert a.confidence == pytest.approx(0.61)


def test_choice_confidence_falls_back_to_the_winning_share():
    a = parse_answer({"choice": "b", "probabilities": {"a": 0.3, "b": 0.7}})
    assert a.kind == "choice" and a.choice == "b"
    assert a.confidence == pytest.approx(0.7)


def test_score_is_a_number_from_zero_and_may_sit_between_levels():
    a = parse_answer({"score": 1.5})
    assert a.kind == "score" and a.score == pytest.approx(1.5)


def test_an_answer_with_no_verdict_is_no_answer():
    assert parse_answer({"probabilities": {"a": 1.0}}) is None
    assert parse_answer("nonsense") is None


def test_unknown_confidence_never_clears_a_bar():
    """`sure` must be False when there is no number. An absent confidence that
    read as confident would let a gate open on nothing at all."""
    assert Answer(kind="score", score=2.0).sure(0.5) is False
    assert Answer(kind="noul", noul=0.9, confidence=0.9).sure(0.5) is True


# ---------------------------------------------------------------------------
# The client's two guarantees
# ---------------------------------------------------------------------------

def test_a_failed_call_returns_no_answers_and_never_raises():
    def boom(url, payload, api_key, timeout):
        raise JevError("bad request", status_code=400)

    jev = Jev(transport=boom)
    assert jev.ask({"a": 1}, {"q": {"type": "noul"}}) == {}
    assert jev.skipped == 1


def test_the_breaker_trips_and_later_calls_are_free():
    calls = []

    def overloaded(url, payload, api_key, timeout):
        calls.append(1)
        raise JevError("busy", status_code=503)

    logged = []
    jev = Jev(transport=overloaded, tries=2, log=logged.append)
    # `with_backoff` sleeps between tries; patch it out via a tiny tries count and
    # accept the one short wait rather than reaching into the module.
    assert jev.ask({}, {"q": {"type": "noul"}}) == {}
    assert jev.available is False
    before = len(calls)
    assert jev.ask({}, {"q": {"type": "noul"}}) == {}
    assert len(calls) == before, "a tripped breaker must not reach the network"
    assert any("disabling Jev" in m for m in logged)


def test_a_reply_without_an_answers_object_is_treated_as_no_answer():
    jev = Jev(transport=lambda *a: {"unexpected": True})
    assert jev.ask({}, {"q": {"type": "noul"}}) == {}
    assert jev.skipped == 1


def test_no_questions_is_no_call():
    calls = []
    jev = Jev(transport=lambda *a: calls.append(1) or {"answers": {}})
    assert jev.ask({"a": 1}, {}) == {}
    assert not calls


# ---------------------------------------------------------------------------
# The gate in front of the semantic judge
# ---------------------------------------------------------------------------

class CountingJudge(SemanticJudge):
    def __init__(self, reason="shared incident"):
        self.calls = 0
        self.reason = reason

    def judge(self, a_text, b_text):
        self.calls += 1
        return self.reason


def test_a_pair_below_the_bar_never_reaches_the_paid_judge():
    inner = CountingJudge()
    gate = JevGate(jev=Jev(transport=transport({"same_work": {"noul": 0.05}})))
    gated = GatedJudge(inner, gate=gate)
    assert gated.judge("a", "b") is None
    assert inner.calls == 0
    assert gated.gated == 1


def test_a_pair_above_the_bar_is_judged_and_carries_the_number():
    inner = CountingJudge()
    gate = JevGate(jev=Jev(transport=transport({
        "same_work": {"noul": 0.87},
        "relation": {"choice": "causal", "probabilities": {"causal": 0.8, "unrelated": 0.2}},
    })))
    reason = GatedJudge(inner, gate=gate).judge("a", "b")
    assert inner.calls == 1
    assert "shared incident" in reason
    assert "jev 0.87" in reason and "causal" in reason


def test_an_unreachable_gate_changes_nothing():
    """The single most important test in this file. No opinion must mean the
    judge runs exactly as it did before the gate existed — never 'no'."""
    inner = CountingJudge()
    dead = Jev(transport=lambda *a: (_ for _ in ()).throw(JevError("nope", status_code=400)))
    gated = GatedJudge(inner, gate=JevGate(jev=dead))
    assert gated.judge("a", "b") == "shared incident"
    assert inner.calls == 1


def test_the_gate_cannot_invent_a_join_the_judge_refused():
    inner = ScriptedJudge([])          # judges nothing the same work
    gate = JevGate(jev=Jev(transport=transport({"same_work": {"noul": 0.99}})))
    assert GatedJudge(inner, gate=gate).judge("a", "b") is None


# ---------------------------------------------------------------------------
# The label screen
# ---------------------------------------------------------------------------

def test_a_transport_label_is_dropped_though_it_is_short_enough_to_pass():
    """'Correspondence Sharing' is two words — the word-count rule waves it
    through — and is exactly what the discovery prompt forbids."""
    screen = LabelScreen(jev=Jev(transport=transport({"is_a_stage": {"score": 0.2}})))
    kept, loose, dropped = screen.cull({"Comms": ["Correspondence Sharing"]})
    assert kept == {} and loose == []
    assert dropped[0][1] == "Correspondence Sharing"


def test_an_achievement_label_survives():
    screen = LabelScreen(jev=Jev(transport=transport({"is_a_stage": {"score": 2.0}})))
    kept, loose, dropped = screen.cull({"Billing": ["Invoice issued"]})
    assert kept == {"Billing": ["Invoice issued"]} and not dropped


def test_a_label_jev_could_not_score_is_kept():
    screen = LabelScreen(jev=Jev(transport=transport({})))
    kept, _, dropped = screen.cull({"Billing": ["Invoice issued"]})
    assert kept == {"Billing": ["Invoice issued"]} and not dropped


def test_a_step_belonging_to_no_process_is_screened_too():
    """The vocabulary allows loose steps, so the screen has to see them — an
    earlier version culled only the nested ones and let every loose label pass."""
    screen = LabelScreen(jev=Jev(transport=transport({"is_a_stage": {"score": 0.1}})))
    kept, loose, dropped = screen.cull({}, ["Correspondence Sharing"])
    assert kept == {} and loose == []
    assert [d[1] for d in dropped] == ["Correspondence Sharing"]


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

class FakeVocab:
    def __init__(self, steps_by_process, loose=()):
        self.steps_by_process = steps_by_process
        self.loose = list(loose)

    @property
    def processes(self):
        return list(self.steps_by_process)

    def steps_for(self, process):
        return self.steps_by_process.get(process, [])


def test_the_option_space_is_process_qualified_so_a_step_cannot_stray():
    vocab = FakeVocab({"Hiring": ["Onsite interview"], "Billing": ["Invoice issued"]})
    options = list(_pair_options(vocab))
    assert "Hiring > Onsite interview" in options
    assert "Billing > Onsite interview" not in options, (
        "a step must be unchoosable for a process it does not belong to")
    assert _read_pair("Hiring > Onsite interview") == ("Hiring", "Onsite interview")


def test_a_peaked_distribution_is_a_placement():
    jev = Jev(transport=transport({"step": {
        "choice": "Hiring > Onsite interview",
        "probabilities": {"Hiring > Onsite interview": 0.91, "Hiring > Offer made": 0.09}}}))
    got = RecordAssigner(jev=jev).step_of({"subject": "onsite"},
                                          FakeVocab({"Hiring": ["Onsite interview",
                                                                "Offer made"]}))
    assert got.process == "Hiring" and got.step == "Onsite interview"
    assert got.placed and not got.ambiguous


def test_a_flat_distribution_is_ambiguous_rather_than_a_placement():
    """The distinction the engine could not previously draw: a record several
    steps fit equally is not the same finding as a record nothing fits."""
    jev = Jev(transport=transport({"step": {
        "choice": "Hiring > Onsite interview",
        "probabilities": {"Hiring > Onsite interview": 0.30, "Hiring > Offer made": 0.28,
                          "Hiring > CV screened": 0.25}}}))
    got = RecordAssigner(jev=jev).step_of({}, FakeVocab({"Hiring": [
        "Onsite interview", "Offer made", "CV screened"]}))
    assert got.ambiguous
    assert [name for name, _ in got.top(2)] == ["Hiring > Onsite interview",
                                                "Hiring > Offer made"]


def test_no_answer_is_no_placement():
    got = RecordAssigner(jev=Jev(transport=transport({}))).step_of(
        {}, FakeVocab({"Hiring": ["Offer made"]}))
    assert not got.placed


def test_the_assigner_is_never_asked_which_process_a_thread_is():
    """The engine's standing rule: the model names, the engine assigns runs by
    counting (`_process_of_case`). A question over processes would be drawing a
    boundary, so no request this module sends may contain one."""
    seen = []
    jev = Jev(transport=recording_transport({"step": {"choice": "Hiring > Offer made"}}, seen))
    RecordAssigner(jev=jev).step_of({}, FakeVocab({"Hiring": ["Offer made"]}))
    assert seen, "expected a request"
    assert set(seen[0]["questions"]) == {"step"}


# ---------------------------------------------------------------------------
# The facade must only ever narrow
# ---------------------------------------------------------------------------

def test_signal_with_no_reading_is_always_worth_reading():
    assert Signal().worth_reading(0.9) is True
    assert Signal(performs_action=0.1).worth_reading(0.25) is False
    assert Signal(performs_action=0.8).worth_reading(0.25) is True


def test_screen_labels_returns_the_vocabulary_untouched_when_jev_is_silent():
    reading = JevReading(jev=Jev(transport=transport({})))
    vocab = FakeVocab({"Hiring": ["Offer made"]})
    assert reading.screen_labels(vocab, lambda m: None) is vocab


def test_sample_pool_keeps_everything_when_nothing_was_read():
    reading = JevReading(jev=Jev(transport=transport({})))
    records = [{"id": str(i)} for i in range(10)]
    assert reading.sample_pool(records, {}, lambda m: None) is records


def test_sample_pool_will_not_shrink_below_what_discovery_needs():
    """A sample chosen for signal that is too small to show the corpus's range
    is a worse sample, not a better one."""
    from induction.abstraction import _DISCOVERY_SAMPLE

    reading = JevReading(jev=Jev(transport=transport({})))
    records = [{"id": str(i)} for i in range(_DISCOVERY_SAMPLE + 20)]
    signals = {r["id"]: Signal(carries_work=0.9 if i < 5 else 0.1)
               for i, r in enumerate(records)}
    assert reading.sample_pool(records, signals, lambda m: None) is records


# ---------------------------------------------------------------------------
# End to end, against the reading tier
# ---------------------------------------------------------------------------

from induction.abstraction import ScriptedRecordClassifier, infer_activities  # noqa: E402
from induction.adapters import email_mbox  # noqa: E402
from induction.pipeline import induce  # noqa: E402

D = "meridian-energy.example"

CLASSIFIER = ScriptedRecordClassifier([
    ("Requested", ["please review the credit terms"]),
    ("Reviewed", ["look fine from our side"]),
    ("Approved", ["approved - proceed"]),
    ("Executed", ["executed and filed"]),
])


def _corpus():
    """Twenty contract runs, each four messages, plus thirty social one-offs.

    Sized to the same shape as `test_reading.py`'s corpus on purpose: the reading
    tier only fires where the verbs demonstrably said nothing, which is measured
    off the corpus (records per distinct activity). A smaller corpus does not
    clear that bar, and a test built on one would pass by never running the code
    it claims to test.
    """
    msgs, n = [], [0]

    def mail(frm, subject, body, day, thread=None):
        n[0] += 1
        mid = f"m{n[0]}"
        head = (f"Message-ID: <{mid}>\nFrom: {frm}@{D}\nTo: desk@{D}\n"
                f"Date: Mon, {day} Oct 2001 09:00:00 -0700\nSubject: {subject}\n")
        if thread:
            head += f"In-Reply-To: <{thread}>\n"
        msgs.append((mid, head + "\n" + body + "\n"))
        return mid

    for i in range(20):
        day = i % 28 + 1
        root = mail("a.okonkwo", f"Master agreement - Counterparty {i}",
                    f"Please review the credit terms on Counterparty {i}.", day)
        mail("l.bergstrom", f"RE: Master agreement - Counterparty {i}",
             f"Credit terms look fine from our side on Counterparty {i}.", day, root)
        mail("p.varga", f"RE: Master agreement - Counterparty {i}",
             f"Approved - proceed to execution for Counterparty {i}.", day, root)
        mail("a.okonkwo", f"RE: Master agreement - Counterparty {i}",
             f"Executed and filed for Counterparty {i}.", day, root)
    odd = "parking badge printer stapler kettle lanyard mug bicycle locker fridge".split()
    for i in range(30):
        mail("r.deniz", f"{odd[i % 10].title()} note {i}",
             f"{odd[i % 10]} {odd[(i + 3) % 10]} {i}", i % 28 + 1)
    return msgs


def _dispatching(step_confidence=0.9, performs=0.9, stage_score=2.0):
    """A scripted endpoint that answers whichever battery it was handed."""
    def send(url, payload, api_key, timeout):
        asked = set(payload["questions"])
        if "performs_action" in asked:
            return {"answers": {"performs_action": {"noul": performs},
                                "machine_generated": {"noul": 0.05},
                                "carries_work": {"noul": 0.8}}}
        if "is_a_stage" in asked:
            return {"answers": {"is_a_stage": {"score": stage_score}}}
        if "step" in asked:
            options = list(payload["questions"]["step"]["criteria"])
            rest = (1.0 - step_confidence) / max(1, len(options) - 1)
            probs = {o: rest for o in options}
            probs[options[0]] = step_confidence
            return {"answers": {"step": {"choice": options[0], "probabilities": probs}}}
        if "verdict" in asked:
            return {"answers": {"verdict": {"choice": "real_process", "confidence": 0.9},
                                "produces_an_artefact": {"noul": 0.9}}}
        return {"answers": {}}
    return send


@pytest.fixture(scope="module")
def baseline():
    """The run exactly as it is without the typed tier — the thing to compare to."""
    m = induce(email_mbox.shape(_corpus(), "meridian"), slug="meridian")
    return m, infer_activities(m, mapper=None, classifier=CLASSIFIER)


def _run_with(transport_fn):
    m = induce(email_mbox.shape(_corpus(), "meridian"), slug="meridian")
    reading = JevReading(jev=Jev(transport=transport_fn))
    return m, infer_activities(m, mapper=None, classifier=CLASSIFIER, jev=reading)


def test_a_dead_jev_leaves_the_reading_byte_for_byte_unchanged(baseline):
    """The guarantee the whole tier rests on. If this ever fails, the typed tier
    is not narrowing the generative one — it is changing its answers."""
    def dead(url, payload, api_key, timeout):
        raise JevError("no", status_code=400)

    _, plain = baseline
    _, gated = _run_with(dead)
    assert gated.by_record == plain.by_record
    assert gated.n_unclassified == plain.n_unclassified
    assert gated.gated == [] and gated.ambiguous == []


def test_a_silent_jev_leaves_the_reading_unchanged(baseline):
    """Answering nothing is not the same as answering no, and must not act like it."""
    _, plain = baseline
    _, quiet = _run_with(lambda *a: {"answers": {}})
    assert quiet.by_record == plain.by_record


def test_gated_records_are_listed_not_deleted():
    """A record the gate held back is a finding with a number on it, and the
    abstention total must not improve just because fewer records were read."""
    _, got = _run_with(_dispatching(performs=0.01))
    assert got.gated, "records that perform nothing should be listed"
    assert all(g["performs_action"] == pytest.approx(0.01) for g in got.gated)
    assert got.by_record == {}, "nothing survived the gate, so nothing was read"


def test_the_typed_confidence_rides_along_with_the_reading():
    _, got = _run_with(_dispatching(step_confidence=0.93))
    assert got.by_record, "the generative classifier should still have read records"
    confidences = [got.confidence_of(eid) for eid in got.by_record]
    assert all(c == pytest.approx(0.93) for c in confidences)
    # and the span — the thing a reader can actually check — is still there
    assert all(got.span_of(eid) for eid in got.by_record)


def test_a_flat_distribution_sends_records_to_the_ambiguous_pile():
    _, got = _run_with(_dispatching(step_confidence=0.30))
    assert got.ambiguous, "a tie should be reported, not silently dropped"
    assert all(row["top"] for row in got.ambiguous)
    assert got.jev_counts["ambiguous"] == len(got.ambiguous)


def test_the_counts_reach_the_inspector():
    from induction.inspector import build_view

    m, got = _run_with(_dispatching(performs=0.01))
    view = build_view(m, activities=got)
    assert view["meta"]["jev"]["gated"] == len(got.gated)
    labels = [row["activity"] for row in view["vocabulary"]]
    assert any(label.startswith("Gated") for label in labels)


def test_a_transport_label_is_culled_from_the_discovered_vocabulary():
    """The screen runs over what discovery proposed, and a label it drops must
    not survive into the vocabulary the records are read against."""
    _, got = _run_with(_dispatching(stage_score=0.1))
    assert got.steps_by_process == {} and got.by_record == {}


# ---------------------------------------------------------------------------
# The wire
# ---------------------------------------------------------------------------
#
# Every test above injects a transport, which is what keeps them fast and
# offline — but it also means none of them touches `_post`, and `_post` is the
# one part of this tier that was written against documentation rather than
# against a live endpoint. So this exercises it for real, over a socket, against
# a server that speaks the Decisions API's shape: the request is built, sent,
# received, decoded and parsed by the same code a real call would use.

import json as _json  # noqa: E402
import threading  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402

from induction.jev_call import _post  # noqa: E402


class _Decisions(BaseHTTPRequestHandler):
    received: dict = {}

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        _Decisions.received = {"payload": _json.loads(body),
                               "auth": self.headers.get("Authorization"),
                               "content_type": self.headers.get("Content-Type")}
        if _Decisions.received["payload"].get("model") == "boom":
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"overloaded")
            return
        out = _json.dumps({"answers": {"q": {"noul": 0.77}}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


@pytest.fixture()
def endpoint():
    server = HTTPServer(("127.0.0.1", 0), _Decisions)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/api/alpha/decisions"
    server.shutdown()
    server.server_close()


def test_the_request_goes_out_in_the_shape_the_api_documents(endpoint, monkeypatch):
    # The real transport is gated on a key, so this also pins that gate: without
    # `OPENROUTER_API_KEY` the client is not available and never reaches a socket.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    jev = Jev(endpoint=endpoint, model="typesafe/jev-1.13")
    answers = jev.ask({"ticket": "payouts failing"},
                      {"q": {"type": "noul", "instructions": "urgent?"}})
    sent = _Decisions.received
    assert sent["auth"].startswith("Bearer ")
    assert sent["content_type"] == "application/json"
    assert sent["payload"]["model"] == "typesafe/jev-1.13"
    assert sent["payload"]["state"] == {"ticket": "payouts failing"}
    assert sent["payload"]["questions"]["q"]["type"] == "noul"
    assert answers["q"].noul == pytest.approx(0.77)
    assert jev.calls == 1


def test_without_a_key_the_real_transport_is_never_reached(endpoint, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _Decisions.received = {}
    jev = Jev(endpoint=endpoint)
    assert jev.available is False
    assert jev.ask({"a": 1}, {"q": {"type": "noul"}}) == {}
    assert _Decisions.received == {}, "no key must mean no request"


def test_an_http_error_carries_its_status_so_backoff_can_classify_it(endpoint):
    with pytest.raises(JevError) as raised:
        _post(endpoint, {"model": "boom"}, "k", 5.0)
    assert raised.value.status_code == 503

    from induction.anthropic_call import is_transient
    assert is_transient(raised.value), "a 503 must be waited out, not given up on"


def test_an_unreachable_host_is_transient_not_a_bad_request():
    """A connection that never lands has no status of its own. It must still read
    as transient — giving up on a momentarily unreachable service after one try
    is the exact bug `with_backoff` exists to prevent."""
    from induction.anthropic_call import is_transient

    with pytest.raises(JevError) as raised:
        _post("http://127.0.0.1:9/decisions", {"model": "x"}, "k", 1.0)
    assert is_transient(raised.value)
