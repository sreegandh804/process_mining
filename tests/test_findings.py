"""Process performance and owner review — measured, guarded, and round-trippable.

The measurements are only worth showing if they refuse to say more than the
records do, so most of these tests are about the refusals: too few runs, no
dates, a step with no date in the middle, one wait posing as a bottleneck. The
rest pin the review loop: an answer given on one run lands on the same claim on
the next, a dispute sits beside the records instead of replacing them, and an
answer that no longer matches anything is reported rather than dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from induction import review
from induction.emit import build_model
from induction.findings import MIN_RUNS, human_duration, measure, parse_ts
from induction.inspector import build_view
from induction.pipeline import run_tabular_pipeline
from run_tabular import sources_for

ROOT = Path(__file__).resolve().parent.parent
CANON = ["raised", "approved", "paid"]


def _run(key, *steps, offsystem=False):
    return {"key": key, "steps": list(steps), "offsystem": offsystem}


def _usual(key, raised="2024-01-01", approved="2024-01-02", paid="2024-01-12", **kw):
    return _run(key, ("raised", raised), ("approved", approved), ("paid", paid), **kw)


# ---- measurement ------------------------------------------------------------

def test_bottleneck_is_the_slow_wait_with_its_share_and_runs():
    runs = [_usual(f"r{i}") for i in range(6)]
    out = measure(runs, CANON, "invoices", "invoice")
    lw = out["metrics"]["longest_wait"]
    assert (lw["from"], lw["to"]) == ("approved", "paid")
    assert lw["median"] == "10 days"
    assert lw["share_pct"] == 91            # 10 of 11 days
    assert lw["runs"] == sorted(r["key"] for r in runs)
    assert out["headline"]["kind"] == "bottleneck"
    assert "Approved to Paid" in out["headline"]["text"]
    assert out["metrics"]["cycle"]["median"] == "11 days"


def test_too_few_runs_is_not_measured_and_says_so():
    out = measure([_usual(f"r{i}") for i in range(MIN_RUNS - 1)], CANON, "invoices", "invoice")
    assert out["enough"] is False
    assert out["metrics"] == {}
    assert out["headline"]["kind"] == "insufficient"
    assert "too few" in out["headline"]["text"]


def test_undated_runs_are_left_out_not_given_a_time():
    runs = [_usual(f"r{i}") for i in range(3)]
    runs += [_run(f"u{i}", ("raised", None), ("approved", None), ("paid", None)) for i in range(4)]
    out = measure(runs, CANON)
    cyc = out["metrics"]["cycle"]
    # 3 dated runs is below the bar: no median is invented from them.
    assert cyc == {"insufficient": True, "measured": 3, "of": 7}
    assert out["metrics"]["longest_wait"] == {"insufficient": True}
    # ...but the route is still counted over all seven.
    assert out["metrics"]["off_route"]["of"] == 7


def test_a_step_without_a_date_breaks_the_chain_rather_than_being_bridged():
    runs = [_run(f"r{i}", ("raised", "2024-01-01"), ("approved", None), ("paid", "2024-01-11"))
            for i in range(6)]
    out = measure(runs, CANON)
    # raised -> paid is NOT a consecutive pair, so no wait is claimed across the gap.
    assert out["metrics"]["waits"] == []
    assert out["metrics"]["cycle"]["median"] == "10 days"


def test_a_single_wait_is_never_called_a_bottleneck():
    runs = [_run(f"r{i}", ("sent", "2024-01-01T09:00:00Z"), ("replied", "2024-01-01T15:00:00Z"))
            for i in range(6)]
    out = measure(runs, ["sent", "replied"])
    assert out["metrics"]["longest_wait"]["share_pct"] == 100
    assert out["headline"]["kind"] != "bottleneck"


def test_a_rare_slow_detour_does_not_headline_over_the_common_wait():
    # Found on a real permit log: a 24-hour detour taken by a handful of cases
    # out-ranked, by median, the wait every case pays. Impact is length x
    # frequency, so the common 2-day wait must win and its share must be of
    # all the time spent, not of the detour's own few runs.
    runs = [_run(f"c{i}", ("raised", "2024-01-01"), ("approved", "2024-01-03"),
                 ("paid", "2024-01-04")) for i in range(40)]
    runs += [_run(f"d{i}", ("raised", "2024-01-01"), ("held", "2024-01-02"),
                  ("approved", "2024-01-12"), ("paid", "2024-01-13")) for i in range(5)]
    m = measure(runs, CANON)["metrics"]
    lw = m["longest_wait"]
    assert (lw["from"], lw["to"]) == ("raised", "approved")
    assert lw["measured"] == 40 and lw["of_timed"] == 45
    held = next(w for w in m["waits"] if w["from"] == "held")
    assert held["median_s"] > lw["median_s"]          # slower, but rarer
    assert held["share_pct"] < lw["share_pct"]


def test_off_route_rework_and_not_completed_are_counted_with_their_runs():
    runs = [_usual(f"ok{i}") for i in range(5)]
    runs.append(_run("skip", ("raised", "2024-01-01"), ("paid", "2024-01-05")))
    runs.append(_run("redo", ("raised", "2024-01-01"), ("approved", "2024-01-02"),
                     ("raised", "2024-01-03"), ("approved", "2024-01-04"), ("paid", "2024-01-09")))
    runs.append(_run("stuck", ("raised", "2024-01-01"), ("approved", "2024-01-02")))
    m = measure(runs, CANON)["metrics"]
    assert m["off_route"]["runs"] == ["skip", "stuck"]      # "redo" has the usual shape
    assert m["off_route"]["most_skipped"]["count"] == 1     # approved (skip) / paid (stuck)
    assert m["rework"]["runs"] == ["redo"] and m["rework"]["most_redone"] in ("raised", "approved")
    assert m["not_completed"]["runs"] == ["stuck"]
    assert m["not_completed"]["final_step"] == "paid"


def test_offsystem_steps_inside_the_bottleneck_are_flagged():
    runs = [_usual(f"r{i}", offsystem=(i < 2)) for i in range(6)]
    h = measure(runs, CANON, "invoices", "invoice")["headline"]
    assert h["caveat"] and h["caveat"].startswith("2 of these invoices")


def test_money_is_never_produced():
    out = measure([_usual(f"r{i}") for i in range(6)], CANON)
    assert out["money"]["value"] is None and out["money"]["why"]


def test_timestamps_and_durations_read_like_a_person_would():
    assert parse_ts("2024-01-05") == (parse_ts("2024-01-05T00:00:00")[0], False)
    a, _ = parse_ts("2024-01-05T10:00:00+02:00")
    b, _ = parse_ts("2024-01-05T08:00:00Z")
    assert a == b
    assert parse_ts("not a date") == (None, False)
    assert human_duration(0, precise=False) == "same day"
    assert human_duration(3 * 3600, precise=True) == "3 hours"
    assert human_duration(1.5 * 86400, precise=True) == "1.5 days"
    assert human_duration(15 * 86400, precise=False) == "15 days"


# ---- on a real sample -------------------------------------------------------

@pytest.fixture(scope="module")
def finance():
    return run_tabular_pipeline(sources_for(ROOT / "samples" / "finance", "csv"), slug="finance")


def _main(view):
    return max((p for p in view["processes"] if not p["leftover"]), key=lambda p: p["count"])


def test_finance_sample_finds_the_payment_wait(finance):
    p = _main(build_view(finance))
    perf = p["performance"]
    assert perf["headline"]["kind"] == "bottleneck"
    lw = perf["metrics"]["longest_wait"]
    assert (lw["from"], lw["to"]) == ("approved", "paid")
    # Coverage is honest: the messy sample has undated rows, so not every
    # invoice could be timed, and the figure says how many were.
    assert perf["metrics"]["cycle"]["measured"] < perf["n_runs"]
    keys = {r["key"] for r in build_view(finance)["runs"]}
    assert set(lw["runs"]) <= keys                 # every run behind it exists


def test_performance_reaches_model_json(finance):
    doc = build_model(finance)
    assert doc["performance"] and all("metrics" in p for p in doc["performance"])
    json.dumps(doc["performance"])                 # serialisable as-is


# ---- owner review -----------------------------------------------------------

def _write(tmp_path, answers):
    f = tmp_path / "corrections.json"
    f.write_text(json.dumps({"version": 1, "answers": answers}))
    return f


def test_claim_ids_are_stable_across_runs(finance):
    a = _main(build_view(finance))["claims"]
    b = _main(build_view(finance))["claims"]
    assert a["process"]["id"] == b["process"]["id"]
    assert {c["id"] for c in a["steps"].values()} == {c["id"] for c in b["steps"].values()}
    assert review.claim_id("step", "X", "paid") == review.claim_id("step", "x", "Paid ")


def test_a_dispute_sits_beside_the_records_and_a_confirm_is_marked(finance, tmp_path):
    p = _main(build_view(finance))
    step = p["claims"]["steps"]["submitted"]
    finding = p["claims"]["finding"]
    f = _write(tmp_path, [
        {"claim_id": step["id"], "claim": step["text"], "verdict": "dispute",
         "note": "we never submit invoices"},
        {"claim_id": finding["id"], "claim": finding["text"], "verdict": "confirm"},
        {"claim_id": "c-gone", "claim": "a step that no longer exists", "verdict": "dispute"},
        {"claim_id": "c-bad", "verdict": "maybe"},            # ignored: not a verdict
    ])
    answers = review.load(f)
    assert set(answers) == {step["id"], finding["id"], "c-gone"}

    view = build_view(finance, corrections=answers)
    p2 = _main(view)
    assert p2["claims"]["steps"]["submitted"]["status"]["state"] == "disputed"
    assert p2["claims"]["finding"]["status"]["state"] == "confirmed"
    # the claim is still there — a dispute never deletes what the records show
    assert "submitted" in p2["flow"]

    div = build_model(finance, corrections=answers)["divergence"]
    assert div["status"] == "live" and div["confirmed"] == 1
    [item] = div["items"]
    assert item["owner_says"] == "we never submit invoices"
    assert item["records_show"].startswith("Submitted appears in ")
    assert [a["claim_id"] for a in div["unmatched_answers"]] == ["c-gone"]
    # earlier answers are carried into the page so an export accumulates
    assert {a["claim_id"] for a in view["review"]["prior"]} == set(answers)


def test_no_review_file_is_an_empty_review_and_a_broken_one_stops():
    assert review.load(None) == {}
    with pytest.raises(SystemExit, match="not a review"):
        review.load(ROOT / "README.md")
    with pytest.raises(SystemExit, match="not found"):
        review.load(ROOT / "no-such-file.json")
