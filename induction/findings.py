"""Process performance — what the runs of one process measurably do.

The inspector already shows WHAT happens: the steps, the routes, every run. This
module answers the next question an operations lead asks — where does the time
go, how often does it go wrong, and how often is work done twice — and it does it
without a model. Every figure here is counted or timed from records that exist.

It is deliberately source-agnostic. It reads only the view-level runs the page
already draws (a run = its steps, in order, with the time each was first seen),
so an invoice ledger, a grants tracker, a mailbox and a code repository all get
the same measurements, with no source-specific code.

The honesty rules, which are the point of the module:

  - A time is only measured between two steps that BOTH carry a real timestamp,
    and the page says how many runs that was ("measured on 41 of 63 invoices").
    A step with no date is not given one.
  - Below `MIN_RUNS`, a measurement is not made at all. "Not enough data" is
    reported; it is never reported as zero.
  - Where a run inside a slow stretch also has a step no system recorded, the
    finding says so: part of that wait may be work that happened off-system.
  - Money is not produced. The data carries no cost, and a figure invented here
    would be the one confidently wrong number the whole engine exists to avoid.

Nothing here decides what to change. It states what the records show, with the
runs behind each figure, so a person can.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import median
from typing import Optional

from induction.steps.variants import shape

# Fewer runs than this and a rate or a median is an anecdote, not a finding.
MIN_RUNS = 5

# A wait is only called THE bottleneck when it is a real share of the whole.
_BOTTLENECK_SHARE = 0.30
_EXCEPTION_RATE = 0.15
_REWORK_RATE = 0.10


def parse_ts(ts: Optional[str]) -> tuple[Optional[datetime], bool]:
    """(`datetime` in UTC, has_time_of_day) — or (None, False) when unreadable.

    Sources disagree on shape (`2024-01-05`, `2024-01-05T09:30:00Z`, offsets), so
    everything is normalised to naive UTC. A date-only value is kept as a date,
    and flagged, so a same-day gap reads as "same day" rather than "0 hours"."""
    if not ts:
        return None, False
    s = str(ts).strip()
    has_time = "T" in s or " " in s
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None, False
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt, has_time


def human_duration(seconds: float, precise: bool) -> str:
    """A duration a non-technical reader parses at a glance."""
    days = seconds / 86400
    if days < 1:
        if not precise:
            return "same day"
        hours = seconds / 3600
        if hours < 1:
            return "under an hour"
        return f"{round(hours)} hour{'s' if round(hours) != 1 else ''}"
    d = round(days, 1) if days < 10 else round(days)
    d = int(d) if float(d).is_integer() else d
    return f"{d} day{'s' if d != 1 else ''}"


def _pct(n: int, total: int) -> int:
    return round(100 * n / total) if total else 0


def _cap(step: str) -> str:
    return step[:1].upper() + step[1:] if step else step


def measure(runs: list[dict], canon: list[str], items: str = "runs", item: str = "run") -> dict:
    """Measure one process from its runs.

    `runs`: each `{"key": str, "steps": [(step_name, iso_ts_or_None), ...],
    "offsystem": bool}` — steps in the run's order, one entry per consecutive
    block of the same step (so a repeat later in the list is a return to it).
    `canon`: the process's usual route (its most common shape), or [] if none.

    Returns a JSON-ready dict: the metrics, the one headline finding, and the
    run keys behind each figure.
    """
    runs = [r for r in runs if r["steps"]]
    n = len(runs)
    out: dict = {"n_runs": n, "min_runs": MIN_RUNS, "enough": n >= MIN_RUNS,
                 "metrics": {}, "headline": None, "money": {
                     "value": None,
                     "why": "Your records carry no cost per step or per hour, so no "
                            "cost is shown. With a rate card, each wait above converts "
                            "directly into money."}}
    if n < MIN_RUNS:
        out["headline"] = {"kind": "insufficient", "text":
                           f"Only {n} {item if n == 1 else items} so far — too few to measure reliably "
                           f"(at least {MIN_RUNS} are needed).", "runs": [r["key"] for r in runs]}
        return out

    # ---- timing -----------------------------------------------------------
    precise = False
    cycle: dict[str, float] = {}                 # run key -> end-to-end seconds
    waits: dict[tuple, dict[str, float]] = defaultdict(dict)   # (A,B) -> key -> s
    for r in runs:
        timed = []
        for name, ts in r["steps"]:
            dt, has_time = parse_ts(ts)
            if dt is not None:
                timed.append((name, dt))
                precise = precise or has_time
        if len(timed) < 2:
            continue
        span = (timed[-1][1] - timed[0][1]).total_seconds()
        if span < 0:
            continue                             # out-of-order dates: not measured
        cycle[r["key"]] = span
        # Waits are between CONSECUTIVE steps of the run, both dated. A step with
        # no date breaks the chain rather than being bridged by a guess.
        seq = r["steps"]
        for (a, ta), (b, tb) in zip(seq, seq[1:]):
            da, _ = parse_ts(ta)
            db, _ = parse_ts(tb)
            if da is None or db is None or a == b:
                continue
            s = (db - da).total_seconds()
            if s >= 0 and r["key"] not in waits[(a, b)]:
                waits[(a, b)][r["key"]] = s

    offsystem = {r["key"] for r in runs if r.get("offsystem")}
    m = out["metrics"]
    if len(cycle) >= MIN_RUNS:
        vals = sorted(cycle.values())
        med = median(vals)
        slow = vals[min(len(vals) - 1, int(round(0.9 * (len(vals) - 1))))]
        m["cycle"] = {"median_s": med, "median": human_duration(med, precise),
                      "slowest_10pct": human_duration(slow, precise),
                      "measured": len(cycle), "of": n, "coverage_pct": _pct(len(cycle), n),
                      "runs": sorted(cycle)}
    else:
        m["cycle"] = {"insufficient": True, "measured": len(cycle), "of": n}

    # WHERE THE TIME GOES is ranked by the total time a wait costs across ALL
    # timed runs, not by its median. Ranked by median, a slow path taken by 13 of
    # 300 permit applications headlined as "83% of the time" while the typical
    # application finished in under an hour. Impact is length x frequency; a
    # sum over every run is exactly that, and its share is of the time spent on
    # the whole process, so a rare detour cannot claim to be the process.
    all_time = sum(cycle.values())
    ranked = []
    for (a, b), per in waits.items():
        if len(per) < MIN_RUNS:
            continue
        med = median(per.values())
        total = sum(per.values())
        ranked.append({"from": a, "to": b, "median_s": med,
                       "median": human_duration(med, precise),
                       "total_s": total,
                       "share_pct": round(100 * total / all_time) if all_time else 0,
                       "measured": len(per), "of_timed": len(cycle),
                       "runs": sorted(per),
                       "offsystem": len(offsystem & set(per))})
    ranked.sort(key=lambda w: (-w["total_s"], -w["median_s"], w["from"], w["to"]))
    m["waits"] = ranked
    m["longest_wait"] = ranked[0] if ranked else {"insufficient": True}

    # ---- route ------------------------------------------------------------
    usual = tuple(canon)
    if usual:
        off = [r for r in runs if shape(tuple(s for s, _ in r["steps"])) != usual]
        skipped = Counter()
        for r in off:
            present = {s for s, _ in r["steps"]}
            for st in usual:
                if st not in present:
                    skipped[st] += 1
        top = skipped.most_common(1)
        m["off_route"] = {"count": len(off), "of": n, "pct": _pct(len(off), n),
                          "runs": sorted(r["key"] for r in off),
                          "most_skipped": ({"step": top[0][0], "count": top[0][1]} if top else None)}
        last = usual[-1]
        unfinished = [r for r in runs if last not in {s for s, _ in r["steps"]}]
        m["not_completed"] = {"count": len(unfinished), "of": n, "pct": _pct(len(unfinished), n),
                              "final_step": last, "runs": sorted(r["key"] for r in unfinished)}
    else:
        m["off_route"] = {"no_usual_route": True, "of": n}
        m["not_completed"] = {"no_usual_route": True, "of": n}

    # ---- rework: a step returned to after something else happened ------------
    rework = []
    for r in runs:
        names = [s for s, _ in r["steps"]]
        if len(names) != len(set(names)):
            again = Counter(names)
            rework.append((r["key"], sorted(s for s, c in again.items() if c > 1)))
    steps_redone = Counter(s for _, ss in rework for s in ss)
    m["rework"] = {"count": len(rework), "of": n, "pct": _pct(len(rework), n),
                   "runs": sorted(k for k, _ in rework),
                   "most_redone": (steps_redone.most_common(1)[0][0] if steps_redone else None)}

    out["headline"] = _headline(m, items)
    out["precise_times"] = precise
    return out


def _headline(m: dict, items: str) -> dict:
    """The single most decision-worthy fact, as a sentence. Chosen by fixed rules
    in a fixed order — never by a model — so it is the same on every run over
    the same data and every number in it is one of the metrics above."""
    lw = m.get("longest_wait") or {}
    # A bottleneck is the slowest of SEVERAL waits. With only one wait measured it
    # is trivially "all of the time", which says nothing about where to look.
    if (not lw.get("insufficient") and len(m.get("waits") or []) >= 2
            and lw.get("share_pct", 0) >= _BOTTLENECK_SHARE * 100):
        text = (f"The wait from {_cap(lw['from'])} to {_cap(lw['to'])} accounts for "
                f"{lw['share_pct']}% of all the time spent on these {items} "
                f"(median {lw['median']}, in {lw['measured']} of {lw['of_timed']}).")
        caveat = None
        if lw.get("offsystem"):
            caveat = (f"{lw['offsystem']} of these {items} also have a step no system recorded, "
                      f"so part of this wait may be work done outside the systems.")
        return {"kind": "bottleneck", "text": text, "caveat": caveat, "runs": lw["runs"],
                "coverage": f"Measured on {lw['measured']} {items} with dates on both steps."}
    off = m.get("off_route") or {}
    if off.get("count") and off["pct"] >= _EXCEPTION_RATE * 100:
        ms = off.get("most_skipped")
        tail = f" The step most often skipped is {_cap(ms['step'])} ({ms['count']})." if ms else ""
        return {"kind": "exception", "runs": off["runs"], "caveat": None,
                "text": f"{off['count']} of {off['of']} {items} ({off['pct']}%) do not follow the most common route.{tail}",
                "coverage": f"Counted over all {off['of']} {items}."}
    rw = m.get("rework") or {}
    if rw.get("count") and rw["pct"] >= _REWORK_RATE * 100:
        return {"kind": "rework", "runs": rw["runs"], "caveat": None,
                "text": (f"{rw['count']} of {rw['of']} {items} ({rw['pct']}%) return to a step already done"
                         + (f", most often {_cap(rw['most_redone'])}." if rw.get("most_redone") else ".")),
                "coverage": f"Counted over all {rw['of']} {items}."}
    return {"kind": "steady", "runs": [], "caveat": None,
            "text": "No single delay, exception or rework pattern stands out in these records.",
            "coverage": ""}
