#!/usr/bin/env python3
"""Measure whether the typed tier's answers hold still.

    python3 check_jev.py --path samples/enron
    python3 check_jev.py --path samples/enron --runs 15 --states 5

Every threshold in the typed tier — the gate's bar, the label screen's, the
ambiguity bar — assumes the number it compares against means something stable.
That assumption has never been checked here, and it is checkable: ask the same
question about the same state several times and see how far the answer moves.

Two numbers per question:

  **std** — how far the probability moved across the repeats. Small means the
  bar you set is the bar that applies. Large means the same record will fall on
  different sides of it on different runs, and no amount of tuning fixes that.

  **flips** — how often the DECISION changed, which is the number that actually
  costs you. A question can wobble by 0.2 and never cross its bar; another can
  wobble by 0.02 sitting right on it and change what the engine does every time.
  A question with a low std and a high flip count is one whose bar is in the
  wrong place, not one that is unstable.

What this cannot tell you is whether any answer is RIGHT. A question that
returns the same wrong number fifteen times scores perfectly here. Repeatability
is a precondition for trusting a threshold, not evidence that the threshold is
well placed — for that, compare the screen's verdicts against what
`_detach_lonely_steps` and `_process_of_case` later prove.

Needs OPENROUTER_API_KEY and spends real calls: `runs × states × batteries`.
The defaults are small on purpose.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path


def _states(path: Path, want: int) -> list[dict]:
    """A spread of real record states from a corpus, not invented ones.

    Drawn through the same `_record_state` the engine uses, so what is measured
    is what the engine actually sends — a hand-written state would measure a
    question this code never asks.
    """
    from induction.abstraction import _record_state, _spread
    from induction.adapters import email_mbox
    from induction.pipeline import induce

    m = induce(email_mbox.load(path, slug=path.name, max_messages=400), slug=path.name)
    entities = {e.id: e for e in m.shaped.entities}
    states = [_record_state(ev, m, entities) for ev in m.shaped.events]
    states = [st for st in states if st]
    return _spread(states, want)


def _repeat(jev, state, questions, runs: int) -> dict[str, list]:
    """`runs` answers per question, in order."""
    out: dict[str, list] = {name: [] for name in questions}
    for _ in range(runs):
        answers = jev.ask(state, questions)
        for name in questions:
            out[name].append(answers.get(name))
    return out


def _summarise(name: str, answers: list, bar: float) -> tuple:
    """(question, n, mean, std, flips, decision) for one question's repeats."""
    values, decisions = [], []
    for a in answers:
        if a is None:
            decisions.append(None)
            continue
        if a.noul is not None:
            values.append(a.noul)
            decisions.append(a.noul >= bar)
        elif a.choice is not None:
            values.append(a.confidence if a.confidence is not None else float("nan"))
            decisions.append(a.choice)
        elif a.score is not None:
            values.append(a.score)
            decisions.append(a.score >= bar)
    if not values:
        return name, 0, None, None, None, "no answer"
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    distinct = {d for d in decisions if d is not None}
    flips = len(distinct) - 1 if distinct else 0
    return (name, len(values), statistics.fmean(values), std, flips,
            " / ".join(str(d) for d in sorted(distinct, key=str)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", required=True, help="a maildir / mbox corpus to draw states from")
    ap.add_argument("--runs", type=int, default=10, help="repeats per question (default 10)")
    ap.add_argument("--states", type=int, default=3, help="records to test (default 3)")
    args = ap.parse_args(argv)

    from induction.jev_call import Jev, have_key
    if not have_key():
        print("[check] OPENROUTER_API_KEY is not set — nothing to measure.", file=sys.stderr)
        return 2

    from induction.jev_reading import (_LABEL_QUESTIONS, _SIGNAL_QUESTIONS, _LABEL_BAR,
                                       SignalFilter)
    jev = Jev(log=lambda m: print(f"  {m}", file=sys.stderr))
    states = _states(Path(args.path), args.states)
    if not states:
        print("[check] no records read from that corpus.", file=sys.stderr)
        return 2

    print(f"corpus: {args.path} · {len(states)} record(s) × {args.runs} repeats\n")
    print(f"{'question':<34}{'n':>4}{'mean':>8}{'std':>8}{'flips':>7}  decisions")
    print("-" * 92)

    gate_bar = SignalFilter(jev=jev).bar
    for i, state in enumerate(states, 1):
        subject = str(state.get("subject") or state.get("title") or "")[:40]
        print(f"[record {i}] {subject}")
        for name, answers in _repeat(jev, {"record": state},
                                     _SIGNAL_QUESTIONS, args.runs).items():
            q, n, mean, std, flips, decision = _summarise(name, answers, gate_bar)
            if n:
                print(f"  {q:<32}{n:>4}{mean:>8.3f}{std:>8.4f}{flips:>7}  {decision}")

    # The label screen is asked about labels, not records, so it gets its own pass.
    print("\n[labels]")
    for process, label in (("Customer Gas Nomination", "Deal entered in Sitara"),
                           ("Customer Gas Nomination", "Monthly requirements faxed in"),
                           ("Gas Invoice Settlement", "Invoice generated for volumes")):
        print(f"  {label!r} in {process}")
        answers = _repeat(jev, {"process": process, "label": label,
                                "vocabulary": ["Customer Gas Nomination",
                                               "Gas Invoice Settlement"]},
                          _LABEL_QUESTIONS, args.runs)
        for name, got in answers.items():
            q, n, mean, std, flips, decision = _summarise(name, got, _LABEL_BAR)
            if n:
                print(f"    {q:<30}{n:>4}{mean:>8.3f}{std:>8.4f}{flips:>7}  {decision}")

    print(f"\n{jev.calls} call(s) made"
          + (f", {jev.skipped} failed" if jev.skipped else ""))
    print("\nA question with a high flip count is one whose bar sits where its answers\n"
          "land. Low std with flips means the bar is misplaced, not the question.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
