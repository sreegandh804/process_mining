#!/usr/bin/env python3
"""Turn an XES event log (the process-mining standard format) into the CSV the
engine reads with `run_tabular.py --file`.

Public benchmark logs (the BPI Challenges, 4TU.ResearchData) ship as XES, often
gzipped. This streams the file, so a large log never has to fit in memory, and
`--max-cases` keeps a demo-sized slice:

    python3 tools/xes_to_csv.py BPI_Challenge_2013_incidents.xes.gz incidents.csv --max-cases 300
    python3 run_tabular.py --file incidents.csv --entity incident

Output columns: case_id, activity, timestamp, resource, plus `lifecycle` and any
other event-level string attributes named with --extra. Only `complete` events
are kept by default (a start/complete pair is one step, not two); pass
--all-lifecycle to keep every transition. Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
import xml.etree.ElementTree as ET


def _open(path: str):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]          # XES files may carry a namespace


def convert(src: str, dst: str, max_cases: int | None = None,
            all_lifecycle: bool = False, extra: tuple[str, ...] = ()) -> tuple[int, int]:
    cols = ["case_id", "activity", "timestamp", "resource", "lifecycle", *extra]
    n_cases = n_events = 0
    with _open(src) as f, open(dst, "w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=cols)
        w.writeheader()
        case_id, depth = None, 0
        for event, el in ET.iterparse(f, events=("start", "end")):
            tag = _local(el.tag)
            if event == "start":
                if tag == "trace":
                    case_id, depth = None, 0
                elif tag == "event":
                    depth += 1
                continue
            if tag == "string" and el.get("key") == "concept:name" and depth == 0 and case_id is None:
                case_id = el.get("value")          # the trace's own name, not an event's
            elif tag == "event":
                depth -= 1
                attrs = {_local(a.tag) + ":" + (a.get("key") or ""): a.get("value") for a in el}
                get = lambda k: next((v for kk, v in attrs.items() if kk.endswith(":" + k)), "")
                life = get("lifecycle:transition")
                if all_lifecycle or not life or life.lower() == "complete":
                    w.writerow({"case_id": case_id, "activity": get("concept:name"),
                                "timestamp": get("time:timestamp"), "resource": get("org:resource"),
                                "lifecycle": life, **{k: get(k) for k in extra}})
                    n_events += 1
                el.clear()
            elif tag == "trace":
                n_cases += 1
                el.clear()
                if max_cases and n_cases >= max_cases:
                    break
    return n_cases, n_events


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help=".xes or .xes.gz")
    ap.add_argument("dst", help="CSV to write")
    ap.add_argument("--max-cases", type=int, help="stop after this many cases (a demo-sized slice)")
    ap.add_argument("--all-lifecycle", action="store_true", help="keep start/assign/… transitions too")
    ap.add_argument("--extra", nargs="*", default=[], help="more event attribute keys to keep as columns")
    args = ap.parse_args(argv)
    cases, events = convert(args.src, args.dst, args.max_cases, args.all_lifecycle, tuple(args.extra))
    print(f"[xes] wrote {args.dst}: {cases} cases, {events} events", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
