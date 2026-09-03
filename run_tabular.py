#!/usr/bin/env python3
"""Run the engine on a spreadsheet corpus — the thin, non-git customer.

The SAME pipeline as git, on CSV/Excel exports. Ships two demo corpora and picks
the right column mapping from whichever files are in --dir:

    python run_tabular.py                       # samples/finance (invoices + bank)
    python run_tabular.py --dir samples/grants  # a grant-making tracker
    python run_tabular.py --profile accounting  # friendly names for finance
    python run_tabular.py --no-llm              # deterministic baseline, no AI naming
    python run_tabular.py --xlsx                # read .xlsx (needs openpyxl)

The model tier (AI naming + activity abstraction) is on by default; it runs when
ANTHROPIC_API_KEY is set and otherwise downshifts to the deterministic baseline
and says so. `--no-llm` forces that baseline.

A new customer's tracker is a new `TableSpec` (see sources_for) — not new code.
Emits out/model.json + out/inspector.html.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from induction.abstraction import infer_activities
from induction.adapters.tabular import EventCol, TableSpec
from induction.emit import write_json
from induction.inspector import write_html
from induction.model_tier import ModelTier, resolve
from induction.naming import infer_names
from induction.pipeline import run_tabular_pipeline
from induction.profiles import ACCOUNTING_PROFILE, GENERIC_PROFILE


def sources_for(base: Path, ext: str):
    """Pick the column mapping from the files present. Each TableSpec is config,
    not code — copy one and rename the columns for your own tracker."""
    if (base / f"grants.{ext}").exists():
        grants = TableSpec(
            source="sheet:acme/grants", entity_type="grant", id_column="grant_id",
            event_columns=[EventCol("applied", "applied", "owner"),
                           EventCol("reviewed", "reviewed"),
                           EventCol("decided", "decided"),
                           EventCol("paid", "paid"),
                           EventCol("reported", "reported")],
            status_column="status", attr_columns=["amount", "owner"])
        return [(grants, base / f"grants.{ext}")]

    if (base / f"invoices.{ext}").exists():
        invoices = TableSpec(
            source="sheet:acme-finance/invoices", entity_type="invoice", id_column="invoice_id",
            event_columns=[EventCol("raised", "raised_date", "raised_by"),
                           EventCol("submitted", "submitted_date"),
                           EventCol("approved", "approved_date", "approved_by"),
                           EventCol("paid", "paid_date")],
            status_column="status", attr_columns=["client", "amount", "po_number"])
        payments = TableSpec(
            source="sheet:acme-finance/payments", entity_type="payment", id_column="payment_ref",
            event_columns=[EventCol("settled", "paid_date")], attr_columns=["amount"],
            ref_columns=[{"column": "invoice_id", "target_type": "invoice"}])
        return [(invoices, base / f"invoices.{ext}"), (payments, base / f"payments.{ext}")]

    raise FileNotFoundError(f"no grants.{ext} or invoices.{ext} in {base}")


def _run_detected(args, tier: ModelTier) -> int:
    """Read a file whose shape we were not told, and say what we decided.

    The detection is an inference like any other here, so it is printed before
    anything is induced from it: a reader who disagrees can see the measurement,
    not just the verdict.
    """
    from induction.adapters.tabular import Detection, detect, read_rows
    from induction.model import direct
    path = Path(args.file)
    if not path.exists():
        print(f"[run] no such file: {path}", file=sys.stderr)
        return 2
    if args.case_column and args.activity_column and args.timestamp_column:
        # Some files hold more than one valid reading — a UI recording is both a
        # log of clicks and a log of the business steps those clicks realise, and
        # only a person knows which one is wanted. Detection is for a file with
        # one obvious reading; this is for the rest.
        from induction.adapters.tabular import EventLogSpec
        cols = [c for c in (read_rows(path)[:1] or [{}])[0]]
        spec = EventLogSpec(
            source=f"log:{path.stem}", entity_type=args.entity,
            case_id_column=args.case_column, activity_column=args.activity_column,
            timestamp_column=args.timestamp_column, actor_column=args.actor_column,
            attr_columns=[c for c in cols if c not in (
                args.case_column, args.activity_column, args.timestamp_column,
                args.actor_column)])
        print(f"[run] reading as an event log you declared: case={args.case_column!r} "
              f"activity={args.activity_column!r} timestamp={args.timestamp_column!r}")
        found = Detection("long", spec, rationale="columns declared, not detected",
                          confidence=direct())
    else:
        found = detect(path, entity_type=args.entity)
    if not found.mode:
        print(f"[run] could not read {path.name} as a process table.\n"
              f"      {found.rationale}", file=sys.stderr)
        return 2
    if found.rationale != "columns declared, not detected":
        print(f"[run] {found.rationale}  [{found.confidence.tier.label}]")

    m = run_tabular_pipeline([(found.spec, path)], slug=path.stem,
                             profile=GENERIC_PROFILE, max_cases=args.max_cases)
    names = infer_names(m, enable=tier.names_enable())
    activities = infer_activities(m, tier.mapper(), tier.classifier())
    out_dir = Path(args.out_dir)
    json_path = write_json(m, out_dir / "model.json")
    html_path = write_html(m, out_dir / "inspector.html", names=names, activities=activities)
    print(_summary(m))
    print(f"[run] wrote {json_path}  ({json_path.stat().st_size // 1024} KB)")
    print(f"[run] wrote {html_path}  — open it in a browser")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="samples/finance")
    ap.add_argument("--file", help="a single CSV/XLSX to read by DETECTING its shape "
                                   "(an event log or a tracker export) instead of a built-in spec")
    ap.add_argument("--entity", default="case", help="what one run is called with --file")
    ap.add_argument("--max-cases", type=int, help="slice cap for a large event log")
    ap.add_argument("--case-column", help="name the case column instead of detecting it")
    ap.add_argument("--activity-column", help="name the activity column")
    ap.add_argument("--timestamp-column", help="name the timestamp column")
    ap.add_argument("--actor-column", help="name the actor column")
    ap.add_argument("--xlsx", action="store_true", help="read .xlsx instead of .csv (needs openpyxl)")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--profile", choices=["generic", "accounting"], default="generic")
    ap.add_argument("--names", choices=["auto", "off", "llm"], default="auto",
                    help="model-tier naming/abstraction (default: auto — on if ANTHROPIC_API_KEY "
                         "is set, else the deterministic baseline). 'llm' insists on it.")
    ap.add_argument("--no-llm", action="store_true",
                    help="force the deterministic baseline (raw verbs, no AI naming/abstraction)")
    args = ap.parse_args(argv)

    tier = resolve(args.names, no_llm=args.no_llm)

    if args.file:
        return _run_detected(args, tier)

    base = Path(args.dir)
    ext = "xlsx" if args.xlsx else "csv"
    try:
        sources = sources_for(base, ext)
    except FileNotFoundError as e:
        print(f"[run] {e}", file=sys.stderr)
        return 2
    for _spec, path in sources:
        if not path.exists():
            print(f"[run] missing sheet: {path}", file=sys.stderr)
            return 2

    profile = {"generic": GENERIC_PROFILE, "accounting": ACCOUNTING_PROFILE}[args.profile]
    slug = base.name
    print(f"[run] inducing processes from {args.dir} ({ext}, profile: {args.profile}) "
          f"· model tier: {tier.label} ...")
    m = run_tabular_pipeline(sources, slug=slug, profile=profile)

    names = infer_names(m, enable=tier.names_enable())
    activities = infer_activities(m, tier.mapper(), tier.classifier())
    out_dir = Path(args.out_dir)
    json_path = write_json(m, out_dir / "model.json")
    html_path = write_html(m, out_dir / "inspector.html", names=names, activities=activities)
    print(_summary(m))
    print(f"[run] wrote {json_path}  ({json_path.stat().st_size // 1024} KB)")
    print(f"[run] wrote {html_path}  — open it in a browser")
    return 0


def _summary(m) -> str:
    from collections import Counter
    lines = ["", "  Induced model:"]
    for k in m.kinds:
        tag = "  [flagged: looks like a process, isn't]" if k.rejected else ""
        common = next((v for v in k.variants if v.role == "common"), None)
        cp = " → ".join(common.signature) if common and common.signature else "—"
        lines.append(f"    {k.name:<28} {len(k.case_ids):>3} runs · common: {cp}{tag}")
    gk = Counter(g.kind for g in m.gaps)
    lines += ["",
              f"    orphans : {len(m.orphans)}   gaps : {len(m.gaps)} {dict(gk)}   "
              f"order-unknown : {sum(1 for c in m.cases.values() if c.order_status=='unknown')}",
              ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
