#!/usr/bin/env python3
"""Run the engine on a spreadsheet corpus — the thin, non-git customer.

The SAME pipeline as git, on CSV/Excel exports. Ships two demo corpora and picks
the right column mapping from whichever files are in --dir:

    python run_tabular.py                       # samples/finance (invoices + bank)
    python run_tabular.py --dir samples/grants  # a grant-making tracker
    python run_tabular.py --profile accounting  # friendly names for finance
    python run_tabular.py --names llm           # let Claude name the processes/steps
    python run_tabular.py --xlsx                # read .xlsx (needs openpyxl)

A new customer's tracker is a new `TableSpec` (see sources_for) — not new code.
Emits out/model.json + out/inspector.html.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from induction.adapters.tabular import EventCol, TableSpec
from induction.emit import write_json
from induction.inspector import write_html
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="samples/finance")
    ap.add_argument("--xlsx", action="store_true", help="read .xlsx instead of .csv (needs openpyxl)")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--profile", choices=["generic", "accounting"], default="generic")
    ap.add_argument("--names", choices=["off", "llm"], default="off",
                    help="'llm' names kinds/steps via the Claude API (needs ANTHROPIC_API_KEY)")
    args = ap.parse_args(argv)

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
    print(f"[run] inducing processes from {args.dir} ({ext}, profile: {args.profile}) ...")
    m = run_tabular_pipeline(sources, slug=slug, profile=profile)

    names = infer_names(m, enable=(args.names == "llm"))
    out_dir = Path(args.out_dir)
    json_path = write_json(m, out_dir / "model.json")
    html_path = write_html(m, out_dir / "inspector.html", names=names)
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
