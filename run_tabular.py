#!/usr/bin/env python3
"""Run the engine on a spreadsheet corpus — the thin, non-git customer.

Demonstrates the SAME pipeline on an accounting firm's invoice tracker (CSV or
Excel) plus a bank-payments export, end-to-end:

    python run_tabular.py                      # samples/finance, generic (unnamed)
    python run_tabular.py --profile accounting # friendly names for this domain
    python run_tabular.py --xlsx               # read the .xlsx sheets (needs openpyxl)

Emits out/model.json + out/inspector.html, exactly like the git path. The point
is that nothing below the adapter changed — new source, same engine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from induction.adapters.tabular import EventCol, TableSpec
from induction.emit import write_json
from induction.inspector import write_html
from induction.pipeline import run_tabular_pipeline
from induction.profiles import ACCOUNTING_PROFILE, GENERIC_PROFILE


def finance_sources(base: Path, ext: str):
    """The two sheets and how their columns map to the canonical model. A new
    tracker (tickets, onboarding, cases) is a new spec like these — not code."""
    invoices = TableSpec(
        source="sheet:acme-finance/invoices",
        entity_type="invoice",
        id_column="invoice_id",
        event_columns=[
            EventCol("raised", "raised_date", "raised_by"),
            EventCol("submitted", "submitted_date"),        # no actor column — stays None
            EventCol("approved", "approved_date", "approved_by"),
            EventCol("paid", "paid_date"),                  # no actor column — stays None
        ],
        status_column="status",
        attr_columns=["client", "amount", "po_number"],
    )
    payments = TableSpec(
        source="sheet:acme-finance/payments",
        entity_type="payment",
        id_column="payment_ref",
        event_columns=[EventCol("settled", "paid_date")],
        attr_columns=["amount"],
        ref_columns=[{"column": "invoice_id", "target_type": "invoice"}],
    )
    return [(invoices, base / f"invoices.{ext}"), (payments, base / f"payments.{ext}")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="samples/finance", help="folder with the sheet files")
    ap.add_argument("--xlsx", action="store_true", help="read .xlsx instead of .csv (needs openpyxl)")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--profile", choices=["generic", "accounting"], default="generic")
    args = ap.parse_args(argv)

    base = Path(args.dir)
    ext = "xlsx" if args.xlsx else "csv"
    sources = finance_sources(base, ext)
    for _spec, path in sources:
        if not path.exists():
            print(f"[run] missing sheet: {path}", file=sys.stderr)
            return 2

    profile = {"generic": GENERIC_PROFILE, "accounting": ACCOUNTING_PROFILE}[args.profile]
    print(f"[run] inducing processes from {args.dir} ({ext}, profile: {args.profile}) ...")
    m = run_tabular_pipeline(sources, slug="acme-finance", profile=profile)

    out_dir = Path(args.out_dir)
    json_path = write_json(m, out_dir / "model.json")
    html_path = write_html(m, out_dir / "inspector.html")
    print(_summary(m))
    print(f"[run] wrote {json_path}  ({json_path.stat().st_size // 1024} KB)")
    print(f"[run] wrote {html_path}  — open it in a browser")
    return 0


def _summary(m) -> str:
    lines = ["", "  Induced model:"]
    for k in m.kinds:
        tag = "  [REJECTED: looks like a process, isn't]" if k.rejected else ""
        common = next((v for v in k.variants if v.role == "common"), None)
        cp = " → ".join(common.signature) if common and common.signature else "—"
        lines.append(f"    {k.name:<34} {len(k.case_ids):>3} runs · common: {cp}{tag}")
    from collections import Counter
    gk = Counter(g.kind for g in m.gaps)
    lines += [
        "",
        f"    orphans (joined to nothing) : {len(m.orphans)}",
        f"    gaps (inferred)             : {len(m.gaps)}  {dict(gk)}",
        f"    order: unknown              : {sum(1 for c in m.cases.values() if c.order_status=='unknown')}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
