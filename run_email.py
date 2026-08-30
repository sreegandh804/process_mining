#!/usr/bin/env python3
"""Run the engine on a mailbox — the corpus the fuzzy pass exists for.

Point it at an Enron-style maildir tree, an .mbox, or the Kaggle emails.csv:

    python run_email.py --path ~/enron/maildir/lay-k          # one exec's mail
    python run_email.py --path emails.csv --max-messages 3000 # Kaggle CSV, sliced
    python run_email.py --path inbox.mbox --names llm

Realism earns more than volume (per the brief): a few thousand messages from a
few mailboxes shows the threading, the fuzzy cross-thread joins, the automated-
notice rejects and the one-off orphans without waiting on all 500k. Emits
out/model.json + out/inspector.html.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from induction.adapters import email_mbox
from induction.emit import write_json
from induction.inspector import write_html
from induction.naming import infer_names
from induction.pipeline import induce


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", required=True, help="maildir dir, .mbox file, or emails.csv")
    ap.add_argument("--slug", default=None, help="a name for this mailbox (default: folder name)")
    ap.add_argument("--max-messages", type=int, default=3000, help="slice cap (newest first not guaranteed)")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--names", choices=["off", "llm"], default="off")
    args = ap.parse_args(argv)

    p = Path(args.path)
    if not p.exists():
        print(f"[run] no such path: {p}", file=sys.stderr)
        return 2
    slug = args.slug or p.stem

    print(f"[run] reading mail from {p} (cap {args.max_messages}) …")
    shaped = email_mbox.load(p, slug=slug, max_messages=args.max_messages)
    n_messages = sum(1 for e in shaped.entities if e.type == "email")
    if not n_messages:
        print("[run] no messages parsed — is this a maildir/.mbox/.csv?", file=sys.stderr)
        return 2

    m = induce(shaped, slug=slug, manifest={"source_kind": "email", "n_messages": n_messages})
    names = infer_names(m, enable=(args.names == "llm"))

    out = Path(args.out_dir)
    write_json(m, out / "model.json")
    write_html(m, out / "inspector.html", names=names)

    print(f"  {n_messages} messages · {len(m.cases)} threads/runs · "
          f"{sum(1 for k in m.kinds if k.rejected)} flagged kinds · {len(m.orphans)} orphans")
    fuzzy = sum(1 for c in m.cases.values()
                if "no shared key" in (c.confidence.rationale or ""))
    print(f"  fuzzy (no-shared-key) joins: {fuzzy}")
    print(f"[run] wrote {out}/inspector.html — open it in a browser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
