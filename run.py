#!/usr/bin/env python3
"""Run the whole induction engine on the cached corpus, end-to-end, one command.

    python run.py                       # pallets/flask, thick + thin sources
    python run.py --slug pallets/click  # a different (held-out) repo
    python run.py --no-thin             # git only, skip the changelog source

Emits ``out/model.json`` (the complete induced model) and ``out/inspector.html``
(the thin inspector) and tells you where they are. If the raw cache is missing
it points you at `ingest.py` rather than silently doing nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from induction.emit import write_json
from induction.inspector import write_html
from induction.pipeline import run_pipeline


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", default="pallets/flask")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--no-thin", action="store_true", help="skip the thin changelog source")
    args = ap.parse_args(argv)

    key = args.slug.replace("/", "__")
    if not Path(args.raw_dir, f"{key}.commits.jsonl").exists():
        print(f"[run] no cached corpus for {args.slug} at {args.raw_dir}/.\n"
              f"      Clone and ingest it first, e.g.:\n"
              f"        GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1000 "
              f"https://github.com/{args.slug} data/corpus/{args.slug.split('/')[-1]}\n"
              f"        python ingest.py --repo-path data/corpus/{args.slug.split('/')[-1]} "
              f"--slug {args.slug}", file=sys.stderr)
        return 2

    print(f"[run] inducing processes from {args.slug} ...")
    m = run_pipeline(args.slug, args.raw_dir, with_thin=not args.no_thin)

    out_dir = Path(args.out_dir)
    json_path = write_json(m, out_dir / "model.json")
    html_path = write_html(m, out_dir / "inspector.html")

    s = _summary(m)
    print(s)
    print(f"[run] wrote {json_path}  ({json_path.stat().st_size // 1024} KB)")
    print(f"[run] wrote {html_path}  — open it in a browser")
    return 0


def _summary(m) -> str:
    lines = ["", "  Induced model:"]
    for k in m.kinds:
        tag = "  [REJECTED: looks like a process, isn't]" if k.rejected else ""
        common = next((v for v in k.variants if v.role == "common"), None)
        cp = " → ".join(common.signature) if common and common.signature else "—"
        lines.append(f"    {k.name:<42} {len(k.case_ids):>4} runs  · common: {cp}{tag}")
    lines += [
        "",
        f"    orphans (joined to nothing) : {len(m.orphans)}",
        f"    gaps (off-system, inferred) : {len(m.gaps)}",
        f"    same-activity merges        : {len(m.merges)}",
        f"    order: unknown (thin source): {sum(1 for c in m.cases.values() if c.order_status=='unknown')}",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
