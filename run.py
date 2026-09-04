#!/usr/bin/env python3
"""Run the whole induction engine on the cached corpus, end-to-end, one command.

    python run.py                       # pallets/flask, thick + thin sources
    python run.py --slug pallets/click  # a different (held-out) repo
    python run.py --no-thin             # git only, skip the changelog source
    python run.py --with-github         # add the Issues/PR corpus (see ingest_github.py)
    python run.py --no-llm              # deterministic baseline, no AI naming/abstraction

The model tier (AI naming + activity abstraction) is **on by default**: it runs
when ANTHROPIC_API_KEY is set, and otherwise downshifts to the deterministic
baseline and says so. `--no-llm` forces that baseline.

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
    ap.add_argument("--with-github", action="store_true",
                    help="also load the cached GitHub Issues/PR corpus, so cross-source "
                         "and fuzzy correlation run against it (see ingest_github.py)")
    ap.add_argument("--profile", choices=["generic", "git", "auto"], default="generic",
                    help="vocabulary overlay. 'generic' (default): unnamed, source-agnostic "
                         "kinds/activities. 'git': friendly names for a git corpus. 'auto': "
                         "pick by source.")
    ap.add_argument("--names", choices=["auto", "off", "llm"], default="auto",
                    help="model-tier naming/abstraction (default: auto — on if ANTHROPIC_API_KEY "
                         "is set, else the deterministic baseline). 'llm' insists on it.")
    ap.add_argument("--no-llm", action="store_true",
                    help="force the deterministic baseline (raw verbs, no AI naming/abstraction)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="stream each inferred join / kind / gap as it is decided")
    ap.add_argument("--quiet", action="store_true", help="suppress stage progress")
    args = ap.parse_args(argv)

    from induction.progress import from_flags
    prog = from_flags(quiet=args.quiet, verbose=args.verbose)

    from induction.profiles import GENERIC_PROFILE, GIT_PROFILE
    profile = {"generic": GENERIC_PROFILE, "git": GIT_PROFILE, "auto": "auto"}[args.profile]

    key = args.slug.replace("/", "__")
    if not Path(args.raw_dir, f"{key}.commits.jsonl").exists():
        print(f"[run] no cached corpus for {args.slug} at {args.raw_dir}/.\n"
              f"      Clone and ingest it first, e.g.:\n"
              f"        GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1000 "
              f"https://github.com/{args.slug} data/corpus/{args.slug.split('/')[-1]}\n"
              f"        python ingest.py --repo-path data/corpus/{args.slug.split('/')[-1]} "
              f"--slug {args.slug}", file=sys.stderr)
        return 2

    # Resolve the model tier only once the run can actually proceed, so a missing
    # cache does not also print a downshift note. May exit(2) on an insisted --names llm.
    from induction.model_tier import resolve
    tier = resolve(args.names, no_llm=args.no_llm)

    print(f"[run] inducing processes from {args.slug} (profile: {args.profile}) "
          f"· model tier: {tier.label} ...")
    m = run_pipeline(args.slug, args.raw_dir, with_thin=not args.no_thin,
                     with_github=args.with_github, profile=profile, progress=prog)

    from induction.abstraction import infer_activities
    from induction.naming import infer_names
    # Abstraction runs BEFORE naming: reading the records can re-segment the
    # corpus (`abstraction._reproject`), and a namer that ran first would hand
    # back names keyed on kind ids that no longer mean the same thing.
    # The verb map groups git's own verbs into activities; the record reader is
    # gated on records-per-activity and simply won't fire on a git corpus.
    activities = infer_activities(m, tier.mapper(log=prog), tier.classifier(log=prog), log=prog)
    names = infer_names(m, enable=tier.names_enable(), log=prog)
    out_dir = Path(args.out_dir)
    json_path = write_json(m, out_dir / "model.json")
    html_path = write_html(m, out_dir / "inspector.html", names=names, activities=activities)

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
