#!/usr/bin/env python3
"""Run the engine over TWO sources at once — a team's GitHub + their mailbox.

This is the corpus the brief is really about: the process lives across systems,
and *nothing announces the relationship*. People email "the export is spinning
again", not "re: #21". So the cross-source joins are earned, at the honest tier
the evidence supports:

    deterministic  a merge that cites an issue, a foreign key           joined
    shared tokens  the same rare words + the same week/people           heuristic
    same *meaning*  paraphrase a language model can read, tokens can't   model

The model tier is **on by default** — the same Claude judge/namer/record-reader
that turns "sent → replied" into real activities. With no ANTHROPIC_API_KEY it
downshifts to the deterministic, offline baseline and says so; `--no-llm` forces
that baseline outright.

    # built-in realistic fixture, offline stand-in model, no keys, no data:
    python run_combined.py --demo

    # your own data (model tier runs if a key is present, else deterministic):
    python run_combined.py --github portal.github.json --mail team.mbox

    # force the deterministic, offline baseline:
    python run_combined.py --github portal.github.json --mail team.mbox --no-llm

    # add the embedding shortlist (needs VOYAGE_API_KEY too):
    python run_combined.py --github portal.github.json --mail team.mbox --hybrid

`--github` is a JSON payload ({"slug","issues","pulls"}, the shape
`ingest_github.py` writes); `--mail` is a maildir dir, an .mbox, or a CSV with a
raw-message column. Emits out/model.json + out/inspector.html.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from induction.abstraction import infer_activities
from induction.adapters import Shaped, email_mbox, github_api
from induction.emit import write_json
from induction.inspector import write_html
from induction.model_tier import resolve
from induction.naming import infer_names
from induction.pipeline import induce
from induction.progress import from_flags
from induction.semantic import SemanticProvider
from induction.steps.correlate import CorrelationPolicy


def _load_demo() -> tuple[Shaped, str]:
    from tests.combined_fixture import GH_PAYLOAD, MAIL, SLUG, MAIL_SLUG
    shaped = Shaped()
    shaped.extend(github_api.shape(GH_PAYLOAD, SLUG))
    shaped.extend(email_mbox.shape(MAIL, MAIL_SLUG))
    return shaped, "northwind"


def _load_real(gh_path: Path, gh_slug: str | None,
               mail_path: Path, mail_slug: str | None) -> tuple[Shaped, str]:
    shaped = Shaped()
    slug = "combined"
    if gh_path:
        payload = json.loads(gh_path.read_text())
        gh_slug = gh_slug or payload.get("slug") or gh_path.stem
        shaped.extend(github_api.shape(payload, gh_slug))
        slug = gh_slug.replace("/", "-")
    if mail_path:
        shaped.extend(email_mbox.load(mail_path, slug=mail_slug or mail_path.stem))
    return shaped, slug


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true",
                    help="run the built-in realistic fixture with an OFFLINE model judge")
    ap.add_argument("--github", help="GitHub payload JSON (issues/pulls)")
    ap.add_argument("--github-slug", help="repo slug, e.g. owner/name (else read from the JSON)")
    ap.add_argument("--mail", help="maildir dir, .mbox file, or emails.csv")
    ap.add_argument("--mail-slug", help="a name for the mailbox")
    ap.add_argument("--no-llm", action="store_true",
                    help="force the deterministic, offline baseline (no naming/abstraction/judge)")
    ap.add_argument("--hybrid", action="store_true",
                    help="add an embedding shortlist to the semantic judge (needs VOYAGE_API_KEY)")
    ap.add_argument("--names", choices=["auto", "off"], default="auto",
                    help="'off' keeps raw activity verbs even when the model tier is on")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="stream each inferred join / kind / gap as it is decided")
    ap.add_argument("--quiet", action="store_true", help="suppress stage progress")
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args(argv)

    prog = from_flags(quiet=args.quiet, verbose=args.verbose)

    if args.demo:
        shaped, slug = _load_demo()
        provider = SemanticProvider(judge=_demo_judge())    # offline scripted judge
        tier_label = "demo (offline model judge)"
        tier = None
    else:
        if not (args.github or args.mail):
            ap.error("give --github and/or --mail, or --demo")
        # One decision drives the whole model tier: the judge, the namer and the
        # record reader. On by default; --no-llm (or no key) -> deterministic, and
        # the label says so. --hybrid adds the embedding shortlist.
        tier = resolve("hybrid" if args.hybrid else "auto", no_llm=args.no_llm)
        shaped, slug = _load_real(Path(args.github) if args.github else None, args.github_slug,
                                  Path(args.mail) if args.mail else None, args.mail_slug)
        provider = tier.semantic(log=prog)
        tier_label = tier.label

    n_records = sum(1 for e in shaped.entities if e.type != "person")
    n_sources = len({e.source for e in shaped.entities if e.type != "person"})
    if not n_records:
        print("[run] no records shaped — check --github / --mail paths", file=sys.stderr)
        return 2

    print(f"[run] {n_records} records from {n_sources} sources · model tier: {tier_label}")
    m = induce(shaped, slug=slug, policy=CorrelationPolicy(semantic=provider),
               manifest={"source_kind": "combined", "n_records": n_records}, progress=prog)

    # Name the kinds (and item) with the model — offline stand-in for --demo, the
    # real Anthropic namer when the tier is on. `--names off` keeps raw verbs.
    if args.demo:
        from tests.combined_fixture import demo_namer
        names = infer_names(m, namer=demo_namer, log=prog)
    else:
        names = infer_names(m, enable=(tier.names_enable() and args.names != "off"), log=prog)

    # AI-first process abstraction: the model groups artefact verbs into the
    # activities the process is made of, and reads each record where the verb is
    # only transport (a mailbox: 761 threads, one verb, one useless "Communicated"
    # step). With the tier off, no abstraction is claimed and the inspector shows
    # the raw artefacts.
    if args.demo:
        from tests.combined_fixture import demo_activity_mapper, demo_record_classifier
        mapper, classifier = demo_activity_mapper(), demo_record_classifier()
    elif args.names == "off":
        mapper = classifier = None
    else:
        mapper, classifier = tier.mapper(log=prog), tier.classifier(log=prog)
    activities = infer_activities(m, mapper, classifier, log=prog)

    out = Path(args.out_dir)
    write_json(m, out / "model.json")
    write_html(m, out / "inspector.html", names=names, activities=activities)

    cross = [c for c in m.cases.values()
             if len({("mail" if e.startswith("email:") else "sys") for e in c.entity_ids}) > 1]
    by_tier: dict[str, int] = {}
    for c in cross:
        by_tier[c.confidence.tier.label] = by_tier.get(c.confidence.tier.label, 0) + 1
    print(f"  {len(m.cases)} cases · {len(cross)} cross-source "
          f"({', '.join(f'{n} {t}' for t, n in by_tier.items()) or 'none'})")
    print(f"  {sum(1 for k in m.kinds if k.rejected)} rejected kinds · {len(m.orphans)} orphans")
    print(f"[run] wrote {out}/inspector.html — open it in a browser")
    return 0


def _demo_judge():
    from tests.combined_fixture import demo_judge
    return demo_judge()


if __name__ == "__main__":
    raise SystemExit(main())
