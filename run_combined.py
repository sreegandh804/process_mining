#!/usr/bin/env python3
"""Run the engine over TWO sources at once — a team's GitHub + their mailbox.

This is the corpus the brief is really about: the process lives across systems,
and *nothing announces the relationship*. People email "the export is spinning
again", not "re: #21". So the cross-source joins are earned, at the honest tier
the evidence supports:

    deterministic  a merge that cites an issue, a foreign key           joined
    shared tokens  the same rare words + the same week/people           heuristic
    same *meaning*  paraphrase a language model can read, tokens can't   model

The model tier is opt-in (`--semantic`), because it sends corpus text to an API.
Off, the engine is fully deterministic and offline.

    # see the shape on a built-in realistic fixture, no keys, no data:
    python run_combined.py --demo

    # your own data, deterministic + shared-token only:
    python run_combined.py --github portal.github.json --mail team.mbox

    # add the model tier (needs ANTHROPIC_API_KEY; hybrid also VOYAGE_API_KEY):
    python run_combined.py --github portal.github.json --mail team.mbox --semantic llm

`--github` is a JSON payload ({"slug","issues","pulls"}, the shape
`ingest_github.py` writes); `--mail` is a maildir dir, an .mbox, or a CSV with a
raw-message column. Emits out/model.json + out/inspector.html.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from induction.abstraction import AnthropicActivityMapper, infer_activities
from induction.adapters import Shaped, email_mbox, github_api
from induction.emit import write_json
from induction.inspector import write_html
from induction.naming import infer_names
from induction.pipeline import induce
from induction.semantic import AnthropicJudge, SemanticProvider, VoyageEmbedder
from induction.steps.correlate import CorrelationPolicy


def _provider(mode: str, demo: bool):
    if mode == "off":
        return None
    if demo:
        from tests.combined_fixture import demo_judge
        return SemanticProvider(judge=demo_judge())
    if mode == "llm":
        return SemanticProvider(judge=AnthropicJudge())
    if mode == "hybrid":
        return SemanticProvider(judge=AnthropicJudge(), embedder=VoyageEmbedder())
    return None


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
    ap.add_argument("--semantic", choices=["off", "llm", "hybrid"], default="off",
                    help="model tier: llm=Anthropic judge, hybrid=+embedding shortlist")
    ap.add_argument("--names", choices=["off", "llm"], default="off")
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args(argv)

    if args.demo:
        shaped, slug = _load_demo()
        provider = _provider("llm", demo=True)   # offline scripted judge
        mode = "demo (offline model judge)"
    else:
        if not (args.github or args.mail):
            ap.error("give --github and/or --mail, or --demo")
        shaped, slug = _load_real(Path(args.github) if args.github else None, args.github_slug,
                                  Path(args.mail) if args.mail else None, args.mail_slug)
        provider = _provider(args.semantic, demo=False)
        mode = args.semantic

    n_records = sum(1 for e in shaped.entities if e.type != "person")
    n_sources = len({e.source for e in shaped.entities if e.type != "person"})
    if not n_records:
        print("[run] no records shaped — check --github / --mail paths", file=sys.stderr)
        return 2

    print(f"[run] {n_records} records from {n_sources} sources · semantic={mode}")
    m = induce(shaped, slug=slug, policy=CorrelationPolicy(semantic=provider),
               manifest={"source_kind": "combined", "n_records": n_records})
    names = infer_names(m, enable=(args.names == "llm"))

    # AI-first process abstraction: the model groups the artefact verbs into the
    # activities the process is made of. Demo uses an offline stand-in; a real
    # semantic run uses the Anthropic mapper; with neither, no abstraction is
    # claimed and the inspector shows the raw artefacts.
    if args.demo:
        from tests.combined_fixture import demo_activity_mapper
        mapper = demo_activity_mapper()
    elif args.semantic in ("llm", "hybrid"):
        mapper = AnthropicActivityMapper()
    else:
        mapper = None
    activities = infer_activities(m, mapper)

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


if __name__ == "__main__":
    raise SystemExit(main())
