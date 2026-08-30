#!/usr/bin/env python3
"""Ingest: GitHub Issues/PR API -> cached raw JSON on disk (reproducible, offline).

Same split as `ingest.py`: this talks to the API once and writes plain JSON under
``data/raw/``; every downstream run reads only that cache. The adapter in
`induction/adapters/github_api.py` is the only code that knows the API's shape.

Usage:
    export GITHUB_TOKEN=ghp_...            # or put GITHUB_TOKEN=... in .env
    python ingest_github.py --slug pallets/flask --max-items 200

    # then induce over git + GitHub together:
    python run.py --slug pallets/flask --with-github

Writes ``data/raw/<owner>__<repo>.github.json``:

    {"slug": ..., "issues": [<issue>, ...], "pulls": [<pr>, ...]}

where each item is the API object verbatim, plus a ``timeline`` key holding its
timeline events. Nothing is reshaped here — reshaping is the adapter's job, and
keeping the cache source-shaped is what lets you diff it against the API when a
correlation looks wrong.

Auth: a token is optional for public repos but the unauthenticated rate limit
(60/hour) will not get you far. Order of lookup: ``--token``, ``$GITHUB_TOKEN``,
``$GH_TOKEN``, then a ``GITHUB_TOKEN=`` line in ``./.env``. The token is never
written to the cache.

Only the standard library is used, matching the rest of the project.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.github.com"
_UA = "induction-engine-ingest"


def _token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key.strip() in ("GITHUB_TOKEN", "GH_TOKEN"):
                return val.strip().strip("'\"")
    return None


def _get(url: str, token: str | None, retries: int = 4):
    """One GET, with rate-limit and transient-failure handling.

    Returns ``(payload, next_url)`` — `next_url` from the Link header, so
    pagination is the API's own rather than a guessed page count.
    """
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": _UA,
        **({"Authorization": f"Bearer {token}"} if token else {}),
    })
    delay = 2
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                return payload, _next_link(resp.headers.get("Link"))
        except urllib.error.HTTPError as exc:
            # Secondary rate limit / abuse detection asks us to back off.
            if exc.code in (403, 429):
                reset = exc.headers.get("X-RateLimit-Reset")
                remaining = exc.headers.get("X-RateLimit-Remaining")
                if remaining == "0" and reset:
                    wait = max(0, int(reset) - int(time.time())) + 1
                    print(f"    rate limited; sleeping {wait}s", file=sys.stderr)
                    time.sleep(min(wait, 900))
                    continue
            if exc.code in (500, 502, 503, 504) and attempt < retries:
                time.sleep(delay); delay *= 2
                continue
            body = exc.read().decode("utf-8", "replace")[:200]
            raise SystemExit(f"GitHub API {exc.code} for {url}\n  {body}")
        except urllib.error.URLError as exc:
            if attempt < retries:
                time.sleep(delay); delay *= 2
                continue
            raise SystemExit(f"network error for {url}: {exc.reason}")
    raise SystemExit(f"gave up on {url}")


def _next_link(header: str | None) -> str | None:
    if not header:
        return None
    for part in header.split(","):
        section = part.split(";")
        if len(section) < 2:
            continue
        if 'rel="next"' in section[1].strip():
            return section[0].strip().strip("<>")
    return None


def _paged(url: str, token: str | None, limit: int):
    """Follow the API's own `next` links until `limit` items are collected."""
    out = []
    while url and len(out) < limit:
        payload, url = _get(url, token)
        if not isinstance(payload, list):
            break
        out.extend(payload)
        print(f"    ...{len(out)}", end="\r", file=sys.stderr)
    return out[:limit]


def fetch(slug: str, token: str | None, max_items: int, with_timelines: bool) -> dict:
    owner, _, repo = slug.partition("/")
    if not owner or not repo:
        raise SystemExit(f"--slug must be owner/repo, got {slug!r}")

    # `/issues` returns issues *and* pull requests; PRs are the ones carrying a
    # `pull_request` key. We split them and then re-fetch the PRs from /pulls,
    # which is the only place merge_commit_sha and merged_at live.
    print(f"  issues+prs from {slug}", file=sys.stderr)
    items = _paged(f"{API}/repos/{owner}/{repo}/issues"
                   f"?state=all&per_page=100&sort=created&direction=desc",
                   token, max_items)
    issues = [i for i in items if "pull_request" not in i]
    pr_numbers = [i["number"] for i in items if "pull_request" in i]
    print(f"  {len(issues)} issues, {len(pr_numbers)} pull requests", file=sys.stderr)

    pulls = []
    for n in pr_numbers:
        pr, _ = _get(f"{API}/repos/{owner}/{repo}/pulls/{n}", token)
        pulls.append(pr)
        print(f"    pr #{n}", end="\r", file=sys.stderr)

    if with_timelines:
        print("  timelines", file=sys.stderr)
        for item in issues + pulls:
            n = item["number"]
            item["timeline"] = _paged(
                f"{API}/repos/{owner}/{repo}/issues/{n}/timeline?per_page=100",
                token, 300)
            print(f"    #{n}", end="\r", file=sys.stderr)
        for pr in pulls:
            # Reviews are not in the timeline in a usable shape; merge them in
            # so the adapter sees one ordered stream.
            n = pr["number"]
            for review in _paged(
                    f"{API}/repos/{owner}/{repo}/pulls/{n}/reviews?per_page=100",
                    token, 100):
                pr["timeline"].append({
                    "event": "reviewed", "user": review.get("user"),
                    "state": review.get("state"), "body": review.get("body"),
                    "submitted_at": review.get("submitted_at"),
                })
            pr["timeline"].sort(key=lambda e: e.get("created_at")
                                or e.get("submitted_at") or "")

    return {"slug": slug, "issues": issues, "pulls": pulls}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", required=True, help="owner/repo, e.g. pallets/flask")
    ap.add_argument("--out-dir", default="data/raw")
    ap.add_argument("--max-items", type=int, default=200,
                    help="cap on issues+PRs fetched (newest first)")
    ap.add_argument("--no-timelines", action="store_true",
                    help="skip per-item timelines (much faster, far thinner traces)")
    ap.add_argument("--token", default=None, help="GitHub token (else env or .env)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing cache")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{args.slug.replace('/', '__')}.github.json"
    if dest.exists() and not args.force:
        print(f"{dest} exists; use --force to refetch", file=sys.stderr)
        return 0

    token = _token(args.token)
    if not token:
        print("  no token found (--token / $GITHUB_TOKEN / .env) — "
              "unauthenticated, 60 requests/hour", file=sys.stderr)

    payload = fetch(args.slug, token, args.max_items, not args.no_timelines)
    dest.write_text(json.dumps(payload, indent=1))
    n_events = sum(len(i.get("timeline") or [])
                   for i in payload["issues"] + payload["pulls"])
    print(f"\nwrote {dest} "
          f"({len(payload['issues'])} issues, {len(payload['pulls'])} pulls, "
          f"{n_events} timeline entries)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
