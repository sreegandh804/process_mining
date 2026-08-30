#!/usr/bin/env python3
"""Ingest: real corpus -> cached raw JSON on disk (reproducible, offline).

We deliberately separate *ingest* from *shape*. This script talks to git once
and writes plain JSON under ``data/raw/``; every downstream run reads only that
cache and never needs the network or the clone again. That is what makes a run
reproducible and offline — and it is where a future GitHub-API or CSV loader
would slot in without touching the pipeline.

Usage:
    python ingest.py                      # default: pallets/flask from data/corpus/flask
    python ingest.py --repo-path PATH --slug owner/name --max-commits 1500

The raw cache is intentionally source-shaped (git fields), NOT canonical. The
adapter in `induction/adapters/git_history.py` is the only code that knows the
git shape; it turns this into Entities/Events/Observations.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Control chars as delimiters: they never appear in commit messages, so parsing
# is unambiguous even when a body contains newlines, quotes, or "#123".
UNIT = "\x1f"   # between fields
REC = "\x00"    # between records (git -z uses NUL)

# Order matters — must match FIELD_NAMES below. Body (%b) is last.
GIT_FORMAT = UNIT.join(
    ["%H", "%h", "%P", "%an", "%ae", "%aI", "%cn", "%ce", "%cI", "%D", "%s", "%b"]
)
FIELD_NAMES = [
    "sha", "short_sha", "parents", "author_name", "author_email", "author_date",
    "committer_name", "committer_email", "committer_date", "refs", "subject", "body",
]


def _git(repo_path: str, *args: str) -> str:
    """Run a git command against the corpus clone, return stdout as text."""
    out = subprocess.run(
        ["git", "-C", repo_path, *args],
        check=True,
        capture_output=True,
    )
    # git message/filename bytes are utf-8 in these repos; be forgiving anyway.
    return out.stdout.decode("utf-8", errors="replace")


def _files_by_sha(repo_path: str, max_commits: int) -> dict[str, list[str]]:
    """Map sha -> list of files it touched. Separate pass keeps parsing simple.

    We use --no-renames so a rename shows as delete+add of concrete paths, which
    is what we want when reasoning about "which artefacts moved".
    """
    raw = _git(
        repo_path, "log", f"-n{max_commits}", "--no-renames", "--name-only",
        "--format=\x1e%H",
    )
    by_sha: dict[str, list[str]] = {}
    for chunk in raw.split("\x1e"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        lines = [ln for ln in chunk.split("\n")]
        sha = lines[0].strip()
        files = [ln.strip() for ln in lines[1:] if ln.strip()]
        if sha:
            by_sha[sha] = files
    return by_sha


def ingest_commits(repo_path: str, max_commits: int) -> list[dict]:
    raw = _git(repo_path, "log", f"-n{max_commits}", "-z", f"--format={GIT_FORMAT}")
    files_map = _files_by_sha(repo_path, max_commits)
    commits: list[dict] = []
    for record in raw.split(REC):
        if not record.strip():
            continue
        parts = record.split(UNIT)
        if len(parts) < len(FIELD_NAMES):
            # Body was empty and trailing field got trimmed — pad it.
            parts += [""] * (len(FIELD_NAMES) - len(parts))
        row = dict(zip(FIELD_NAMES, parts))
        parents = row["parents"].split() if row["parents"].strip() else []
        commit = {
            "sha": row["sha"].strip(),
            "short_sha": row["short_sha"].strip(),
            "parents": parents,
            "is_merge": len(parents) > 1,
            "author": {"name": row["author_name"], "email": row["author_email"].strip().lower()},
            "committer": {"name": row["committer_name"], "email": row["committer_email"].strip().lower()},
            "author_date": row["author_date"].strip(),
            "committer_date": row["committer_date"].strip(),
            "refs": [r.strip() for r in row["refs"].split(",") if r.strip()],
            "subject": row["subject"],
            "body": row["body"].strip("\n"),
            "files": files_map.get(row["sha"].strip(), []),
        }
        commits.append(commit)
    return commits


def ingest_tags(repo_path: str) -> list[dict]:
    """Tags -> the release skeleton. Each carries the commit it points at and a
    date, so a release is anchored to real evidence, not asserted."""
    fmt = "%(refname:short)\x1f%(objectname)\x1f%(creatordate:iso-strict)\x1f%(*objectname)"
    raw = _git(repo_path, "for-each-ref", f"--format={fmt}", "refs/tags")
    tags = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        name, obj, date, deref = (line.split("\x1f") + ["", "", "", ""])[:4]
        tags.append({
            "name": name.strip(),
            # annotated tags: the commit is the dereferenced object
            "commit": (deref.strip() or obj.strip()),
            "date": date.strip(),
        })
    return tags


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-path", default="data/corpus/flask",
                    help="path to the cloned corpus repo")
    ap.add_argument("--slug", default="pallets/flask",
                    help="owner/name, used in evidence source and cache filenames")
    ap.add_argument("--max-commits", type=int, default=1500,
                    help="bound the corpus for reproducibility and speed")
    ap.add_argument("--out-dir", default="data/raw")
    ap.add_argument("--force", action="store_true",
                    help="re-ingest even if the cache already exists")
    args = ap.parse_args(argv)

    repo_path = args.repo_path
    if not Path(repo_path, ".git").exists():
        print(f"[ingest] no git repo at {repo_path!r}. Clone the corpus first, e.g.:\n"
              f"  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1000 "
              f"https://github.com/{args.slug} {repo_path}", file=sys.stderr)
        return 2

    slug_key = args.slug.replace("/", "__")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    commits_path = out_dir / f"{slug_key}.commits.jsonl"
    tags_path = out_dir / f"{slug_key}.tags.json"
    changes_path = out_dir / f"{slug_key}.CHANGES.rst"
    manifest_path = out_dir / f"{slug_key}.manifest.json"

    if commits_path.exists() and not args.force:
        print(f"[ingest] cache exists ({commits_path}); use --force to re-ingest.")
        return 0

    head = _git(repo_path, "rev-parse", "HEAD").strip()
    print(f"[ingest] {args.slug} @ {head[:10]} (max {args.max_commits} commits)")

    commits = ingest_commits(repo_path, args.max_commits)
    with commits_path.open("w") as f:
        for c in commits:
            f.write(json.dumps(c) + "\n")
    print(f"[ingest] wrote {len(commits)} commits -> {commits_path}")

    tags = ingest_tags(repo_path)
    tags_path.write_text(json.dumps(tags, indent=2))
    print(f"[ingest] wrote {len(tags)} tags -> {tags_path}")

    # Thin source: the changelog, copied verbatim so evidence locators resolve.
    src_changes = Path(repo_path, "CHANGES.rst")
    if src_changes.exists():
        changes_path.write_text(src_changes.read_text(errors="replace"))
        print(f"[ingest] copied changelog -> {changes_path}")
    else:
        print("[ingest] no CHANGES.rst found (thin source will be skipped)")

    manifest = {
        "slug": args.slug,
        "head": head,
        "max_commits": args.max_commits,
        "n_commits": len(commits),
        "n_tags": len(tags),
        "has_changelog": src_changes.exists(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[ingest] wrote manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
