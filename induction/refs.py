"""Reference extraction — how one artefact points at another.

Correlation (step 2) lives or dies on these. We keep the patterns here, in one
place, so the *confidence tier* attached to each kind of reference is explicit
and defensible rather than buried in the correlator.

The tiering rule we defend:
- A reference that follows a *structural convention* GitHub itself writes
  ("Merge pull request #N", the squash-merge "subject (#N)") is `joined`: it is
  effectively a deterministic key, not a guess.
- A reference that depends on a *human writing a keyword* ("fixes #N",
  "closes #N") is still `joined` — the keyword makes the intent explicit.
- A *bare* "#N" mentioned in prose is `heuristic`: a mention is not a link.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# "Merge pull request #123 from someuser/branch"  — GitHub's own merge message.
_MERGE_PR = re.compile(r"Merge pull request #(\d+)\b")
# Squash-merge convention: the PR number is appended to the subject as "(#123)".
_SQUASH_PR = re.compile(r"\(#(\d+)\)")
# Explicit issue-closing keywords understood by GitHub.
_ISSUE_KW = re.compile(
    r"(?i)\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\b[:\s]+#(\d+)"
)
# Any bare "#123" mention.
_BARE = re.compile(r"#(\d+)\b")
# Backport / release-train merges carry no PR number.
_MERGE_BRANCH = re.compile(r"Merge (?:remote-tracking )?branch '([^']+)'")


@dataclass(frozen=True)
class Reference:
    kind: str        # "pr" or "issue"
    number: int
    tier: str        # "joined" | "heuristic" (see module docstring)
    via: str         # which pattern matched — the *why*, recorded on the join
    snippet: str     # the exact text the reference was read from


def extract_references(subject: str, body: str) -> list[Reference]:
    """Pull every believable cross-reference out of a commit message.

    We are deliberately conservative about calling something a PR vs an issue:
    the merge/squash conventions name a PR; closing-keywords name an issue;
    everything else is a bare mention we keep but tier down.
    """
    refs: list[Reference] = []
    seen: set[tuple[str, int]] = set()

    def add(kind: str, number: int, tier: str, via: str, snippet: str) -> None:
        key = (kind, number)
        # Keep the strongest tier if the same number shows up multiple ways.
        if key in seen:
            return
        seen.add(key)
        refs.append(Reference(kind, number, tier, via, snippet.strip()[:200]))

    # Strongest first so the dedupe keeps the structural interpretation.
    m = _MERGE_PR.search(subject)
    if m:
        add("pr", int(m.group(1)), "joined", "merge-pull-request", subject)

    for m in _SQUASH_PR.finditer(subject):
        add("pr", int(m.group(1)), "joined", "squash-subject", subject)

    for m in _ISSUE_KW.finditer(subject + "\n" + body):
        add("issue", int(m.group(1)), "joined", "closing-keyword", m.group(0))

    # Bare mentions become heuristic references to whichever kind we haven't
    # already pinned down. A mention is evidence of a *relationship*, not proof
    # of which run it belongs to — so it is tiered down and stays inspectable.
    for text in (subject, body):
        for m in _BARE.finditer(text):
            n = int(m.group(1))
            if ("pr", n) in seen or ("issue", n) in seen:
                continue
            add("issue", n, "heuristic", "bare-mention", _line_of(text, m.start()))

    return refs


def merged_branch_name(subject: str) -> Optional[str]:
    """The branch name from a "Merge branch 'X'" message, or None.

    These backport/release-train merges are the spine of the *release* process
    kind (segment step 0), which is why we surface them separately from PRs.
    """
    m = _MERGE_BRANCH.search(subject)
    return m.group(1) if m else None


def _line_of(text: str, offset: int) -> str:
    """The single line containing ``offset`` — a tidy snippet for evidence."""
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    if end == -1:
        end = len(text)
    return text[start:end].strip()
