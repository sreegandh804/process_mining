"""GitHub Issues/PR adapter — the "everything is connected" source.

This is the customer whose process genuinely spans systems: an issue is filed,
a pull request is opened that may or may not say which issue it serves, review
happens, a merge lands a commit that git also knows about, and a release note
mentions it later. Nothing but convention ties those together.

**This file is the entire GitHub integration.** There is no `correlate_github`,
no `run_github_pipeline`, no GitHub branch anywhere downstream — adding this
source touched no other module. That is the contract `adapters/__init__.py`
states, and it only became true once correlation stopped being written once per
source (see `induction/links.py`).

What it shapes, and the judgements it makes:

- **An issue/PR is an Entity whose timeline yields many Events.** Never one
  "issue event". `opened`, `commented`, `labeled`, `review_requested`,
  `reviewed`, `merged`, `closed`, `reopened` are separate timed changes with
  their own actors — that is what makes a real review loop visible as a trace,
  and what lets an *exception* variant (reopened, closed unmerged) read
  differently from the common path.

- **Entity ids are shared with the git adapter on purpose.** A PR is `pr:{n}`
  and an issue is `issue:{slug}:{n}` — exactly the ids git *materialises* when a
  commit says "(#123)". So when both sources are loaded, git's thin reference
  and GitHub's thick record are the same entity: the reference resolves to the
  real record instead of inventing an inferred one. One PR, two sources, no
  reconciliation step.

- **A PR's `merge_commit_sha` links to `commit:{sha}`.** This is a genuine
  cross-source deterministic join — GitHub's record of a merge meeting git's
  record of the same commit — and it costs one `Link`.

- **Actors are `person:gh:{login}`, deliberately NOT unified with git's
  `person:{email}`.** GitHub's API does not reliably expose the commit email
  behind a login, so merging them would be a guess presented as an identity.
  Cross-source actor resolution is a real upgrade (and a fuzzy one); pretending
  it is already done would be exactly the dishonesty this engine exists to
  avoid. The consequence is visible and correct: the same human may appear as
  two people until that is built.

Live pulls are `ingest_github.py`. This module only reads the cached JSON, so
the pipeline stays reproducible and offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from induction import refs as refs_mod
from induction.adapters import Shaped
from induction.links import Link, declare
from induction.model import Entity, Event, Tier, direct

# Timeline event names GitHub emits that are real, timed changes to the item.
# Anything not listed is skipped rather than guessed at — an unrecognised
# timeline entry is not evidence of an activity we can name.
_TIMELINE_ACTIONS = {
    "commented": "commented",
    "labeled": "labeled",
    "unlabeled": "unlabeled",
    "assigned": "assigned",
    "unassigned": "unassigned",
    "review_requested": "review_requested",
    "review_request_removed": "review_request_removed",
    "reviewed": "reviewed",
    "committed": "committed",
    "head_ref_force_pushed": "force_pushed",
    "ready_for_review": "ready_for_review",
    "convert_to_draft": "converted_to_draft",
    "renamed": "renamed",
    "closed": "closed",
    "reopened": "reopened",
    "merged": "merged",
    "referenced": "referenced",
    "cross-referenced": "cross_referenced",
    "connected": "connected",
    "disconnected": "disconnected",
}

# Bot logins. Marked at the source so the generic segmenter can separate
# machine-driven runs, exactly as the git adapter does.
_BOT_SUFFIXES = ("[bot]", "-bot")
_BOT_LOGINS = {"dependabot", "github-actions", "renovate", "pre-commit-ci", "codecov"}


def load(raw_dir: str | Path, slug: str) -> Shaped:
    """Load a cached GitHub corpus for ``slug`` into canonical records."""
    raw_dir = Path(raw_dir)
    key = slug.replace("/", "__")
    path = raw_dir / f"{key}.github.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no cached GitHub corpus at {path}. "
            f"Run `python ingest_github.py --slug {slug}` first."
        )
    return shape(json.loads(path.read_text()), slug)


def shape(payload: dict, slug: str) -> Shaped:
    """Shape an already-loaded GitHub payload. Split out so tests can drive it
    from a fixture without touching disk or the network."""
    source = f"github:{slug}"
    out = Shaped()
    people: dict[str, Entity] = {}

    for item in payload.get("issues", []):
        _shape_item(out, people, source, slug, item, is_pr=False)
    for item in payload.get("pulls", []):
        _shape_item(out, people, source, slug, item, is_pr=True)

    out.entities.extend(people.values())
    return out


# ---------------------------------------------------------------------------
# Shaping one issue or pull request
# ---------------------------------------------------------------------------

def _shape_item(out: Shaped, people: dict, source: str, slug: str,
                item: dict, is_pr: bool) -> None:
    number = item.get("number")
    if number is None:
        return
    kind = "pr" if is_pr else "issue"
    # Shared with the git adapter on purpose — see the module docstring.
    ent_id = f"pr:{number}" if is_pr else f"issue:{slug}:{number}"
    locator = item.get("html_url") or f"{kind}/{number}"
    title = item.get("title") or ""
    body = item.get("body") or ""
    labels = [lab.get("name", "") if isinstance(lab, dict) else str(lab)
              for lab in (item.get("labels") or [])]

    entity = Entity(
        id=ent_id, source=source, type=kind,
        attrs={
            "number": number,
            "title": title,            # the fuzzy pass reads this
            "state": item.get("state"),
            "labels": labels,
            "draft": bool(item.get("draft")),
            "is_bot": _is_bot(_login(item.get("user"))),
            "merged": bool(item.get("merged_at")),
            "merge_commit_sha": item.get("merge_commit_sha") or "",
            "comments": item.get("comments", 0),
        },
        confidence=direct(),
        evidence=[_ev(source, locator, title)],
        raw=item,
    )

    # An issue or PR owns its own timeline, so it is a run in its own right.
    # A PR outranks an issue when both are in one component: git already makes
    # that call (a run of commits is "PR #1"), and staying consistent is what
    # lets the two sources describe the same run without disagreeing about its
    # name. The issue's own claim survives when no PR is present.
    declare(entity, Link(
        target=ent_id, method="record-identity", tier=Tier.DIRECT,
        rationale=f"{kind} carries its own timeline and is a run in its own right",
        locator=locator, snippet=title,
        anchors=True, anchor_rank=0 if is_pr else 5,
        anchor_attrs={"type": kind, "number": number},
    ))

    # Cross-references written in the title/body: "closes #12", "see #9". The
    # tiering (explicit keyword vs bare mention) is refs.py's judgement, shared
    # with the git adapter so the same convention scores the same way whichever
    # source read it.
    for ref in refs_mod.extract_references(title, body):
        target = (f"pr:{ref.number}" if ref.kind == "pr"
                  else f"issue:{slug}:{ref.number}")
        if target == ent_id:
            continue
        declare(entity, Link(
            target=target, method=ref.via, tier=Tier.from_label(ref.tier),
            rationale=(f"{kind} #{number} body references #{ref.number} ({ref.via})"),
            locator=locator, snippet=ref.snippet,
        ))

    _shape_timeline(out, people, source, slug, item, entity, ent_id, locator, is_pr)
    out.entities.append(entity)


def _shape_timeline(out: Shaped, people: dict, source: str, slug: str, item: dict,
                    entity: Entity, ent_id: str, locator: str, is_pr: bool) -> None:
    """Expand the item's timeline into Events — never flattened into one."""
    kind = "pr" if is_pr else "issue"
    seen: set[str] = set()

    def emit(action: str, when: Optional[str], actor_login: Optional[str],
             snippet: str = "", attrs: Optional[dict] = None) -> None:
        eid = f"evt:gh:{ent_id}:{action}:{len(seen)}"
        seen.add(eid)
        out.events.append(Event(
            id=eid, entity_id=ent_id, action=action, source=source,
            confidence=direct(), evidence=[_ev(source, locator, snippet or action)],
            timestamp=when or None,          # never fabricated
            actor=_person(people, source, actor_login) if actor_login else None,
            attrs=attrs or {},
        ))

    emit("opened", item.get("created_at"), _login(item.get("user")),
         item.get("title") or "", {"kind": kind})

    for entry in item.get("timeline") or []:
        raw_event = entry.get("event") or ""
        action = _TIMELINE_ACTIONS.get(raw_event)
        if action is None:
            continue
        when = (entry.get("created_at") or entry.get("submitted_at")
                or (entry.get("commit_id") and item.get("updated_at")))
        who = _login(entry.get("actor") or entry.get("user"))
        snippet = (entry.get("body") or entry.get("label", {}).get("name", "")
                   if isinstance(entry.get("label"), dict) else entry.get("body") or "")
        attrs = {}
        if action == "reviewed" and entry.get("state"):
            attrs["review_state"] = entry["state"]
        emit(action, when, who, str(snippet or raw_event)[:200], attrs)

        # A cross-reference in the timeline is a link GitHub itself drew.
        src = entry.get("source") or {}
        other = (src.get("issue") or {}).get("number")
        if action in ("cross_referenced", "connected") and other:
            other_is_pr = bool((src.get("issue") or {}).get("pull_request"))
            declare(entity, Link(
                target=f"pr:{other}" if other_is_pr else f"issue:{slug}:{other}",
                method=f"timeline-{raw_event}", tier=Tier.JOINED,
                rationale=f"GitHub timeline records a {raw_event} to #{other}",
                locator=locator, snippet=str(snippet)[:200],
            ))

    # Terminal states, taken from the item itself so they survive a corpus
    # cached without timelines.
    if item.get("merged_at") and not _has(out, ent_id, "merged"):
        emit("merged", item["merged_at"], _login(item.get("merged_by")), "merged")
    if item.get("closed_at") and not _has(out, ent_id, "closed"):
        emit("closed", item["closed_at"], _login(item.get("closed_by")), "closed")

    # The merge commit is the same artefact git has: a real cross-source key.
    sha = item.get("merge_commit_sha")
    if sha:
        declare(entity, Link(
            target=f"commit:{sha}", method="merge-commit-sha", tier=Tier.JOINED,
            rationale="pull request records the SHA of the commit it merged",
            locator=locator, snippet=sha[:12],
        ))


def _has(out: Shaped, ent_id: str, action: str) -> bool:
    return any(e.entity_id == ent_id and e.action == action for e in out.events)


# ---------------------------------------------------------------------------
# Actors
# ---------------------------------------------------------------------------

def _login(user) -> Optional[str]:
    if isinstance(user, dict):
        return user.get("login")
    if isinstance(user, str):
        return user
    return None


def _is_bot(login: Optional[str]) -> bool:
    if not login:
        return False
    low = login.lower()
    return low in _BOT_LOGINS or any(low.endswith(s) for s in _BOT_SUFFIXES)


def _person(people: dict, source: str, login: str) -> str:
    """A GitHub actor. NOT merged with git's email-keyed people — see docstring."""
    pid = f"person:gh:{login.lower()}"
    ent = people.get(pid)
    if ent is None:
        people[pid] = Entity(
            id=pid, source=source, type="person",
            attrs={"name": login, "login": login, "is_bot": _is_bot(login),
                   "identity_scope": "github"},
            confidence=direct(),
        )
    return pid


def _ev(source: str, locator: str, snippet: str):
    from induction.model import Evidence
    return Evidence(source=source, locator=locator, snippet=(snippet or "")[:200] or None)
