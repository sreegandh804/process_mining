"""Git-history adapter (step 1, Shape) — the *thick* source.

Turns cached raw commits + tags into canonical Entities/Events/Observations.
This is the only module that knows what a commit looks like.

The core design commitment (brief §2): a commit is an **Entity** whose timeline
yields several **Events** — we never flatten it into one event. Concretely a
single commit can produce:

  - `authored`   (actor = author, at author-date)                  — always
  - `authored`   (actor = each co-author, at author-date)          — the same
                  activity done by more than one person; Label merges these.
  - `committed`  (actor = committer, at committer-date)            — ONLY when
                  the committer differs from the author, i.e. a real handoff
                  (a maintainer applied someone else's patch; a web merge).
  - `merged`     (actor = committer)   — when the commit is a PR/branch merge.
  - `reverted`   (actor = author)      — when the commit undoes earlier work.

Every field here is either read straight from git (tier `direct`) or left
absent. We never invent an actor or a timestamp.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from induction.adapters import Shaped
from induction.model import Entity, Event, Observation, direct
from induction import refs as refs_mod

# Bots produce recurring commits that *look like* a process. We tag them at the
# source so step 0/§6 can put the look-alikes to the "is this really a process?"
# test rather than silently rolling them into real work.
_BOT_MARKERS = ("[bot]", "dependabot", "pre-commit-ci", "github-actions", "renovate")
_COAUTHOR = re.compile(r"(?im)^\s*co-authored-by:\s*(.+?)\s*<([^>]+)>\s*$")


def _is_bot(name: str, email: str) -> bool:
    blob = f"{name} {email}".lower()
    return any(m in blob for m in _BOT_MARKERS)


def _person_id(name: str, email: str) -> str:
    """A stable identity for an actor.

    Email is the better key (names get typo'd and reformatted), but some commits
    carry only a name. We never merge two people we cannot prove are the same,
    so a missing email falls back to the name — and that ambiguity is itself
    honest (two 'David' with no email stay separate, not falsely merged).
    """
    email = (email or "").strip().lower()
    if email and "noreply" not in email.split("@")[0]:
        return f"person:{email}"
    if email:
        return f"person:{email}"
    return f"person:name:{name.strip().lower()}"


def _co_authors(body: str) -> list[tuple[str, str]]:
    return [(m.group(1).strip(), m.group(2).strip().lower()) for m in _COAUTHOR.finditer(body or "")]


class _People:
    """Deduplicating registry of person entities across all commits."""

    def __init__(self, source: str):
        self.source = source
        self._by_id: dict[str, Entity] = {}

    def ensure(self, name: str, email: str, evidence_locator: str) -> str:
        pid = _person_id(name, email)
        ent = self._by_id.get(pid)
        if ent is None:
            self._by_id[pid] = Entity(
                id=pid,
                source=self.source,
                type="person",
                attrs={
                    "name": name.strip(),
                    "email": (email or "").strip().lower(),
                    "is_bot": _is_bot(name, email),
                    "commit_count": 1,
                },
                confidence=direct(),
                evidence=[],
            )
        else:
            ent.attrs["commit_count"] += 1
            # Keep the most recent human-readable name we saw.
            if name.strip():
                ent.attrs["name"] = name.strip()
        return pid

    def entities(self) -> list[Entity]:
        return list(self._by_id.values())


def load(raw_dir: str | Path, slug: str) -> Shaped:
    """Load the cached git corpus for ``slug`` into canonical records."""
    raw_dir = Path(raw_dir)
    key = slug.replace("/", "__")
    source = f"git:{slug}"
    commits_path = raw_dir / f"{key}.commits.jsonl"
    tags_path = raw_dir / f"{key}.tags.json"
    if not commits_path.exists():
        raise FileNotFoundError(
            f"no cached corpus at {commits_path}. Run `python ingest.py --slug {slug} ...` first."
        )

    out = Shaped()
    people = _People(source)

    for line in commits_path.open():
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        _shape_commit(c, source, people, out)

    if tags_path.exists():
        _shape_tags(json.loads(tags_path.read_text()), source, out)

    out.entities.extend(people.entities())
    return out


def _shape_commit(c: dict, source: str, people: _People, out: Shaped) -> None:
    sha = c["sha"]
    short = c.get("short_sha", sha[:8])
    ent_id = f"commit:{sha}"
    locator = sha  # a sha resolves back to the raw artefact via `git show <sha>`
    subject = c.get("subject", "")
    body = c.get("body", "")

    references = refs_mod.extract_references(subject, body)
    merge_branch = refs_mod.merged_branch_name(subject) if c.get("is_merge") else None
    is_revert = subject.strip().lower().startswith("revert") or subject.strip().lower().startswith("re-revert")
    author = c["author"]
    committer = c["committer"]
    is_bot = _is_bot(author["name"], author["email"]) or _is_bot(committer["name"], committer["email"])

    author_id = people.ensure(author["name"], author["email"], locator)
    committer_id = people.ensure(committer["name"], committer["email"], locator)

    # The commit entity — read directly, so tier `direct`.
    out.entities.append(Entity(
        id=ent_id,
        source=source,
        type="commit",
        attrs={
            "short_sha": short,
            "subject": subject,
            "is_merge": bool(c.get("is_merge")),
            "is_revert": is_revert,
            "is_bot": is_bot,
            "parents": c.get("parents", []),
            "n_files": len(c.get("files", [])),
            "files": c.get("files", [])[:50],
            "refs": c.get("refs", []),
            # references materialise into pr/issue entities during correlate.
            "references": [r.__dict__ for r in references],
            "merge_branch": merge_branch,
        },
        confidence=direct(),
        evidence=[_ev(source, locator, subject)],
        raw=c,
    ))

    # Decide the commit's primary activity. Merge and revert are distinct
    # actions on purpose — they are what makes an *exception* trace read
    # differently from the common path.
    if c.get("is_merge") and (any(r.kind == "pr" for r in references) or merge_branch):
        primary_action = "merged"
        primary_actor = committer_id
        primary_ts = c["committer_date"] or None
    elif is_revert:
        primary_action = "reverted"
        primary_actor = author_id
        primary_ts = c["author_date"] or None
    else:
        primary_action = "authored"
        primary_actor = author_id
        primary_ts = c["author_date"] or None

    out.events.append(Event(
        id=f"evt:{sha}:{primary_action}",
        entity_id=ent_id,
        action=primary_action,
        source=source,
        confidence=direct(),
        evidence=[_ev(source, locator, subject)],
        timestamp=primary_ts,
        actor=primary_actor,
        attrs={"is_bot": is_bot},
        raw=None,
    ))

    # Co-authors: the SAME authoring activity performed by more than one person.
    # Emitted as sibling `authored` events so Label's same-activity merge can
    # fold them into one Step with several Members, keeping every record.
    if primary_action == "authored":
        for i, (co_name, co_email) in enumerate(_co_authors(body)):
            co_id = people.ensure(co_name, co_email, locator)
            if co_id == primary_actor:
                continue
            out.events.append(Event(
                id=f"evt:{sha}:coauthored:{i}",
                entity_id=ent_id,
                action="authored",
                source=source,
                confidence=direct(),
                evidence=[_ev(source, locator, f"Co-authored-by: {co_name} <{co_email}>")],
                timestamp=primary_ts,
                actor=co_id,
                attrs={"role": "co-author"},
            ))

    # A committer who is not the author is a real second timeline point: someone
    # else applied or merged this work. That handoff is the signal, so we record
    # it — but only when it actually happened (else it is redundant noise).
    handoff = (committer_id != author_id) or (c["committer_date"] != c["author_date"])
    if handoff and primary_action != "merged":
        out.events.append(Event(
            id=f"evt:{sha}:committed",
            entity_id=ent_id,
            action="committed",
            source=source,
            confidence=direct(),
            evidence=[_ev(source, locator, subject)],
            timestamp=c["committer_date"] or None,
            actor=committer_id,
            attrs={
                "handoff": committer_id != author_id,
                "authored_by": author_id,
            },
        ))


def _shape_tags(tags: list[dict], source: str, out: Shaped) -> None:
    """A tag is an Observation of a release plus a `released` Event.

    The tagger's identity is not reliably the process actor, so actor stays
    None rather than being guessed — the release's *author* is recovered later
    by correlating the release entity to the commit it points at.
    """
    for t in tags:
        name = t.get("name", "")
        commit = t.get("commit", "")
        date = t.get("date", "") or None
        rel_id = f"release:{name}"
        locator = f"tag:{name}"
        out.entities.append(Entity(
            id=rel_id,
            source=source,
            type="release",
            attrs={"name": name, "commit": commit},
            confidence=direct(),
            evidence=[_ev(source, locator, f"tag {name} -> {commit[:10]}")],
            raw=t,
        ))
        out.events.append(Event(
            id=f"evt:release:{name}",
            entity_id=rel_id,
            action="released",
            source=source,
            confidence=direct(),
            evidence=[_ev(source, locator, f"tag {name}")],
            timestamp=date,
            actor=None,  # unknown from the tag alone — not fabricated
            attrs={"tag": name, "commit": commit},
        ))


def _ev(source: str, locator: str, snippet: str):
    from induction.model import Evidence
    return Evidence(source=source, locator=locator, snippet=(snippet or "")[:200] or None)
