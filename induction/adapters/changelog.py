"""Changelog adapter (step 4b) — the *thin* source, end-to-end and real.

The git corpus is the thickest possible source: an actor and a timestamp on
every event. So the `Observation` type and the `order: unknown` path would only
ever be exercised by fixtures unless a genuinely thin source runs through the
*same* pipeline. This adapter is that source.

A changelog is honestly thin: it records **what** shipped in a version, but not
**when** each change was made, **by whom**, or in what **order** within the
version. So each bullet becomes an `Observation` with:
    seen_at = None      (we do NOT borrow the release date — it is not the time
                         the change was made; inventing per-item time is exactly
                         what the brief forbids)
    actor   = None      (a changelog names no one)
and the version's "Released / Unreleased" line is kept only as a *status*, the
way a thin CSV export carries a status column.

Where a bullet cites a PR/issue that the git side already has a case for, the
two sources correlate on that shared key (tier `joined`) — thin data enriching
thick. Where it does not, the bullet falls back to a thin-only, order-`unknown`
case per version: graceful degradation, proven on real data rather than asserted.

Both of those are declared as `Link`s and resolved by the one shared correlator.
This adapter used to carry its own `correlate_thin()`, which is how the engine
ended up with a correlator per source — the exact trap that makes cross-source
joining impossible, since each copy could only ever see its own records.
"""

from __future__ import annotations

import re
from pathlib import Path

from induction.adapters import Shaped
from induction.links import Link, declare
from induction.model import Confidence, Entity, Evidence, Observation, Tier, direct

_VERSION = re.compile(r"^Version\s+(\S+)\s*$")
_STATUS = re.compile(r"^(Released\b.*|Unreleased)\s*$")
_REF = re.compile(r":(pr|issue):`(\d+)`")
_BARE = re.compile(r"#(\d+)\b")


def load(raw_dir: str | Path, slug: str) -> Shaped:
    raw_dir = Path(raw_dir)
    key = slug.replace("/", "__")
    path = raw_dir / f"{key}.CHANGES.rst"
    source = f"changelog:{slug}"
    out = Shaped()
    if not path.exists():
        return out

    lines = path.read_text(errors="replace").splitlines()
    version = None
    status = None
    idx = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        mv = _VERSION.match(line)
        if mv and i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}:
            version = mv.group(1)
            status = None
            idx = 0
            i += 2
            continue
        ms = _STATUS.match(line.strip())
        if ms and version:
            status = ms.group(1).split()[0]  # "Released" | "Unreleased" — no date kept
            i += 1
            continue
        if line.startswith("-   ") and version:
            # Gather the (possibly multi-line) bullet.
            bullet_lines = [line[4:]]
            start_lineno = i + 1
            j = i + 1
            while j < len(lines) and (lines[j].startswith("    ") and not lines[j].startswith("    -")):
                bullet_lines.append(lines[j].strip())
                j += 1
            text = " ".join(bl.strip() for bl in bullet_lines).strip()
            _emit_bullet(out, source, slug, version, status, idx, text, start_lineno, key)
            idx += 1
            i = j
            continue
        i += 1

    return out


def _emit_bullet(out, source, slug, version, status, idx, text, lineno, key) -> None:
    refs = [{"kind": k, "number": int(n)} for k, n in _REF.findall(text)]
    seen_refs = {(r["kind"], r["number"]) for r in refs}
    for n in _BARE.findall(text):
        if ("pr", int(n)) not in seen_refs and ("issue", int(n)) not in seen_refs:
            refs.append({"kind": "issue", "number": int(n)})

    ent_id = f"changelog_entry:{slug}:{version}:{idx}"
    locator = f"{key}.CHANGES.rst:L{lineno}"
    snippet = text[:200]

    entity = Entity(
        id=ent_id, source=source, type="changelog_entry",
        attrs={"version": version, "status": status, "text": text, "references": refs},
        confidence=direct(),
        evidence=[Evidence(source, locator, snippet)],
    )

    # A cited PR/issue number is a shared key with whatever other source knows
    # that artefact — thin data meeting thick. We deliberately do NOT materialise
    # the target: a changelog citing issue #999 is not evidence that an issue
    # record exists anywhere, only that the note's author believed one did. If no
    # source has it, the citation is recorded as an unresolved reference and the
    # bullet falls back to its version section below.
    for r in refs:
        target = f"pr:{r['number']}" if r["kind"] == "pr" else f"issue:{slug}:{r['number']}"
        declare(entity, Link(
            target=target, method="changelog-citation", tier=Tier.JOINED,
            rationale=f"changelog bullet and git share {r['kind']} #{r['number']}",
            locator=locator, snippet=snippet,
        ))

    # Fallback: bullets nothing else claimed group by the version section they
    # were printed under. `fallback` is what keeps that from swallowing the real
    # cases — applied eagerly, one leftover bullet would fuse every run it shares
    # a version with into a single invented mega-case.
    declare(entity, Link(
        target=f"notes:{slug}:{version}", method="version-section", tier=Tier.HEURISTIC,
        rationale=("a changelog version section with no timestamps, "
                   "actors, or intra-version order"),
        locator=locator, snippet=snippet,
        anchors=True, virtual=True, fallback=True,
        anchor_attrs={"type": "release_notes", "version": version},
    ))

    out.entities.append(entity)
    out.observations.append(Observation(
        id=f"obs:{ent_id}",
        entity_id=ent_id,
        state={"version": version, "status": status, "text": text, "references": refs},
        source=source,
        # We DID read this state directly; the honesty is in the nulls below and
        # in the unknown order — not in pretending the read was uncertain.
        confidence=Confidence(Tier.DIRECT,
                              "state read from a changelog line; no actor, no time, "
                              "and no intra-version order are recoverable"),
        evidence=[Evidence(source, locator, snippet)],
        seen_at=None,   # deliberately not the release date — see module docstring
    ))
