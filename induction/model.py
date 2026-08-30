"""The canonical model — the substrate everything normalises into (brief §2).

Every source is normalised into three record types **before** any mining. The
three-way split is a design decision we defend explicitly:

- An `Entity` is a *thing with identity* (a commit, a pull request, a person).
  A GitHub/git artefact is NOT one event: it is an Entity whose *timeline*
  yields many Events. We never flatten an entity into a single event.

- An `Event` is a *timed change* to an entity (authored, committed, merged,
  reverted). `timestamp`, `actor` and `case_id` are all nullable and, when
  present-by-inference, carry a confidence.

- An `Observation` is a *state seen at a point*, with no action and maybe no
  time. This is how a changelog line or a spreadsheet row enters the system:
  it has no actor and often no timestamp. We record the absence; we never
  invent the missing field.

Provenance is not optional. Any field that is *inferred* rather than *read*
carries a `Confidence` (an ordinal tier — never a fabricated 0.83 decimal) and
points to `Evidence` that resolves back to the raw artefact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable, Optional


class Tier(IntEnum):
    """A claim's confidence IS its tier. Ordinal, strongest -> weakest.

    We use an ``IntEnum`` so tiers compare and so the confidence of a *chain*
    of inferences is simply ``min(...)`` of its links — the weakest link. The
    integer values are ordering only; they are never emitted as scores. If a
    number is ever genuinely needed it must be *calibrated* against the golden
    fixture (see README), not invented here.
    """

    MODEL = 1      # embedding / LLM inference (weakest)
    HEURISTIC = 2  # rule-based inference (reference similarity, actor+time proximity)
    JOINED = 3     # deterministic join on a shared key (commit <-> PR number, etc.)
    DIRECT = 4     # read straight from the source (present as data — strongest)

    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def from_label(cls, label: str) -> "Tier":
        return cls[label.strip().upper()]


@dataclass(frozen=True)
class Evidence:
    """A pointer that must resolve back to the raw artefact.

    ``locator`` is a url / api id / file+offset / commit sha — anything a
    reader (or a test) can follow to see the source with their own eyes. If a
    claim cannot produce Evidence, it does not get made.
    """

    source: str          # e.g. "git:pallets/flask" or "changelog:pallets/flask"
    locator: str         # commit sha, "CHANGES.rst:L120", a url — must resolve
    snippet: Optional[str] = None  # the exact text/line the claim was read from

    def to_dict(self) -> dict:
        d = {"source": self.source, "locator": self.locator}
        if self.snippet is not None:
            d["snippet"] = self.snippet
        return d


@dataclass(frozen=True)
class Confidence:
    """A tier plus (for inferred claims) *why* that tier.

    The brief models Confidence as ``{tier}``. We keep an optional
    ``rationale`` because the brief also requires that a fuzzy/heuristic join
    "records *why* it joined". For a `direct` read the rationale is usually
    None — the evidence speaks for itself.
    """

    tier: Tier
    rationale: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"tier": self.tier.label}
        if self.rationale:
            d["rationale"] = self.rationale
        return d

    @staticmethod
    def weakest(confidences: Iterable["Confidence"], rationale: Optional[str] = None) -> "Confidence":
        """Confidence of a chain = its weakest link. Used when a claim depends
        on several sub-claims (e.g. a case assignment that hops commit -> PR ->
        issue is only as strong as its weakest hop)."""
        tiers = [c.tier for c in confidences]
        if not tiers:
            return Confidence(Tier.HEURISTIC, rationale)
        return Confidence(min(tiers), rationale)


# Convenience constructors — read almost like prose at the call site.
def direct(rationale: Optional[str] = None) -> Confidence:
    return Confidence(Tier.DIRECT, rationale)


def joined(rationale: Optional[str] = None) -> Confidence:
    return Confidence(Tier.JOINED, rationale)


def heuristic(rationale: Optional[str] = None) -> Confidence:
    return Confidence(Tier.HEURISTIC, rationale)


def model(rationale: Optional[str] = None) -> Confidence:
    return Confidence(Tier.MODEL, rationale)


@dataclass
class Entity:
    """A thing with identity: commit, pr, issue, person, file, release, ...

    Per the brief, every field is optional except ``id`` and ``source``.
    ``confidence``/``evidence`` are None for entities *read* from the source
    (a commit we can see). They are populated for entities discovered *by
    inference* — e.g. a pull request we never saw directly, only because a
    commit's subject references ``(#1234)``. Marking those as inferred is the
    honest thing to do: we believe PR #1234 exists, but our only evidence is a
    reference, and its own timeline is a gap.
    """

    id: str
    source: str
    type: str = "unknown"
    attrs: dict = field(default_factory=dict)
    raw: Any = None
    confidence: Optional[Confidence] = None
    evidence: list[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict:
        # The full file list is an in-memory detail used by segmentation; it is
        # not part of the induced model, so we keep the count and drop the list
        # from the emitted JSON (evidence still resolves the commit to its diff).
        attrs = self.attrs
        if "files" in attrs:
            attrs = {k: v for k, v in attrs.items() if k != "files"}
        d: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "attrs": attrs,
        }
        if self.confidence is not None:
            d["confidence"] = self.confidence.to_dict()
        if self.evidence:
            d["evidence"] = [e.to_dict() for e in self.evidence]
        return d


@dataclass
class Event:
    """A TIMED CHANGE to an entity: opened, committed, merged, reverted, closed.

    ``timestamp``, ``actor`` and ``case_id`` are nullable. We *never* fabricate
    a missing one — its absence is a finding.

    Two separate confidences live here on purpose:
      - ``confidence``       : how sure we are the *event itself* happened
                               (a git-read event is `direct`).
      - ``case_confidence``  : how sure we are this event belongs to ``case_id``
                               (assigned by correlation, step 2, and scored
                               *per link* — a deterministic join is `joined`, a
                               fuzzy/proximity join is `heuristic`/`model`).
    Conflating the two would hide exactly the uncertainty the task cares about.
    """

    id: str
    entity_id: str
    action: str
    source: str
    confidence: Confidence
    evidence: list[Evidence] = field(default_factory=list)
    timestamp: Optional[str] = None   # ISO-8601, or None when genuinely unknown
    actor: Optional[str] = None       # a person entity id, or None (never invented)
    case_id: Optional[str] = None     # assigned by correlate (step 2)
    case_confidence: Optional[Confidence] = None
    attrs: dict = field(default_factory=dict)
    raw: Any = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id,
            "entity_id": self.entity_id,
            "action": self.action,
            "source": self.source,
            "confidence": self.confidence.to_dict(),
            "timestamp": self.timestamp,
            "actor": self.actor,
            "case_id": self.case_id,
            "attrs": self.attrs,
        }
        if self.case_confidence is not None:
            d["case_confidence"] = self.case_confidence.to_dict()
        if self.evidence:
            d["evidence"] = [e.to_dict() for e in self.evidence]
        return d


@dataclass
class Observation:
    """A STATE SEEN at a point, with no action and maybe no time.

    This is how thin data enters the system. A changelog line "``**Version
    3.1.0**`` ... fixed X (#123)" tells us a state (this change shipped in this
    version) but records no actor and no wall-clock time and no order relative
    to its siblings. ``actor`` stays None, ``seen_at`` stays None, confidence
    is low. The engine's job is to *surface* that thinness, not paper over it.
    """

    id: str
    entity_id: str
    state: dict
    source: str
    confidence: Confidence
    evidence: list[Evidence] = field(default_factory=list)
    seen_at: Optional[str] = None
    case_id: Optional[str] = None
    case_confidence: Optional[Confidence] = None
    raw: Any = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id,
            "entity_id": self.entity_id,
            "state": self.state,
            "source": self.source,
            "confidence": self.confidence.to_dict(),
            "seen_at": self.seen_at,
            "case_id": self.case_id,
        }
        if self.case_confidence is not None:
            d["case_confidence"] = self.case_confidence.to_dict()
        if self.evidence:
            d["evidence"] = [e.to_dict() for e in self.evidence]
        return d


# A single record stream flows through the pipeline. Downstream steps see only
# these three types and never the source format — that is the whole point of
# the thin adapters.
Record = Any  # Entity | Event | Observation (kept loose to avoid a heavy Union everywhere)


def to_json(obj: Any, **kwargs) -> str:
    """JSON-encode any of our dataclasses (or a plain container of them)."""

    def default(o):
        if hasattr(o, "to_dict"):
            return o.to_dict()
        if isinstance(o, Tier):
            return o.label
        raise TypeError(f"not serialisable: {type(o)!r}")

    return json.dumps(obj, default=default, **kwargs)
