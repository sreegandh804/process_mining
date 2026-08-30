"""The link vocabulary — how an adapter says "this record points at that one".

This is the seam that keeps correlation source-agnostic. Before it, every source
grew its own correlator: git walked the commit DAG, the spreadsheet path matched
foreign keys, the changelog matched PR numbers, and a GitHub adapter would have
made four. Each one re-implemented union-find, re-decided when an inference is
`joined` vs `heuristic`, and — worse — could only ever join records *within* its
own source. Cross-source correlation was structurally impossible.

A `Link` is the one shape all of them collapse into:

    "commit f1 belongs to pr:1, because the DAG says it is reachable from the
     merge but not from the trunk — that is structural, so `joined`."
    "payment:88 points at invoice:INV-312, because a foreign-key column says
     so — `joined`."
    "email:41 points at thread:re-march-invoices — `joined`."

The adapter supplies the *domain judgement* (what points at what, and how sure
that makes us). The correlator supplies the *mechanics* (resolve, union, anchor,
score) and never learns a source's name. Adding a source is then an adapter and
nothing else — the contract `adapters/__init__.py` has claimed all along.

Three flags carry the cases that would otherwise justify a bespoke correlator:

- ``materialise`` — the target is real but we never saw its record (a PR known
  only because a commit cites it). Create it as an *inferred* entity, so its
  existence reads as inference and its empty timeline becomes a gap. Without
  this, an adapter would have to synthesise entities itself and the "why is this
  here?" trail would break. When it is None an unresolvable link is recorded as
  an unresolved reference — a reconciliation finding, never a silent drop.

- ``anchors`` — the target names the case. A run of commits is "PR #1", not
  "commit f1". Adapters know which end of a link is the process instance, and
  ``anchor_rank`` settles it when two of them claim the same run (a PR names a
  run more strongly than the issue it closes). That judgement is the adapter's
  because it is domain knowledge; the correlator only compares the numbers.

- ``fallback`` — apply only if nothing stronger claimed this record. This is how
  "group the leftovers" rules (changelog bullets with no PR fall back to their
  version; emails with no In-Reply-To fall back to their subject thread) stay
  declarative. Applied eagerly they would fuse every real case that contains one
  leftover into a single mega-case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from induction.model import Confidence, Tier


@dataclass(frozen=True)
class Link:
    """One declared pointer from the entity carrying it to ``target``."""

    target: str                       # entity id this record points at
    method: str                       # how we know ("merge-topology", "foreign-key", ...)
    tier: Tier                        # how strong that makes the claim
    rationale: str                    # prose — becomes the join's recorded reason
    snippet: Optional[str] = None     # the exact text the link was read from
    locator: Optional[str] = None     # where to look to check it
    materialise: Optional[str] = None       # entity type to create if target is absent
    materialise_attrs: dict = field(default_factory=dict)
    virtual: bool = False             # target is a case identity, not a record
    anchors: bool = False             # the TARGET should name the case
    anchor_rank: int = 0              # 0 = strongest claim; breaks ties between anchors
    anchor_attrs: dict = field(default_factory=dict)   # how that case describes itself
    fallback: bool = False            # apply only if the source is still unlinked

    @property
    def confidence(self) -> Confidence:
        return Confidence(self.tier, self.rationale)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "target": self.target, "method": self.method,
            "tier": self.tier.label, "rationale": self.rationale,
        }
        if self.virtual:
            d["virtual"] = True
        for key in ("snippet", "locator", "materialise"):
            val = getattr(self, key)
            if val is not None:
                d[key] = val
        if self.materialise_attrs:
            d["materialise_attrs"] = self.materialise_attrs
        if self.anchors:
            d["anchors"] = True
        if self.anchor_rank:
            d["anchor_rank"] = self.anchor_rank
        if self.anchor_attrs:
            d["anchor_attrs"] = self.anchor_attrs
        if self.fallback:
            d["fallback"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Link":
        return cls(
            target=d["target"], method=d.get("method", "reference"),
            tier=Tier.from_label(d.get("tier", "heuristic")),
            rationale=d.get("rationale", ""),
            snippet=d.get("snippet"), locator=d.get("locator"),
            materialise=d.get("materialise"),
            materialise_attrs=d.get("materialise_attrs", {}) or {},
            virtual=bool(d.get("virtual")),
            anchors=bool(d.get("anchors")),
            anchor_rank=int(d.get("anchor_rank", 0)),
            anchor_attrs=d.get("anchor_attrs", {}) or {},
            fallback=bool(d.get("fallback")),
        )


# Links live in ``entity.attrs`` as plain dicts so the emitted model stays JSON
# and a reader can see, on the record itself, every claim made about it.
ATTR = "links"


def declare(entity, *links: Link) -> None:
    """Attach one or more links to an entity. Adapters call this; nothing else does."""
    if not links:
        return
    bucket = entity.attrs.setdefault(ATTR, [])
    seen = {(d["target"], d.get("method")) for d in bucket}
    for link in links:
        key = (link.target, link.method)
        if key in seen:
            continue          # the same claim twice is still one claim
        seen.add(key)
        bucket.append(link.to_dict())


def links_of(entity) -> list[Link]:
    """Decode the links declared on an entity."""
    return [Link.from_dict(d) for d in entity.attrs.get(ATTR, [])]


def unresolved(entity) -> list[dict]:
    """Links whose target never turned up — a reconciliation finding, not a drop."""
    return entity.attrs.get("unresolved_links", [])


def record_unresolved(entity, link: Link) -> None:
    entity.attrs.setdefault("unresolved_links", []).append(link.to_dict())
