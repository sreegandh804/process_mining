"""The induced-model vocabulary — the shapes the engine *populates* from evidence.

`model.py` holds the substrate (Entity/Event/Observation — what we read).
This module holds what we *induce* on top of it and maps onto the target schema
(brief §3):

    a Case      = a Process INSTANCE (one run)
    a Variant   = one distinct way that kind of process actually ran (a real
                  trace + how often it happened)
    a ProcessKind = a Process DEFINITION (the shape shared across instances)
    a Step      = a named activity (a labelled Event or merged group)
    a Member    = an actor (a person entity)
    a Gap       = an inferred off-system step (never asserted as fact)
    an Orphan   = a record that joined to no case (surfaced, never dropped)

Everything here that is not read straight from the data carries a Confidence
and Evidence, exactly like the substrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from induction.model import Confidence, Evidence


@dataclass
class Case:
    """One run of a process (a Process instance).

    ``confidence`` is the confidence of the *grouping* — how sure we are these
    records belong together — which is the weakest link used to assemble it.
    """

    id: str
    kind_hint: str                       # "pr" | "issue" | "integration" | "release"
    anchor: dict                         # {"type": "pr", "number": 123}
    confidence: Confidence
    event_ids: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    # filled by later steps:
    ordered_event_ids: list[str] = field(default_factory=list)
    order_status: str = "unordered"      # "ordered" | "partial" | "unknown"
    trace_signature: tuple = ()          # the sequence of step-names (for variants)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind_hint": self.kind_hint,
            "anchor": self.anchor,
            "confidence": self.confidence.to_dict(),
            "n_events": len(self.event_ids),
            "order_status": self.order_status,
            "ordered_event_ids": self.ordered_event_ids,
            "trace_signature": list(self.trace_signature),
            "entity_ids": self.entity_ids,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass
class Variant:
    """One distinct trace shape within a process kind, with its real frequency."""

    signature: tuple
    frequency: int
    case_ids: list[str]
    role: str = "one-off"                # "common" | "exception" | "one-off"

    def to_dict(self) -> dict:
        return {
            "signature": list(self.signature),
            "frequency": self.frequency,
            "role": self.role,
            "example_case_ids": self.case_ids[:5],
            "n_cases": len(self.case_ids),
        }


@dataclass
class Step:
    """A named activity, traceable to the events it summarises."""

    id: str
    name: str
    action: str
    confidence: Confidence
    member_ids: list[str] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "action": self.action,
            "confidence": self.confidence.to_dict(),
            "members": self.member_ids,
            "n_events": len(self.event_ids),
            "evidence": [e.to_dict() for e in self.evidence],
            "attrs": self.attrs,
        }


@dataclass
class Gap:
    """An inferred off-system step. Rendered dashed/low-confidence — never fact."""

    id: str
    case_id: str
    kind: str                            # what sort of gap (e.g. "off_system_review")
    description: str
    confidence: Confidence               # heuristic/model — always an inference
    evidence: list[Evidence] = field(default_factory=list)
    between: Optional[tuple] = None      # (event_id_before, event_id_after)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "case_id": self.case_id,
            "kind": self.kind,
            "description": self.description,
            "inferred": True,
            "confidence": self.confidence.to_dict(),
            "between": list(self.between) if self.between else None,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass
class Orphan:
    """A record that joined to no case. Surfaced, with why."""

    record_id: str
    record_type: str                     # "event" | "observation"
    entity_id: str
    reason: str
    evidence: list[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "record_type": self.record_type,
            "entity_id": self.entity_id,
            "reason": self.reason,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass
class ProcessKind:
    """A Process definition — a kind of process, its variants and its verdict."""

    id: str
    name: str
    rationale: str
    confidence: Confidence
    case_ids: list[str] = field(default_factory=list)
    variants: list[Variant] = field(default_factory=list)
    dfg: dict = field(default_factory=dict)          # {"nodes":[...], "edges":[...]}
    steps: list[str] = field(default_factory=list)   # step ids seen in this kind
    features: dict = field(default_factory=dict)     # the structural features it clustered on
    rejected: bool = False
    reject_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "rationale": self.rationale,
            "confidence": self.confidence.to_dict(),
            "n_cases": len(self.case_ids),
            "case_ids": self.case_ids,
            "variants": [v.to_dict() for v in self.variants],
            "dfg": self.dfg,
            "steps": self.steps,
            "features": self.features,
            "rejected": self.rejected,
            "reject_reason": self.reject_reason,
        }
