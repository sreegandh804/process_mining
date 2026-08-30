"""Thin adapters: one per source, ``raw record -> [Entity | Event | Observation]``.

Everything downstream of an adapter sees ONLY the three canonical types and
never the source format. Adding a new source (a GitHub-API loader, a CSV, a
mailbox) means adding one adapter here and nothing else in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from induction.model import Entity, Event, Observation


@dataclass
class Shaped:
    """What an adapter emits. The pipeline concatenates these across sources."""

    entities: list[Entity] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)

    def extend(self, other: "Shaped") -> None:
        self.entities.extend(other.entities)
        self.events.extend(other.events)
        self.observations.extend(other.observations)
