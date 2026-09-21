"""What a typed model was asked, and what the engine did with the answer.

A number on a page is not traceability. `0.92` beside a step tells a reader that
something was confident about something, and gives them no way to disagree: they
cannot see what question was asked, which options were on the table, or what the
engine did once the answer came back. A `model`-tier join in this codebase has
always carried a sentence a reader can argue with; a typed decision needs the
same, and its sentence is the question itself.

So every typed decision that shows up in the output is recorded here, with:

  - the question, as the model was asked it;
  - every option it could have picked, and what each one scored;
  - what it picked, and how sure;
  - and what the ENGINE then did — which is the part a reader most often wants,
    because the engine is free to ignore the answer and frequently does.

The ledger lives on the `InducedModel` and is written into `model.json`, not
just rendered in the inspector. That order matters: `model.json` is the artefact
and the page is a view of it, so a decision that only existed in HTML would be a
claim no downstream tool could audit.

**Not every call is recorded, on purpose.** A decision belongs here when a reader
can see its consequence and might dispute it: which step a record performs, why
two records are one case, why a proposed step is missing from the vocabulary, why
a cluster is flagged as a look-alike. A call that only narrowed what the engine
looked at — the gate, the discovery sample — asserts nothing about any record,
and recording it per-record would bury the four that matter under thousands that
do not. Those stay as counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# The four decision kinds a reader can see the consequence of.
STEP = "step"        # this record performs this stage
JOIN = "join"        # these two records are the same piece of work
LABEL = "label"      # this proposed step is not a step, and was dropped
REJECT = "reject"    # this cluster looks like a process and is not


@dataclass(frozen=True)
class Decision:
    """One typed question, its answer, and what the engine did with it."""

    kind: str                 # one of the four above
    about: str                # the record / pair / label / cluster it concerns
    question: str             # as the model was asked it
    qtype: str                # noul | choice | score
    answer: Any               # what came back
    outcome: str              # what the ENGINE did — never assumed from `answer`
    confidence: Optional[float] = None
    distribution: dict[str, float] = field(default_factory=dict)
    # Extra numbers the engine computed FROM the answer (e.g. mass per process).
    # Kept apart from `distribution` so what the model said stays distinguishable
    # from what this code worked out afterwards.
    derived: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {
            "kind": self.kind,
            "about": self.about,
            "question": self.question,
            "type": self.qtype,
            "answer": self.answer,
            "outcome": self.outcome,
        }
        if self.confidence is not None:
            out["confidence"] = round(self.confidence, 4)
        if self.distribution:
            out["distribution"] = {k: round(v, 4) for k, v in
                                   sorted(self.distribution.items(), key=lambda kv: -kv[1])}
        if self.derived:
            out["derived"] = {k: round(v, 4) for k, v in
                              sorted(self.derived.items(), key=lambda kv: -kv[1])}
        return out


class Ledger:
    """Where decisions accumulate during a run.

    One per invocation, shared by every seam that makes a recordable decision, so
    the artefact holds them in the order the run made them. Appending is the only
    operation: nothing here edits or removes a decision once taken, because the
    record of a decision the engine later overruled is exactly the record worth
    keeping.
    """

    def __init__(self):
        self._rows: list[Decision] = []

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def add(self, decision: Decision) -> None:
        self._rows.append(decision)

    def record(self, **kwargs) -> None:
        self.add(Decision(**kwargs))

    def of_kind(self, kind: str) -> list[Decision]:
        return [row for row in self._rows if row.kind == kind]

    def to_list(self) -> list[dict]:
        return [row.to_dict() for row in self._rows]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in self._rows:
            out[row.kind] = out.get(row.kind, 0) + 1
        return out
