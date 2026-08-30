"""Tabular adapter (Shape) — the thin, non-git customer: CSV / Excel exports.

This is the "accounting firm with spreadsheets" end of the market. The whole
argument of the engine is that the pipeline downstream of an adapter is
source-agnostic; this adapter is the proof, in a domain that shares nothing with
git.

It is **spec-driven**, on purpose: a spreadsheet adapter that hardcoded invoice
columns would be a toy. A `TableSpec` says which column is the identity, which
columns are timestamped events (and who performed them), which is the current
status, and which columns are foreign keys to other rows. A new customer's
tracker — tickets, onboarding, cases — is a new *spec*, not new code.

The honest, thin-customer realities are first-class, not swept up:
  - a blank date column means the step is simply not recorded — no event, no
    invented time;
  - a blank actor column means the actor is unknown — `actor = None`, never
    guessed;
  - a row with a status but no dated activity is an `Observation` (state seen,
    order unknown);
  - a row with no identity, or a foreign key to a row that does not exist, is
    surfaced (as an orphan / an unresolved reference), never dropped.

CSV is read with the standard library (zero dependencies). `.xlsx` is read with
openpyxl if it is installed (`pip install openpyxl`); the CSV path needs nothing.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from induction.adapters import Shaped
from induction.links import Link, declare
from induction.model import Confidence, Entity, Evidence, Event, Observation, Tier, direct, heuristic

_BOT_ACTORS = ("system", "auto", "automated", "bot", "robot", "batch")
# Tried in order. Day-first before month-first: a UK finance export is far more
# likely to be dd/mm than mm/dd. Real deployments would set this per source.
_DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%Y/%m/%d", "%m/%d/%Y")


@dataclass
class EventCol:
    action: str
    date_column: str
    actor_column: Optional[str] = None


@dataclass
class TableSpec:
    source: str                                   # "excel:acme-finance/invoices"
    entity_type: str                              # "invoice"
    id_column: str
    event_columns: list[EventCol] = field(default_factory=list)
    status_column: Optional[str] = None
    attr_columns: list[str] = field(default_factory=list)
    # Which column(s) hold free text a fuzzy join can read (a description, a
    # note, a client name) — a name or a list of them. Copied to `summary`
    # because that is an attribute name the correlator looks for; the sheet's own
    # column names are arbitrary and the correlator must never learn them.
    text_column: Optional[str | list[str]] = None
    # foreign keys: {"column": "invoice_id", "target_type": "invoice"}
    ref_columns: list[dict] = field(default_factory=list)


def parse_date(raw: str) -> Optional[str]:
    """Return an ISO date, or None if blank/unparseable (never a guess)."""
    s = (raw or "").strip()
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s).date().isoformat()
    except ValueError:
        return None


def _is_bot(name: str) -> bool:
    n = name.strip().lower()
    return any(b in n for b in _BOT_ACTORS)


class _People:
    def __init__(self, source: str):
        self.source = source
        self._by_id: dict[str, Entity] = {}

    def ensure(self, name: str) -> Optional[str]:
        name = (name or "").strip()
        if not name:
            return None
        pid = f"person:{name.lower()}"
        ent = self._by_id.get(pid)
        if ent is None:
            self._by_id[pid] = Entity(
                id=pid, source=self.source, type="person",
                attrs={"name": name, "is_bot": _is_bot(name), "commit_count": 1},
                confidence=direct(),
            )
        else:
            ent.attrs["commit_count"] += 1
        return pid

    def entities(self):
        return list(self._by_id.values())


def read_rows(path: str | Path) -> list[dict]:
    """Read a CSV or .xlsx into a list of {column: value} dicts (strings)."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        return _read_xlsx(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items() if k})
        return rows


def _read_xlsx(path: Path) -> list[dict]:
    try:
        import openpyxl  # optional dependency, only for .xlsx
    except ImportError as e:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"reading {path.name} needs openpyxl (`pip install openpyxl`); or export the "
            f"sheet to CSV, which needs no dependency."
        ) from e
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb.active
    it = ws.iter_rows(values_only=True)
    headers = [str(h).strip() if h is not None else "" for h in next(it)]
    rows = []
    for r in it:
        row = {}
        for i, v in enumerate(r):
            if i >= len(headers) or not headers[i]:
                continue
            if isinstance(v, datetime):
                v = v.date().isoformat()
            row[headers[i]] = "" if v is None else str(v).strip()
        rows.append(row)
    return rows


def load(spec: TableSpec, path: str | Path) -> Shaped:
    out = Shaped()
    people = _People(spec.source)
    filename = Path(path).name
    seen_ids: dict[str, int] = {}

    for i, row in enumerate(read_rows(path), start=1):
        if not any(v for v in row.values()):
            continue  # a truly blank line carries nothing to surface
        locator = f"{filename}:row{i}"
        _shape_row(spec, row, i, locator, people, seen_ids, out)

    out.entities.extend(people.entities())
    return out


def _summary(spec, row: dict) -> str:
    """The free text this row offers a fuzzy join, in the spec's column order."""
    if not spec.text_column:
        return ""
    columns = ([spec.text_column] if isinstance(spec.text_column, str)
               else list(spec.text_column))
    return " ".join(str(row.get(c) or "").strip() for c in columns).strip()


def _shape_row(spec, row, rownum, locator, people, seen_ids, out: Shaped) -> None:
    rid = (row.get(spec.id_column) or "").strip()
    snippet = " | ".join(f"{k}={v}" for k, v in row.items() if v)[:200]

    # A row with no identity key cannot be an entity — surface it, do not drop it.
    if not rid:
        out.observations.append(Observation(
            id=f"obs:{spec.source}:row{rownum}",
            entity_id=f"row:{spec.source}:{rownum}",
            state=dict(row),
            source=spec.source,
            confidence=heuristic("row has no identity key — cannot be attributed to a case"),
            evidence=[Evidence(spec.source, locator, snippet)],
            seen_at=None,
        ))
        out.entities.append(Entity(
            id=f"row:{spec.source}:{rownum}", source=spec.source, type="orphan_row",
            attrs={"reason": "no identity key"}, confidence=heuristic(),
            evidence=[Evidence(spec.source, locator, snippet)]))
        return

    ent_id = f"{spec.entity_type}:{rid}"
    dup = rid in seen_ids
    seen_ids[rid] = seen_ids.get(rid, 0) + 1

    refs = []
    for rc in spec.ref_columns:
        val = (row.get(rc["column"]) or "").strip()
        if val:
            refs.append({"type": rc["target_type"], "key": val})

    existing = next((e for e in out.entities if e.id == ent_id), None)
    if existing is None:
        entity = Entity(
            id=ent_id, source=spec.source, type=spec.entity_type,
            attrs={
                **{c: row.get(c, "") for c in spec.attr_columns},
                "status": row.get(spec.status_column, "") if spec.status_column else "",
                "summary": _summary(spec, row),
                "references": refs,
                "n_rows": 1,
            },
            confidence=direct(),
            evidence=[Evidence(spec.source, locator, snippet)],
            raw=dict(row),
        )
        # A foreign key is a declared link like any other — the same shape a
        # commit's "(#123)" or a mail header produces, resolved by the same
        # correlator. The referenced row *anchors* the run (a payment belongs to
        # its invoice, not the other way round). Nothing is materialised: a
        # dangling invoice_id is a data-quality finding, not licence to invent
        # the invoice it points at.
        # A row with an identity key is a run in its own right — an invoice
        # owns "raised -> approved -> paid" whether or not anything points at
        # it. Declared at a weak anchor rank so that when something *does* point
        # at it, the pointed-at record names the run (the payment belongs to the
        # invoice, not the reverse).
        declare(entity, Link(
            target=ent_id, method="row-identity", tier=Tier.DIRECT,
            rationale="row carries its own identity key and owns the records on it",
            locator=locator, snippet=snippet, anchors=True, anchor_rank=5,
        ))
        for ref in refs:
            declare(entity, Link(
                target=f"{ref['type']}:{ref['key']}", method="foreign-key",
                tier=Tier.JOINED,
                rationale="records share a row identity / foreign key",
                locator=locator, snippet=snippet, anchors=True,
            ))
        out.entities.append(entity)
    else:
        existing.attrs["n_rows"] += 1
        existing.attrs["duplicate_id"] = True   # two rows claim the same identity
        existing.evidence.append(Evidence(spec.source, locator, snippet))

    dated = 0
    for ec in spec.event_columns:
        ts = parse_date(row.get(ec.date_column, ""))
        if ts is None:
            continue
        dated += 1
        actor = people.ensure(row.get(ec.actor_column, "")) if ec.actor_column else None
        out.events.append(Event(
            id=f"evt:{spec.source}:{rid}:{ec.action}:{seen_ids[rid]}",
            entity_id=ent_id,
            action=ec.action,
            source=spec.source,
            confidence=direct(),
            evidence=[Evidence(spec.source, locator, f"{ec.date_column}={row.get(ec.date_column)}")],
            timestamp=ts,
            actor=actor,
            attrs={"duplicate_row": dup} if dup else {},
        ))

    # No dated activity at all → this row is a *state seen*, order unknown.
    if dated == 0:
        out.observations.append(Observation(
            id=f"obs:{spec.source}:{rid}:{seen_ids[rid]}",
            entity_id=ent_id,
            state={"status": row.get(spec.status_column, "") if spec.status_column else "",
                   **{c: row.get(c, "") for c in spec.attr_columns}},
            source=spec.source,
            confidence=Confidence(Tier.DIRECT,
                                 "state read from the row; no dated activity, so no order"),
            evidence=[Evidence(spec.source, locator, snippet)],
            seen_at=None,
        ))
