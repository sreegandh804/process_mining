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


@dataclass
class EventLogSpec:
    """The OTHER shape a table can have: one row per event, not per case.

    `TableSpec` above maps COLUMN NAMES to activities — right for a tracker
    export, where a row is an invoice and `approved_date` is a column. It cannot
    express an event log, where the activity is a VALUE in a cell and one case
    spans many rows. That is not a formatting quibble: a wide table has exactly
    one `approved_date` column, so a step can happen at most once, and rework
    loops — a step repeating — are unrepresentable in it.

    Long format is the field's standard (case id / activity / timestamp; the XES
    triple). Between the two shapes, tabular data is covered: everything after
    this is a spec, not code.
    """

    source: str                                   # "log:acme/accounts-payable"
    entity_type: str                              # "invoice"
    case_id_column: str
    activity_column: str
    timestamp_column: str
    actor_column: Optional[str] = None
    attr_columns: list[str] = field(default_factory=list)
    text_column: Optional[str | list[str]] = None
    status_column: Optional[str] = None


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


# An event log orders several events per case per day, so unlike a tracker's
# date columns the time of day is load-bearing. Day-first before month-first, to
# match `_DATE_FORMATS`; the reading is reported so an ambiguous corpus can be
# spotted rather than silently mis-ordered.
_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
    "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M",
    "%d/%m/%Y %I:%M %p", "%m/%d/%Y %I:%M %p",
)


def parse_timestamp(raw: str) -> Optional[str]:
    """An ISO timestamp, keeping the time when the source gives one.

    Falls back to `parse_date` (midnight-free, date only) rather than inventing a
    time — an unrecorded hour is unknown, not 00:00.
    """
    s = (raw or "").strip()
    if not s:
        return None
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(s, fmt).isoformat()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s).isoformat()
    except ValueError:
        pass
    return parse_date(s)


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
        reader = csv.DictReader(f, delimiter=_sniff_delimiter(path))
        rows = []
        for row in reader:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items() if k})
        return rows


def _sniff_delimiter(path: Path) -> str:
    """`,` or `;` — an export from a European locale uses the semicolon, and
    read with the wrong one every row becomes a single unusable column."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        header = f.readline()
    return max((",", ";", "\t"), key=header.count) if header else ","


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


def load_event_log(spec: EventLogSpec, path: str | Path, max_cases: Optional[int] = None) -> Shaped:
    """One row per event: group the rows into cases, one Event each.

    Nothing downstream learns a new concept — this produces the same
    `Entity`/`Event` records the wide reader does. The activity is the cell's own
    value, kept verbatim, which is the engine's standing rule: an activity is
    named by the label the source itself gave it.
    """
    out = Shaped()
    people = _People(spec.source)
    filename = Path(path).name
    cases: dict[str, Entity] = {}
    dropped = 0

    for i, row in enumerate(read_rows(path), start=1):
        if not any(v for v in row.values()):
            continue
        locator = f"{filename}:row{i}"
        case_id = (row.get(spec.case_id_column) or "").strip()
        activity = (row.get(spec.activity_column) or "").strip()
        if not case_id or not activity:
            # A row with no case or no activity cannot be placed. Surfaced, in
            # the same queue a wide row with no identity goes to — never dropped
            # quietly, never attached to a neighbouring case because it happens
            # to sit next to one in the file.
            dropped += 1
            out.entities.append(Entity(
                id=f"row:{spec.source}:{i}", source=spec.source, type="orphan_row",
                attrs={"reason": "no case id" if not case_id else "no activity"},
                confidence=heuristic(),
                evidence=[Evidence(spec.source, locator, "; ".join(
                    f"{k}={v}" for k, v in list(row.items())[:4] if v)[:160])]))
            continue

        ent_id = f"{spec.entity_type}:{case_id}"
        ent = cases.get(ent_id)
        if ent is None:
            if max_cases is not None and len(cases) >= max_cases:
                continue                      # slice cap: stop taking NEW cases
            attrs = {c: (row.get(c) or "").strip() for c in spec.attr_columns
                     if (row.get(c) or "").strip()}
            summary = _summary(spec, row)
            if summary:
                attrs["summary"] = summary
            if spec.status_column:
                status = (row.get(spec.status_column) or "").strip()
                if status:
                    attrs["status"] = status
            ent = Entity(id=ent_id, source=spec.source, type=spec.entity_type,
                         attrs=attrs, confidence=direct(),
                         evidence=[Evidence(spec.source, locator, case_id)])
            # The case id IS the correlation, given to us by the source — the one
            # place in this engine where a run needs no inference at all. Declared
            # like any other link so the one correlator resolves it, exactly as it
            # resolves a merge DAG or a foreign key.
            declare(ent, Link(
                target=ent_id, method="case-id", tier=Tier.DIRECT,
                rationale="the log states which case this event belongs to",
                locator=locator, snippet=case_id, anchors=True, anchor_rank=5,
            ))
            cases[ent_id] = ent
            out.entities.append(ent)
        elif ent_id not in cases:
            continue                          # capped out; its later rows go too

        actor = people.ensure(row.get(spec.actor_column) or "") if spec.actor_column else None
        out.events.append(Event(
            # One case repeats an activity (rework), so the row number — not the
            # activity — is what keeps event ids unique.
            id=f"evt:{spec.source}:{case_id}:{i}",
            entity_id=ent_id, action=activity, source=spec.source,
            timestamp=parse_timestamp(row.get(spec.timestamp_column) or ""),
            actor=actor, confidence=direct(),
            evidence=[Evidence(spec.source, locator, f"{case_id} · {activity}")]))

    out.entities.extend(people.entities())
    if dropped:
        print(f"[tabular] {dropped} row(s) had no case id or no activity — surfaced, not dropped")
    return out


# ---------------------------------------------------------------------------
# Detection — which shape is this table, and which columns carry the triple
# ---------------------------------------------------------------------------
# A real tool does not ask the user to declare the shape; it reads the file and
# proposes a mapping. But a proposal is an INFERENCE, so it arrives here like
# every other inference in this engine: at `heuristic`, saying exactly what it
# measured, so a reader can disagree with the evidence rather than the verdict.
#
# The decisive signal is not how many distinct values a column has — 'Payment
# Status' and 'Event Name' both have few. It is whether the column VARIES INSIDE
# A CASE. A status is one value per invoice; an activity is eight. Nothing else
# separates them as cleanly, and nothing about it is domain knowledge.

_DETECT_SAMPLE = 5000
_ACTOR_HINTS = ("user", "actor", "resource", "performer", "who", "operator", "agent", "owner")
# A case is a run, not a bucket: beyond this many rows per group the column is an
# attribute everything shares, not an identity.
_MAX_EVENTS_PER_CASE = 100
# More labels than this and it is free text, not a step vocabulary.
_MAX_ACTIVITIES = 100
# What share of a case's rows must carry a distinct activity value.
_MIN_ACTIVITY_DENSITY = 0.5
# How unique an identity column must be. Not 1.0: a real export has duplicate
# and blank keys, and those are findings, not grounds to reject the file.
_MIN_ID_UNIQUENESS = 0.85


@dataclass
class Detection:
    mode: Optional[str]                # "long" | "wide" | None
    spec: object = None                # EventLogSpec | TableSpec
    rationale: str = ""
    confidence: Confidence = field(default_factory=lambda: heuristic("shape not determined"))


def detect(path: str | Path, source: Optional[str] = None,
           entity_type: str = "case") -> Detection:
    """Read the head of a table and propose how to read it."""
    rows = read_rows(path)[:_DETECT_SAMPLE]
    if not rows:
        return Detection(None, rationale="the file has no data rows")
    source = source or f"table:{Path(path).stem}"
    cols = list(rows[0].keys())
    values = {c: [(r.get(c) or "").strip() for r in rows] for c in cols}
    distinct = {c: len({v for v in values[c] if v}) for c in cols}

    ts_col, ts_rate = _best_timestamp_column(values)
    long_ = _detect_long(rows, cols, values, distinct, ts_col, source, entity_type)
    if long_:
        return long_
    wide = _detect_wide(rows, cols, values, distinct, source, entity_type)
    if wide:
        return wide
    return Detection(None, rationale=(
        f"neither shape fits: no column varies within a group the way an activity "
        f"does (so not an event log), and there is no mostly-unique identity "
        f"column beside two or more date columns (so not a tracker export). "
        f"Best timestamp candidate was {ts_col or 'none'} at {ts_rate:.0%}"))


def _best_timestamp_column(values: dict) -> tuple[Optional[str], float]:
    best, best_rate = None, 0.0
    for c, vals in values.items():
        filled = [v for v in vals if v]
        if len(filled) < max(3, 0.5 * len(vals)):
            continue                       # too sparse to be THE timestamp
        rate = sum(1 for v in filled if parse_timestamp(v)) / len(filled)
        if rate > best_rate:
            best, best_rate = c, rate
    return best, best_rate


def _detect_long(rows, cols, values, distinct, ts_col, source, entity_type):
    """Find the (case, activity) pair, if the table has one.

    Three properties separate a real pair from a coincidence, and all three are
    needed — the first attempt used only the middle one and confidently reported
    that 'Payment Method' was the case id, because with three groups covering
    every row, EVERY column "varies a lot" inside a group:

      1. a case is SHORT — a handful of events, not a bucket of thousands;
      2. within one case the activity nearly always changes — a status is one
         value per case, an activity is one per row;
      3. there are far more cases than activities. A log is many runs of a few
         steps; the reverse is a lookup table.
    """
    if not ts_col:
        return None
    n = len(rows)
    case_cands = [c for c in cols if c != ts_col and 2 <= distinct[c] <= n * 0.95
                  and 1.5 <= n / max(1, distinct[c]) <= _MAX_EVENTS_PER_CASE]
    act_cands = [c for c in cols if c != ts_col and 2 <= distinct[c] <= _MAX_ACTIVITIES]
    best = None
    for case_c in case_cands:
        groups: dict[str, list[int]] = {}
        for i, v in enumerate(values[case_c]):
            if v:
                groups.setdefault(v, []).append(i)
        if not groups:
            continue
        for act_c in act_cands:
            if act_c == case_c or distinct[case_c] <= distinct[act_c]:
                continue                   # (3) more activities than cases: not a log
            # (2) scale-free: what FRACTION of a case's rows carry a distinct
            # activity. ~1.0 for a real activity column, ~0 for an attribute.
            per_group = [len({values[act_c][i] for i in idx if values[act_c][i]}) / len(idx)
                         for idx in groups.values()]
            density = sum(per_group) / len(per_group)
            spread = sum(len({values[act_c][i] for i in idx if values[act_c][i]})
                         for idx in groups.values()) / len(groups)
            if density < _MIN_ACTIVITY_DENSITY or spread < 2.0:
                continue
            score = (round(density, 3), distinct[case_c])
            if best is None or score > best[0]:
                best = (score, case_c, act_c, len(groups), spread, density)
    if best is None:
        return None
    _, case_c, act_c, n_groups, spread, density = best
    actor_c = next((c for c in cols if c not in (case_c, act_c, ts_col)
                    and any(h in c.lower() for h in _ACTOR_HINTS)), None)
    attrs = [c for c in cols if c not in (case_c, act_c, ts_col, actor_c)]
    filled = sum(1 for v in values[ts_col] if v)
    rate = sum(1 for v in values[ts_col] if v and parse_timestamp(v)) / max(1, filled)
    spec = EventLogSpec(
        source=source, entity_type=entity_type, case_id_column=case_c,
        activity_column=act_c, timestamp_column=ts_col, actor_column=actor_c,
        attr_columns=attrs)
    return Detection("long", spec, confidence=heuristic("table shape inferred, not declared"),
                     rationale=(
        f"read as an event log: {case_c!r} groups the sample into {n_groups} cases of "
        f"~{n / max(1, n_groups):.1f} rows, and {act_c!r} — {distinct[act_c]} values — "
        f"changes on {density:.0%} of the rows within a case ({spread:.1f} distinct per "
        f"case), which an attribute or a status does not; {ts_col!r} parsed on "
        f"{rate:.0%} of rows (day-first where ambiguous)"
        + (f"; {actor_c!r} read as the actor" if actor_c else "; no actor column found")))


def _detect_wide(rows, cols, values, distinct, source, entity_type):
    """One row per case, one date column per step.

    The identity column is *mostly* unique, not perfectly: a real export has
    duplicate keys and blank ones, and those are findings this engine exists to
    surface — demanding a perfect key would reject exactly the messy sheet it is
    for. Coverage breaks ties, so a well-populated id beats a sparse one that
    happens to look tidier.
    """
    n = len(rows)
    date_cols = []
    for c in cols:
        filled = [v for v in values[c] if v]
        if filled and sum(1 for v in filled if parse_date(v)) / len(filled) >= 0.8:
            date_cols.append(c)
    if len(date_cols) < 2 or n < 2:
        return None

    id_col = None
    best = ()
    for c in cols:
        if c in date_cols:
            continue
        filled = sum(1 for v in values[c] if v)
        if filled < n * 0.5 or not filled:
            continue
        if distinct[c] / filled < _MIN_ID_UNIQUENESS:
            continue
        score = (filled, distinct[c])
        if not best or score > best:
            best, id_col = score, c
    if not id_col:
        return None

    actors = {c for c in cols if any(h in c.lower() for h in _ACTOR_HINTS)
              or c.lower().endswith("_by")}
    events = [EventCol(action=_action_from(c), date_column=c,
                       actor_column=_actor_for(c, actors)) for c in date_cols]
    attrs = [c for c in cols if c not in date_cols and c != id_col]
    spec = TableSpec(source=source, entity_type=entity_type, id_column=id_col,
                     event_columns=events, attr_columns=attrs)
    uniq = distinct[id_col] / max(1, sum(1 for v in values[id_col] if v))
    return Detection("wide", spec, confidence=heuristic("table shape inferred, not declared"),
                     rationale=(
        f"read as a tracker export: {id_col!r} is {uniq:.0%} unique across "
        f"{n} rows, and {len(date_cols)} columns parse as dates "
        f"({', '.join(date_cols[:4])}{'...' if len(date_cols) > 4 else ''}), so a "
        f"row is one case and each date column is one step"))


def _action_from(column: str) -> str:
    a = column.strip().lower()
    for suffix in ("_date", "_at", "_on", " date", "date"):
        if a.endswith(suffix) and len(a) > len(suffix):
            a = a[: -len(suffix)]
            break
    return a.strip(" _-") or column.strip().lower()


def _actor_for(date_column: str, actors: set) -> Optional[str]:
    stem = _action_from(date_column)
    return next((a for a in actors if stem and stem in a.lower()), None)
