"""Owner review — confirm or dispute what the engine surfaced, and keep both.

A process owner has to recognise a process as theirs before they will change it,
and they are sometimes wrong about it: they describe the official version while
the records show another. This module is where both of those survive.

Each statement on the page (a process, one of its steps, its headline finding) is
a *claim* with a stable id derived from what it says — not from a counter — so an
answer given on one run attaches to the same claim on the next. The inspector
exports the answers as `corrections.json`; the next run reads it back:

  - confirmed  -> the claim is shown as "Confirmed by process owner".
  - disputed   -> the claim is NOT removed. It is shown beside what the records
                  still say ("Records show this step in 52 of 63 invoices"), and
                  written to `model.json` under `divergence` — belief vs data.
  - unmatched  -> an answer whose claim no longer exists (the data changed, or a
                  name did) is listed, never silently dropped.

The file format, kept deliberately small so a person can read and edit it:

    {"version": 1, "answers": [
        {"claim_id": "c-3f2a…", "claim": "<what the page said>",
         "verdict": "confirm" | "dispute", "note": "…", "at": "<iso time>"}]}
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

VERDICTS = ("confirm", "dispute")


def claim_id(*parts: str) -> str:
    """A stable id from a claim's content: same statement, same id, every run."""
    raw = "\x1f".join(str(p).strip().lower() for p in parts)
    return "c-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def load(path: Optional[str | Path]) -> dict[str, dict]:
    """Read a corrections file into {claim_id: answer}. The last answer wins.

    A missing path is an empty review, not an error. A malformed file IS an
    error: silently ignoring someone's review would be worse than stopping."""
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"[run] corrections file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise SystemExit(f"[run] {p} is not a review exported from the inspector: {e}")
    answers = data.get("answers", []) if isinstance(data, dict) else []
    out: dict[str, dict] = {}
    for a in answers:
        if not isinstance(a, dict) or a.get("verdict") not in VERDICTS or not a.get("claim_id"):
            continue
        out[a["claim_id"]] = {"claim_id": a["claim_id"], "claim": a.get("claim", ""),
                              "verdict": a["verdict"], "note": (a.get("note") or "").strip(),
                              "at": a.get("at", "")}
    return out


def status_of(cid: str, answers: dict[str, dict]) -> dict:
    """What the page shows for one claim."""
    a = answers.get(cid)
    if not a:
        return {"state": "pending"}
    return {"state": "confirmed" if a["verdict"] == "confirm" else "disputed",
            "note": a["note"], "at": a["at"]}


def reconcile(claims: list[dict], answers: dict[str, dict]) -> dict:
    """Split the answers against this run's claims.

    `claims`: each `{"id", "text", "evidence"}` where `evidence` is what the
    records currently say about it, in words. Returns the divergence items (a
    disputed claim beside its evidence) and the answers that matched nothing."""
    by_id = {c["id"]: c for c in claims}
    divergence, unmatched = [], []
    for cid, a in answers.items():
        c = by_id.get(cid)
        if c is None:
            unmatched.append(a)
        elif a["verdict"] == "dispute":
            divergence.append({"claim_id": cid, "claim": c["text"], "owner_says": a["note"],
                               "records_show": c.get("evidence", ""), "at": a["at"]})
    return {"divergence": divergence, "unmatched": unmatched,
            "n_confirmed": sum(1 for cid, a in answers.items()
                               if a["verdict"] == "confirm" and cid in by_id),
            "n_disputed": len(divergence), "n_claims": len(claims)}
