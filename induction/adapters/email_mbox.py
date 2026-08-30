"""Email adapter (Shape) — the source the fuzzy pass was built for.

A mailbox is the hardest correlation case in the brief: mostly no structural key
at all. This adapter reads real RFC-822 messages (an Enron-style maildir tree, an
.mbox, or the Kaggle `emails.csv` with a raw-message column) and declares the two
kinds of link the one correlator already knows how to resolve:

  - **In-Reply-To / References** (tier `joined`): a real, structural thread key.
    Most correlation, when it exists, comes free from this.
  - **subject thread** (tier `heuristic`, a *fallback* + *virtual anchor*): the
    "Re: March invoices" grouping, applied only to a message nothing stronger
    claimed — the same declarative-fallback trick the changelog uses, so one
    stray message never fuses every thread that shares a word.

And where neither exists — a new mail about the same matter, no reply-link, a
changed subject — the correlator's own fuzzy pass (subject/body text + sender or
time proximity, tier `heuristic`) is what connects them. That is the whole point
of pointing this at email: it is the corpus a git DAG or a foreign key lets you
avoid, and it exercises every honesty path (one-off orphans, automated-notice
rejects, `order: unknown` on undated messages).

Standard library only (`email`, `mailbox`).
"""

from __future__ import annotations

import email
import email.utils
import hashlib
import mailbox
import re
from email.message import Message
from pathlib import Path
from typing import Iterable, Optional

from induction.adapters import Shaped
from induction.links import Link, declare
from induction.model import Entity, Evidence, Event, Tier, direct

_BOT_HINTS = ("noreply", "no-reply", "donotreply", "do-not-reply", "mailer-daemon",
              "postmaster", "notifications", "automated", "auto-confirm", "bounce")
_SUBJECT_PREFIX = re.compile(r"^\s*((re|fwd?|fw|aw|sv)(\[\d+\])?\s*:\s*)+", re.I)
_WS = re.compile(r"\s+")


def _addr(raw: str) -> Optional[str]:
    name, addr = email.utils.parseaddr(raw or "")
    return addr.lower() or None


def _addrs(raw: str) -> list[str]:
    return [a.lower() for _, a in email.utils.getaddresses([raw or ""]) if a]


def _is_bot(addr: str) -> bool:
    return any(h in (addr or "").lower() for h in _BOT_HINTS)


def _msgid(raw: str) -> Optional[str]:
    raw = (raw or "").strip()
    m = re.search(r"<([^>]+)>", raw)
    return (m.group(1) if m else raw) or None


def _norm_subject(subject: str) -> str:
    s = _SUBJECT_PREFIX.sub("", subject or "")
    return _WS.sub(" ", s).strip().lower()


def _iso(date_hdr: str) -> Optional[str]:
    try:
        dt = email.utils.parsedate_to_datetime(date_hdr)
        return dt.isoformat() if dt else None
    except (TypeError, ValueError, IndexError):
        return None


def _body_snippet(msg: Message, limit: int = 400) -> str:
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True) or b""
                    return payload.decode("utf-8", "replace")[:limit]
            return ""
        payload = msg.get_payload(decode=True)
        text = payload.decode("utf-8", "replace") if payload else str(msg.get_payload())
        return text[:limit]
    except Exception:
        return ""


class _People:
    def __init__(self, source: str):
        self.source = source
        self._by_id: dict[str, Entity] = {}

    def ensure(self, addr: Optional[str]) -> Optional[str]:
        if not addr:
            return None
        pid = f"person:mail:{addr}"
        if pid not in self._by_id:
            self._by_id[pid] = Entity(
                id=pid, source=self.source, type="person",
                attrs={"name": addr, "is_bot": _is_bot(addr), "commit_count": 1},
                confidence=direct())
        else:
            self._by_id[pid].attrs["commit_count"] += 1
        return pid

    def entities(self):
        return list(self._by_id.values())


def load(path: str | Path, slug: Optional[str] = None, max_messages: Optional[int] = None) -> Shaped:
    """Read a maildir tree, an .mbox file, or a CSV with a raw-message column."""
    path = Path(path)
    slug = slug or path.stem
    return shape(_read(path, max_messages), slug)


def _read(path: Path, max_messages: Optional[int]) -> Iterable[tuple[str, str]]:
    """Yield (locator, raw_message) pairs from whatever mailbox form is given."""
    n = 0

    def capped(it):
        nonlocal n
        for x in it:
            if max_messages and n >= max_messages:
                return
            n += 1
            yield x

    if path.is_dir():                                   # maildir tree (Enron layout)
        files = sorted(p for p in path.rglob("*") if p.is_file())
        yield from capped((str(p), p.read_text("utf-8", "replace")) for p in files)
    elif path.suffix.lower() == ".csv":                 # Kaggle emails.csv
        import csv
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            for i, row in enumerate(csv.DictReader(f)):
                msg = row.get("message") or row.get("raw") or ""
                if msg:
                    yield from capped([(row.get("file", f"row{i}"), msg)])
    elif path.suffix.lower() in (".mbox", ""):          # mbox file
        box = mailbox.mbox(str(path))
        yield from capped((f"{path.name}:{i}", m.as_string()) for i, m in enumerate(box))
    else:
        yield from capped([(str(path), path.read_text("utf-8", "replace"))])


def shape(messages: Iterable[tuple[str, str]], slug: str) -> Shaped:
    """The pure function: (locator, raw) pairs -> canonical records. `ingest` and
    tests both call this, so nothing about the mailbox layout leaks downstream."""
    source = f"mail:{slug}"
    out = Shaped()
    people = _People(source)

    for locator, raw in messages:
        msg = email.message_from_string(raw)
        _shape_message(msg, locator, slug, source, people, out)

    out.entities.extend(people.entities())
    return out


def _shape_message(msg: Message, locator: str, slug: str, source: str,
                   people: _People, out: Shaped) -> None:
    subject = str(msg.get("Subject", "") or "")
    frm = _addr(msg.get("From", ""))
    date = _iso(msg.get("Date", ""))
    mid = _msgid(msg.get("Message-ID", "")) or hashlib.sha1(
        f"{frm}|{msg.get('Date','')}|{subject}".encode()).hexdigest()[:16]
    ent_id = f"email:{slug}:{mid}"
    body = _body_snippet(msg)
    in_reply_to = _msgid(msg.get("In-Reply-To", ""))
    to = _addrs(msg.get("To", ""))
    cc = _addrs(msg.get("Cc", ""))

    from_id = people.ensure(frm)
    for a in to + cc:                       # participants become Members, not events
        people.ensure(a)

    ent = Entity(
        id=ent_id, source=source, type="email",
        attrs={"subject": subject, "body": body, "from": frm, "to": to, "cc": cc,
               "message_id": mid, "in_reply_to": in_reply_to},
        confidence=direct(), evidence=[Evidence(source, locator, subject[:200])], raw=None)
    out.entities.append(ent)

    # Deterministic thread key: the message this one replied to.
    if in_reply_to:
        declare(ent, Link(
            target=f"email:{slug}:{in_reply_to}", method="in-reply-to", tier=Tier.JOINED,
            rationale=f"In-Reply-To <{in_reply_to}>", locator=locator, snippet=subject[:120]))
    # Fallback thread by normalised subject — only if nothing stronger claims it.
    # NOT for automated senders: a daily notice with a fixed subject would fuse
    # every day's copy into one fake "thread"; kept separate, N copies read as the
    # recurring, produces-nothing pattern the reject rule is meant to catch.
    norm = _norm_subject(subject)
    if norm and not _is_bot(frm or ""):
        declare(ent, Link(
            target=f"thread:{slug}:{norm}", method="subject-thread", tier=Tier.HEURISTIC,
            rationale=f"subject thread '{norm[:60]}'", locator=locator, snippet=subject[:120],
            virtual=True, anchors=True, fallback=True,
            anchor_attrs={"type": "email_thread", "subject": subject}))
    else:
        # An automated notice stands as its own instance (see above); many copies
        # then cluster and get put to the "looks like a process, isn't?" test.
        declare(ent, Link(
            target=ent_id, method="standalone", tier=Tier.DIRECT,
            rationale="automated notice — its own instance", anchors=True))

    action = "replied" if in_reply_to else ("forwarded" if _is_forward(subject) else "sent")
    out.events.append(Event(
        id=f"evt:{source}:{mid}:{action}", entity_id=ent_id, action=action, source=source,
        confidence=direct(), evidence=[Evidence(source, locator, subject[:200])],
        timestamp=date, actor=from_id,
        attrs={"to": to, "cc": cc, "is_bot": _is_bot(frm or "")}))


def _is_forward(subject: str) -> bool:
    return bool(re.match(r"^\s*(fwd?|fw)\s*:", subject or "", re.I))
