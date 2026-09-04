"""AI-first process abstraction — turning artefact verbs into the activities a
process is actually made of.

A ticket is not a step. `sent → sent → opened → opened → labeled → reviewed →
merged → closed` is the systems' record of *artefacts*; the process is the
*activities* those artefacts are evidence of — `Raised → Reproduced → Fixed →
Reviewed → Shipped`. Collapsing many artefact verbs, across sources, into one
named activity is judgement about **equivalence and naming** — squarely the model
tier's lane (naming.py's guardrail: the model may name and group, never add,
drop, or reorder an artefact), and nothing the deterministic skeleton should
guess.

So this layer is **AI-first on purpose**. It assumes a model, asks it to map each
distinct (artefact-type, verb) to the activity that verb realises, and abstracts
each run over that map — keeping **every artefact as the evidence beneath its
activity**. There is no deterministic 'fallback naming': without a mapper the
engine does not *claim* a process abstraction, it shows the raw artefacts. The
mapper is injected, so the real Anthropic path is the one built, and the whole
layer is tested offline against a stand-in that simulates the model's answer.

The map is global per (artefact, verb) — one short model call for the whole
corpus, not one per run — so it is cheap and stable across runs of a kind.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional


def _key(artefact: str, action: str) -> str:
    return f"{artefact}/{action}"


class ActivityMapper:
    """Maps each distinct (artefact-type, verb) to the activity it realises.

    ``map`` takes the corpus vocabulary — ``[{"artefact","action","examples"}]`` —
    and returns ``{"<artefact>/<verb>": "Activity Name"}``, where many verbs may
    share one name (that sharing is the whole point: an issue *opened* and an
    email *sent* are both "Raised"). One call for the corpus; injected so the
    layer is testable without a key.
    """

    def map(self, vocab: list[dict]) -> dict[str, str]:
        raise NotImplementedError


class ScriptedActivityMapper(ActivityMapper):
    """An offline stand-in for the model — for tests and ``--demo``. Returns a
    fixed map, standing in for the model's grouping so the abstraction is exercised
    end-to-end without a key or a network. Live, ``AnthropicActivityMapper`` makes
    the real call; the engine itself holds no such vocabulary."""

    def __init__(self, mapping: dict[str, str]):
        self._m = dict(mapping)

    def map(self, vocab: list[dict]) -> dict[str, str]:
        # honours only the pairs actually present, exactly as the model's answer
        # would be filtered — a stand-in should not smuggle in unseen vocabulary.
        present = {_key(v["artefact"], v["action"]) for v in vocab}
        return {k: name for k, name in self._m.items() if k in present}


class AnthropicActivityMapper(ActivityMapper):
    """The real mapper. One short call for the whole corpus, same guardrail
    naming.py lives under: it groups and names the vocabulary it is handed and
    returns JSON; anything else it says is ignored, and any failure yields an
    empty map (no abstraction claimed) rather than breaking the run."""

    _SYSTEM = (
        "You turn a company's raw system-event vocabulary into the ACTIVITIES its "
        "process is made of. You are given distinct (artefact type, verb) pairs with "
        "example text. Group the pairs that are the SAME real activity — across "
        "artefact types and systems (an issue 'opened' and an email 'sent' reporting "
        "a bug are both 'Raised'; a pull request 'merged' and an issue 'closed' are "
        "'Shipped' — these illustrate the SHAPE of the grouping, not a vocabulary to "
        "reuse) — and give each group a short human activity name drawn from this "
        "corpus's own subject matter. You may NAME and GROUP only: do not invent an "
        "activity no pair evidences.\n"
        "NEVER return a name that just restates how the record travelled or was "
        "filed — 'Correspondence Sent', 'Email Forwarded', 'Message Posted' are the "
        "verb in more words and say nothing about the work. If a verb genuinely "
        "carries no activity (a system that records only that something was sent, "
        "posted or uploaded), map it to the single plainest word you can and STOP; a "
        "later pass reads the record itself. Padding the verb hides from that pass "
        "that it is needed.\n"
        'Return ONLY JSON: {"map": {"<artefact>/<verb>": "<Activity Name>"}} '
        "covering every pair."
    )

    def __init__(self, api_model: Optional[str] = None, log=None):
        self.api_model = api_model
        self._log = log or (lambda m: None)

    def map(self, vocab: list[dict]) -> dict[str, str]:
        if not vocab or not os.environ.get("ANTHROPIC_API_KEY"):
            return {}
        from induction.anthropic_call import client, with_backoff
        try:
            api = client()
        except ImportError:
            self._log("[abstraction] activity mapping needs the Anthropic SDK: pip install anthropic")
            return {}
        try:
            msg = with_backoff(
                lambda: api.messages.create(
                    model=self.api_model or os.environ.get("INDUCTION_ACTIVITY_MODEL", "claude-opus-5"),
                    max_tokens=_MAP_TOKENS,
                    system=self._SYSTEM,
                    messages=[{"role": "user", "content":
                               "Vocabulary:\n" + json.dumps(vocab, indent=2) +
                               "\n\nReturn the JSON map."}],
                ),
                label="activity map", log=self._log)
            text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            got = _parse_map(text)
            present = {_key(v["artefact"], v["action"]) for v in vocab}
            # Guardrail: keep only string→string entries for pairs we actually gave.
            return {str(k): str(v) for k, v in got.items() if k in present and v}
        except Exception as e:  # abstraction is a convenience; never break the run
            self._log(f"[abstraction] activity mapping skipped ({type(e).__name__}: {e})")
            return {}


def infer_activities(m, mapper: Optional[ActivityMapper],
                     classifier: "Optional[RecordClassifier]" = None,
                     log=None) -> "Abstraction":
    """Name the corpus's activities — the verb map first, the record classifier
    only where the verb map had nothing to say.

    Returns an `Abstraction`. With no mapper and no classifier it is empty, and
    the engine shows raw artefacts, claiming no abstraction. `log` (a `msg->None`
    sink) reports progress; default is silent.
    """
    log = log or (lambda m: None)
    types = {e.id: e.type for e in m.shaped.entities}
    vocab: dict[str, dict] = {}
    for ev in m.shaped.events:
        artefact = types.get(ev.entity_id, "record")
        k = _key(artefact, ev.action)
        entry = vocab.setdefault(k, {"artefact": artefact, "action": ev.action, "examples": []})
        snip = ev.evidence[0].snippet if ev.evidence else None
        if snip and len(entry["examples"]) < 3 and snip not in entry["examples"]:
            entry["examples"].append(snip[:80])

    if mapper is not None:
        log(f"abstraction: mapping {len(vocab)} (artefact, verb) pairs to activities")
    by_vocab = mapper.map(list(vocab.values())) if mapper is not None else {}
    abstraction = Abstraction(by_vocab=by_vocab)
    if classifier is None:
        return abstraction

    events = _events_needing_a_reading(m, by_vocab, types)
    if not events:
        return abstraction          # the verbs discriminated; nothing to read
    _read_the_records(abstraction, m, events, classifier, log)
    return abstraction


# ---------------------------------------------------------------------------
# Tier 2 — reading the record, for sources whose verb is transport not meaning
# ---------------------------------------------------------------------------
# The verb map above assumes THE VERB IS THE ACTIVITY. That holds for git
# (`authored`, `merged`, `reverted`) and for a tracker (`raised`, `approved`,
# `paid`). It fails completely for a mailbox, a chat log or a document library,
# where every record is `sent` / `posted` / `uploaded` — the verb is the
# transport, and the activity is in the text. A per-verb map cannot recover it:
# asked what `sent` means across 761 threads, the honest answer is one word, and
# a process whose every step is "Communicated" is not a process model.
#
# So: read the record instead — but only where the verbs demonstrably said
# nothing, which is a property of the vocabulary and never of the word "email".
#
# The reading answers TWO questions per record, because the transport verb was
# hiding two things, not one:
#
#   what the record DOES     -> Requested, Reviewed, Approved   -> the STEPS
#   what the record is ABOUT -> Contract execution, Invoice dispute -> the KINDS
#
# The second half exists because the steps were only half the failure. Kinds are
# clustered on `(automated, case.kind_hint)`, which for a mailbox is
# `(False, 'email')` for every run in the corpus — one kind, always — with token
# overlap over pooled thread text as the only fallback. On the repo's own Enron
# sample that fallback named its kinds `shirley, shall, time` and `hey, ena,
# work`. Perfect steps inside one process called "Correspondence Sharing" is not
# a process model either, so `_reproject` re-runs segmentation over what was read.
#
# The division of labour is the point, and it is what keeps this honest:
#   the model  LABELS a record that exists, and quotes the text it read;
#   the engine PLACES each run, by counting its own records' labels
#              (`_process_of_case`) — no model is ever asked "what process is
#              this thread?", because answering that is drawing a boundary;
#   the engine FINDS what is absent, deterministically, by comparing a run
#              against its kind's common path (steps/gaps_generic.py).
# The model never invents a missing step. It cannot: it only ever assigns a name
# to a record already in the corpus, and a record it will not commit to keeps
# its raw verb rather than being folded into a step it does not evidence.

# How many records to show the vocabulary-discovery pass. Enough to see the
# corpus's real range, small enough to be one call.
# Both are cost/quality judgements, not measurements: enough of the corpus to
# see its range in one call, and a batch small enough that a truncated reply
# loses 25 readings rather than all of them.
_DISCOVERY_SAMPLE = 150
_CLASSIFY_BATCH = 25

# Why these caps are far above the JSON they have to fit.
#
# `max_tokens` is a CEILING, not a spend — an unused token costs nothing — but a
# reply that hits it is truncated mid-JSON, and truncated JSON is an empty dict
# two frames later. Current models think before they answer, adaptively and by
# default, and **thinking tokens count against this ceiling**. So a budget sized
# to the answer ("4-8 short names, call it 1500") is a budget the answer never
# reaches: on samples/enron the discovery pass spent its whole 1500 thinking and
# emitted nothing, and the run reported "returned 0 activities" for two days
# before the truncation was visible at all.
#
# Size these to the answer PLUS room to think, and let `_call` say so when a
# reply is cut off anyway. The alternative — turning thinking down per model —
# would need this layer to know which model it is talking to, and it does not.
_DISCOVER_TOKENS = 8000
_MAP_TOKENS = 8000
_CLASSIFY_TOKENS = 16000


@dataclass
class Abstraction:
    """What the activity layer concluded, and on what evidence.

    `by_vocab` is the verb-level map (tier 1). `by_record` is the per-record
    reading (tier 2), each entry carrying the span of text that justified it, so
    a step in the inspector opens to the sentence behind it. `vocabulary` is the
    audit view — every activity, how many records got it, and the phrases it was
    read from — because the classification is the most falsifiable thing here and
    a reader has to be able to check it in bulk, not one message at a time.
    """

    by_vocab: dict[str, str] = field(default_factory=dict)
    by_record: dict[str, dict] = field(default_factory=dict)
    vocabulary: list[dict] = field(default_factory=list)
    n_unclassified: int = 0
    # The process families the reading proposed, and where each run landed.
    # `by_case` is filled by `_reproject` — a run's process is a COUNT over its
    # records' readings, never a label a model attached to the run itself.
    processes: list[str] = field(default_factory=list)
    by_case: dict[str, str] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.by_vocab or self.by_record)

    def activity_of(self, event_id: str, artefact: str, action: str) -> Optional[str]:
        """The reading first, then the verb map, then nothing (caller falls back
        to the raw verb — never a guess)."""
        rec = self.by_record.get(event_id)
        if rec:
            return rec["activity"]
        return self.by_vocab.get(_key(artefact, action))

    def span_of(self, event_id: str) -> Optional[str]:
        rec = self.by_record.get(event_id)
        return rec.get("span") if rec else None

    def process_of(self, event_id: str) -> Optional[str]:
        rec = self.by_record.get(event_id)
        return rec.get("process") if rec else None

    @classmethod
    def of(cls, value) -> "Abstraction":
        """Accept a bare ``{artefact/verb: Activity}`` dict as a tier-1-only
        abstraction, so every existing caller keeps working."""
        if isinstance(value, cls):
            return value
        return cls(by_vocab=dict(value or {}))


@dataclass
class ReadVocabulary:
    """What the discovery pass proposed, derived from a sample of the corpus.

    **Two lists, because one verb was hiding two different things.** A mailbox
    records `sent`, and that single word conceals both what a message *does* and
    what it is *about*:

      - `activities` — what the record DOES: Requested, Reviewed, Approved,
        Escalated. These become the STEPS inside a process.
      - `processes` — what the record is ABOUT: contract execution, invoice
        dispute, campus recruiting. These become the PROCESS KINDS.

    Before this, only the first list existed, and kinds were clustered on
    `(automated, case.kind_hint)` — which for mail is `(False, 'email')` for
    every case in the corpus, i.e. exactly one kind, always. The fallback was
    token overlap over pooled thread text, and on a real mailbox it produced
    kinds called `shirley, shall, time` and `hey, ena, work`. Vocabulary is not
    subject matter, and a reader cannot act on either of those.

    The guardrail is unchanged and is the reason `processes` is a *closed list
    proposed once*, rather than a free-text label per record: the model NAMES the
    families, the engine ASSIGNS runs to them by counting (`_process_of_case`)
    and draws every boundary itself. A model that could invent a label per record
    would be partitioning the corpus, which is structural work and not its lane.
    """

    activities: list[str] = field(default_factory=list)
    processes: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.activities)

    @classmethod
    def of(cls, value) -> "ReadVocabulary":
        """Accept a bare ``["Requested", ...]`` as an activities-only vocabulary,
        the same courtesy `Abstraction.of` extends to a bare verb map."""
        if isinstance(value, cls):
            return value
        return cls(activities=list(value or []))


class RecordClassifier:
    """Reads what a single record *does* and is *about*, when its verb says neither.

    Two passes, both batched: `discover` proposes the vocabulary from a sample of
    the corpus itself (never a taxonomy this engine hardcodes), and `classify`
    assigns each record an activity, a process, and the span it read them from.
    Injected, exactly like `ActivityMapper`, so the layer is testable offline.
    """

    def discover(self, samples: list[str]) -> ReadVocabulary:
        raise NotImplementedError

    def classify(self, records: list[dict], vocabulary: ReadVocabulary) -> dict[str, dict]:
        """``[{"id","text"}]`` -> ``{id: {"activity","process","span"}}``.

        A record the classifier will not commit to must be OMITTED, not guessed
        at. `process` may be omitted on its own — a record can plainly perform an
        activity while belonging to no family the sample named.
        """
        raise NotImplementedError


def _parse_map(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return {}
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    if isinstance(obj, list):
        return obj                       # a bare array is a valid answer for discovery
    inner = obj.get("map", obj)
    return inner if isinstance(inner, dict) else {}


def _first_named(obj, keys) -> list[str]:
    """Names under the first of `keys` the model actually used, else []."""
    if not isinstance(obj, dict):
        return []
    for key in keys:
        if key in obj:
            names = _names_from(obj[key])
            if names:
                return names
    return []


def _names_from(value) -> list[str]:
    """Pull a list of names out of whatever shape the model answered in.

    The prompt asks for ``{"activities": ["Requested", ...]}`` and the guardrail
    used to accept only that — a list of bare strings. Every other shape a model
    reasonably reaches for parsed fine and was then filtered to nothing:

        {"activities": [{"name": "Requested", "evidence": "..."}]}
        {"Requested": "a party asks for something", "Approved": "..."}

    That is how a run reported "returned 0 activities" while the model had
    answered perfectly well. Read the name out of each shape instead. The
    guardrail does not move: the result is still only ever names, and
    `_clean_readings` still refuses any activity that is not on this list.
    """
    def one(item) -> Optional[str]:
        if isinstance(item, str):
            return item.strip() or None
        if isinstance(item, dict):
            for k in ("name", "activity", "label", "title", "process", "step"):
                v = item.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return None

    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [n for n in (one(i) for i in value) if n]
    if isinstance(value, dict):
        # a name -> description map is a list of names wearing a different hat
        if value and all(isinstance(v, str) for v in value.values()):
            return [k.strip() for k in map(str, value) if k.strip()]
        for v in value.values():
            if isinstance(v, (list, dict)):
                names = _names_from(v)
                if names:
                    return names
    return []


# When a kind's records outnumber its distinct activities by this much, the verbs
# are not marking stages — they are marking transmissions. Measured, not guessed;
# these are median records-per-activity over the corpora in the repo:
#
#     samples/finance   1.00     samples/grants          1.00
#     flask commit kind 1.00     flask main kind         1.33
#     a mailbox         2.00  <- the only one that needs reading
#
# A stage happens once in a run (raised, approved, paid). A transmission recurs
# all the way through it (sent, forwarded, sent, forwarded). That is the whole
# difference, and it is a property of the data, never of the word "email".
_TRANSPORT_RATIO = 2.0
# Below this many multi-record runs the median means nothing — flask has a kind
# whose ratio is 4.50 on a sample of exactly one.
_MIN_RUNS_TO_JUDGE = 10


def _events_needing_a_reading(m, by_vocab: dict, types: dict) -> list:
    """The gate. Events in kinds whose verbs turned out to be transport.

    Three ways a kind qualifies, all read off the data:
      - its runs draw on ONE activity — the verb map collapsed it, or the source
        only ever records one verb; or
      - its records outnumber its distinct activities (see `_TRANSPORT_RATIO`); or
      - the CORPUS as a whole clears that bar, even if this kind is too small to
        answer for itself.

    That third test is not a loosening, it is a correction. Kinds are formed
    before anything has been read, and on a mailbox they come from token overlap
    over pooled thread text — so a corpus whose verbs are plainly transport can be
    diced into kinds that each fall below `_MIN_RUNS_TO_JUDGE` and never get read
    at all. On a synthetic two-family mailbox that is exactly what happened: the
    contract kind had a median ratio of 2.0 and was skipped for having 8 runs
    rather than 10, and the reading that would have replaced those token-noise
    boundaries never ran. The weak fallback was vetoing its own replacement.

    Whether a source records transport or stages is a property of ITS VERBS, not
    of any one clustering of its runs, so it is fair — and more honest — to ask
    the question of the whole corpus too. It stays a measurement either way.
    Corpus-wide medians over every corpus in the repo, which is what this branch
    turns on:

        samples/finance   1.00 (13 runs)     pallets/flask   1.33 (312 runs)
        samples/grants    1.00 (20 runs)     pallets/click   1.50 (372 runs)
        unifyai/ivy       1.50 (440 runs)    samples/enron   2.00 (57 runs)

    Only the mailbox reaches the bar, and it reaches it on its own verbs. Note
    the margin, though, because it is the honest weakness here: the two busiest
    git corpora sit at 1.50, half a step below the line, and a repo with chattier
    commit verbs could cross it. This branch WIDENS what gets read, and it is
    justified on six corpora — a small sample. Re-measure it when a seventh
    arrives rather than assuming it still holds; a source that trips it wrongly
    pays in tokens and in relabelled steps, and `n_unclassified` is where that
    would show.

    A rejected kind is left alone: reading a nightly build notice more closely
    will not make it a process.
    """
    from statistics import median

    events_by_id = {e.id: e for e in m.shaped.events}

    def activity(ev):
        return by_vocab.get(_key(types.get(ev.entity_id, "record"), ev.action)) or ev.action

    def events_of(case_id):
        return [events_by_id[eid] for eid in m.cases[case_id].event_ids if eid in events_by_id]

    def ratios_of(runs):
        return [len(evs) / len({activity(e) for e in evs}) for evs in runs if len(evs) >= 2]

    live = [k for k in m.kinds if not k.rejected]
    runs_of = {k.id: [events_of(cid) for cid in k.case_ids] for k in live}

    corpus_ratios = [r for k in live for r in ratios_of(runs_of[k.id])]
    corpus_transport = (len(corpus_ratios) >= _MIN_RUNS_TO_JUDGE
                        and median(corpus_ratios) >= _TRANSPORT_RATIO)

    qualifying: list = []
    for kind in live:
        runs = runs_of[kind.id]
        alphabet = {activity(ev) for evs in runs for ev in evs}
        ratios = ratios_of(runs)

        transport = corpus_transport or len(alphabet) <= 1 or (
            len(ratios) >= _MIN_RUNS_TO_JUDGE and median(ratios) >= _TRANSPORT_RATIO)
        if transport:
            qualifying.extend(ev for evs in runs for ev in evs)
    return qualifying


def _record_text(ev, m) -> str:
    """The record behind an event, as text — the same attributes the correlator
    and the topic step read, so no adapter needs to know this layer exists."""
    from induction.steps.correlate import DEFAULT_POLICY, _text_of
    ent = next((e for e in m.shaped.entities if e.id == ev.entity_id), None)
    if ent is None:
        return ""
    return _text_of(ent, DEFAULT_POLICY.fuzzy.text_attrs)[:1200]


def _spread(records: list, n: int) -> list:
    """`n` records drawn evenly across the corpus, not the first `n` of it.

    The discovery pass proposes the whole corpus's vocabulary from this sample,
    so what the sample over-represents, the vocabulary over-fits. Records arrive
    grouped — by kind, and within a kind by whatever order the adapter walked the
    source (for a maildir, one mailbox and one folder at a time). Taking the head
    therefore asks one corner of the corpus to name the processes for all of it:
    on the repo's Enron sample the first 150 of 263 records are dominated by the
    earliest folders, and the later mailboxes get no vote at all.

    An even stride is the cheapest fix that has no parameters to tune and no
    randomness to make a run irreproducible.
    """
    if n <= 0 or not records:
        return []
    if len(records) <= n:
        return list(records)
    step = len(records) / n
    return [records[int(i * step)] for i in range(n)]


def _read_the_records(abstraction: "Abstraction", m, events, classifier, log=None) -> None:
    """Discover the corpus's activity vocabulary, then read each record into it."""
    log = log or (lambda m: None)
    records = []
    seen_ids = set()
    for ev in events:
        if ev.id in seen_ids:
            continue
        text = _record_text(ev, m)
        if not text.strip():
            continue                      # nothing to read; the verb stands
        seen_ids.add(ev.id)
        records.append({"id": ev.id, "text": text})
    if not records:
        return

    log(f"abstraction: verbs are transport, reading {len(records)} records "
        f"(discovering the vocabulary from a sample of {min(len(records), _DISCOVERY_SAMPLE)})")
    sample = [r["text"] for r in _spread(records, _DISCOVERY_SAMPLE)]
    try:
        vocab = classifier.discover(sample)
    except Exception as e:                # the tier is a convenience, never a blocker
        log(f"[abstraction] activity discovery skipped ({type(e).__name__}: {e})")
        return
    activities = [v for v in vocab.activities if isinstance(v, str) and v.strip()]
    processes = [v for v in vocab.processes if isinstance(v, str) and v.strip()]
    if len(activities) < 2:
        # One activity is what the verb map already told us; claiming it again,
        # more expensively, is not an improvement. But say so — a silent return
        # here is indistinguishable from the tier never having been asked, and on
        # a real run that cost an evening of wondering which had happened.
        log(f"[abstraction] {len(records)} records needed reading, but activity "
            f"discovery returned {len(activities)} activities "
            f"({activities or 'none'}) — nothing to classify into, so the steps "
            f"stay as the source's own verbs")
        return
    vocab = ReadVocabulary(activities=activities, processes=processes)
    n_batches = (len(records) + _CLASSIFY_BATCH - 1) // _CLASSIFY_BATCH
    log(f"abstraction: reading {len(records)} records into {len(activities)} "
        f"activities ({', '.join(activities)}) — {n_batches} batch(es)")
    if processes:
        log(f"abstraction: and into {len(processes)} processes "
            f"({', '.join(processes)}) — these become the kinds")

    got: dict[str, dict] = {}
    for i in range(0, len(records), _CLASSIFY_BATCH):
        batch = records[i:i + _CLASSIFY_BATCH]
        b = i // _CLASSIFY_BATCH + 1
        log(f"abstraction: classifying batch {b}/{n_batches} ({len(batch)} records)")
        try:
            got.update(_clean_readings(classifier.classify(batch, vocab), batch, vocab))
        except Exception as e:
            log(f"[abstraction] batch {b} skipped ({type(e).__name__}: {e})")

    abstraction.by_record = got
    abstraction.processes = processes
    abstraction.n_unclassified = len(records) - len(got)
    abstraction.vocabulary = _audit_rows(activities, got, abstraction.n_unclassified)
    n_with_process = sum(1 for r in got.values() if r.get("process"))
    log(f"abstraction: read {len(got)} of {len(records)} records "
        f"({abstraction.n_unclassified} declined) · {n_with_process} placed in a process")
    if got:
        _reproject(m, abstraction, log)


def _clean_readings(raw: dict, batch: list[dict], vocab: "ReadVocabulary") -> dict[str, dict]:
    """Guardrail, the same shape as naming.py's `_clean`: a reading may only
    label a record we asked about, with an activity — and a process — we
    proposed. Anything else (an invented name, an id we never sent, a missing
    span) is dropped, and a dropped record keeps its raw verb rather than being
    guessed at.

    The activity and the process are dropped INDEPENDENTLY: a record can plainly
    perform an activity while belonging to no family the sample named, and
    forcing it into one would be the invention this whole layer exists to avoid.
    A reading with no process simply leaves its run to be placed by the others.
    """
    vocab = ReadVocabulary.of(vocab)
    allowed_ids = {r["id"] for r in batch}
    allowed_acts = {v.lower(): v for v in vocab.activities}
    allowed_procs = {v.lower(): v for v in vocab.processes}
    out: dict[str, dict] = {}
    for rid, val in (raw or {}).items():
        if rid not in allowed_ids or not isinstance(val, dict):
            continue
        act = allowed_acts.get(str(val.get("activity", "")).strip().lower())
        span = str(val.get("span", "")).strip()
        if not (act and span):
            continue
        reading = {"activity": act, "span": span[:160]}
        proc = allowed_procs.get(str(val.get("process", "")).strip().lower())
        if proc:
            reading["process"] = proc
        out[rid] = reading
    return out


def _audit_rows(vocabulary: list[str], readings: dict, n_unclassified: int) -> list[dict]:
    """The audit view: every activity, how many records it claimed, and the
    phrases it was read from. The `Unclassified` row is the trust number — if it
    is most of the corpus, the classifier did not understand this data, and a
    reader must be able to see that at a glance rather than infer it."""
    rows: dict[str, dict] = {v: {"activity": v, "n": 0, "phrases": []} for v in vocabulary}
    for r in readings.values():
        row = rows.get(r["activity"])
        if row is None:
            continue
        row["n"] += 1
        span = r["span"]
        if len(row["phrases"]) < 6 and span not in row["phrases"]:
            row["phrases"].append(span)
    out = sorted((r for r in rows.values() if r["n"]), key=lambda r: -r["n"])
    if n_unclassified:
        out.append({"activity": "Unclassified", "n": n_unclassified,
                    "phrases": [], "unclassified": True})
    return out


class ScriptedRecordClassifier(RecordClassifier):
    """Offline stand-in for the reading model — tests and ``--demo``.

    Rules are ``(activity, [phrase, ...])`` or ``(activity, [phrase, ...],
    process)``; the first whose phrase appears in the record wins, and the phrase
    it matched becomes the span, exactly as the real model returns the text it
    read. A record no rule matches is OMITTED — the abstention path is the one
    most worth exercising offline, and so is the rule that names no process.
    """

    def __init__(self, rules: list[tuple]):
        self._rules = []
        for rule in rules:
            name, phrases = rule[0], rule[1]
            process = rule[2] if len(rule) > 2 else None
            self._rules.append((name, [p.lower() for p in phrases], process))

    def discover(self, samples: list[str]) -> ReadVocabulary:
        seen: list[str] = []
        for _, _, process in self._rules:
            if process and process not in seen:
                seen.append(process)
        return ReadVocabulary(activities=[name for name, _, _ in self._rules],
                              processes=seen)

    def classify(self, records: list[dict], vocabulary: "ReadVocabulary") -> dict[str, dict]:
        out = {}
        for rec in records:
            low = rec["text"].lower()
            for name, phrases, process in self._rules:
                hit = next((p for p in phrases if p in low), None)
                if hit:
                    out[rec["id"]] = {"activity": name, "span": hit}
                    if process:
                        out[rec["id"]]["process"] = process
                    break
        return out


class AnthropicRecordClassifier(RecordClassifier):
    """The real reading tier. Two batched calls' worth of work, not one per record.

    Both prompts are written to make abstention cheap and invention expensive:
    the discovery pass must derive its vocabulary from the sample it is shown,
    and the classification pass must quote the span it read or say nothing.
    """

    _DISCOVER_SYSTEM = (
        "You are given a sample of raw records from ONE system of record. Each records "
        "that something happened, but the system's own verb for it (such as 'sent', "
        "'posted', 'uploaded', 'logged') describes only how the record was filed, not "
        "WHAT was done. Read the sample and return TWO vocabularies, both derived from "
        "THIS sample only.\n"
        "\n"
        "The two lists are DIFFERENT KINDS OF THING, and that is the whole point. One "
        "is the stages work passes through; the other is what the work is about.\n"
        "\n"
        "1. ACTIVITIES — the STAGES. Past-tense verbs, one word where possible. "
        "Prefer 4-8 for the whole corpus.\n"
        "   THE TEST: an activity must be reusable in EVERY process you list below. "
        "If it fits only one of them, it is subject matter and not a stage — "
        "generalise it or drop it. 'Termination Log Update' fails the test; "
        "'Recorded' passes. (Requested / Reviewed / Approved / Escalated / Declined "
        "illustrate the FORM a stage takes — do not reuse those words unless this "
        "sample actually shows them.)\n"
        "\n"
        "2. PROCESSES — the SUBJECT MATTER. The families of work this corpus is a "
        "record of. Prefer 3-8.\n"
        "   THE TEST: a process must fit ONLY itself and no other, it must RECUR "
        "across many records (never one specific case, person or counterparty), and "
        "every process must be expressible as a path through the activities above. "
        "A process you cannot walk with those stages means one of the two lists is "
        "wrong — fix it before answering.\n"
        "\n"
        "The domain is whatever the sample says it is — engineering, clinical, "
        "logistics, legal, manufacturing, support. Take the PROCESS names from the "
        "sample's own subject matter and vocabulary; do not reach for the vocabulary "
        "of office administration, or any other domain, unless the records are "
        "actually about it, and do not return a generic taxonomy.\n"
        "\n"
        "NEVER return a name that merely restates how the record travelled or was "
        "filed. 'Sent', 'Forwarded', 'Posted', 'Correspondence Sharing' are the "
        "system's own verb in more words: they say a message moved, not that anything "
        "was accomplished, and returning one defeats the point of reading the text at "
        "all. A stage says what the record ACHIEVED. Do not return a name the sample "
        "does not show. If a record only passes something along, that is for the "
        "classifier to decline, not for you to name.\n"
        'Return ONLY JSON: {"activities": ["<Name>", ...], "processes": ["<Name>", ...]}'
    )
    _CLASSIFY_SYSTEM = (
        "You assign each record exactly one ACTIVITY from the activity list and, where "
        "it is clear, one PROCESS from the process list — and you quote the span of the "
        "record's own text that justifies the activity. If a record does not clearly "
        "perform any activity on the list, OMIT the record entirely; if it performs one "
        "but belongs to no process on the list, give the activity and omit the "
        "\"process\" field. Omission is correct and expected; a guess is not. Never "
        "invent an activity or a process outside the lists given. Return ONLY JSON: "
        '{"<record id>": {"activity": "<Name>", "process": "<Name>", '
        '"span": "<quoted text>"}}'
    )

    def __init__(self, api_model: Optional[str] = None, log=None):
        self.api_model = api_model
        self._log = log or (lambda m: None)

    def _call(self, system: str, content: str, max_tokens: int) -> tuple:
        """Returns ``(parsed, raw_text)``.

        The raw text comes back too so a caller left holding nothing can SAY what
        it was handed. A silent empty parse cost a run once: the log read
        "activity discovery returned 0 activities" with no sight of the reply,
        which is a symptom, not a diagnosis.
        """
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return {}, ""
        from induction.anthropic_call import client, with_backoff
        api = client()
        msg = with_backoff(
            lambda: api.messages.create(
                model=self.api_model or os.environ.get("INDUCTION_ACTIVITY_MODEL", "claude-opus-5"),
                max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": content}],
            ),
            label="activity reading", log=self._log)
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        if getattr(msg, "stop_reason", None) == "max_tokens":
            # A truncated reply is unparseable JSON, and unparseable JSON is an
            # empty dict two frames later. Say it here, where the cause is known.
            self._log(f"[abstraction] the reply hit its {max_tokens}-token cap and was "
                      f"cut off mid-JSON — raise the cap or shrink the batch")
        return _parse_map(text), text

    # Keys a model plausibly answers under, the documented one first.
    _ACTIVITY_KEYS = ("activities", "activity", "steps", "vocabulary")
    _PROCESS_KEYS = ("processes", "process", "kinds", "families", "workstreams")

    def discover(self, samples: list[str]) -> ReadVocabulary:
        got, raw = self._call(
            self._DISCOVER_SYSTEM,
            "Records:\n" + json.dumps([s[:400] for s in samples], indent=1),
            max_tokens=_DISCOVER_TOKENS)
        acts = _first_named(got, self._ACTIVITY_KEYS)
        procs = _first_named(got, self._PROCESS_KEYS)
        if not acts:
            # A reply that is just the one list, under no key we know or none at all.
            acts = _names_from(got)
        if not acts and raw:
            self._log("[abstraction] activity discovery could not read a vocabulary out "
                      f"of the model's reply, which was: {raw[:300]!r}")
        return ReadVocabulary(activities=acts, processes=procs)

    def classify(self, records: list[dict], vocabulary: ReadVocabulary) -> dict[str, dict]:
        ask = ("Activities: " + json.dumps(vocabulary.activities))
        if vocabulary.processes:
            ask += "\nProcesses: " + json.dumps(vocabulary.processes)
        got, _ = self._call(
            self._CLASSIFY_SYSTEM,
            ask
            + "\n\nRecords:\n" + json.dumps(records, indent=1)
            + "\n\nReturn the JSON. Omit any record you are not sure about.",
            max_tokens=_CLASSIFY_TOKENS)
        return got if isinstance(got, dict) else {}


def _process_of_case(m, abstraction: "Abstraction") -> dict[str, str]:
    """Place each run in a process family, by COUNTING its records' readings.

    This is the line that keeps the model in its lane. The model named the
    families and read each record into one; the run's family is then whichever
    the plurality of its own records evidenced — arithmetic the engine does, over
    labels the model supplied. Nothing ever asks a model "what process is this
    thread?", because answering that is drawing a boundary, and drawing
    boundaries is structural work.

    A tie is broken by the earliest record, so the answer does not depend on dict
    order. A run whose records were all declined gets no family and stays in its
    structural cluster.
    """
    from collections import Counter

    out: dict[str, str] = {}
    for case in m.cases.values():
        votes = Counter()
        first_seen: dict[str, int] = {}
        for i, eid in enumerate(case.ordered_event_ids):
            proc = abstraction.process_of(eid)
            if proc:
                votes[proc] += 1
                first_seen.setdefault(proc, i)
        if votes:
            out[case.id] = min(votes, key=lambda p: (-votes[p], first_seen[p]))
    return out


def _reproject(m, abstraction: "Abstraction", log=None) -> None:
    """Re-derive the kinds, the spine, the variants and the findings over what
    we just read.

    Without this the reading is a view-layer decoration: `induce()` computed
    every case's trace, every kind's variants and every gap from the RAW VERBS,
    long before anything read a record. So the cards would show
    `Requested → Reviewed → Executed` while `steps/gaps_generic.py` was still
    comparing `sent → replied` against a one-step common path and finding, by
    construction, nothing.

    The fix is not a second detector. It is to re-run the existing ones over the
    corrected spine, so the variants a reader sees, the `Differs how` column, and
    `model.json` all say the same thing — and the finding ("executed with no
    approval on record") is a real `Gap` with evidence, not a label in a chart.

    **Segmentation is re-run too**, and that is the half that was missing. Kinds
    were clustered on `(automated, case.kind_hint)` before a single record had
    been read — which for a mailbox is `(False, 'email')` for the entire corpus,
    one kind, always. Re-segmenting over the process families the reading found
    is the only way "what are the processes here?" gets an answer from the
    records rather than from the envelope.
    """
    from collections import defaultdict

    from induction.honesty import apply_reject
    from induction.model import direct, model
    from induction.process import Step
    from induction.steps.gaps_generic import infer_missing_step_gaps
    from induction.steps.segment import segment
    from induction.steps.variants import induced_variants

    log = log or (lambda msg: None)
    types = {e.id: e.type for e in m.shaped.entities}
    events = {e.id: e for e in m.shaped.events}

    def activity(event_id):
        ev = events.get(event_id)
        if ev is None:
            return None
        return abstraction.activity_of(
            ev.id, types.get(ev.entity_id, "record"), ev.action) or ev.action

    # 1. the spine — consecutive records realising one activity are one step,
    #    matching exactly how the inspector folds them. Done FIRST, because
    #    segment() reads each case's trace to compute its variants.
    for case in m.cases.values():
        seq: list[str] = []
        for eid in case.ordered_event_ids:
            act = activity(eid)
            if act and (not seq or seq[-1] != act):
                seq.append(act)
        if seq:
            case.trace_signature = tuple(seq)

    # 2. the kinds, re-clustered on what each run was READ to be about
    abstraction.by_case = _process_of_case(m, abstraction)
    if abstraction.by_case:
        from induction.profiles import GENERIC_PROFILE
        profile = getattr(m, "profile", None) or GENERIC_PROFILE
        before = len(m.kinds)
        kinds = segment(m.shaped, m.correlation, profile,
                        case_process=abstraction.by_case)
        apply_reject(kinds, profile)
        m.kinds = kinds
        placed = len(abstraction.by_case)
        log(f"abstraction: re-segmented on what the records say — {before} kind(s) "
            f"from the envelope, {len(kinds)} from the reading "
            f"({placed} of {len(m.cases)} runs placed in a process)")

    # 3. variants, recounted over the new alphabet
    for kind in m.kinds:
        kind.variants, kind.dfg = induced_variants(kind.case_ids, m.cases)
        kind.steps = sorted({a for v in kind.variants for a in v.signature})

    # 4. the step catalogue. `label.py` built it during induce(), keyed on the raw
    #    verb — so without this `model.json` ships two disagreeing vocabularies:
    #    `steps: [step:sent]` beside cases whose traces say `Requested`, and a
    #    consumer joining on `step:{action}` finds nothing for a read activity.
    #    The tier moves with the provenance: a step every one of whose records was
    #    read is a `model` claim, not the `direct` one label.py could make.
    by_activity: dict[str, list] = defaultdict(list)
    for ev in m.shaped.events:
        by_activity[activity(ev.id) or ev.action].append(ev)
    m.steps = []
    for name, evs in sorted(by_activity.items(), key=lambda kv: -len(kv[1])):
        members = sorted({e.actor for e in evs if e.actor})
        n_read = sum(1 for e in evs if e.id in abstraction.by_record)
        m.steps.append(Step(
            id=f"step:{name}", name=name, action=name,
            confidence=(model(f"read from the records' own words ({n_read} of "
                              f"{len(evs)} records)") if n_read
                        else direct("named from the source's own action verb")),
            member_ids=members, event_ids=[e.id for e in evs],
            evidence=[e.evidence[0] for e in evs[:3] if e.evidence],
            attrs={"count": len(evs), "n_members": len(members), "n_read": n_read},
        ))

    # 5. the detectors, re-run now that a common path has more than one step
    m.gaps = [g for g in m.gaps if g.kind != "missing_expected_step"]
    m.gaps.extend(infer_missing_step_gaps(m.correlation, m.kinds))
