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
        "'Shipped') — and give each group a short human activity name. You may NAME "
        "and GROUP only: do not invent an activity no pair evidences.\n"
        "NEVER return a name that just restates how the record travelled — "
        "'Correspondence Sent', 'Email Forwarded', 'Message Posted' are the verb in "
        "more words and say nothing about the work. If a verb genuinely carries no "
        "activity (a mailbox records only that something was sent), map it to the "
        "single plainest word you can and STOP; a later pass reads the record "
        "itself. Padding the verb hides from that pass that it is needed.\n"
        'Return ONLY JSON: {"map": {"<artefact>/<verb>": "<Activity Name>"}} '
        "covering every pair."
    )

    def __init__(self, api_model: Optional[str] = None):
        self.api_model = api_model

    def map(self, vocab: list[dict]) -> dict[str, str]:
        if not vocab or not os.environ.get("ANTHROPIC_API_KEY"):
            return {}
        try:
            import anthropic
        except ImportError:
            print("[abstraction] activity mapping needs the Anthropic SDK: pip install anthropic")
            return {}
        try:
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=self.api_model or os.environ.get("INDUCTION_ACTIVITY_MODEL", "claude-opus-5"),
                max_tokens=1200,
                system=self._SYSTEM,
                messages=[{"role": "user", "content":
                           "Vocabulary:\n" + json.dumps(vocab, indent=2) +
                           "\n\nReturn the JSON map."}],
            )
            text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
            got = _parse_map(text)
            present = {_key(v["artefact"], v["action"]) for v in vocab}
            # Guardrail: keep only string→string entries for pairs we actually gave.
            return {str(k): str(v) for k, v in got.items() if k in present and v}
        except Exception as e:  # abstraction is a convenience; never break the run
            print(f"[abstraction] activity mapping skipped ({type(e).__name__}: {e})")
            return {}


def infer_activities(m, mapper: Optional[ActivityMapper],
                     classifier: "Optional[RecordClassifier]" = None) -> "Abstraction":
    """Name the corpus's activities — the verb map first, the record classifier
    only where the verb map had nothing to say.

    Returns an `Abstraction`. With no mapper and no classifier it is empty, and
    the engine shows raw artefacts, claiming no abstraction.
    """
    types = {e.id: e.type for e in m.shaped.entities}
    vocab: dict[str, dict] = {}
    for ev in m.shaped.events:
        artefact = types.get(ev.entity_id, "record")
        k = _key(artefact, ev.action)
        entry = vocab.setdefault(k, {"artefact": artefact, "action": ev.action, "examples": []})
        snip = ev.evidence[0].snippet if ev.evidence else None
        if snip and len(entry["examples"]) < 3 and snip not in entry["examples"]:
            entry["examples"].append(snip[:80])

    by_vocab = mapper.map(list(vocab.values())) if mapper is not None else {}
    abstraction = Abstraction(by_vocab=by_vocab)
    if classifier is None:
        return abstraction

    events = _events_needing_a_reading(m, by_vocab, types)
    if not events:
        return abstraction          # the verbs discriminated; nothing to read
    _read_the_records(abstraction, m, events, classifier)
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
# The division of labour is the point, and it is what keeps this honest:
#   the model  LABELS a record that exists, and quotes the text it read;
#   the engine FINDS what is absent, deterministically, by comparing a run
#              against its kind's common path (steps/gaps_generic.py).
# The model never invents a missing step. It cannot: it only ever assigns a name
# to a record already in the corpus, and a record it will not commit to keeps
# its raw verb rather than being folded into a step it does not evidence.

# How many records to show the vocabulary-discovery pass. Enough to see the
# corpus's real range, small enough to be one call.
_DISCOVERY_SAMPLE = 150
_CLASSIFY_BATCH = 25


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

    @classmethod
    def of(cls, value) -> "Abstraction":
        """Accept a bare ``{artefact/verb: Activity}`` dict as a tier-1-only
        abstraction, so every existing caller keeps working."""
        if isinstance(value, cls):
            return value
        return cls(by_vocab=dict(value or {}))


class RecordClassifier:
    """Reads what a single record *does*, when its verb does not say.

    Two passes, both batched: `discover` proposes the activity vocabulary from a
    sample of the corpus itself (never a taxonomy this engine hardcodes), and
    `classify` assigns each record one of those activities plus the span it read.
    Injected, exactly like `ActivityMapper`, so the layer is testable offline.
    """

    def discover(self, samples: list[str]) -> list[str]:
        raise NotImplementedError

    def classify(self, records: list[dict], vocabulary: list[str]) -> dict[str, dict]:
        """``[{"id","text"}]`` -> ``{id: {"activity","span"}}``. A record the
        classifier will not commit to must be OMITTED, not guessed at."""
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

    Two ways a kind qualifies, both read off the data:
      - its runs draw on ONE activity — the verb map collapsed it, or the source
        only ever records one verb; or
      - its records outnumber its distinct activities (see `_TRANSPORT_RATIO`).

    A rejected kind is left alone: reading a nightly build notice more closely
    will not make it a process.
    """
    from statistics import median

    events_by_id = {e.id: e for e in m.shaped.events}

    def activity(ev):
        return by_vocab.get(_key(types.get(ev.entity_id, "record"), ev.action)) or ev.action

    def events_of(case_id):
        return [events_by_id[eid] for eid in m.cases[case_id].event_ids if eid in events_by_id]

    qualifying: list = []
    for kind in m.kinds:
        if kind.rejected:
            continue
        runs = [events_of(cid) for cid in kind.case_ids]
        alphabet = {activity(ev) for evs in runs for ev in evs}
        ratios = [len(evs) / len({activity(e) for e in evs}) for evs in runs if len(evs) >= 2]

        transport = len(alphabet) <= 1 or (
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


def _read_the_records(abstraction: "Abstraction", m, events, classifier) -> None:
    """Discover the corpus's activity vocabulary, then read each record into it."""
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

    sample = [r["text"] for r in records[:_DISCOVERY_SAMPLE]]
    try:
        vocabulary = [v for v in classifier.discover(sample) if isinstance(v, str) and v.strip()]
    except Exception as e:                # the tier is a convenience, never a blocker
        print(f"[abstraction] activity discovery skipped ({type(e).__name__}: {e})")
        return
    if len(vocabulary) < 2:
        # One activity is what the verb map already told us; claiming it again,
        # more expensively, is not an improvement. But say so — a silent return
        # here is indistinguishable from the tier never having been asked, and on
        # a real run that cost an evening of wondering which had happened.
        print(f"[abstraction] {len(records)} records needed reading, but activity "
              f"discovery returned {len(vocabulary)} activities "
              f"({vocabulary or 'none'}) — nothing to classify into, so the steps "
              f"stay as the source's own verbs")
        return
    print(f"[abstraction] reading {len(records)} records into "
          f"{len(vocabulary)} activities: {', '.join(vocabulary)}")

    got: dict[str, dict] = {}
    for i in range(0, len(records), _CLASSIFY_BATCH):
        batch = records[i:i + _CLASSIFY_BATCH]
        try:
            got.update(_clean_readings(classifier.classify(batch, vocabulary), batch, vocabulary))
        except Exception as e:
            print(f"[abstraction] batch {i // _CLASSIFY_BATCH} skipped ({type(e).__name__}: {e})")

    abstraction.by_record = got
    abstraction.n_unclassified = len(records) - len(got)
    abstraction.vocabulary = _audit_rows(vocabulary, got, abstraction.n_unclassified)
    if got:
        _reproject(m, abstraction)


def _clean_readings(raw: dict, batch: list[dict], vocabulary: list[str]) -> dict[str, dict]:
    """Guardrail, the same shape as naming.py's `_clean`: a reading may only
    label a record we asked about, with an activity we proposed. Anything else —
    an invented activity, an id we never sent, a missing span — is dropped, and a
    dropped record keeps its raw verb rather than being guessed at."""
    allowed_ids = {r["id"] for r in batch}
    allowed_acts = {v.lower(): v for v in vocabulary}
    out: dict[str, dict] = {}
    for rid, val in (raw or {}).items():
        if rid not in allowed_ids or not isinstance(val, dict):
            continue
        act = allowed_acts.get(str(val.get("activity", "")).strip().lower())
        span = str(val.get("span", "")).strip()
        if act and span:
            out[rid] = {"activity": act, "span": span[:160]}
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

    Rules are (activity, [phrase, ...]); the first whose phrase appears in the
    record wins, and the phrase it matched becomes the span, exactly as the real
    model returns the text it read. A record no rule matches is OMITTED — the
    abstention path is the one most worth exercising offline.
    """

    def __init__(self, rules: list[tuple[str, list[str]]]):
        self._rules = [(name, [p.lower() for p in phrases]) for name, phrases in rules]

    def discover(self, samples: list[str]) -> list[str]:
        return [name for name, _ in self._rules]

    def classify(self, records: list[dict], vocabulary: list[str]) -> dict[str, dict]:
        out = {}
        for rec in records:
            low = rec["text"].lower()
            for name, phrases in self._rules:
                hit = next((p for p in phrases if p in low), None)
                if hit:
                    out[rec["id"]] = {"activity": name, "span": hit}
                    break
        return out


class AnthropicRecordClassifier(RecordClassifier):
    """The real reading tier. Two batched calls' worth of work, not one per record.

    Both prompts are written to make abstention cheap and invention expensive:
    the discovery pass must derive its vocabulary from the sample it is shown,
    and the classification pass must quote the span it read or say nothing.
    """

    _DISCOVER_SYSTEM = (
        "You are given a sample of raw records from ONE company system (emails, chat "
        "messages, notes). Each records that something happened, but the system's own "
        "verb ('sent') says nothing about WHAT. Read the sample and return the short "
        "list of distinct ACTIVITIES these records actually perform — the kind of thing "
        "a process analyst would call a step (e.g. Requested, Reviewed, Approved, "
        "Escalated, Confirmed, Informed). Derive them from THIS sample only; do not "
        "return a generic taxonomy, and do not return an activity the sample does not "
        "show. Prefer 4-8 activities.\n"
        "NEVER return an activity that merely restates how the record travelled — "
        "'Corresponded by Email', 'Relayed to Others', 'Sent a Message', 'Forwarded' "
        "are the system's verb in more words, and naming them defeats the point of "
        "reading the text at all. Every activity must say what the WORK was. If a "
        "record only passes something along, that is for the classifier to decline, "
        "not for you to name.\n"
        'Return ONLY JSON: {"activities": ["<Name>", ...]}'
    )
    _CLASSIFY_SYSTEM = (
        "You assign each record exactly one ACTIVITY from the list given, and quote the "
        "span of the record's own text that justifies it. If a record does not clearly "
        "perform any activity on the list, OMIT it entirely — omission is correct and "
        "expected; a guess is not. Never invent an activity outside the list. Return "
        'ONLY JSON: {"<record id>": {"activity": "<Name>", "span": "<quoted text>"}}'
    )

    def __init__(self, api_model: Optional[str] = None):
        self.api_model = api_model

    def _call(self, system: str, content: str, max_tokens: int) -> dict:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return {}
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=self.api_model or os.environ.get("INDUCTION_ACTIVITY_MODEL", "claude-opus-5"),
            max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return _parse_map(text)

    def discover(self, samples: list[str]) -> list[str]:
        got = self._call(self._DISCOVER_SYSTEM,
                         "Records:\n" + json.dumps([s[:400] for s in samples], indent=1),
                         max_tokens=1500)
        if isinstance(got, list):                       # a bare JSON array
            acts = got
        elif isinstance(got, dict):
            # the documented key, else the only list-valued key it returned
            acts = got.get("activities")
            if not isinstance(acts, list):
                acts = next((v for v in got.values() if isinstance(v, list)), [])
        else:
            acts = []
        return [str(a) for a in acts if isinstance(a, str) and a.strip()]

    def classify(self, records: list[dict], vocabulary: list[str]) -> dict[str, dict]:
        return self._call(
            self._CLASSIFY_SYSTEM,
            "Activities: " + json.dumps(vocabulary)
            + "\n\nRecords:\n" + json.dumps(records, indent=1)
            + "\n\nReturn the JSON. Omit any record you are not sure about.",
            max_tokens=4000)


def _reproject(m, abstraction: "Abstraction") -> None:
    """Re-derive the spine, the variants and the findings over what we just read.

    Without this the reading is a view-layer decoration: `induce()` computed
    every case's trace, every kind's variants and every gap from the RAW VERBS,
    long before anything read a record. So the cards would show
    `Requested → Reviewed → Executed` while `steps/gaps_generic.py` was still
    comparing `sent → replied` against a one-step common path and finding, by
    construction, nothing.

    The fix is not a second detector. It is to re-run the existing one over the
    corrected spine, so the variants a reader sees, the `Differs how` column, and
    `model.json` all say the same thing — and the finding ("executed with no
    approval on record") is a real `Gap` with evidence, not a label in a chart.
    """
    from induction.steps.gaps_generic import infer_missing_step_gaps
    from induction.steps.variants import induced_variants

    types = {e.id: e.type for e in m.shaped.entities}
    events = {e.id: e for e in m.shaped.events}

    def activity(event_id):
        ev = events.get(event_id)
        if ev is None:
            return None
        return abstraction.activity_of(
            ev.id, types.get(ev.entity_id, "record"), ev.action) or ev.action

    # 1. the spine — consecutive records realising one activity are one step,
    #    matching exactly how the inspector folds them.
    for case in m.cases.values():
        seq: list[str] = []
        for eid in case.ordered_event_ids:
            act = activity(eid)
            if act and (not seq or seq[-1] != act):
                seq.append(act)
        if seq:
            case.trace_signature = tuple(seq)

    # 2. variants, recounted over the new alphabet
    for kind in m.kinds:
        kind.variants, kind.dfg = induced_variants(kind.case_ids, m.cases)
        kind.steps = sorted({a for v in kind.variants for a in v.signature})

    # 3. the one detector, re-run now that a common path has more than one step
    m.gaps = [g for g in m.gaps if g.kind != "missing_expected_step"]
    m.gaps.extend(infer_missing_step_gaps(m.correlation, m.kinds))
