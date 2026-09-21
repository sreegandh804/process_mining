"""Model-tier correlation — connecting records that mean the same thing in
different words. The weakest, most-inferential rung of the ladder.

The heuristic pass (`text.py`) matches on shared tokens. It cannot see that
"signed out moments after login" and "SSO login fails after token refresh" are
the same work — that is paraphrase, and token overlap is blind to it. On a real,
un-annotated corpus that blindness is *most* of the cross-source joins there are
to make: people describe the work, they don't quote its ticket number.

This module supplies the two external-service seams that close that gap, and
nothing else. The correlator still owns the mechanics (components, proximity,
same-shape, union, tiering); this only provides *proposed reasons*. Every join it
leads to is tier `model`: opt-in, surgical (only the leftovers determinism and
the heuristic pass could not explain), and honest — it carries the model's own
one sentence, it can be overruled, and it can never override a stronger join.

Two stages — the "hybrid" the design calls for:

  1. **Shortlist / embed** (`Embedder`) — cheaply rank candidate pairs so the
     expensive judge runs on a handful, not O(n²). An embedding model does this
     best at scale; with none configured the correlator falls back to token
     overlap at a *lenient* bar (good enough to shortlist, never to decide).

  2. **Judge** (`SemanticJudge`) — the LLM gives the reasoned yes/no on each
     shortlisted pair: "same piece of work?", not "same topic?".

Providers are injected, so the whole pass is testable offline with a scripted
stand-in and only reaches the network when a real provider is wired in. The real
ones follow the Anthropic Python SDK / an embeddings SDK; both are lazy imports
so the engine's default path needs neither.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# The judge seam
# ---------------------------------------------------------------------------

def as_text(record) -> str:
    """One side of a pair as prose, for a judge that reads prose.

    The correlator now hands the judge FIELDS — `when`, `who`, `text`, and the
    component id — because a typed question can point at them. A generative judge
    still wants a paragraph, and this is where the paragraph is made: once, here,
    rather than in every judge that needs one. A plain string passes through
    unchanged, so a stub or a test may still hand over text.
    """
    if not isinstance(record, dict):
        return str(record or "")
    head = []
    if record.get("when"):
        head.append(f"When: {record['when']}")
    who = record.get("who")
    if who:
        head.append("Who: " + (", ".join(who) if isinstance(who, (list, tuple)) else str(who)))
    body = record.get("text", "")
    return ("\n".join(head) + "\n\n" if head else "") + body


def record_id(record) -> Optional[str]:
    """The component id, when the caller supplied one. Never asked about — it is
    carried so a decision about the pair can be found again, and a judge that
    decided anything on an identifier would be deciding on the wrong thing."""
    return record.get("id") if isinstance(record, dict) else None


class SemanticJudge:
    """Decides whether two records are the *same piece of work*.

    ``judge`` returns a one-line reason when they are, or None. That is the whole
    contract — a subclass may reach a network, a stub may be a table.

    Each side arrives as a dict of fields (`when`, `who`, `text`, `id`) or, for a
    caller that has only prose, as a string. `as_text` collapses either into the
    paragraph a generative judge reads; a typed judge reads the fields directly.
    """

    def judge(self, a_text, b_text) -> Optional[str]:
        raise NotImplementedError


class ScriptedJudge(SemanticJudge):
    """An offline stand-in for a real model — for tests and the `--semantic demo`.

    It judges *text* (like the real judge), deterministically: a rule fires when a
    shared concept term appears on **both** sides. It is emphatically not engine
    logic — the engine holds no such vocabulary — it is a transparent simulation
    of the verdicts a real LLM returns, so the whole pipeline (and the honesty of
    a `model`-tier join) can be exercised without a key or a network.
    """

    def __init__(self, rules: Sequence[tuple[Sequence[str], str]]):
        # each rule: (concept terms, the reason to record if both sides mention one)
        self._rules = [(tuple(t.lower() for t in terms), reason) for terms, reason in rules]

    def judge(self, a_text, b_text) -> Optional[str]:
        a, b = as_text(a_text).lower(), as_text(b_text).lower()
        for terms, reason in self._rules:
            if any(t in a for t in terms) and any(t in b for t in terms):
                return reason
        return None


class AnthropicJudge(SemanticJudge):
    """The real judge. One short call per shortlisted pair, same guardrail
    `naming.py` lives under: it decides the pair it is handed and returns JSON;
    anything else it says is ignored, and any failure downgrades to "no join"
    rather than breaking the run.

    Resilience: each call retries transient API overload (529 / 429 / 5xx) with
    backoff. If the API stays overloaded *through* the retries, a **circuit
    breaker** trips — the judge disables itself for the rest of the run instead of
    retrying every remaining pair for a minute apiece, says so once (`self._log`),
    and the correlator carries on with deterministic + fuzzy joins. `skipped`
    counts the pairs it could not judge, for a one-line summary at the call site.
    """

    _SYSTEM = (
        "You decide whether two records from a company's systems are the SAME piece "
        "of work — the same task, incident, or change — not merely the same topic. "
        "Two invoices to one customer are not the same work. A bug report and the pull "
        "request that fixes it ARE. Two threads about the same counterparty, weeks "
        "apart, with different people on them, are two runs of work about one "
        "subject — NOT the same work. Each record is headed by WHEN it happened and "
        "WHO was on it: use both. Judge only the two records shown; do not invent "
        "facts about either. Answer ONLY compact JSON: "
        '{"same": true|false, "reason": "<one short clause>"}.'
    )

    def __init__(self, api_model: Optional[str] = None, max_reason: int = 160,
                 log=None, tries: int = 5):
        self.api_model = api_model
        self.max_reason = max_reason
        self._log = log or (lambda m: None)
        self._tries = tries
        self._client = None
        self._tripped = False   # the breaker: API stayed overloaded through retries
        self.skipped = 0        # pairs we could not judge (transient or otherwise)

    def judge(self, a_text, b_text) -> Optional[str]:
        a_text, b_text = as_text(a_text), as_text(b_text)
        if self._tripped or not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        from induction.anthropic_call import client, is_transient, with_backoff
        if self._client is None:
            try:
                self._client = client()
            except ImportError:
                self._log("[semantic] judge needs the Anthropic SDK: pip install anthropic")
                self._tripped = True
                return None
        try:
            msg = with_backoff(
                lambda: self._client.messages.create(
                    model=self.api_model or os.environ.get("INDUCTION_SEMANTIC_MODEL", "claude-haiku-4-5"),
                    # One short JSON verdict — but thinking tokens count against
                    # this too, and at 200 a thinking model never reaches the
                    # JSON at all: every pair silently reads as 'not the same
                    # work'. See abstraction.py's note on the caps.
                    max_tokens=4000,
                    system=self._SYSTEM,
                    messages=[{"role": "user", "content":
                               f"Record A:\n{a_text[:1500]}\n\nRecord B:\n{b_text[:1500]}\n\n"
                               "Same piece of work?"}],
                ),
                tries=self._tries, label="semantic judge", log=self._log)
        except Exception as e:  # a correlation convenience; never break the run
            self.skipped += 1
            if is_transient(e):
                self._tripped = True
                self._log(f"[semantic] API still overloaded after {self._tries} retries — "
                          "disabling the semantic judge for the rest of this run "
                          "(deterministic + fuzzy joins still apply).")
            else:
                self._log(f"[semantic] judge skipped ({type(e).__name__}: {e})")
            return None
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        verdict = _parse_json(text)
        if verdict.get("same") is True:
            reason = str(verdict.get("reason", "") or "model judged these the same work")
            return reason[:self.max_reason]
        return None


# ---------------------------------------------------------------------------
# The Jev gate — a typed verdict in front of the paid judge
# ---------------------------------------------------------------------------

# Two records, two questions, one call. Decomposed on purpose: "are these the
# same work" and "if they are related, by what" are different judgements, and a
# single broad question hides the second behind the first. Asking both costs one
# round trip — questions over one state are evaluated in parallel — and the pair
# is exactly what the engine's hardest failure mode needs: a high `same_work`
# sitting on top of `same_subject` is the "two threads, one counterparty, weeks
# apart" mistake, and it is invisible if you only ever see the boolean.
_PAIR_QUESTIONS: dict = {
    "same_work": {
        "type": "noul",
        "instructions": {
            "question": "Are `a` and `b` records of the SAME piece of work?",
            "compare": ["`a.text`", "`b.text`"],
            "focus": "One task, incident or change — not merely one subject.",
        },
        "criteria": {
            "true": {
                "what": "One piece of work seen from two places",
                "examples": ["A bug report and the pull request that fixes it",
                             "A request and the record of it being fulfilled"],
            },
            "false": {
                "what": "Two separate runs of work, or nothing in common",
                "not_for": "Two records of one run that happen to use different words",
                "examples": ["Two invoices to one customer",
                             "Two threads about one counterparty, weeks apart, "
                             "with different people on them"],
            },
        },
    },
    "relation": {
        "type": "choice",
        "instructions": {
            "question": "If `a` and `b` are connected at all, what connects them?",
            "compare": ["`a.text`", "`b.text`"],
            "focus": "Name the strongest connection, not every connection.",
        },
        "criteria": {
            "causal": {"what": "One exists because the other happened",
                       "examples": ["A fix that exists because of a report"]},
            "sequential": {"what": "Consecutive stages of one run of work",
                           "examples": ["A quote, then the order placed against it"]},
            "same_subject": {"what": "One topic, customer or component — separate runs",
                             "not_for": "Two records of a single run",
                             "examples": ["Two unrelated tickets about one product"]},
            "same_actor": {"what": "The same people, on work that is not the same",
                           "examples": ["One engineer's two unrelated changes"]},
            "unrelated": {"what": "No meaningful connection", "examples": []},
        },
    },
    # The third question, and the reason the state is fields rather than a blob.
    # "The same counterparty a fortnight apart, different people on it" is the
    # engine's hardest false positive, and it is a question about `a.when`,
    # `b.when` and the two `who` lists — not something to be hoped for from a
    # header pasted on top of some text.
    "one_span_of_work": {
        "type": "noul",
        "instructions": {
            "question": "Are `a.when` and `b.when` consistent with ONE continuous "
                        "piece of work, given who was on each?",
            "compare": ["`a.when`", "`b.when`", "`a.who`", "`b.who`"],
            "focus": "Judge the gap and the people, not the subject matter.",
        },
        "criteria": {
            "true": {
                "what": "The dates sit inside one run's rhythm, or the same people "
                        "carry it across the gap",
                "examples": ["A report and its fix days apart, same engineer",
                             "A quote and the order placed against it that week"],
            },
            "false": {
                "what": "A gap longer than one run of this kind takes, with different "
                        "people on each side",
                "not_for": "A genuinely long-running piece of work whose participants "
                           "stay the same",
                "examples": ["Two dealings with one counterparty a fortnight apart, "
                             "no one in common"],
            },
        },
    },
}


@dataclass(frozen=True)
class PairVerdict:
    """What the gate learned about one candidate pair."""

    same: float
    relation: Optional[str] = None
    relation_p: Optional[float] = None
    one_span: Optional[float] = None

    @property
    def same_subject_apart(self) -> bool:
        """The engine's hardest false positive, now detectable as a SHAPE.

        Two threads about one counterparty, weeks apart, different people: the
        text reads as one piece of work and it is two runs. Before, that was one
        number to be argued with. Now it is a pattern across three answers — the
        text says yes, the relation says the subject is shared, and the dates and
        people say otherwise — and a pattern is something the engine can act on
        rather than hope about.
        """
        return (self.relation == "same_subject"
                and self.one_span is not None and self.one_span < 0.5)

    def note(self) -> str:
        """The one fragment appended to a join's reason, so the numbers a join
        rests on travel with it everywhere the reason is shown."""
        tail = f" · {self.relation}" if self.relation else ""
        if self.relation and self.relation_p is not None:
            tail = f" · {self.relation} {self.relation_p:.2f}"
        if self.one_span is not None:
            tail += f" · one span {self.one_span:.2f}"
        return f"jev {self.same:.2f}{tail}"


def _side(record) -> dict:
    """One side of the pair, as the fields the questions point at.

    `id` is deliberately NOT sent. It is carried alongside so the decision can be
    found again, and a question that could see it might decide on it — which is
    the one thing a judge of content must never do.
    """
    if not isinstance(record, dict):
        return {"text": str(record or "")[:1500]}
    side = {"text": str(record.get("text", ""))[:1500]}
    for key in ("when", "who"):
        if record.get(key):
            side[key] = record[key]
    return side


class JevGate:
    """Scores a candidate pair before the generative judge is paid to read it.

    This is the `Confidence-Gated Routing` shape: a cheap typed decision chooses
    whether an expensive one happens. It is deliberately NOT a replacement for
    the judge — the judge writes the one-line reason a `model`-tier join carries,
    and a probability cannot be argued with the way a sentence can.

    `bar` is where the gate stops a pair. It is set low on purpose: the gate's job
    is to throw out the clearly-unrelated, not to make the call. A pair it passes
    is still judged, and can still be refused.
    """

    def __init__(self, jev=None, bar: float = 0.40, log=None):
        self._log = log or (lambda m: None)
        if jev is None:
            from induction.jev_call import Jev
            jev = Jev(log=self._log)
        self._jev = jev
        self.bar = bar

    @property
    def available(self) -> bool:
        return self._jev.available

    def verdict(self, a, b) -> Optional[PairVerdict]:
        """A typed reading of the pair, or None when Jev could not be reached —
        and None must mean "no opinion", never "no": a gate that cannot run has
        to let the pair through, or an unreachable service would silently delete
        every model-tier join in the run."""
        answers = self._jev.ask({"a": _side(a), "b": _side(b)}, _PAIR_QUESTIONS)
        same = answers.get("same_work")
        if same is None or same.noul is None:
            return None
        rel = answers.get("relation")
        span = answers.get("one_span_of_work")
        return PairVerdict(same=same.noul,
                           relation=rel.choice if rel else None,
                           relation_p=rel.confidence if rel else None,
                           one_span=span.noul if span is not None else None)


class GatedJudge(SemanticJudge):
    """A `SemanticJudge` with a Jev gate in front of it.

    The contract is unchanged — a reason string or None — so the correlator needs
    no modification and the tiering is untouched: a join this produces is still
    `model` tier, still overrulable, still unable to beat a stronger join. What
    changes is how many pairs reach the paid judge, and that every join it does
    make now carries the number the gate gave it.

    Degrading is one-directional by design. No key, a tripped breaker, an
    unparseable answer — all mean the underlying judge runs exactly as it did
    before this class existed. The gate can only ever save a call, never cause a
    join the judge did not make.
    """

    def __init__(self, judge: SemanticJudge, gate: Optional[JevGate] = None,
                 bar: float = 0.40, log=None, ledger=None):
        self._judge = judge
        self._gate = gate if gate is not None else JevGate(bar=bar, log=log)
        self.bar = bar
        self._ledger = ledger
        self.gated = 0      # pairs the gate answered for, so the judge never saw them
        self.passed = 0     # pairs the gate let through to the judge

    def judge(self, a, b) -> Optional[str]:
        verdict = self._gate.verdict(a, b)
        if verdict is None:
            return self._judge.judge(a, b)            # no opinion: unchanged behaviour
        if verdict.same < self.bar:
            self.gated += 1
            self._record(a, b, verdict,
                         "not judged — below the gate, so no join was proposed")
            return None
        self.passed += 1
        reason = self._judge.judge(a, b)
        if not reason:
            self._record(a, b, verdict,
                         "passed the gate, and the judge still declined — no join")
            return None
        if verdict.same_subject_apart:
            # The text reads as one piece of work, the subject is shared, and the
            # dates and people say two runs. The generative judge cannot see that
            # shape — it is offered one paragraph and asked one question — so the
            # note goes on the join rather than overruling it. Flagging beats
            # vetoing here: the pattern is a suspicion, and a reader who can see
            # the suspicion can settle it; a join silently withheld cannot be
            # argued with at all.
            reason = (f"{reason} — but read as one SUBJECT across two spans of "
                      f"time with different people, which is often two runs")
        self._record(a, b, verdict,
                     f"joined at tier `model`, with the judge's reason: {reason}")
        return f"{reason} [{verdict.note()}]"

    def _record(self, a, b, verdict: "PairVerdict", outcome: str) -> None:
        """Record what the gate was asked and what the engine did with the answer.

        Recorded for BOTH outcomes, including the pairs that were gated out. A
        ledger that only held the joins would show a reader every connection the
        engine made and none of the ones it decided against, which is the half
        more likely to be wrong.
        """
        if self._ledger is None:
            return
        from induction.decisions import JOIN, Decision

        derived = {}
        if verdict.one_span is not None:
            derived["one span of work"] = verdict.one_span
        if verdict.same_subject_apart:
            derived["same subject, two spans"] = 1.0
        self._ledger.add(Decision(
            kind=JOIN,
            about=pair_key(a, b),
            question=_PAIR_QUESTIONS["same_work"]["instructions"]["question"],
            qtype="noul",
            answer=round(verdict.same, 4),
            confidence=max(verdict.same, 1.0 - verdict.same),
            distribution=({verdict.relation: verdict.relation_p}
                          if verdict.relation and verdict.relation_p is not None else {}),
            derived=derived,
            outcome=outcome,
        ))


def pair_key(a, b) -> str:
    """A stable name for the pair a decision was about.

    Built from the two component ids when the caller carried them, sorted, so the
    same pair keys the same way whichever order it was judged in — that is what
    lets a page find the decision behind a join it is rendering. The ids never
    reach the model; they ride alongside the fields it is shown.

    Falls back to the first line of each side's text when there are no ids, which
    identifies the decision well enough to read but is not something the page can
    look up. A caller that wants the join traceable supplies ids.
    """
    ids = (record_id(a), record_id(b))
    if all(ids):
        return "↔".join(sorted(ids))

    def head(record) -> str:
        text = record.get("text", "") if isinstance(record, dict) else str(record or "")
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line[:60]
        return text[:60]

    return f"{head(a)} ↔ {head(b)}"


# ---------------------------------------------------------------------------
# The embed / shortlist seam
# ---------------------------------------------------------------------------

class Embedder:
    """Maps texts to vectors so candidate pairs can be ranked before the judge.

    Optional: the correlator falls back to token overlap when there is no
    embedder. A subclass wraps whatever provider is available.
    """

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


class VoyageEmbedder(Embedder):
    """Voyage AI embeddings (Anthropic has no first-party embeddings API). Lazy
    import, gated on VOYAGE_API_KEY, so it costs nothing unless wired in."""

    def __init__(self, model: str = "voyage-3"):
        self.model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        import voyageai  # lazy; only when a hybrid run actually asks for it
        client = voyageai.Client()
        return client.embed(list(texts), model=self.model).embeddings


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (na * nb)


def shortlist_by_embedding(embedder: Embedder, texts: dict[str, str],
                           candidates: list[tuple[str, str]], min_cos: float = 0.55
                           ) -> Optional[list[tuple[str, str]]]:
    """Rank candidate component pairs by embedding cosine; keep those above the
    bar. Returns None (→ let the correlator use its token-overlap fallback) if the
    embedder is unavailable or errors — a shortlist is an optimisation, never a
    gate the run depends on."""
    if embedder is None:
        return None
    ids = list(texts)
    try:
        vecs = {i: v for i, v in zip(ids, embedder.embed([texts[i] for i in ids]))}
    except Exception as e:
        print(f"[semantic] embedding shortlist skipped ({type(e).__name__}: {e})")
        return None
    kept = [(a, b) for a, b in candidates
            if a in vecs and b in vecs and cosine(vecs[a], vecs[b]) >= min_cos]
    kept.sort(key=lambda p: -cosine(vecs[p[0]], vecs[p[1]]))
    return kept


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# What the correlator receives
# ---------------------------------------------------------------------------

class SemanticProvider:
    """Bundles the two seams for the correlator: a judge (required) and an
    embedder (optional). This is the single object a policy carries."""

    def __init__(self, judge: SemanticJudge, embedder: Optional[Embedder] = None,
                 min_cos: float = 0.55):
        self.judge = judge
        self.embedder = embedder
        self.min_cos = min_cos

    def shortlist(self, texts: dict[str, str],
                  candidates: list[tuple[str, str]]) -> Optional[list[tuple[str, str]]]:
        return shortlist_by_embedding(self.embedder, texts, candidates, self.min_cos)
