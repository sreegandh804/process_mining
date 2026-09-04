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
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# The judge seam
# ---------------------------------------------------------------------------

class SemanticJudge:
    """Decides whether two records are the *same piece of work*.

    ``judge`` returns a one-line reason when they are, or None. That is the whole
    contract — a subclass may reach a network, a stub may be a table. The
    correlator supplies only the two records' text and turns a reason into a
    ``model``-tier link.
    """

    def judge(self, a_text: str, b_text: str) -> Optional[str]:
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

    def judge(self, a_text: str, b_text: str) -> Optional[str]:
        a, b = a_text.lower(), b_text.lower()
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

    def judge(self, a_text: str, b_text: str) -> Optional[str]:
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
