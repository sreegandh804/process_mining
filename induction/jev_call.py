"""Resilient Jev calls — the engine's *second* model seam, and a different kind
of model from the first.

`anthropic_call.py` reaches a generative model: it is asked an open question and
answers in prose that this codebase then has to parse, repair and guard. Jev
(TypeSafe's "System One") is the opposite shape. It generates no text at all. It
is handed a `state` and a dict of **typed questions**, and returns one typed
answer per question, drawn only from the options the caller supplied:

  - `noul`   — a calibrated probability in [0, 1] that some property holds;
  - `choice` — one label from a set the caller named, plus the full distribution
               over that set;
  - `score`  — a position on an ordered rubric the caller wrote, as a number from
               zero (and it may land between two levels).

Two consequences, and they are the whole reason this module exists:

  1. **There is nothing to parse.** A malformed reply is not a failure mode, so
     none of `abstraction.py`'s tolerant-JSON machinery is needed here.
  2. **The answer is comparable.** A Choice's options are normalised against each
     other, so a flat distribution is a *reading of doubt*, not a missing answer.
     That is a thing this engine could not previously express: a record the
     classifier declined and a record it found genuinely ambiguous arrived at the
     same place — dropped.

What Jev CANNOT do bounds every use of it in this repo. It returns values, never
sentences, so it can never produce a `span`, a join's one-line reason, or a name.
Everywhere this engine emits a claim *with its evidence*, the generative model
still writes the evidence. Jev goes in FRONT of that call, to decide whether it
is worth making and to narrow what it is asked — never instead of it.

Transport: the OpenRouter Decisions API, over `urllib` so the stdlib-only
baseline stays stdlib-only (see requirements.txt). Retries reuse
`anthropic_call.with_backoff`, so both model seams wait out a busy API in the
same visible, offline-testable way, and both trip a circuit breaker rather than
retry every remaining item for a minute apiece.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# The Decisions API — a different endpoint from OpenRouter's chat completions,
# because the request is a `state` + `questions` document, not a message list.
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "typesafe/jev-1.13"


class JevError(Exception):
    """A failed Jev call, carrying the HTTP status so the shared
    `anthropic_call.is_transient` can classify it.

    A transport-level failure (no route, DNS, read timeout) has no status of its
    own; it is reported as 503, which is exactly what it means to the caller —
    the service could not be reached right now, and waiting may fix it.
    """

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# One typed answer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Answer:
    """One question's answer, normalised.

    `kind` says which field carries the verdict. `confidence` is the API's own
    calibrated certainty where it supplies one; where it does not, it is derived
    from the distribution — for a Noul that is `max(p, 1-p)` (0.5 is maximal
    doubt, and both 0.02 and 0.98 are confident answers), and for a Choice it is
    the winning option's share. A Score has no distribution to derive from, so
    its confidence stays None rather than being invented.
    """

    kind: str
    noul: Optional[float] = None
    choice: Optional[str] = None
    probabilities: dict[str, float] = field(default_factory=dict)
    score: Optional[float] = None
    confidence: Optional[float] = None

    def sure(self, bar: float) -> bool:
        """True when the answer is at least `bar` certain. Unknown confidence is
        NOT confident — an absent number never clears a gate."""
        return self.confidence is not None and self.confidence >= bar


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _probabilities(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out = {}
    for k, v in raw.items():
        p = _as_float(v)
        if p is not None:
            out[str(k)] = p
    return out


def parse_answer(raw: Any) -> Optional[Answer]:
    """One answer object -> `Answer`, or None if it carries no verdict.

    Deliberately shape-tolerant about *which* keys are present — the response is
    normalised by OpenRouter across providers, and `confidence`/`probabilities`
    are conveniences we can live without — but never about the verdict itself: an
    answer with no noul, choice or score is no answer, and returning None lets
    the call site treat it the same as a question that was never asked.
    """
    if not isinstance(raw, dict):
        return None
    conf = _as_float(raw.get("confidence"))
    probs = _probabilities(raw.get("probabilities"))

    noul = _as_float(raw.get("noul"))
    if noul is not None:
        return Answer(kind="noul", noul=noul, probabilities=probs,
                      confidence=conf if conf is not None else max(noul, 1.0 - noul))

    if raw.get("choice") is not None:
        choice = str(raw["choice"])
        if conf is None and probs:
            conf = probs.get(choice, max(probs.values()))
        return Answer(kind="choice", choice=choice, probabilities=probs, confidence=conf)

    score = _as_float(raw.get("score"))
    if score is not None:
        return Answer(kind="score", score=score, probabilities=probs, confidence=conf)

    return None


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------

def have_key() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def _post(url: str, payload: dict, api_key: str, timeout: float) -> dict:
    """POST JSON, return the decoded body. Raises `JevError` with a status."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Optional on OpenRouter; identifies the caller on their leaderboards.
            "X-Title": "process-mining induction engine",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:200]
        except Exception:  # noqa: BLE001 — the status is the part that matters
            pass
        raise JevError(f"HTTP {e.code} from the decisions API: {detail}",
                       status_code=e.code) from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        # Could not reach the service at all. Reported as 503: transient, and
        # worth the same wait-and-retry as a server that answered 503 itself.
        raise JevError(f"could not reach the decisions API ({type(e).__name__}: {e})",
                       status_code=503) from e


class Jev:
    """One Jev endpoint, with this codebase's two standing guarantees.

    **Never breaks a run.** `ask` returns an empty dict on any failure, exactly as
    the semantic judge returns None: a decision the engine could not get is a
    decision it does without, and `skipped` counts them for a one-line summary.

    **Never retries a dead service forever.** If the API stays transient *through*
    the backoff, the breaker trips: the instance disables itself for the rest of
    the run, says so once, and every later `ask` is a free no-op.

    `transport` is injected so the whole battery is exercisable offline with a
    scripted stand-in — the same seam `ScriptedJudge` gives the generative tier.
    """

    def __init__(self, model: Optional[str] = None, log: Optional[Callable[[str], None]] = None,
                 tries: int = 5, timeout: float = 30.0, endpoint: Optional[str] = None,
                 transport: Optional[Callable[..., dict]] = None):
        self.model = model or os.environ.get("INDUCTION_JEV_MODEL", DEFAULT_MODEL)
        self.endpoint = endpoint or os.environ.get("INDUCTION_JEV_ENDPOINT", ENDPOINT)
        self._log = log or (lambda m: None)
        self._tries = tries
        self._timeout = timeout
        self._transport = transport or _post
        self._tripped = False
        self.calls = 0      # requests actually sent
        self.skipped = 0    # requests that could not be answered

    @property
    def available(self) -> bool:
        """False once the breaker has tripped, or with no key to call with. A
        scripted transport needs no key — that is what makes the tests offline."""
        return not self._tripped and (have_key() or self._transport is not _post)

    def ask(self, state: Any, questions: dict[str, dict]) -> dict[str, Answer]:
        """Ask every question in `questions` about one `state`, in a single call.

        Questions over one state are evaluated independently and in parallel
        server-side, so asking six narrow questions costs one round trip, not six.
        That is what makes decomposition free here, and decomposition is the whole
        technique: a broad question hides several judgements behind one number.

        Returns `{name: Answer}` for every question that came back with a verdict,
        and `{}` for a call that could not be made at all.
        """
        if not self.available or not questions:
            return {}
        from induction.anthropic_call import is_transient, with_backoff

        payload = {"model": self.model, "state": state, "questions": questions}
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        try:
            data = with_backoff(
                lambda: self._transport(self.endpoint, payload, api_key, self._timeout),
                tries=self._tries, label="jev", log=self._log)
            self.calls += 1
        except Exception as e:  # noqa: BLE001 — a decision, never the run
            self.skipped += 1
            if is_transient(e):
                self._tripped = True
                self._log(f"[jev] the decisions API stayed unavailable through "
                          f"{self._tries} retries — disabling Jev for the rest of this "
                          f"run; the engine falls back to what it did before it.")
            else:
                self._log(f"[jev] call skipped ({type(e).__name__}: {e})")
            return {}

        answers = (data or {}).get("answers")
        if not isinstance(answers, dict):
            self.skipped += 1
            self._log("[jev] a reply carried no answers object — treated as no answer")
            return {}
        out: dict[str, Answer] = {}
        for name, raw in answers.items():
            parsed = parse_answer(raw)
            if parsed is not None:
                out[str(name)] = parsed
        return out
