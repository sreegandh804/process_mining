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
        "and GROUP only: do not invent an activity no pair evidences. Return ONLY "
        'JSON: {"map": {"<artefact>/<verb>": "<Activity Name>"}} covering every pair.'
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


def infer_activities(m, mapper: Optional[ActivityMapper]) -> dict[str, str]:
    """Ask the mapper to name the corpus's activities, returning
    ``{"<artefact>/<verb>": "Activity Name"}`` — or ``{}`` when no mapper is wired
    in (the engine then shows raw artefacts, claiming no abstraction)."""
    if mapper is None:
        return {}
    types = {e.id: e.type for e in m.shaped.entities}
    vocab: dict[str, dict] = {}
    for ev in m.shaped.events:
        artefact = types.get(ev.entity_id, "record")
        k = _key(artefact, ev.action)
        entry = vocab.setdefault(k, {"artefact": artefact, "action": ev.action, "examples": []})
        snip = ev.evidence[0].snippet if ev.evidence else None
        if snip and len(entry["examples"]) < 3 and snip not in entry["examples"]:
            entry["examples"].append(snip[:80])
    return mapper.map(list(vocab.values()))


def _parse_map(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}
    inner = obj.get("map", obj)
    return inner if isinstance(inner, dict) else {}
