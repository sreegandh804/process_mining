"""Optional LLM naming (tier `model`) — friendly names for kinds and steps.

This is the *only* place an LLM touches the system, and it is bounded to the
guardrail the whole design rests on: **the LLM may name, never do structural
work.** It receives the already-induced actions and a few example records and
returns human labels. It cannot change a correlation, an order, a variant, or a
finding — those are computed deterministically before this runs and are not sent
back in.

It is entirely optional. With no `ANTHROPIC_API_KEY` (and no `--names llm`) the
engine runs exactly as before with generic/profile names. When enabled, the
names it returns are inference — surfaced as "names suggested by AI" so a reader
knows they are a `model`-tier convenience, not read from the data.

Model + SDK usage follow the Anthropic Python SDK (`pip install anthropic`).
"""

from __future__ import annotations

import json
import os
from collections import Counter

_SYSTEM = (
    "You NAME things; you do NOT analyse, infer, or add process facts. You are given "
    "activity verbs and process clusters that were already discovered from data, plus a "
    "few example records. Return short, human, domain-appropriate names for them, using "
    "ONLY the information given. Never invent a step, a stage, or a claim that is not in "
    "the input. Return ONLY a JSON object, no prose."
)


def _clean(names: dict) -> dict:
    """Guardrail: keep only naming keys, ignore anything else the model returned.
    A namer may name; it may not add a step, a run, or a finding."""
    return {
        "item": str(names.get("item", "")) or None,
        "items": str(names.get("items", "")) or None,
        "steps": {str(k): str(v) for k, v in (names.get("steps") or {}).items()},
        "kinds": {str(k): str(v) for k, v in (names.get("kinds") or {}).items()},
        "_ai": True,
    }


def infer_names(model, enable: bool = False, api_model: str | None = None, namer=None) -> dict:
    """Return {item, items, steps:{action:Name}, kinds:{id:Name}} or {} if disabled.

    `namer` (a callable ``payload -> raw names``) is the injection seam: the demo
    passes an offline stand-in, tests a scripted one, and the whole naming path is
    exercised without a key. With no namer, the real Anthropic path runs — and
    only when `enable` is true (an LLM call spends money) AND a key resolves.
    """
    payload = _payload(model)
    if namer is not None:
        try:
            return _disambiguate(_clean(namer(payload)), model)
        except Exception as e:  # naming is a convenience; never break the run
            print(f"[names] namer skipped ({type(e).__name__}: {e})")
            return {}

    if not enable or not os.environ.get("ANTHROPIC_API_KEY"):
        return {}
    try:
        import anthropic
    except ImportError:
        print("[names] --names llm needs the Anthropic SDK: pip install anthropic")
        return {}
    try:
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=api_model or os.environ.get("INDUCTION_NAMING_MODEL", "claude-opus-5"),
            max_tokens=1500,
            system=_SYSTEM,
            messages=[{"role": "user", "content": _prompt(payload)}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return _disambiguate(_clean(_parse(text)), model)
    except Exception as e:  # naming is a convenience; never break the run
        print(f"[names] LLM naming skipped ({type(e).__name__}: {e})")
        return {}


def _disambiguate(names: dict, model) -> dict:
    """Two kinds may not wear the same name.

    A model handed two clusters it cannot tell apart returns one name twice, and
    the reader sees two identical cards with no way to know which is which — that
    happened on a real mailbox ("Send then forward email", twice). The prompt asks
    for distinct names; this guarantees it. The tie is broken by what actually
    separated the kinds — their own subject terms — falling back to a numeral only
    when there is nothing better to say.
    """
    kinds = (names or {}).get("kinds") or {}
    if not kinds:
        return names
    terms_of = {k.id: list(k.features.get("topic_terms", [])) for k in model.kinds}
    taken: set[str] = set()
    out: dict[str, str] = {}
    for kid, name in kinds.items():
        if name not in taken:
            out[kid], _ = name, taken.add(name)
            continue
        for term in terms_of.get(kid, []):
            candidate = f"{name} ({term})"
            if candidate not in taken:
                break
        else:
            n = 2
            while f"{name} ({n})" in taken:
                n += 1
            candidate = f"{name} ({n})"
        out[kid] = candidate
        taken.add(candidate)
    names["kinds"] = out
    return names


def _payload(model) -> dict:
    events_by_id = {e.id: e for e in model.shaped.events}
    actions = sorted({e.action for e in model.shaped.events})
    kinds = []
    for k in model.kinds:
        # a few short example snippets, purely to hint the domain
        examples = []
        for cid in k.case_ids[:3]:
            case = model.cases.get(cid)
            if not case:
                continue
            snip = None
            for eid in case.event_ids[:1]:
                ev = events_by_id.get(eid)
                if ev and ev.evidence:
                    snip = ev.evidence[0].snippet
            if snip:
                examples.append(snip[:120])
        kinds.append({
            "id": k.id,
            "n_runs": len(k.case_ids),
            "activities_in_order": k.features.get("dominant_actions", k.steps),
            "automated": bool(k.features.get("automated")),
            # What the cluster is ABOUT. Without this the model can only see the
            # verb sequence, and names a mail cluster "Send then forward email" —
            # the transport, not the work.
            "subject_terms": k.features.get("topic_terms", []),
            "examples": examples,
        })
    return {"activities": actions, "kinds": kinds}


def _prompt(payload: dict) -> str:
    return (
        "Here is a process model discovered from a company's own records.\n\n"
        "Name each kind for the WORK IT DOES — read its subject_terms and examples. "
        "Never name it after the verb sequence: 'Send then forward email' describes "
        "how a message travelled, not what anyone was doing. Every kind must get a "
        "DIFFERENT name; if two look alike, their subject_terms are what separate "
        "them.\n\n"
        + json.dumps(payload, indent=2)
        + "\n\nReturn JSON exactly like:\n"
        '{\n'
        '  "item": "<singular noun for one run, e.g. grant / invoice / pull request>",\n'
        '  "items": "<plural>",\n'
        '  "steps": { "<activity verb>": "<short human step name>", ... },\n'
        '  "kinds": { "<kind id>": "<short human process name>", ... }\n'
        "}\n"
        "Rules: keep names 1-3 words; base them only on the verbs/examples given; "
        "if a kind is automated, its name may say so; do not add steps."
    )


def _parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    return json.loads(text[start:end + 1])
