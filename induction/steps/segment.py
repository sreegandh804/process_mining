"""Step 0 — Segment: separate the different *kinds* of process.

Hold the line the reviewers will probe:
  - Segment (step 0) = different *kinds* of process — distinct processes.
  - Variant (step 4) = different *runs of one kind*.

Nothing in the data announces where one kind ends and the next begins, so every
boundary here is an **inference** (tier `heuristic`) and correctable.

The DEFAULT baseline is deliberately source-agnostic and **unnamed**: we cluster
runs by structure alone — whether they are automated vs human-driven, and their
correlation anchor — and leave the resulting kinds as `kind_1`, `kind_2`, … with
a *data-derived* rationale. We will not invent domain names ("dependency bump",
"release") for data we know nothing about; naming is deferred to a source
`Profile` or a human. A profile (e.g. the git one) can refine the clustering and
supply real names — but that is an overlay, never a requirement.

Structure alone is not always enough to say anything at all. Where a source has
exactly one shape — a mailbox, where every run is `(human, email_thread)` — the
structural key collapses the whole corpus into one kind, and "they all send
mail" is not a process model. `steps/topics.py` is the fallback for precisely
that: over a cluster that came out large and undifferentiated, it groups runs by
what they are *about*, using the rarity-weighted token overlap already built for
the fuzzy pass. It declines unless the vocabulary genuinely separates the
cluster, so a source whose structure did the work is untouched.

Embeddings would *propose* finer boundaries still — the documented upgrade path,
deliberately not built (similar text != same process, which is why a
topic-refined boundary stays `heuristic` and names the words that made it).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from induction.adapters import Shaped
from induction.model import heuristic
from induction.process import ProcessKind
from induction.profiles import GENERIC_PROFILE, Profile
# `_text_of` and the pooling cap are shared with the correlator on purpose: the
# text a kind is grouped on must be the same text a join is scored on, or the
# two steps would be reading different corpora.
from induction.steps.correlate import Correlation, DEFAULT_POLICY, _MAX_POOLED_TEXT, _text_of
from induction.steps.topics import DEFAULT_TOPIC_POLICY, TopicPolicy
from induction.steps.topics import refine as refine_by_topic
from induction.steps.variants import induced_variants


def segment(shaped: Shaped, corr: Correlation, profile: Profile = GENERIC_PROFILE,
            topics: TopicPolicy = DEFAULT_TOPIC_POLICY) -> list[ProcessKind]:
    entities_by_id = {e.id: e for e in shaped.entities}
    is_bot = {e.id: e.attrs.get("is_bot", False) for e in shaped.entities if e.type == "person"}
    events_by_id = {e.id: e for e in shaped.events}

    # --- per-case generic features + optional domain features ---
    clusters: dict[tuple, list[str]] = defaultdict(list)
    per_case: dict[str, dict] = {}
    for case in corr.cases.values():
        evs = [events_by_id[e] for e in case.event_ids if e in events_by_id]
        # "Automated" means a machine touched it and no human did. This is
        # domain-general and robust to blank actors (an unrecorded actor is
        # unknown, not automated) — unlike a majority vote, which a `None`
        # committer/merger would dilute.
        actor_ids = [e.actor for e in evs if e.actor]
        has_bot = any(is_bot.get(a) for a in actor_ids)
        has_human = any(not is_bot.get(a) for a in actor_ids)
        automated = has_bot and not has_human
        cf = profile.case_features(case, entities_by_id) or {}
        per_case[case.id] = {"automated": automated, "events": evs, "cf": cf}
        # generic structural key: (automated?, correlation anchor). A profile may
        # refine it (git clusters by its own subkind).
        generic_key = (automated, case.kind_hint)
        clusters[profile.cluster_key(generic_key, cf)].append(case.id)

    # --- refine a large, undifferentiated cluster by what its runs are about ---
    clusters, topic_terms = _refine_by_topic(clusters, corr, entities_by_id, topics)

    # --- assemble a ProcessKind per cluster, biggest first (loudest reads first) ---
    kinds: list[ProcessKind] = []
    seen_ids: Counter = Counter()
    ordered = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    for idx, (key, case_ids) in enumerate(ordered, 1):
        kf = _aggregate_features(case_ids, per_case, corr, entities_by_id)
        terms = topic_terms.get(key, ())
        if terms:
            kf["topic_terms"] = list(terms)

        # A profile names a kind from its features, so two topic-refined
        # siblings can claim the same name. Keep the profile's word and make the
        # *id* unique, rather than dropping to an anonymous one.
        base_kid = profile.name_kind(kf) or f"kind_{idx}"
        seen_ids[base_kid] += 1
        kid = base_kid if seen_ids[base_kid] == 1 else f"{base_kid}_{seen_ids[base_kid]}"
        display = profile.display_name(base_kid) or f"Kind {idx}"
        rationale = profile.rationale(kf) or _data_derived_rationale(kf)
        why = "kind boundary inferred by structural clustering, not read"
        if terms:
            # The terms go in the name because a reader picking between kinds
            # needs to know which is which before opening either.
            display = f"{display} — {', '.join(terms[:3])}"
            why = (f"kind boundary inferred by structural clustering, then grouped by "
                   f"shared vocabulary ({', '.join(terms)}); not read")

        variants, dfg = induced_variants(case_ids, corr.cases)
        steps_seen = sorted({a for v in variants for a in v.signature})
        pk = ProcessKind(
            id=kid, name=display, rationale=rationale,
            confidence=heuristic(why),
            case_ids=case_ids, variants=variants, dfg=dfg, steps=steps_seen,
        )
        pk.features = kf
        kinds.append(pk)
    return kinds


def _case_text(case, entities_by_id) -> str:
    """A run's pooled text — every text attribute of every record in it."""
    parts, seen = [], set()
    for eid in case.entity_ids:
        ent = entities_by_id.get(eid)
        if ent is None:
            continue
        text = _text_of(ent, DEFAULT_POLICY.fuzzy.text_attrs)
        if text and text not in seen:
            seen.add(text)
            parts.append(text)
    return " ".join(parts)[:_MAX_POOLED_TEXT]


def _refine_by_topic(clusters: dict, corr, entities_by_id, policy: TopicPolicy):
    """Split each structural cluster by topic, where the vocabulary supports it.

    Applied per cluster rather than corpus-wide, so a topic never reaches across
    a structural boundary the data *did* establish: an automated cluster and a
    human one stay separate however alike they read. And a cluster only qualifies
    at all if it is essentially the whole corpus — see `TopicPolicy.min_dominance`.
    """
    if not policy.enabled:
        return clusters, {}

    n_cases = sum(len(ids) for ids in clusters.values())
    out: dict[tuple, list[str]] = {}
    terms_by_key: dict[tuple, tuple[str, ...]] = {}
    for key, case_ids in clusters.items():
        texts = {cid: _case_text(corr.cases[cid], entities_by_id) for cid in case_ids}
        topic_of, terms = refine_by_topic(texts, n_cases, policy)
        if not topic_of:                       # structure stands, unrefined
            out[key] = case_ids
            continue
        for cid in case_ids:
            topic = topic_of.get(cid)
            # A run whose vocabulary matched nothing stays in the parent kind —
            # an unexplained run is not a one-run process.
            sub_key = key + (("topic", topic),) if topic else key
            out.setdefault(sub_key, []).append(cid)
            if topic:
                terms_by_key[sub_key] = terms[topic]
    return out, terms_by_key


def _aggregate_features(case_ids, per_case, corr, entities_by_id) -> dict:
    actions: Counter = Counter()
    etypes: Counter = Counter()
    automated_votes = 0
    merged: dict = {}
    for cid in case_ids:
        info = per_case[cid]
        automated_votes += 1 if info["automated"] else 0
        for e in info["events"]:
            actions[e.action] += 1
        for eid in corr.cases[cid].entity_ids:
            ent = entities_by_id.get(eid)
            if ent is not None:
                etypes[ent.type] += 1
        _merge(merged, info["cf"])
    kind_hint = Counter(corr.cases[cid].kind_hint for cid in case_ids).most_common(1)[0][0]
    return {
        "automated": automated_votes * 2 > len(case_ids),
        "kind_hint": kind_hint,
        "n_cases": len(case_ids),
        "dominant_actions": [a for a, _ in actions.most_common(4)],
        "entity_types": dict(etypes),
        **merged,
    }


def _merge(dst: dict, cf: dict) -> None:
    """Aggregate per-case domain features across a cluster: sum counters, keep
    the first scalar (constant within a profile-defined cluster)."""
    for k, v in cf.items():
        if isinstance(v, dict):
            dst[k] = Counter(dst.get(k, Counter())) + Counter(v)
        elif k not in dst:
            dst[k] = v


def _data_derived_rationale(kf: dict) -> str:
    acts = ", ".join(kf["dominant_actions"]) or "—"
    types = ", ".join(sorted(kf["entity_types"])) or "—"
    topic = ""
    if kf.get("topic_terms"):
        # Say what separated this from its siblings, and say what that is worth:
        # shared words are evidence of a shared subject, not proof of a shared
        # process, and the reader is the one who can tell the difference.
        topic = (f" Grouped by shared vocabulary: {', '.join(kf['topic_terms'])} — "
                 f"structure alone did not separate these runs, so the boundary is "
                 f"read off what they are about, and is revisable.")
    return (f"{kf['n_cases']} runs · {'automated' if kf['automated'] else 'human-driven'} · "
            f"anchor: {kf['kind_hint']} · dominant activities: {acts} · touches: {types}. "
            f"Unnamed data-derived cluster — naming is deferred to a source profile or a human."
            f"{topic}")
