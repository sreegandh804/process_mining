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

Token overlap is the *weakest* form of "what it is about", and on a real mailbox
it shows: refining 57 Enron threads produced kinds called `shirley, shall, time`
and `hey, ena, work`. Shared vocabulary is not shared subject matter. So where
`abstraction.py`'s reading tier has actually READ each record and placed it in
one of the process families it derived from the corpus, those readings are
passed in as `case_process` and they take precedence over the token fallback.
The engine still draws every boundary — it counts each run's records and places
the run by majority (`_reproject`) — the model only ever supplied the names.
Such a boundary is tier `model`, and says so.

Embeddings would *propose* finer boundaries still — the documented upgrade path,
deliberately not built (similar text != same process, which is why a
topic-refined boundary stays `heuristic` and names the words that made it).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from induction.adapters import Shaped
from induction.model import heuristic
from induction.model import model as model_tier
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
            topics: TopicPolicy = DEFAULT_TOPIC_POLICY,
            case_process: dict | None = None) -> list[ProcessKind]:
    """Cluster runs into kinds.

    `case_process` — ``{case_id: "Contract execution"}`` — is the reading tier's
    answer to "what is this run about", available only after `abstraction.py` has
    read the records. When present it replaces the token-overlap fallback for the
    runs it covers: a run it does not name stays in its structural cluster rather
    than being forced into a family nothing evidenced.
    """
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

    # --- refine by what each run is ABOUT ---
    # The reading tier wins where it has an answer: it read the records, and the
    # token fallback only counted their words. Where it has none, the fallback
    # gets its usual (deliberately reluctant) turn.
    read_terms: dict[tuple, str] = {}
    if case_process:
        clusters, read_terms = _split_by_read_process(clusters, case_process,
                                                      cases=corr.cases)
        topic_terms: dict[tuple, tuple[str, ...]] = {}
    else:
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
        read_process = read_terms.get(key)
        if read_process:
            kf["read_process"] = read_process
            kf["project"] = ("project", read_process) in key

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

        if read_process and kf.get("project"):
            display = read_process
            pk_confidence = model_tier(
                f"a project, not a process: {len(case_ids)} run(s) of {read_process} "
                f"with a real multi-step arc, but nothing that recurs — it happened, "
                f"it had stages, it is not how work is usually done here")
        elif read_process:
            # A boundary drawn from records the model actually read. The name is
            # the family it named; the tier is `model`, because that is the claim.
            display = read_process
            pk_confidence = model_tier(
                f"grouped by what each run is about, read from its records "
                f"({read_process}); the boundary is a count over those readings, "
                f"not a label a model put on the run")
        else:
            pk_confidence = heuristic(why)

        variants, dfg = induced_variants(case_ids, corr.cases)
        steps_seen = sorted({a for v in variants for a in v.signature})
        pk = ProcessKind(
            id=kid, name=display, rationale=rationale,
            confidence=pk_confidence,
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


# A one-off needs this many distinct read steps to count as a project rather than
# noise. Three: a request and its answer is correspondence; a request, a review
# and a decision is an arc.
_PROJECT_MIN_STEPS = 3


def _split_by_read_process(clusters: dict, case_process: dict,
                           policy: TopicPolicy = DEFAULT_TOPIC_POLICY,
                           cases: dict | None = None):
    """Split each structural cluster by the process its runs were READ into.

    Two refusals, both taken from `_refine_by_topic` — not because a mailbox
    needs them, but because the question ("is this group big enough to be a
    kind?") is the same question whatever read the group, and two paths answering
    it with different numbers is how a corpus starts getting special-cased:

      - a run the reading did not place stays in its parent cluster; an
        unexplained run is not a one-run process, and
      - a family below `max(min_topic_cases, len * min_topic_share)` is folded
        back into the parent. **Both** terms matter, and using only the floor was
        a bug: `topics.py` puts the reason plainly — 3 out of 40 is a topic,
        3 out of 509 is a splinter. A flat floor is tuned to whatever corpus was
        in front of you when you picked it, which is exactly the overfitting this
        engine is supposed to refuse. The share makes it scale-free.
    """
    out: dict[tuple, list[str]] = {}
    terms_by_key: dict[tuple, str] = {}
    for key, case_ids in clusters.items():
        floor = max(policy.min_topic_cases,
                    round(len(case_ids) * policy.min_topic_share))
        by_process: dict[str, list[str]] = defaultdict(list)
        unplaced: list[str] = []
        for cid in case_ids:
            proc = case_process.get(cid)
            (by_process[proc] if proc else unplaced).append(cid)
        for proc, ids in sorted(by_process.items()):
            if len(ids) >= floor:
                sub_key = key + (("process", proc),)
                out.setdefault(sub_key, []).extend(ids)
                terms_by_key[sub_key] = proc
                continue
            # Below the floor it is not a PROCESS — nothing recurs. But it may be a
            # PROJECT: one run (or two) with a real multi-step arc. Upgrading the
            # office chairs is contact the seller -> negotiate -> proposal reviewed
            # -> paid -> delivered, once. That is work with structure, and binning
            # it with "Congratulations" and forwarded MIME blobs is a lie of
            # omission. Repetition makes a process; structure without repetition
            # is a project; neither is noise.
            if cases is not None and any(
                    len(set(cases[cid].trace_signature)) >= _PROJECT_MIN_STEPS
                    for cid in ids if cid in cases):
                sub_key = key + (("project", proc),)
                out.setdefault(sub_key, []).extend(ids)
                terms_by_key[sub_key] = proc
                continue
            unplaced.extend(ids)              # no repetition, no arc: not a thing
        if unplaced:
            out.setdefault(key, []).extend(unplaced)
    return out, terms_by_key


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
    if kf.get("read_process"):
        return (f"{kf['n_cases']} runs whose records were read as {kf['read_process']}. "
                f"Steps seen: {acts}. The boundary is a majority count over each run's "
                f"own records — the model named the family, it did not draw the line.")
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
