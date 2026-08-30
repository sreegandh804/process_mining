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

Embeddings would *propose* finer boundaries — the documented upgrade path,
deliberately not built (similar text != same process).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from induction.adapters import Shaped
from induction.model import heuristic
from induction.process import ProcessKind
from induction.profiles import GENERIC_PROFILE, Profile
from induction.steps.correlate import Correlation
from induction.steps.variants import induced_variants


def segment(shaped: Shaped, corr: Correlation, profile: Profile = GENERIC_PROFILE) -> list[ProcessKind]:
    entities_by_id = {e.id: e for e in shaped.entities}
    is_bot = {e.id: e.attrs.get("is_bot", False) for e in shaped.entities if e.type == "person"}
    events_by_id = {e.id: e for e in shaped.events}

    # --- per-case generic features + optional domain features ---
    clusters: dict[tuple, list[str]] = defaultdict(list)
    per_case: dict[str, dict] = {}
    for case in corr.cases.values():
        evs = [events_by_id[e] for e in case.event_ids if e in events_by_id]
        n = len(evs)
        bot_n = sum(1 for e in evs if e.actor and is_bot.get(e.actor))
        automated = n > 0 and bot_n > n / 2
        cf = profile.case_features(case, entities_by_id) or {}
        per_case[case.id] = {"automated": automated, "events": evs, "cf": cf}
        # generic structural key: (automated?, correlation anchor). A profile may
        # refine it (git clusters by its own subkind).
        generic_key = (automated, case.kind_hint)
        clusters[profile.cluster_key(generic_key, cf)].append(case.id)

    # --- assemble a ProcessKind per cluster, biggest first (loudest reads first) ---
    kinds: list[ProcessKind] = []
    ordered = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    for idx, (_key, case_ids) in enumerate(ordered, 1):
        kf = _aggregate_features(case_ids, per_case, corr, entities_by_id)
        kid = profile.name_kind(kf) or f"kind_{idx}"
        display = profile.display_name(kid) or f"Kind {idx}"
        rationale = profile.rationale(kf) or _data_derived_rationale(kf)

        variants, dfg = induced_variants(case_ids, corr.cases)
        steps_seen = sorted({a for v in variants for a in v.signature})
        pk = ProcessKind(
            id=kid, name=display, rationale=rationale,
            confidence=heuristic("kind boundary inferred by structural clustering, not read"),
            case_ids=case_ids, variants=variants, dfg=dfg, steps=steps_seen,
        )
        pk.features = kf
        kinds.append(pk)
    return kinds


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
    return (f"{kf['n_cases']} runs · {'automated' if kf['automated'] else 'human-driven'} · "
            f"anchor: {kf['kind_hint']} · dominant activities: {acts} · touches: {types}. "
            f"Unnamed data-derived cluster — naming is deferred to a source profile or a human.")
