"""Step 2 — Correlate: group records into cases (process instances).

**One correlator, every source.** This is the weakest claim the engine makes, so
it gets the most care — and the care would be worthless if each source got its
own copy of it. Adapters declare `Link`s (see `induction/links.py`); this module
resolves them. It contains no source name, no commit, no invoice, no email.

Two layers, and the difference between them is the whole point:

1. **Deterministic** (`joined`) — a link the data itself asserts: a merge that
   says it closed PR #1, a foreign-key column, a mail ``In-Reply-To``. Resolved
   strongest-tier-first so a record lands in the best-evidenced case available,
   and first-claim-wins so the reason recorded against it is its strongest one.

2. **Fuzzy** (`heuristic`) — records with **no shared key at all**, joined on
   text similarity *plus* actor-or-time proximity. This is what a git DAG or a
   foreign key lets you dodge, and what an email/document corpus is made of. It
   is deliberately the *second* pass: it only ever sees what determinism could
   not explain, so it can never override a real key. Every fuzzy link records
   its score and the exact tokens it matched on, so it reads as the inference it
   is and a reader can overrule it.

What joins to nothing stays joined to nothing — it becomes an orphan (§6), never
padded into a case to make the output look tidier.

A link whose target never appears is resolved three ways, chosen by the adapter
because only the adapter knows which is true:
  * ``materialise`` — the thing is real, we just never saw its record. Create it
    as an *inferred* entity (its existence is then legible as inference, and its
    empty timeline becomes a gap in step 6).
  * ``virtual`` — the target is a case identity, not a thing ("this version's
    release notes", "this backport run"). It names the case without inventing an
    entity to hang it on.
  * neither — an unresolved reference, recorded on the source record as a
    reconciliation finding. Never silently dropped.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Optional

from induction.adapters import Shaped
from induction.links import Link, links_of, record_unresolved
from induction.model import Confidence, Entity, Evidence, Event, heuristic
from induction.process import Case
from induction.text import Similarity, TokenStats, similar


@dataclass
class Correlation:
    cases: dict[str, Case] = field(default_factory=dict)
    # records left with case_id=None are orphans; the honesty step collects them.

    def case_of(self, event: Event) -> Optional[Case]:
        return self.cases.get(event.case_id) if event.case_id else None


@dataclass(frozen=True)
class FuzzyPolicy:
    """When two records with no shared key may be treated as the same run.

    Every knob is a *threshold*, not a branch: there is no per-source code path
    here. Defaults are conservative — the fuzzy pass should recover the joins a
    key-based one visibly misses, not manufacture a tidy-looking process.
    """

    enabled: bool = True
    # Attribute names the fuzzy pass will read text from, tried in order. Any
    # adapter can populate one; none is required to.
    text_attrs: tuple[str, ...] = (
        "title", "subject", "summary", "memo", "description", "text", "name",
    )
    min_score: float = 0.30
    min_shared_tokens: int = 1
    # Proximity is the second half of the evidence: similar text alone is a
    # topic, not a case. Same actor, or opened within this window.
    require_proximity: bool = True
    proximity_days: int = 30
    # A record joins at most one fuzzy partner: chains of guesses compound into
    # a mega-case that no single link would justify.
    cross_source_only: bool = False


@dataclass(frozen=True)
class CorrelationPolicy:
    """Declarative knobs. Data, not code — the reason a new source needs neither."""

    # Types that are never process instances in their own right (an actor is not
    # a run). Adapters mark them; the correlator just honours the list.
    skip_types: frozenset[str] = frozenset({"person", "orphan_row"})
    fuzzy: FuzzyPolicy = FuzzyPolicy()


DEFAULT_POLICY = CorrelationPolicy()


# ---------------------------------------------------------------------------
# The correlator
# ---------------------------------------------------------------------------

def correlate(shaped: Shaped, policy: CorrelationPolicy | None = None) -> Correlation:
    policy = policy or DEFAULT_POLICY
    entities: dict[str, Entity] = {
        e.id: e for e in shaped.entities if e.type not in policy.skip_types
    }
    events_by_entity: dict[str, list] = defaultdict(list)
    obs_by_entity: dict[str, list] = defaultdict(list)
    for ev in shaped.events:
        if ev.entity_id in entities:
            events_by_entity[ev.entity_id].append(ev)
    for ob in shaped.observations:
        if ob.entity_id in entities:
            obs_by_entity[ob.entity_id].append(ob)

    graph = _Components()
    for eid in entities:
        graph.add(eid)

    materialised: dict[str, Entity] = {}
    # A "virtual" node is a case identity with no entity behind it — the notes
    # for a version, one backport run. It anchors and names, but is not a thing.
    virtual: dict[str, Link] = {}
    anchor_claims: dict[str, Link] = {}      # node id -> the link that named it
    attach: dict[str, Confidence] = {}       # entity id -> why it is in its case

    def resolve(source_id: str, link: Link) -> Optional[str]:
        """Find (or legitimately invent) the node a link points at."""
        if link.target in entities or link.target in virtual:
            return link.target
        if link.materialise:
            ent = _materialise(link, entities[source_id].source)
            entities[ent.id] = ent
            materialised[ent.id] = ent
            graph.add(ent.id)
            return ent.id
        if link.virtual:
            virtual[link.target] = link
            graph.add(link.target)
            return link.target
        # Nothing to point at and nothing the adapter says we may invent: a
        # dangling reference is a reconciliation finding, not a silent drop.
        record_unresolved(entities[source_id], link)
        return None

    def apply(source_id: str, link: Link) -> None:
        if link.target == source_id:
            # A self-link is an adapter saying "this record is a run in its own
            # right" — an invoice owns its lifecycle; a commit does not own one.
            # Nothing to union, but it does license a case to exist.
            if link.anchors:
                anchor_claims.setdefault(source_id, link)
                attach.setdefault(source_id, link.confidence)
            return
        target = resolve(source_id, link)
        if target is None:
            return
        graph.union(source_id, target)
        # First claim wins, and claims arrive strongest-first, so a record keeps
        # the strongest reason it has for being where it is.
        attach.setdefault(source_id, link.confidence)
        if link.anchors:
            anchor_claims.setdefault(target, link)

    # ---- pass 1: deterministic links, strongest tier first ----------------
    declared: list[tuple[str, Link]] = [
        (eid, link) for eid, ent in entities.items() for link in links_of(ent)
    ]
    firm = [(eid, l) for eid, l in declared if not l.fallback]
    firm.sort(key=lambda pair: -int(pair[1].tier))   # stable: declaration order within a tier
    for eid, link in firm:
        apply(eid, link)

    # ---- pass 2: fallback links, only for what nothing has claimed --------
    # Eager application would fuse every real case containing one leftover into
    # a single mega-case, so these wait until the firm links have settled.
    for eid, link in [(e, l) for e, l in declared if l.fallback]:
        if graph.size(eid) == 1:
            apply(eid, link)

    # ---- pass 3: fuzzy — no shared key, so text + proximity or nothing ----
    fuzzy_links: list[tuple[str, str, Confidence]] = []
    if policy.fuzzy.enabled:
        fuzzy_links = _fuzzy_pass(policy.fuzzy, entities, events_by_entity,
                                  obs_by_entity, graph)
        for a, b, conf in fuzzy_links:
            graph.union(a, b)
            attach.setdefault(a, conf)
            attach.setdefault(b, conf)
            # A fuzzy pair is a run nobody declared, so it must license its own
            # case — but at the weakest anchor rank there is, so any real claim
            # in the component names it instead.
            claim = Link(target=a, method="fuzzy-match", tier=conf.tier,
                         rationale=conf.rationale or "", anchors=True, anchor_rank=9)
            anchor_claims.setdefault(a, claim)
            anchor_claims.setdefault(b, replace(claim, target=b))

    # ---- assemble ---------------------------------------------------------
    shaped.entities.extend(materialised.values())
    return _build_cases(graph, entities, virtual, anchor_claims, attach,
                        events_by_entity, obs_by_entity, policy)


# ---------------------------------------------------------------------------
# Union-find over "nodes" (entities plus virtual case identities)
# ---------------------------------------------------------------------------

class _Components:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._size: dict[str, int] = {}

    def add(self, node: str) -> None:
        self._parent.setdefault(node, node)
        self._size.setdefault(node, 1)

    def find(self, node: str) -> str:
        self.add(node)
        root = node
        while self._parent[root] != root:
            self._parent[root] = self._parent[self._parent[root]]
            root = self._parent[root]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        self._parent[ra] = rb
        self._size[rb] = self._size[rb] + self._size[ra]

    def size(self, node: str) -> int:
        return self._size[self.find(node)]

    def groups(self, order: list[str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for node in order:
            out[self.find(node)].append(node)
        return out


def _materialise(link: Link, source: str) -> Entity:
    """An entity we never saw a record for, but a link asserts exists.

    Marked inferred with the link's own tier and evidence, so its existence
    reads as the inference it is — and so step 6 can flag its empty timeline.
    """
    attrs = {"known_via": "reference", **(link.materialise_attrs or {})}
    return Entity(
        id=link.target, source=source, type=link.materialise, attrs=attrs,
        confidence=Confidence(link.tier,
                              f"existence inferred from a reference ({link.method}); "
                              f"no direct record in {source}"),
        evidence=[Evidence(source, link.locator or link.target, link.snippet)],
    )


# ---------------------------------------------------------------------------
# The fuzzy pass
# ---------------------------------------------------------------------------

def _fuzzy_pass(policy: FuzzyPolicy, entities, events_by_entity, obs_by_entity,
                graph) -> list[tuple[str, str, Confidence]]:
    """Join records that share no key, on text similarity + actor/time proximity.

    Only records that carry their own timeline (events or observations) and that
    determinism left unexplained are candidates — the fuzzy pass never second-
    guesses a real key, and never invents a case out of two bare references.

    Greedy best-first with each record taking at most one partner: chains of
    guesses compound into a mega-case that no single link would justify.
    """
    candidates = [
        eid for eid, ent in entities.items()
        if graph.size(eid) == 1
        and (events_by_entity.get(eid) or obs_by_entity.get(eid))
        and _text_of(ent, policy.text_attrs)
    ]
    if len(candidates) < 2:
        return []

    # Rarity is measured over the candidate pool itself — the honest scope, and
    # what stops shared boilerplate ("invoice", "fix") from carrying a join.
    stats = TokenStats()
    for eid in candidates:
        stats.add(_text_of(entities[eid], policy.text_attrs))

    scored: list[tuple[float, str, str, Similarity, str]] = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            ea, eb = entities[a], entities[b]
            if policy.cross_source_only and ea.source == eb.source:
                continue
            sim = similar(_text_of(ea, policy.text_attrs),
                          _text_of(eb, policy.text_attrs), stats)
            if sim.score < policy.min_score or len(sim.shared) < policy.min_shared_tokens:
                continue
            why = _proximity(a, b, events_by_entity, obs_by_entity, policy)
            if policy.require_proximity and why is None:
                continue
            scored.append((sim.score, a, b, sim, why or "no proximity signal"))

    scored.sort(key=lambda t: (-t[0], t[1], t[2]))   # best first, deterministic ties
    taken: set[str] = set()
    out: list[tuple[str, str, Confidence]] = []
    for score, a, b, sim, why in scored:
        if a in taken or b in taken:
            continue
        taken.add(a)
        taken.add(b)
        out.append((a, b, heuristic(
            f"no shared key; text overlap {score:.2f} on {sim.describe()}, and {why}"
        )))
    return out


def _text_of(entity, attrs: tuple[str, ...]) -> str:
    """The first populated text attribute — adapters choose what to expose."""
    for name in attrs:
        val = entity.attrs.get(name)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _when_who(eid, events_by_entity, obs_by_entity) -> tuple[Optional[datetime], Optional[str]]:
    """The earliest moment we can see this record, and who was behind it."""
    best_ts, actor = None, None
    for ev in events_by_entity.get(eid, []):
        ts = _dt(ev.timestamp)
        if ts is not None and (best_ts is None or ts < best_ts):
            best_ts, actor = ts, ev.actor or actor
        elif actor is None:
            actor = ev.actor
    if best_ts is None:
        for ob in obs_by_entity.get(eid, []):
            ts = _dt(ob.seen_at)
            if ts is not None and (best_ts is None or ts < best_ts):
                best_ts = ts
    return best_ts, actor


def _proximity(a, b, events_by_entity, obs_by_entity, policy: FuzzyPolicy) -> Optional[str]:
    """Same person, or close enough in time — the half that turns topic into case."""
    ta, aa = _when_who(a, events_by_entity, obs_by_entity)
    tb, ab = _when_who(b, events_by_entity, obs_by_entity)
    if aa and ab and aa == ab:
        return f"same actor ({aa.split(':')[-1]})"
    if ta is not None and tb is not None:
        days = abs((ta - tb).days)
        if days <= policy.proximity_days:
            return f"{days} days apart"
    return None


def _dt(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Case assembly
# ---------------------------------------------------------------------------

def _build_cases(graph, entities, virtual, anchor_claims, attach,
                 events_by_entity, obs_by_entity, policy) -> Correlation:
    order = list(entities) + list(virtual)
    corr = Correlation()

    for members in graph.groups(order).values():
        real = [m for m in members if m in entities]
        has_records = any(events_by_entity.get(m) or obs_by_entity.get(m) for m in real)
        if not has_records:
            continue        # a cluster of bare references is not a run
        if not any(m in anchor_claims for m in members):
            # Nothing declared this a process instance: no link claimed it, it
            # claimed nothing, and no fuzzy pass could place it. A lone record is
            # then an *orphan*, not a one-step "process" — padding the output
            # with singleton cases is exactly the dishonesty §6 exists to stop.
            continue

        anchor_id = _pick_anchor(members, entities, virtual, anchor_claims,
                                 events_by_entity, obs_by_entity)
        anchor_link = anchor_claims.get(anchor_id)
        case = Case(
            id=f"case:{anchor_id}",
            kind_hint=_kind_of(anchor_id, entities, anchor_link),
            anchor=_anchor_attrs(anchor_id, entities, anchor_link),
            confidence=_case_confidence(members, anchor_id, anchor_claims, attach),
        )
        corr.cases[case.id] = case

        for m in real:
            evs, obs = events_by_entity.get(m, []), obs_by_entity.get(m, [])
            link_conf = attach.get(m) or case.confidence
            if m not in case.entity_ids:
                case.entity_ids.append(m)
            for ev in evs:
                ev.case_id = case.id
                ev.case_confidence = link_conf
                if ev.id not in case.event_ids:
                    case.event_ids.append(ev.id)
            for ob in obs:
                ob.case_id = case.id
                ob.case_confidence = link_conf
            ent = entities[m]
            if ent.evidence and ent.evidence[0] not in case.evidence:
                case.evidence.append(ent.evidence[0])
    return corr


def _pick_anchor(members, entities, virtual, anchor_claims, events_by_entity,
                 obs_by_entity) -> str:
    """Which member names the run.

    A run of commits is "PR #1", not "commit f1"; a payment run is the invoice,
    not the payment. Adapters mark the candidates (``anchors=True``) and rank
    them (``anchor_rank``) because which end of a link names a run is domain
    knowledge. Everything below that is mechanical tie-breaking, so the
    correlator can stay ignorant of what any of these records are.
    """
    claimed = [m for m in members if m in anchor_claims]
    pool = claimed or members

    def rank(node: str):
        link = anchor_claims.get(node)
        records = len(events_by_entity.get(node, [])) + len(obs_by_entity.get(node, []))
        return (
            link.anchor_rank if link is not None else 99,   # the adapter's judgement
            0 if node in virtual or node not in entities else 1,  # identity over record
            -records,                                        # then weight of evidence
            node,                                            # then id, for determinism
        )

    return min(pool, key=rank)


def _kind_of(node: str, entities, link: Link | None) -> str:
    ent = entities.get(node)
    if ent is not None:
        return ent.type
    if link is not None and link.materialise:
        return link.materialise
    return node.split(":", 1)[0]


def _anchor_attrs(node: str, entities, link: Link | None) -> dict:
    if link is not None and link.anchor_attrs:
        return dict(link.anchor_attrs)
    ent = entities.get(node)
    node_type = _kind_of(node, entities, link)
    attrs = {"type": node_type, "key": node.split(":", 1)[1] if ":" in node else node}
    if ent is not None and "number" in ent.attrs:
        attrs["number"] = ent.attrs["number"]
    return attrs


def _case_confidence(members, anchor_id, anchor_claims, attach) -> Confidence:
    """How sure we are this run exists at all.

    The link that *named* the case is the claim that it is a run, so that is the
    confidence reported — and where several members were attached by weaker
    links, the case cannot be stronger than its weakest member. Weakest-link is
    the same rule `Confidence.weakest` applies to any chain of inference.
    """
    named = anchor_claims.get(anchor_id)
    tiers = [attach[m] for m in members if m in attach]
    if named is not None:
        floor = min([named.confidence] + tiers, key=lambda c: c.tier)
        return named.confidence if floor.tier >= named.tier else floor
    if tiers:
        return min(tiers, key=lambda c: c.tier)
    return heuristic("records grouped with no declared link")
