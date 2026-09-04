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
from datetime import datetime, timezone
from typing import Optional

from induction.adapters import Shaped
from induction.links import Link, links_of, record_unresolved
from induction.model import Confidence, Entity, Evidence, Event, heuristic, model
from induction.process import Case
from induction.semantic import SemanticProvider
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
        "title", "subject", "summary", "memo", "description", "text", "name", "body",
    )
    min_score: float = 0.30
    min_shared_tokens: int = 1
    # Proximity is the second half of the evidence: similar text alone is a
    # topic, not a case. Same actor, or opened within this window.
    require_proximity: bool = True
    proximity_days: int = 30
    cross_source_only: bool = False
    # Two components whose traces have the same *shape* are two runs of one
    # kind, not two halves of one run — however alike their text. Above this
    # overlap of activity signatures the pass declines to join them.
    max_activity_overlap: float = 0.8
    # A run can span more than two systems (a ledger, a delivery sheet and the
    # mail thread about both), and one round joins at most two components. Each
    # further round must clear the same bar against the *merged* evidence, so
    # later joins are held to a higher standard, not a lower one.
    max_rounds: int = 3


@dataclass(frozen=True)
class CorrelationPolicy:
    """Declarative knobs. Data, not code — the reason a new source needs neither."""

    # Types that are never process instances in their own right (an actor is not
    # a run). Adapters mark them; the correlator just honours the list.
    skip_types: frozenset[str] = frozenset({"person", "orphan_row"})
    fuzzy: FuzzyPolicy = FuzzyPolicy()
    # Optional model-tier pass (embeddings + LLM). None => off, and the engine
    # stays fully deterministic and offline. See induction/semantic.py.
    semantic: Optional[SemanticProvider] = None
    # Undo an over-eager guessed key (see steps/sessions.py). On by default: a
    # case a weak key stretched across a year-long silence is wrong in every
    # corpus, not merely untidy in one. Deterministic-key cases are never
    # touched, so this is a no-op wherever the keys mean something.
    split_quiet_sessions: bool = True


# Pooled component text is capped so one enormous thread cannot dominate the
# corpus-wide rarity statistics.
_MAX_POOLED_TEXT = 4000

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
    # Two different questions, kept apart because conflating them launders a
    # guess into a fact: `attach` is "why is this record in a case with the
    # *others*" (the connecting link), `self_conf` is "why is this record a run
    # at all" (its own claim on itself). A record that anchors its own case is
    # certain of its own events; a record dragged into someone else's case is
    # only ever as sure as the link that dragged it.
    attach: dict[str, Confidence] = {}
    self_conf: dict[str, Confidence] = {}

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
                self_conf.setdefault(source_id, link.confidence)
            return
        target = resolve(source_id, link)
        if target is None:
            return
        graph.union(source_id, target)
        # Both ends are in this case *because of this link*, so both are scored
        # by it — scoring only the source would leave the pointed-at record
        # looking certain of a membership it owes entirely to the pointer.
        # First claim wins and claims arrive strongest-first, so each record
        # keeps the strongest reason it has for being where it is.
        attach.setdefault(source_id, link.confidence)
        attach.setdefault(target, link.confidence)
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
    # Which component each record sat in *before* any guessing. Anything that
    # ends up on the far side of a fuzzy link from its case's anchor reached that
    # case through the guess, and cannot be more certain than the guess was —
    # `Confidence.weakest` applied to a chain, exactly as the model defines it.
    # Without this, a deterministically-linked record dragged across a fuzzy
    # bridge would keep reporting `joined`, laundering the guess.
    island: dict[str, str] = {eid: graph.find(eid) for eid in entities}
    bridge: dict[str, Confidence] = {}

    if policy.fuzzy.enabled:
        for _ in range(max(1, policy.fuzzy.max_rounds)):
            round_links = _fuzzy_pass(policy.fuzzy, entities, events_by_entity,
                                      obs_by_entity, graph)
            if not round_links:
                break
            for a, b, conf in round_links:
                _apply_inferred(a, b, conf, "fuzzy-match", bridge, graph, anchor_claims)

    # ---- pass 4: model — same-work paraphrase the token pass cannot see -----
    # Opt-in, and strictly a pass over the leftovers: it sees only components that
    # no key and no shared-token overlap could join, and each join it makes is
    # `model` — the weakest tier — carrying the model's own reason. It rides the
    # same bridge machinery as the fuzzy pass, so a case assembled across a model
    # link reports `model`, never launders the guess into something stronger.
    if policy.semantic is not None:
        for _ in range(max(1, policy.fuzzy.max_rounds)):
            round_links = _semantic_pass(policy.semantic, policy.fuzzy, entities,
                                         events_by_entity, obs_by_entity, graph)
            if not round_links:
                break
            for a, b, conf in round_links:
                _apply_inferred(a, b, conf, "model-match", bridge, graph, anchor_claims)

    # ---- assemble ---------------------------------------------------------
    shaped.entities.extend(materialised.values())
    corr = _build_cases(graph, entities, virtual, anchor_claims, attach, self_conf,
                        island, bridge, events_by_entity, obs_by_entity)

    # ---- pass 5: split a weak key that spans a silence ---------------------
    # Every pass above JOINS. None of them can undo a join that a guessed key
    # made too eagerly — and a guessed key (a shared subject, a shared title)
    # has no idea whether it is looking at one run or twenty. See
    # `steps/sessions.py`: it touches only cases whose weakest link is a guess,
    # so a real key spanning a year is left exactly as it was.
    if policy.split_quiet_sessions:
        from induction.steps.sessions import split_quiet_sessions
        split_quiet_sessions(corr, events_by_entity, obs_by_entity, policy)
    return corr


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
    """Join *components* that share no key, on text similarity + proximity.

    Components, not records, and that distinction is the whole usefulness of the
    pass. By the time it runs, determinism has already assembled what it can: a
    mail thread is one component (linked by In-Reply-To), an invoice and its
    payments are another (linked by a foreign key). The join worth making is
    between those two — and a record-level pass could never make it, because
    neither side is a lone record any more. It would only ever rescue debris.

    A component's text is its members' pooled, so a thread is matched on
    everything said in it rather than on whichever message happens to be first.
    Rarity is measured across the components actually in play, which is what
    stops shared boilerplate ("invoice", "widget", "fix") from carrying a join
    in a corpus where every record says it.

    Greedy best-first, one partner per component: chains of guesses compound
    into a mega-case no single link would justify.
    """
    pooled: dict[str, list[str]] = defaultdict(list)
    for eid, ent in entities.items():
        if not (events_by_entity.get(eid) or obs_by_entity.get(eid)):
            continue
        text = _text_of(ent, policy.text_attrs)
        if text:
            pooled[graph.find(eid)].append(text)
    if len(pooled) < 2:
        return []

    texts = {root: " ".join(parts)[:_MAX_POOLED_TEXT] for root, parts in pooled.items()}
    stats = TokenStats()
    for text in texts.values():
        stats.add(text)

    members: dict[str, list[str]] = defaultdict(list)
    for eid in entities:
        members[graph.find(eid)].append(eid)

    roots = list(texts)
    scored = []
    for i, ra in enumerate(roots):
        for rb in roots[i + 1:]:
            if policy.cross_source_only and _sources(members[ra], entities) & _sources(
                    members[rb], entities):
                continue
            sim = similar(texts[ra], texts[rb], stats)
            if sim.score < policy.min_score or len(sim.shared) < policy.min_shared_tokens:
                continue
            why = _proximity(members[ra], members[rb], events_by_entity, obs_by_entity, policy)
            if policy.require_proximity and why is None:
                continue
            if _same_shape(members[ra], members[rb], events_by_entity,
                           policy.max_activity_overlap):
                continue
            scored.append((sim.score, ra, rb, sim, why or "no proximity signal"))

    scored.sort(key=lambda t: (-t[0], t[1], t[2]))   # best first, deterministic ties
    taken: set[str] = set()
    out: list[tuple[str, str, Confidence]] = []
    for score, ra, rb, sim, why in scored:
        if ra in taken or rb in taken:
            continue
        taken.add(ra)
        taken.add(rb)
        out.append((ra, rb, heuristic(
            f"no shared key; text overlap {score:.2f} on {sim.describe()}, and {why}"
        )))
    return out


def _apply_inferred(a: str, b: str, conf: Confidence, method: str,
                    bridge: dict, graph: "_Components", anchor_claims: dict) -> None:
    """Union two components across an *inferred* bridge (fuzzy or model) and record
    it, so every record that reached its case across the bridge is scored no higher
    than the guess was — ``Confidence.weakest`` applied to the chain.

    The joined pair is a run nobody declared, so it licenses its own case — but at
    the weakest anchor rank there is, so any real claim in the merged component
    names it instead of the bridge.
    """
    for side in (a, b):
        prior = bridge.get(side)
        bridge[side] = conf if prior is None or conf.tier < prior.tier else prior
    graph.union(a, b)
    claim = Link(target=a, method=method, tier=conf.tier,
                 rationale=conf.rationale or "", anchors=True, anchor_rank=9)
    anchor_claims.setdefault(a, claim)
    anchor_claims.setdefault(b, replace(claim, target=b))


def _semantic_pass(provider: SemanticProvider, policy: FuzzyPolicy, entities,
                   events_by_entity, obs_by_entity, graph,
                   cap: int = 200) -> list[tuple[str, str, Confidence]]:
    """One round of model-tier joins over the *current* components.

    Mirrors the fuzzy pass's guards exactly — pooled component text, the proximity
    requirement, and the same-shape veto — because the point of the model tier is a
    better *judge*, not a laxer one: the model is only ever asked about pairs that
    are already close in people-or-time and are not two runs of a single kind. The
    only thing that changes is which oracle decides a shortlisted pair — a language
    model that can read paraphrase, in place of token overlap that cannot.

    Greedy, one partner per component per round (the correlator re-runs it), so a
    chain of model guesses cannot fuse a mega-case in a single sweep.
    """
    pooled: dict[str, list[str]] = defaultdict(list)
    for eid, ent in entities.items():
        if not (events_by_entity.get(eid) or obs_by_entity.get(eid)):
            continue
        text = _text_of(ent, policy.text_attrs)
        if text:
            pooled[graph.find(eid)].append(text)
    if len(pooled) < 2:
        return []

    texts = {root: " ".join(parts)[:_MAX_POOLED_TEXT] for root, parts in pooled.items()}
    members: dict[str, list[str]] = defaultdict(list)
    for eid in entities:
        members[graph.find(eid)].append(eid)

    roots = list(texts)
    when_who = {r: _component_when_who(members[r], events_by_entity, obs_by_entity)
                for r in roots}
    candidates: list[tuple[str, str]] = []
    for i, ra in enumerate(roots):
        for rb in roots[i + 1:]:
            if policy.require_proximity and _proximity(
                    members[ra], members[rb], events_by_entity, obs_by_entity, policy) is None:
                continue
            if _same_shape(members[ra], members[rb], events_by_entity, policy.max_activity_overlap):
                continue
            # SAME TOPIC IS NOT SAME RUN. The judge reads text, and text is what
            # a topic is made of: two threads about "the Dominion dispute", a
            # fortnight apart, with entirely different people on them, read as
            # the same work to a model and are two runs to anyone in the room.
            # On samples/enron that fused four subject threads into one 24-message
            # case whose trace interleaved two processes. Time proximity alone
            # cannot catch it — a fortnight is close. So a model join needs the
            # second, independent signal a real key would carry: someone in
            # common.
            #
            # Applied WITHIN ONE SOURCE only. Across sources the identity
            # namespaces do not line up — GitHub `maria` and maria@… are the same
            # person and different ids — so "nobody in common" is not evidence
            # there, and the cross-source join is the one the judge exists for.
            # Within a source it is decisive: two mail threads with disjoint
            # senders are two conversations.
            _, who_a = when_who[ra]
            _, who_b = when_who[rb]
            same_source = (_sources(members[ra], entities) == _sources(members[rb], entities)
                           and len(_sources(members[ra], entities)) == 1)
            if same_source and who_a and who_b and not (who_a & who_b):
                continue
            candidates.append((ra, rb))
    if not candidates:
        return []

    def with_context(root: str) -> str:
        """What the judge is shown: the text, headed by the facts that separate a
        run from a subject — when it happened and who was on it."""
        whens, whos = when_who[root]
        head = []
        if whens:
            lo, hi = min(whens), max(whens)
            head.append(f"When: {lo.date()}" + (f" to {hi.date()}" if hi != lo else ""))
        if whos:
            head.append("Who: " + ", ".join(sorted(w.split(":")[-1] for w in whos)[:6]))
        return ("\n".join(head) + "\n\n" if head else "") + texts[root]

    # Shortlist so the judge runs on a handful. An embedder ranks best; with none,
    # judge them all when few, else token-rank down to the cap — a budget when there
    # is no embedder, never the paraphrase filter the judge exists to beat.
    shortlist = provider.shortlist(texts, candidates)
    if shortlist is None:
        shortlist = candidates
        if len(shortlist) > cap:
            stats = TokenStats()
            for text in texts.values():
                stats.add(text)
            shortlist = sorted(
                shortlist, key=lambda p: -similar(texts[p[0]], texts[p[1]], stats).score)[:cap]

    taken: set[str] = set()
    out: list[tuple[str, str, Confidence]] = []
    for ra, rb in shortlist:
        if ra in taken or rb in taken:
            continue
        reason = provider.judge.judge(with_context(ra), with_context(rb))
        if not reason:
            continue
        taken.add(ra)
        taken.add(rb)
        out.append((ra, rb, model(f"same work (model): {reason}")))
    return out


def _sources(member_ids, entities) -> set[str]:
    return {entities[m].source for m in member_ids if m in entities}


def _same_shape(members_a, members_b, events_by_entity, threshold: float) -> bool:
    """Are these two components the same run, or two runs of the same kind?

    The distinction the fuzzy pass most needs and text cannot make. Two
    dependency-bump runs read almost identically — same words, same bot, days
    apart — and joining them invents a two-commit "process instance" that never
    happened, while destroying the evidence that this is a *recurring* pattern
    (which is what makes it a look-alike non-process worth rejecting).

    Two halves of one run look different: an issue is opened, labelled and
    commented; the pull request that serves it is reviewed and merged. Their
    activity signatures barely overlap, and that is the signal — complementary
    roles, not duplicated ones. Compared as a Jaccard overlap of the actions
    each side actually performed, so it needs no vocabulary of its own.
    """
    def actions(members):
        return {ev.action for m in members for ev in events_by_entity.get(m, [])}

    sig_a, sig_b = actions(members_a), actions(members_b)
    if not sig_a or not sig_b:
        return False           # nothing to compare is not evidence of sameness
    return len(sig_a & sig_b) / len(sig_a | sig_b) > threshold


def _text_of(entity, attrs: tuple[str, ...]) -> str:
    """Every populated text attribute an adapter exposes, joined.

    Not the first one: a mail message's evidence is its subject *and* its body,
    and an invoice's is its customer *and* its memo. Taking only the first would
    silently discard the half that usually carries the match. Repeated values are
    dropped so an adapter that exposes the same string twice does not
    double-weight it.
    """
    seen, parts = set(), []
    for name in attrs:
        val = entity.attrs.get(name)
        if isinstance(val, str) and val.strip() and val not in seen:
            seen.add(val)
            parts.append(val.strip())
    return " ".join(parts)


def _proximity(members_a, members_b, events_by_entity, obs_by_entity,
               policy: FuzzyPolicy) -> Optional[str]:
    """Same people, or close enough in time — what turns a shared topic into a case.

    Text alone is a topic: two invoices to the same customer a year apart are
    about the same thing and are not the same run. Requiring a second, different
    kind of signal is what keeps the fuzzy pass from merging a corpus by subject
    matter.
    """
    when_a, who_a = _component_when_who(members_a, events_by_entity, obs_by_entity)
    when_b, who_b = _component_when_who(members_b, events_by_entity, obs_by_entity)

    shared_actors = who_a & who_b
    if shared_actors:
        names = sorted(a.split(":")[-1] for a in shared_actors)[:2]
        return f"shared actor{'s' if len(names) > 1 else ''} ({', '.join(names)})"
    if when_a and when_b:
        # Nearest points of the two spans: a long thread overlapping an invoice's
        # life is close, even if their first events are months apart.
        days = min(abs((x - y).days) for x in when_a for y in when_b)
        if days <= policy.proximity_days:
            return f"{days} days apart"
    return None


def _component_when_who(member_ids, events_by_entity, obs_by_entity):
    """Every timestamp and actor the component can show, never invented."""
    whens, whos = [], set()
    for eid in member_ids:
        for ev in events_by_entity.get(eid, []):
            ts = _dt(ev.timestamp)
            if ts is not None:
                whens.append(ts)
            if ev.actor:
                whos.add(ev.actor)
        for ob in obs_by_entity.get(eid, []):
            ts = _dt(ob.seen_at)
            if ts is not None:
                whens.append(ts)
    return whens, whos


def _dt(raw: Optional[str]) -> Optional[datetime]:
    """Parse a timestamp for comparison, or None. Never raises on bad input.

    Sources disagree about timezones and always will: git and most APIs carry an
    offset, a spreadsheet cell carries a bare date with no timezone at all. A
    naive value is read as UTC purely so the two can be *compared* — the
    alternative is a TypeError the moment a corpus mixes a sheet with an API,
    which is precisely the corpus this engine is for. Nothing is written back, so
    no record gains a timezone it did not have; the only use is measuring how far
    apart two records are, against a window measured in days, where a sub-day
    ambiguity cannot change the answer.
    """
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Case assembly
# ---------------------------------------------------------------------------

def _build_cases(graph, entities, virtual, anchor_claims, attach, self_conf,
                 island, bridge, events_by_entity, obs_by_entity) -> Correlation:
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
            confidence=_case_confidence(members, anchor_id, anchor_claims,
                                        attach, island, bridge),
        )
        corr.cases[case.id] = case

        for m in real:
            evs, obs = events_by_entity.get(m, []), obs_by_entity.get(m, [])
            if m == anchor_id:
                # The anchor's own records are in its own case by definition.
                link_conf = self_conf.get(m) or attach.get(m) or case.confidence
            else:
                link_conf = attach.get(m) or self_conf.get(m) or case.confidence
            crossed = _crossed_bridge(m, anchor_id, island, bridge)
            if crossed is not None and crossed.tier < link_conf.tier:
                # It got here through a guess; it is only as sure as the guess.
                link_conf = crossed
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


def _crossed_bridge(member, anchor_id, island, bridge) -> Optional[Confidence]:
    """The fuzzy link a member had to cross to reach its case's anchor, if any."""
    if member == anchor_id or island.get(member) == island.get(anchor_id):
        return None
    return bridge.get(island.get(member)) or bridge.get(island.get(anchor_id))


def _case_confidence(members, anchor_id, anchor_claims, attach, island, bridge) -> Confidence:
    """How sure we are this run exists *as a run*, holding together.

    The link that named the case is the claim that it is a run — but a case
    assembled with one fuzzy member is a fuzzy case, however certain its anchor
    is of itself. So the reported confidence is the weakest link holding it
    together, the same rule `Confidence.weakest` applies to any chain of
    inference. Reporting the anchor's own certainty instead would let a guess
    ride into the output wearing `direct`, which is the single most damaging
    thing this engine could do.
    """
    named = anchor_claims.get(anchor_id)
    links = [attach[m] for m in members if m in attach]
    # Any fuzzy link used to assemble this case counts against it, even when
    # every individual record reached its own side of the bridge deterministically.
    for root in {island.get(m) for m in members if m in island}:
        if root in bridge:
            links.append(bridge[root])
    if named is not None:
        links.append(named.confidence)
    if links:
        return min(links, key=lambda c: c.tier)
    return heuristic("records grouped with no declared link")
