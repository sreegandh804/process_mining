"""Topic refinement for Segment (step 0) — what a run is *about*, when its
*shape* says nothing.

The segmenter clusters runs by structure: automated vs human-driven, and the
correlation anchor. That discriminates well when a source has several structural
shapes (git has bot merges, squash landings and release trains; a finance export
has invoices and payments). It discriminates **not at all** when every run has
the same shape — a mailbox, where all 761 threads are `(human, email_thread)`
and collapse into one kind whose "process" reads `sent → forwarded`. Structure
had nothing to say, and the honest response is not to assert one kind.

So this is a **fallback**, in exactly the sense the changelog and mail adapters
already use for their thread keys: it runs only over a structural cluster that
came out large and undifferentiated, and it declines unless the vocabulary
actually separates that cluster into groups.

`FuzzyPolicy` puts the licence for this plainly, from the other side:

    "similar text alone is a topic, not a case"

Correct — which is why the correlator refuses to *join runs* on text alone, and
why grouping *kinds* on text alone is the legitimate use of the same signal. A
kind is a family of runs about the same sort of work; that is what a topic is.
No new machinery is needed for it: `induction/text.py` already scores overlap
with corpus-derived rarity (built for the fuzzy pass) and already reports the
tokens that produced a score, so every kind this creates can name the words that
made it one.

**What this is not.** It groups by *vocabulary*, not by process. Two different
processes discussed in the same words will merge; one process discussed in two
vocabularies will split. So a topic-refined boundary stays `heuristic`, says
"grouped by shared vocabulary" in its rationale, and lists its own terms — a
reader who disagrees can see exactly what to disagree with. Embeddings are the
next rung and slot in behind `similar()`; they would find paraphrase this misses
and would cost the auditability, which is why they are not the first rung.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from induction.text import TokenStats, similar, tokenize


@dataclass(frozen=True)
class TopicPolicy:
    """Thresholds, not branches — the same discipline as `FuzzyPolicy`.

    Every default here is set to *decline*: refinement has to earn its place
    against a cluster big enough for a split to mean something, with groups big
    enough not to be noise.
    """

    enabled: bool = True
    # THE gate, and the one that matters most. Refine only a cluster that *is*
    # essentially the whole corpus — the operational meaning of "structure said
    # nothing". Measured, not guessed: pallets/flask segments into 5 structural
    # clusters at 49/24/16/10/1%, and refining its 49% one produces nonsense
    # kinds ("method, add, flask") because every commit is about something
    # different — similar text is not the same process. A mailbox segments into
    # exactly one cluster at 100%, because every run is `(human, email_thread)`.
    # Between those two the engine should keep its mouth shut, so the bar sits
    # near the top: only a corpus that structure genuinely failed to partition.
    min_dominance: float = 0.9
    # Below this a cluster is one kind until proven otherwise — a handful of
    # runs cannot evidence a split, and over-fitting a tiny corpus into topics
    # is exactly the invented structure this engine exists to avoid.
    min_cases: int = 25
    # A group smaller than this is not a kind; its runs stay in the parent.
    min_topic_cases: int = 3
    # Same bar as the fuzzy pass: enough shared word-mass to be about one thing.
    min_score: float = 0.30
    # Boilerplate exclusion, and deliberately loose. This is a *blocking* device
    # — it decides which pairs are worth scoring, not which are joined — so it
    # only has to drop the near-universal ("re", "enron", "invoice"); `similar()`
    # already discounts a common token by its rarity weight when it scores. Set
    # tight it silently caps how big a topic can be: at 0.25 a corpus of three
    # equal topics finds none of them, because each one *is* a third of it.
    max_df_ratio: float = 0.6
    # Blocking: a posting list longer than this costs more pairs than it can be
    # worth. Rarest tokens are spent first, so this bites only the common tail.
    max_postings: int = 400
    # A budget, so blocking degrades gracefully instead of falling back to the
    # O(n^2) scan it exists to avoid. Rarest tokens are spent first.
    max_pairs: int = 200_000
    # Beyond this the tail is noise; the rest stay in the parent kind.
    max_topics: int = 24


DEFAULT_TOPIC_POLICY = TopicPolicy()


class _Union:
    """Minimal union-find. The correlator's is bound to its own graph; this one
    is over case ids and nothing else."""

    def __init__(self):
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def refine(case_texts: dict[str, str], n_corpus_cases: int,
           policy: TopicPolicy = DEFAULT_TOPIC_POLICY,
           ) -> tuple[dict[str, int], dict[int, tuple[str, ...]]]:
    """Split one structural cluster by what its runs are about.

    ``case_texts`` is ``{case_id: pooled text}`` for a single structural
    cluster; ``n_corpus_cases`` is the size of the whole corpus, which is what
    decides whether structure partitioned it or merely failed to. Returns
    ``({case_id: topic_index}, {topic_index: terms})`` covering only the cases
    that landed in a topic — a case absent from the mapping stays in the parent
    kind, which is the honest place for a run whose vocabulary matched nothing.

    Returns empty mappings when refinement should not apply at all: the cluster
    is one partition among several that structure *did* find, too few cases, no
    text to read, or a vocabulary that failed to separate them. That last case
    matters as much as the gate — a cluster of 500 invoices that all say the
    same words comes back unrefined, because it *is* one kind.
    """
    if not policy.enabled:
        return {}, {}
    # Structure found a real partition; this cluster is one side of it, and what
    # a run is *about* is not licence to cut across an answer we already have.
    if not n_corpus_cases or len(case_texts) < n_corpus_cases * policy.min_dominance:
        return {}, {}
    texts = {cid: t for cid, t in case_texts.items() if t and t.strip()}
    if len(texts) < policy.min_cases:
        return {}, {}

    stats = TokenStats()
    for text in texts.values():
        stats.add(text)

    # --- blocking: only compare cases that share a rare-enough token ---------
    # Without this the pass is O(n^2) over the cluster, which is the cost that
    # already makes the fuzzy pass slow on a real mailbox. An inverted index
    # over discriminating tokens turns it into "compare what could possibly
    # match", and the pair budget keeps the worst case bounded.
    postings: dict[str, list[str]] = defaultdict(list)
    for cid, text in texts.items():
        content, strong = tokenize(text)
        for tok in content | strong:
            postings[tok].append(cid)

    df_cap = max(2, int(len(texts) * policy.max_df_ratio))
    usable = [tok for tok, ids in postings.items()
              if 2 <= len(ids) <= min(df_cap, policy.max_postings)]
    # Rarest first, so the budget is spent on the most discriminating evidence.
    usable.sort(key=lambda t: (-stats.weight(t), t))

    candidates: set[tuple[str, str]] = set()
    for tok in usable:
        ids = sorted(postings[tok])
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                candidates.add((a, b))
        if len(candidates) >= policy.max_pairs:
            break

    # --- score, and union what clears the bar --------------------------------
    graph = _Union()
    joined_terms: dict[tuple[str, str], tuple[str, ...]] = {}
    for a, b in candidates:
        sim = similar(texts[a], texts[b], stats)
        if sim.score < policy.min_score:
            continue
        graph.union(a, b)
        joined_terms[(a, b)] = sim.shared

    components: dict[str, list[str]] = defaultdict(list)
    for cid in texts:
        components[graph.find(cid)].append(cid)

    kept = sorted((ids for ids in components.values() if len(ids) >= policy.min_topic_cases),
                  key=lambda ids: (-len(ids), ids[0]))[:policy.max_topics]
    # The self-guard: one group (or none) means the vocabulary did not separate
    # this cluster, and the structural answer stands. Never split into one.
    if len(kept) < 2:
        return {}, {}

    topic_of: dict[str, int] = {}
    terms_of: dict[int, tuple[str, ...]] = {}
    for idx, ids in enumerate(kept, 1):
        members = set(ids)
        for cid in ids:
            topic_of[cid] = idx
        terms_of[idx] = _terms_for(members, joined_terms, stats)
    return topic_of, terms_of


def _terms_for(members: set[str], joined_terms, stats: TokenStats,
               limit: int = 4) -> tuple[str, ...]:
    """The words that actually made this a group.

    Counted over the joins *inside* the component rather than over all the text
    in it, so the terms name the evidence for the boundary and not merely the
    component's most frequent vocabulary.
    """
    weight: dict[str, float] = defaultdict(float)
    for (a, b), shared in joined_terms.items():
        if a in members and b in members:
            for tok in shared:
                # `similar()` labels a prefix match as "blueprint's~blueprint".
                # That is the right amount of detail for a join's rationale and
                # the wrong amount for a kind's name; keep the stem.
                stem = min(tok.split("~"), key=len)
                weight[stem] += stats.weight(stem)
    ranked = sorted(weight.items(), key=lambda kv: (-kv[1], kv[0]))
    return tuple(tok for tok, _ in ranked[:limit])
