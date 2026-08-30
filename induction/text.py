"""Text similarity for the fuzzy correlation pass — stdlib only, and *legible*.

The fuzzy join has to survive the question "why did you connect these two?", so
every primitive here is one a reader can check by hand:

- **Content tokens.** Lowercase words, stopwords and the reply/forward noise
  that mail corpora are full of ("re", "fwd") removed.

- **Strong tokens.** Whole identifiers (``INV-2024-0312``), money amounts,
  dates and email addresses. Two records sharing ``4250.00`` is a real
  coincidence; two sharing "invoice" is not. Extracted with **span masking** so
  a compound identifier is never shredded into its parts — otherwise every
  invoice raised in the same year would "share an identifier" (``2024``) and
  the fuzzy pass would join the entire corpus to itself.

- **Rarity weighting (IDF).** Computed over the corpus actually being
  correlated, not a general-language table we would have to ship and defend.
  "Invoice" on every record is worth ~nothing; "acme" on four is worth a lot.
  Without this, fuzzy matching on business documents degenerates into matching
  boilerplate.

Everything returns the tokens it matched on, because the correlator writes them
into the join's rationale. A join a reader cannot audit is a join we do not make.
This is the transparent tier: it is `heuristic`, never `model`. Embeddings are
the next rung up and slot in behind the same `similar()` signature.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

# Deliberately small and general. Domain vocabulary belongs in a Profile, not
# here — this module must not learn what an "invoice" is.
_STOPWORDS = frozenset("""
a an the and or but if then else for of to in on at by with from as is are was
were be been being do does did done have has had having it its this that these
those i you he she they we not no yes so than too very can will just about into
over under again further once here there when where why how all any both each
few more most other some such only own same s t don now re fwd fw reply forward
""".split())

# Ordered strongest-first: an earlier pattern's span is masked out so a later
# one cannot re-extract a fragment of it.
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_DATEISH = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")
# An identifier: letters then digits, optionally hyphen/underscore separated and
# possibly multi-part ("INV-2024-0312", "PO_88231", "ACME-77").
_IDENT = re.compile(r"\b[a-z]{2,}[-_/]?\d{2,}(?:[-_/]\w+)*\b", re.I)
_MONEY = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|(?<![\w.])\d+\.\d{2}(?![\w.])")
_NUMBER = re.compile(r"(?<![\w.])\d{3,}(?![\w.])")
_WORD = re.compile(r"[a-z][a-z']+", re.I)

_STRONG_PATTERNS = (
    ("email", _EMAIL),
    ("date", _DATEISH),
    ("ident", _IDENT),
    ("money", _MONEY),
    ("number", _NUMBER),
)

# A bare year is a calendar fact, not an identifier — every document produced in
# 2024 would otherwise "share" one.
_YEARISH = re.compile(r"^(19|20)\d{2}$")

# Below this length a prefix match is meaningless ("in" prefixes "invoice").
_MIN_PREFIX = 4
# A prefix match ("corp" ~ "corporation") is real evidence but weaker than an
# exact one, so it is discounted rather than counted whole.
_PREFIX_DISCOUNT = 0.6
# How fast shared identifiers saturate the lift they give.
_STRONG_SATURATION = 1.5


def _normalise_amount(tok: str) -> str:
    """"4,250.00", "4250.00" and "4250" all become one token."""
    cleaned = tok.replace(",", "")
    try:
        val = float(cleaned)
    except ValueError:
        return cleaned.lower()
    return f"{val:.2f}".rstrip("0").rstrip(".")


def strong_tokens(text: str) -> set[str]:
    """Identifiers, money, dates, emails — the high-evidence half.

    Spans are masked as they are consumed, so ``INV-2024-0312`` yields exactly
    ``inv-2024-0312`` and never also ``2024`` or ``0312``.
    """
    if not text:
        return set()
    masked = list(text)
    out: set[str] = set()
    for kind, pat in _STRONG_PATTERNS:
        for m in pat.finditer("".join(masked)):
            raw = m.group(0)
            tok = _normalise_amount(raw) if kind in ("money", "number") else raw.lower().strip()
            if kind == "number" and _YEARISH.match(tok):
                continue          # a year is not an identifier
            if tok:
                out.add(tok)
            for i in range(m.start(), m.end()):
                masked[i] = " "   # consumed — later patterns cannot see it
    return out


def content_tokens(text: str) -> set[str]:
    """Lowercased words, stopwords removed. The low-evidence half."""
    if not text:
        return set()
    return {w.lower() for w in _WORD.findall(text)
            if len(w) > 1 and w.lower() not in _STOPWORDS}


def tokenize(text: str) -> tuple[set[str], set[str]]:
    """``(content, strong)`` for one blob of text."""
    return content_tokens(text), strong_tokens(text)


@dataclass
class TokenStats:
    """Corpus-wide rarity, so common boilerplate stops dominating the score.

    Built from the very records being correlated — which is the honest scope. A
    weight is ``ln(N / df)``: a token on every record scores 0, a token on one
    record scores high. Clamped at >= 0 so nothing goes negative.
    """

    n_docs: int = 0
    df: Counter = field(default_factory=Counter)

    def add(self, text: str) -> None:
        self.n_docs += 1
        content, strong = tokenize(text)
        for tok in content | strong:
            self.df[tok] += 1

    def weight(self, token: str) -> float:
        if self.n_docs <= 1:
            return 1.0
        df = self.df.get(token, 0) or 1
        return max(0.0, math.log(self.n_docs / df))


@dataclass(frozen=True)
class Similarity:
    """The result of one comparison — the score *and* what produced it."""

    score: float
    shared_content: tuple[str, ...] = ()
    shared_strong: tuple[str, ...] = ()

    @property
    def shared(self) -> tuple[str, ...]:
        return self.shared_strong + self.shared_content

    def describe(self, limit: int = 4) -> str:
        """The phrase that goes into a join's rationale."""
        bits = []
        if self.shared_strong:
            bits.append("identifiers " + ", ".join(repr(t) for t in self.shared_strong[:limit]))
        if self.shared_content:
            bits.append("words " + ", ".join(repr(t) for t in self.shared_content[:limit]))
        return " and ".join(bits) if bits else "nothing"


def _prefix_pairs(only_a: set[str], only_b: set[str]) -> list[tuple[str, str]]:
    """Unmatched words where one is a prefix of the other ("corp" ~ "corporation").

    Generic morphology, not domain knowledge: abbreviation is how real corpora
    differ between a mail subject and a ledger memo. Longest match wins, and
    each token is used at most once so one long word cannot absorb several.
    """
    pairs = []
    used_b: set[str] = set()
    for a in sorted(only_a, key=len, reverse=True):
        best = None
        for b in only_b:
            if b in used_b or a == b:
                continue
            short, long_ = (a, b) if len(a) <= len(b) else (b, a)
            if len(short) >= _MIN_PREFIX and long_.startswith(short):
                if best is None or len(b) > len(best):
                    best = b
        if best is not None:
            used_b.add(best)
            pairs.append((a, best))
    return pairs


def similar(a: str, b: str, stats: TokenStats | None = None) -> Similarity:
    """Overlap of two texts, in [0, 1], with the matched tokens that produced it.

    Content words and identifiers are scored **separately and then combined**,
    which matters more than it sounds:

    - *Content* is a weighted Dice coefficient — shared word-mass over total
      word-mass. Unshared words genuinely count against the match, which is what
      stops two long unrelated documents scoring on incidental overlap.

    - *Identifiers* only ever **lift** the score (a saturating noisy-OR over
      their rarity). Pooling them into the Dice denominator would mean an
      invoice rich in reference numbers scores *worse* against the email that
      names it — penalising a record for carrying more evidence. Sharing an
      identifier is positive evidence; not sharing one is evidence of nothing.

    The two halves answer different questions — "are these about the same
    thing?" and "do they cite the same thing?" — and either can carry a join.
    """
    ac, as_ = tokenize(a)
    bc, bs = tokenize(b)
    if not (ac or as_) or not (bc or bs):
        return Similarity(0.0)

    def w(tok: str) -> float:
        return stats.weight(tok) if stats else 1.0

    # --- content: weighted Dice; unshared words penalise ---
    shared_c = ac & bc
    shared_mass = sum(w(t) for t in shared_c)
    matched_labels = list(shared_c)
    for pa, pb in _prefix_pairs(ac - shared_c, bc - shared_c):
        shared_mass += min(w(pa), w(pb)) * _PREFIX_DISCOUNT
        matched_labels.append(f"{pa}~{pb}")
    mass_a, mass_b = sum(w(t) for t in ac), sum(w(t) for t in bc)
    content_score = (2.0 * shared_mass / (mass_a + mass_b)) if (mass_a + mass_b) > 0 else 0.0

    # --- identifiers: saturating lift; unshared ones cost nothing ---
    shared_s = as_ & bs
    strong_mass = sum(w(t) for t in shared_s)
    strong_lift = 1.0 - math.exp(-strong_mass / _STRONG_SATURATION) if strong_mass else 0.0

    score = content_score + (1.0 - content_score) * strong_lift

    def by_rarity(toks):
        return tuple(sorted(toks, key=lambda t: -w(t.split("~")[0])))

    return Similarity(min(1.0, score), by_rarity(matched_labels), by_rarity(shared_s))
