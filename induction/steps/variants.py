"""Step 4 — Variants: the different *runs* of one kind of process.

We present variants as **real observed traces with real frequencies**, not just
a merged directly-follows graph. A DFG is convenient but it over-generalises: it
draws edges for transitions that co-occur across different runs and so implies
paths that never actually happened. So the trace list is primary and the DFG is
secondary, carried with that caveat.

Roles make the shape readable at a glance (brief §4):
  - `common`    : the single most frequent trace shape — the loud path.
  - `exception` : a repeated-but-not-common shape, or any shape containing a
                  revert (an undo is an exception by nature).
  - `one-off`   : a shape seen exactly once — "someone's own way", kept visible
                  but quiet.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from induction.process import Case, Variant

START = "__start__"
END = "__end__"


def induced_variants(case_ids: list[str], cases_by_id: dict[str, Case]) -> tuple[list[Variant], dict]:
    cases = [cases_by_id[c] for c in case_ids if c in cases_by_id]
    by_sig: dict[tuple, list[str]] = defaultdict(list)
    for c in cases:
        by_sig[c.trace_signature].append(c.id)

    freqs = {sig: len(ids) for sig, ids in by_sig.items()}
    top_sig = max(freqs, key=lambda s: (freqs[s], -len(s))) if freqs else None

    variants: list[Variant] = []
    for sig, ids in by_sig.items():
        freq = len(ids)
        has_revert = "reverted" in sig
        if has_revert or (freq > 1 and sig != top_sig):
            role = "exception"
        elif sig == top_sig and freq > 1:
            role = "common"
        elif freq == 1:
            role = "one-off"
        else:
            role = "common"  # single distinct shape that is also the whole kind
        variants.append(Variant(signature=sig, frequency=freq, case_ids=ids, role=role))

    variants.sort(key=lambda v: (-v.frequency, len(v.signature)))
    dfg = _dfg(cases)
    return variants, dfg


def _dfg(cases: list[Case]) -> dict:
    """Directly-follows graph over the observed traces. Kept as a secondary,
    explicitly-caveated view (see module docstring)."""
    node_count: Counter = Counter()
    edge_count: Counter = Counter()
    for c in cases:
        sig = c.trace_signature
        if not sig:
            continue
        prev = START
        for action in sig:
            node_count[action] += 1
            edge_count[(prev, action)] += 1
            prev = action
        edge_count[(prev, END)] += 1

    nodes = [{"action": a, "count": n} for a, n in node_count.most_common()]
    edges = [{"from": a, "to": b, "count": n}
             for (a, b), n in sorted(edge_count.items(), key=lambda kv: -kv[1])]
    return {
        "nodes": nodes,
        "edges": edges,
        "caveat": "A directly-follows graph over-generalises: an edge means 'B "
                  "followed A in some run', not that any single run took every "
                  "edge. The variant traces above are the ground truth.",
    }
