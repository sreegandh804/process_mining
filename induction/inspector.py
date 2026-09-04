"""The inspector — the surface a non-technical operations lead actually reads.

Design commitments (from the redesign brief):
  - Plain, calm, one screen. No engine jargon on the face of it.
  - EVERY run, not a sample. The filters are just views over the full list; each
    row opens to its own timeline, and every line resolves to its source record.
  - Adaptive to any source. Nothing here is git- or finance-shaped: it renders
    whatever kinds/runs/steps the model induced, with words from the profile or
    (optionally) an LLM naming pass — falling back to the raw activity verbs.
  - Inference stays legible as inference: a step we didn't see is dashed and
    labelled; findings read as "what's missing", never asserted as fact.

`build_view` projects the InducedModel into exactly what the page needs; the
page is a dumb renderer over that projection.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from induction.abstraction import Abstraction
from induction.steps.variants import shape
from induction.emit import disclaimers_for
from induction.pipeline import InducedModel

# Friendly nouns for common anchor types (LLM naming overrides these when on).
_ITEM_WORDS = {
    "invoice": ("invoice", "invoices"), "grant": ("grant", "grants"),
    "payment": ("payment", "payments"), "pr": ("pull request", "pull requests"),
    "issue": ("issue", "issues"), "commit": ("change", "changes"),
    "release": ("release", "releases"), "case": ("case", "cases"),
    "integration": ("integration", "integrations"), "ticket": ("ticket", "tickets"),
    "email_thread": ("email thread", "email threads"), "email": ("email", "emails"),
}


def build_view(m: InducedModel, names: dict | None = None, activities: dict | None = None) -> dict:
    names = names or {}
    activities = activities or {}
    shaped = m.shaped
    events_by_id = {e.id: e for e in shaped.events}
    obs_by_id = {o.id: o for o in shaped.observations}
    ents = {e.id: e for e in shaped.entities}
    pname = {e.id: e.attrs.get("name", e.id) for e in shaped.entities if e.type == "person"}
    base_step = {s.action: s.name for s in m.steps}

    def step_label(a: str) -> str:
        return (names.get("steps", {}) or {}).get(a) or base_step.get(a) or a.replace("_", " ").title()

    abstraction = Abstraction.of(activities)
    read_ran = bool(abstraction.by_case)

    def kind_label(k) -> str:
        # A kind the reading tier formed already wears the family name the model
        # read out of its own records. The namer must not overwrite that: it ran
        # against the pre-reading kinds, so its map is keyed on ids that no
        # longer mean the same thing, and its whole purpose (rescuing
        # "Send then forward email") is already served here.
        feats = getattr(k, "features", {})
        if feats.get("read_process"):
            return k.name
        if read_ran:
            # The leftover kind, once the reading has run: the runs it declined
            # to place. It must NOT be handed to the namer, which will look at
            # its subject terms and christen the reject pile something confident
            # and false — on a real mailbox it produced "Supplier & Sales
            # Inquiries" for 36 threads, 21 of which had every single record
            # declined. A bucket of "we could not read these" presented as a
            # business process is the exact dishonesty the rest of this page is
            # built to avoid.
            return "Not placed in a process"
        return (names.get("kinds", {}) or {}).get(k.id) or k.name

    gaps_by_case = defaultdict(list)
    for g in m.gaps:
        gaps_by_case[g.case_id].append(g)
    kind_of_case = {cid: k for k in m.kinds for cid in k.case_ids}

    # Collective noun. Labelling every run by the single most common anchor type
    # ("pull request") overfits the moment a corpus is heterogeneous — a company
    # runs many processes across many systems and products, not one kind of item.
    # Use the specific noun ONLY when one type genuinely dominates; otherwise a
    # neutral "run", and let the kinds section below carry the real diversity.
    type_counts = Counter(c.anchor.get("type", "run") for c in m.cases.values())
    dom_type, dom_n = (type_counts.most_common(1) or [("run", 0)])[0]
    total = sum(type_counts.values()) or 1
    if len(type_counts) <= 1 or dom_n / total >= 0.8:
        item, items = _ITEM_WORDS.get(dom_type, (dom_type, dom_type + "s"))
    else:
        item, items = "run", "runs"      # mixed corpus — claim no single product
    item = names.get("item") or item
    items = names.get("items") or items

    runs = []
    dev_counter = Counter()
    dev_meta = {}
    for case in m.cases.values():
        kind = kind_of_case.get(case.id)
        rv = _run_view(case, kind, m, events_by_id, obs_by_id, ents, pname, step_label,
                       gaps_by_case, activities)
        # which induced process this run is an instance of — the link between the
        # "processes we found" section and the flat list, and how a multi-product
        # corpus is navigated (one kind per distinct process/offering).
        rv["kind"] = kind_label(kind) if kind else "Ungrouped"
        rv["kind_id"] = kind.id if kind else "none"
        rv["kind_attn"] = bool(kind and kind.rejected)
        runs.append(rv)
        dev_counter[rv["dev_key"]] += 1
        dev_meta[rv["dev_key"]] = (rv["dev_label"], rv["dev_attn"])
    # order: usual first, then attention items, then the rest — but the table
    # itself is sortable/filterable, so just keep a stable, readable order.
    runs.sort(key=lambda r: (0 if r["dev_key"] == "usual" else 1, r["id"]))

    # Mark every step as read or not BEFORE the cards are built: `_process_view`
    # draws its path from the read steps only, so it has to know which those are.
    # (Computing this after the cards silently gave every card an unread count of
    # zero and put the raw verbs back in the paths.)
    vocabulary = _naming_provenance(m, abstraction)
    read_names = {row["activity"] for row in abstraction.vocabulary
                  if not row.get("unclassified")}
    for r in runs:
        # A run carries an unread record when one of its steps is still the
        # source's own verb — i.e. the classifier declined it. Surfacing that per
        # run is what makes the abstention count checkable instead of a total.
        for n in r["activities"]:
            n["unread_step"] = bool(read_names) and n["name"] not in read_names
        r["unread"] = any(n["unread_step"] for n in r["activities"])

    # The kind cards' flow and variants are built from the runs' ACTIVITY spines —
    # the same abstraction the detail shows — so "the processes we found" reads as
    # activities (Raised → Fixed → Shipped), not the raw artefact verbs.
    runs_by_kind = defaultdict(list)
    for rv in runs:
        runs_by_kind[rv["kind_id"]].append(rv)
    processes = [_process_view(k, m, kind_label, runs_by_kind.get(k.id, []), read_ran,
                               abstraction.steps_by_process)
                 for k in m.kinds]

    # Filter buttons carry a GENERIC label per group (rows keep their specific
    # one), so "Ended early" doesn't masquerade as one particular end point.
    group_label = {"usual": "Follows usual path", "missing": "Missing a step",
                   "ended": "Ended early", "unmatched": "Unmatched record",
                   "incomplete": "Incomplete", "automated": "Automated"}
    filters = [{"key": "all", "label": f"All {items}", "count": len(runs), "attn": False}]
    # By process kind — the primary way to navigate a multi-process / multi-product
    # corpus (only shown when there is more than one kind to choose between).
    kind_count = Counter(r["kind_id"] for r in runs)
    kmeta = {r["kind_id"]: (r["kind"], r["kind_attn"]) for r in runs}
    if len(kind_count) > 1:
        for kid, n in kind_count.most_common():
            label, attn = kmeta[kid]
            filters.append({"key": f"kind:{kid}", "label": label, "count": n, "attn": attn})
    # then by how a run deviates from its kind's common path
    for key, n in dev_counter.most_common():
        _, attn = dev_meta[key]
        filters.append({"key": key, "label": group_label.get(key, key), "count": n, "attn": attn})

    orphans = [{"id": o.entity_id.split(":")[-1], "reason": o.reason,
                "src": (o.evidence[0].locator if o.evidence else "")} for o in m.orphans]


    leftover_ids = {k.id for k in m.kinds if read_ran and not k.features.get("read_process")}
    for r in runs:
        # Why a run sits in the leftover — so the leftover pane can group by cause
        # instead of listing 37 rows that all look the same.
        if r["kind_id"] in leftover_ids:
            any_read = any(not n.get("unread_step") for n in r["activities"])
            r["reason"] = ("some records read, but too few to sit inside any process"
                           if any_read else "no record in it named a step — every one kept the source's own verb")
    n_projects = sum(1 for k in m.kinds if k.features.get("project"))
    n_unplaced = sum(1 for r in runs if r["kind_id"] in leftover_ids)

    return {
        "meta": {
            "title": "How the work runs" if items == "runs" else f"How your {items} run",
            "n_records": sum(1 for e in shaped.entities if e.type not in ("person", "orphan_row")),
            "n_runs": len(runs),
            "n_processes": len([k for k in m.kinds if k.id not in leftover_ids and not k.features.get("project")]),
            "n_projects": n_projects,
            "n_unplaced": n_unplaced,
            "read_ran": read_ran,
            "corpus": _corpus_line(m, items),
            "scope": disclaimers_for(m),
            "ai_named": bool(names.get("_ai")),
            "ai_steps": bool(abstraction),
            "item": item, "items": items,
        },
        "processes": processes,
        "runs": runs,
        "filters": filters,
        "vocabulary": vocabulary,
        "orphans": orphans,
    }


def _naming_provenance(m, abstraction) -> list[dict]:
    """Where every step's name came from — not only the ones a model read.

    The first version listed the record-reading tier and nothing else, so a run
    whose steps were all named by the VERB MAP showed no legend at all: the page
    said "Correspondence Sent" with no hint a model had invented the phrase.

    The second version read `m.steps`, and was wrong in the same way for the
    opposite reason — that catalogue is only re-derived when the reading tier
    fires, so a verb-map-only run had cards saying "Correspondence Sent" above a
    legend saying `sent`, "the source's own word". A legend that disagrees with
    the thing it explains is worse than none.

    So this derives from the DISPLAYED activity, exactly as `_activity()` does,
    and attributes each name to the tier that actually produced it.
    """
    types = {e.id: e.type for e in m.shaped.entities}
    groups: dict[str, dict] = {}
    for ev in m.shaped.events:
        artefact = types.get(ev.entity_id, "record")
        reading = abstraction.by_record.get(ev.id)
        mapped = abstraction.by_vocab.get(f"{artefact}/{ev.action}")
        if reading:
            name, tier, how = reading["activity"], "model", "read"
        elif mapped:
            name, tier, how = mapped, "model", "mapped"
        else:
            name, tier, how = ev.action, "direct", "source"
        row = groups.setdefault(name, {"activity": name, "n": 0, "tier": tier,
                                       "how_kind": how, "n_read": 0,
                                       "verbs": set(), "phrases": []})
        row["n"] += 1
        row["verbs"].add(ev.action)
        if reading:
            row["n_read"] += 1
            row["tier"], row["how_kind"] = "model", "read"
            if reading["span"] not in row["phrases"] and len(row["phrases"]) < 6:
                row["phrases"].append(reading["span"])

    rows = []
    for row in groups.values():
        verbs = ", ".join(sorted(row.pop("verbs")))
        kind = row.pop("how_kind")
        if kind == "read":
            row["how"] = f"read from {row['n_read']} of {row['n']} records' own words"
        elif kind == "mapped":
            row["how"] = f"grouped and named by the model from the verb {verbs!r}"
        else:
            row["how"] = "the source's own word for it"
        rows.append(row)
    rows.sort(key=lambda r: -r["n"])
    if abstraction.n_unclassified:
        rows.append({"activity": "Unclassified", "n": abstraction.n_unclassified,
                     "tier": "", "phrases": [], "unclassified": True, "n_read": 0,
                     "how": "the model would not commit — kept the source's own verb"})
    return rows


def _canon(kind) -> list[str]:
    if not kind or not kind.variants:
        return []
    sig = max(kind.variants, key=lambda v: (v.frequency, len(v.signature))).signature
    out = []
    for a in sig:
        if a not in out:
            out.append(a)
    return out


# Below this many runs, "every run took a different path" is not a finding — it
# is a sample of two. A judgement call in the same spirit as
# `abstraction._MIN_RUNS_TO_JUDGE`: state the absence of a usual way only where
# there were enough runs for one to have shown up.
_MIN_RUNS_FOR_NO_COMMON = 4


def _process_view(k, m, kind_label, kind_runs, read_ran: bool = False,
                  steps_by_process: dict | None = None) -> dict:
    # actors from the kind's events (richer than one-per-run)
    ev_by_id = {e.id: e for e in m.shaped.events}
    pname = {e.id: e.attrs.get("name", e.id) for e in m.shaped.entities if e.type == "person"}
    actors = Counter()
    for cid in k.case_ids:
        case = m.cases.get(cid)
        for eid in (case.event_ids if case else []):
            ev = ev_by_id.get(eid)
            if ev and ev.actor:
                actors[pname.get(ev.actor, ev.actor)] += 1

    # Flow and variants at the ACTIVITY level, taken from the runs' own inferred
    # spines — so the card shows the process (Raised → Fixed → Shipped), not the
    # raw artefact verbs the systems happened to record. A variant is a distinct
    # activity path; its frequency is how many runs took it.
    ident = lambda x: x
    leftover = read_ran and not k.features.get("read_process")

    # THE PATH IS THE STEPS, NOT THE RECORDS.
    #
    # A record the reading declined keeps its source's own verb — `Sent` — and
    # that is the honest thing to do with it. Rendering it INLINE IN THE PATH is
    # not: it puts a claim ("Requested": the model read this and quoted the line)
    # and the absence of a claim ("Sent": nobody knows what this was) in one
    # chain, as if they were the same kind of thing. On a real mailbox that
    # produced `Requested -> Sent -> Reviewed -> Sent -> Sent`, which reads as a
    # process with a step called Sent and is not what anyone meant.
    #
    # A process is `Screened -> Interviewed -> Offered`. So the path shows the
    # steps that were actually read, and the records that were not are counted
    # beside it rather than drawn into it. Nothing is hidden — every message is
    # still in the run detail, still in `model.json`, and the count is on the
    # card — but the shape of the process is no longer diluted by the records
    # that could not contribute to it.
    def read_seq(rv):
        return tuple(n["name"] for n in rv["activities"] if not n.get("unread_step"))

    n_unread = sum(1 for rv in kind_runs for n in rv["activities"]
                   if n.get("unread_step") for _ in n["arts"])
    n_records = sum(len(n["arts"]) for rv in kind_runs for n in rv["activities"])
    runs_with_steps = [rv for rv in kind_runs if read_seq(rv)]

    # Variants at the SHAPE level — distinct steps in first-occurrence order —
    # so a loop-back or a re-sent reply is not a new "different order" row. The
    # exact traces are kept per shape and shown as detail. See variants.shape.
    exact: dict[tuple, Counter] = defaultdict(Counter)
    for rv in runs_with_steps:
        seq = read_seq(rv)
        exact[shape(seq)][seq] += 1
    sigs = Counter({sh: sum(c.values()) for sh, c in exact.items()})
    max_freq = max(sigs.values(), default=1)

    # A common path only exists if some path is actually COMMON. Where every run
    # took a different route, `max()` broke the all-tied-at-one tie BY LENGTH, so
    # the card presented the single longest, most chaotic run as the canonical
    # process — a 21-step chain labelled "most common" above six runs that shared
    # nothing. That is the worst available answer to "how does this usually go",
    # and it is a claim the data does not support.
    #
    # Two corrections, because "nothing repeated" means different things at
    # different sizes. With enough runs it is a real finding — say there is no
    # usual way and let the list below carry the truth. With two runs it is
    # unremarkable, and refusing to summarise would be over-correcting; show a
    # representative path, but break the tie by the SHORTEST rather than the
    # longest, since length-bias is what produced the 21-step monster.
    all_unique = max_freq == 1 and len(sigs) > 1
    no_common = all_unique and len(runs_with_steps) >= _MIN_RUNS_FOR_NO_COMMON
    if not sigs:
        canon = []
    elif no_common:
        canon = []
    elif all_unique:
        canon = list(min(sigs, key=lambda s: (len(s), s)))
    else:
        canon = list(max(sigs, key=lambda s: (sigs[s], len(s))))

    # THE CARD'S HEADLINE IS THE PROCESS, NOT ONE RUN OF IT.
    #
    # `canon` above is the most frequent TRACE, and that is the right thing to
    # label "most common" in the list of variants below. It is the wrong thing to
    # put at the top of the card as the process's shape, and on a corpus with any
    # abstention it collapses to a single chip: "Research Collaboration:
    # Approved" — which means three of its runs had exactly one readable record
    # and it happened to be an approval. Seven runs that between them show
    # Requested, Approved and Escalated were summarised as one word.
    #
    # A process is its STEPS. Hiring is `Screened -> Interviewed -> Offered`; it
    # is not whichever single stage the most candidates happen to share. So the
    # headline is every step this kind's runs perform, ordered by where those
    # runs typically put it — the same ordering `gaps_generic._canonical_order`
    # uses, minus its majority bar, because describing a process is a weaker
    # claim than accusing a run of skipping part of one.
    #
    # The variant list underneath still carries what actually happened, run by
    # run, so the summary can never be mistaken for a claim that every run took
    # all of these steps.
    # Ordering comes only from runs that HAVE an order. A one-step run says
    # nothing about where its step belongs — but scored naively it says
    # "position 0.0", so any step that often appears alone gets dragged to the
    # front: seven Research Collaboration runs came out `Approved -> Requested
    # -> Escalated`, approved before requested, purely because three runs had a
    # lone approval in them. Such runs still contribute the STEP; they just do
    # not get a vote on where it goes.
    seen_steps: dict[str, int] = defaultdict(int)
    where: dict[str, list[float]] = defaultdict(list)
    for rv in runs_with_steps:
        seq = read_seq(rv)
        for step in seq:
            seen_steps[step] += 1
        if len(seq) < 2:
            continue
        span = len(seq) - 1
        for i, step in enumerate(seq):
            where[step].append(i / span)

    def _at(step):
        # A step only ever seen alone sorts last rather than first: we know it
        # happens, and we do not know when.
        return sum(where[step]) / len(where[step]) if where.get(step) else 1.1

    flow = sorted(seen_steps, key=lambda st: (_at(st), -seen_steps[st], st))

    # WHERE A PROCESS DEFINITION EXISTS, IT DEFINES THE HEADLINE.
    #
    # Two things go wrong when the headline is derived from the runs instead:
    #
    #  - the ORDER is noise. Discovery already returned each process's steps in
    #    the order they happen; re-deriving that from three runs with 60%
    #    abstention produced `Deal booked -> Invoice shortfall pursued ->
    #    Payment received -> Manual invoice created`, which is the right steps
    #    in nearly the wrong order. The definition's order is a claim the model
    #    actually made; the observed order here is a small, gappy sample of it.
    #
    #  - the MEMBERSHIP leaks. A run is placed in a kind by majority vote over
    #    its records, so a run can sit in one process while carrying records read
    #    into another's steps. The card then advertises steps belonging to a
    #    different process — "Academic Visit" was listing "Research idea scoped
    #    with faculty", which is University Research Sponsorship's step.
    #
    # So: the definition's steps, in the definition's order, filtered to those
    # some run here actually performed. Steps a run performed that belong to
    # another process stay in the variant list below, where they are evidence of
    # what happened rather than a claim about this process.
    defined = (steps_by_process or {}).get(k.features.get("read_process") or "")
    if defined:
        flow = [st for st in defined if st in seen_steps]

    # The leftover kind is not a process and must not be drawn as one. Its runs
    # are the ones the reading declined to place; a flow across them would be a
    # shape assembled from whatever happened to be readable in a reject pile.
    if leftover:
        flow = []
    paths = [{
        "count": freq,
        "seq": list(sig),
        "label": "" if no_common else _path_label(sig, canon, ident),
        "rare": freq == 1 and len(sigs) > 1,
        "width": max(6, round(freq / max_freq * 180)),
        # how many distinct exact traces sit under this shape (loops, re-sends)
        "n_traces": len(exact[sig]),
    } for sig, freq in sigs.most_common()]

    # The leftover kind is the runs the reading could not place. It has a count
    # and a reason; it does not have paths. Three variant rows assembled from
    # nine readable records out of 105 is a shape made of noise.
    if leftover:
        paths, no_common = [], False

    return {
        "id": k.id,
        "name": kind_label(k),
        "count": len(k.case_ids),
        "actors": [a for a, _ in actors.most_common(6)],
        "flow": flow,
        "canon": canon,
        "paths": paths,
        # Said out loud rather than left for a reader to infer from N paths at 1x.
        "no_common": no_common,
        "n_paths": len(sigs),
        # What the path does NOT account for, so the omission is visible.
        "n_unread": n_unread,
        "n_records": n_records,
        "n_runs_unread": len(kind_runs) - len(runs_with_steps),
        "flagged": k.rejected,
        "flag_note": k.reject_reason or "",
        # Where this process boundary came from. Nothing in the data announces
        # one, so the card has to say who drew it and on what — the same
        # discipline every other claim on the page is held to.
        "tier": k.confidence.tier.label,
        "why": _boundary_why(k, kind_runs, read_ran),
        "leftover": leftover,
        # A one-off with a real arc: shown, and labelled for what it is.
        "project": bool(k.features.get("project")),
    }


def _boundary_why(k, kind_runs, read_ran: bool) -> str:
    """What to say about where this kind's boundary came from.

    For a read kind, the confidence rationale already says it. For the LEFTOVER
    kind once the reading has run, the rationale ("structural clustering, not
    read") is true but tells a reader nothing about why these runs are here —
    they are here because the reading DECLINED them, and how many were declined
    outright is the number that says whether to care about this card at all.
    """
    if not (read_ran and not k.features.get("read_process")):
        return k.confidence.rationale or ""
    fully = sum(1 for rv in kind_runs
                if rv.get("activities") and all(n.get("unread_step") for n in rv["activities"]))
    n = len(kind_runs)
    return (f"Not a process the records evidence — these are the {n} runs the reading "
            f"could not place in one"
            + (f", {fully} of which had every single record declined. " if fully else ". ")
            + "They keep the structural grouping and their source's own verbs.")


def _path_label(sig, canon, step_label) -> str:
    if not sig:
        return "no dated steps"
    if list(sig) == canon:
        return "most common"
    cset, sset = list(canon), set(sig)
    missing = [a for a in canon if a not in sset]
    extra = [a for a in sig if a not in set(canon)]
    if list(sig) == canon[:len(sig)] and len(sig) < len(canon):
        return f"ended at {step_label(sig[-1])}"
    if missing and not extra and len(sig) == len(canon) - len(missing):
        return "no " + ", ".join(step_label(a) for a in missing[:2]) + " step"
    if len(sig) < len(canon):
        return "fewer steps"
    return "different order"


# How sure we are a run holds together, in words a non-technical reader can act
# on. Only the two *inferred* tiers get a chip — a deterministic run is the norm
# and needs no badge; the point is to make a guess look like a guess.
_TIER_CHIP = {
    "heuristic": ("matched on wording", "t-soft"),
    "model": ("AI judged same work", "t-ai"),
}
_SRC_WORD = {"github": "GitHub", "mail": "email", "git": "git history",
             "changelog": "changelog", "tabular": "spreadsheet", "finance": "spreadsheet"}


def _run_sources(case, ents) -> list[str]:
    """The distinct systems a run spans — the honest headline of a cross-source
    case ('this one activity lived in email AND GitHub')."""
    words: list[str] = []
    for eid in case.entity_ids:
        e = ents.get(eid)
        if not e:
            continue
        w = _SRC_WORD.get(e.source.split(":")[0], e.source.split(":")[0])
        if w not in words:
            words.append(w)
    return words


def _run_title(case, ents, disp: str) -> str:
    """A human name for the run — the anchor's title/subject, so an ops lead sees
    'SSO login fails after token refresh', not 'case:pr:15'."""
    anchor_id = case.id[5:] if case.id.startswith("case:") else case.id
    for eid in [anchor_id, *case.entity_ids]:
        e = ents.get(eid)
        if e:
            t = e.attrs.get("title") or e.attrs.get("subject")
            if t:
                return t.strip()
    return disp


def _run_view(case, kind, m, events_by_id, obs_by_id, ents, pname, step_label,
              gaps_by_case, activities=None) -> dict:
    canon = _canon(kind)
    gaps = gaps_by_case.get(case.id, [])
    abstraction = Abstraction.of(activities)

    def _activity(ev) -> str:
        """The activity a record realises — the reading of the record itself when
        there is one, else the verb map, else the verb's own label. The reading is
        what turns 400 emails that all say 'sent' into Requested / Reviewed /
        Approved; the map is what turns 'issue opened' + 'email sent' into one
        'Raised'. With neither we show the artefact's own verb, claiming no
        abstraction we did not earn."""
        artefact = ents[ev.entity_id].type if ev.entity_id in ents else "record"
        return abstraction.activity_of(ev.id, artefact, ev.action) or step_label(ev.action)

    # action -> activity name where a verb maps to one activity across artefact
    # types, so a MISSING step (which carries only a verb) still reads as the
    # process activity, not the raw verb.
    act2activity: dict = {}
    for key, name in abstraction.by_vocab.items():
        a = key.split("/", 1)[-1]
        act2activity[a] = None if (a in act2activity and act2activity[a] != name) else name

    def _act_name(action: str) -> str:
        return act2activity.get(action) or action.replace("_", " ")

    # artefacts, in the run's real order — the evidence that will sit under the
    # activities. Each keeps its own verb, time, owner and source locator.
    arts = []
    actors = Counter()
    for rid in case.ordered_event_ids:
        if rid in events_by_id:
            ev = events_by_id[rid]
            who = pname.get(ev.actor) if ev.actor else None
            if who:
                actors[who] += 1
            ent = ents.get(ev.entity_id)
            art_type = ent.type if ent else "record"
            num = ent.attrs.get("number") if ent else None
            src = ev.evidence[0].locator if ev.evidence else ""
            arts.append({
                "activity": _activity(ev),
                "artefact": _ITEM_WORDS.get(art_type, (art_type, art_type))[0],
                "ref": f"#{num}" if num else "",
                # the artefact's own action, described plainly — this is evidence
                # detail ("review requested"), not the inferred step name
                "verb": ev.action.replace("_", " "),
                "when": (ev.timestamp or "").split("T")[0] or "—",
                # A read activity shows the span it was read from: the difference
                # between a claim you can check and one you have to accept.
                "who": who, "inferred": False, "note": abstraction.span_of(ev.id) or "",
                "src": src, "is_url": src.startswith("http"),
                "src_kind": _SRC_WORD.get(ev.source.split(":")[0], ev.source.split(":")[0]),
            })
        elif rid in obs_by_id:
            ob = obs_by_id[rid]
            src = ob.evidence[0].locator if ob.evidence else ""
            arts.append({
                "activity": "State recorded", "artefact": "record", "ref": "", "verb": "recorded",
                "when": (ob.seen_at or "").split("T")[0] or "no date", "who": None,
                "inferred": False, "note": str(ob.state.get("status", "") or ob.state.get("text", ""))[:80],
                "src": src, "is_url": src.startswith("http"),
                "src_kind": _SRC_WORD.get(ob.source.split(":")[0], ob.source.split(":")[0]),
            })

    # Fold consecutive artefacts that realise the SAME activity into one step; the
    # artefacts become that step's evidence. THIS is the process — a spine of
    # activities — with the systems' records hanging beneath, not the raw verbs.
    nodes: list[dict] = []
    for it in arts:
        if nodes and nodes[-1]["name"] == it["activity"]:
            nodes[-1]["arts"].append(it)
        else:
            nodes.append({"name": it["activity"], "arts": [it]})
    for n in nodes:
        whens = [a["when"] for a in n["arts"] if a["when"] not in ("—", "no date")]
        n["when"] = min(whens) if whens else "—"
        n["sources"] = sorted({a["src_kind"] for a in n["arts"] if a["src_kind"]})
        n["n"] = len(n["arts"])

    # inferred (missing / off-system) steps — the confidently-incomplete part
    inferred = []
    for g in gaps:
        if g.kind == "missing_expected_step":
            act = g.id.split(":")[-1]
            inferred.append({"name": _act_name(act), "when": "not in the records",
                             "note": "expected here, no record",
                             "src": (g.evidence[0].locator if g.evidence else "")})
        elif g.kind == "reconciliation":
            inferred.append({"name": "No matching record", "when": "—",
                             "note": g.description[:90],
                             "src": (g.evidence[0].locator if g.evidence else "")})

    dev_key, dev_label, dev_attn = _deviation(case, kind, canon, gaps, step_label)

    a = case.anchor
    disp = (f"#{a['number']}" if "number" in a else a.get("key") or a.get("branch")
            or case.id.replace("case:", ""))
    status = ""
    for eid in case.entity_ids:
        e = ents.get(eid)
        if e and e.attrs.get("status"):
            status = e.attrs["status"]
            break
    if not status:
        status = (nodes[-1]["name"] if nodes else "—")

    tier = case.confidence.tier.label
    chip_word, chip_class = _TIER_CHIP.get(tier, ("", ""))
    sources = _run_sources(case, ents)

    return {
        "id": disp,
        "title": _run_title(case, ents, disp),
        "actor": (actors.most_common(1)[0][0] if actors else "—"),
        "status": status,
        # the spine is the activities, not the raw verbs
        "path": " → ".join(n["name"] for n in nodes) or "—",
        "dev_key": dev_key, "dev_label": dev_label, "dev_attn": dev_attn,
        # Honesty on the face of the row: an inferred join wears a chip and, when
        # it is the model tier, the model's own reason; a deterministic run wears
        # nothing. Cross-source runs say which systems they crossed.
        "tier": tier, "chip_word": chip_word, "chip_class": chip_class,
        "why": (case.confidence.rationale or "") if tier in ("heuristic", "model") else "",
        "sources": sources, "cross": len(sources) > 1,
        "activities": nodes, "inferred": inferred,
    }


def _deviation(case, kind, canon, gaps, step_label):
    if kind and kind.rejected:
        return "automated", "Automated", False
    present = [a for a in case.trace_signature]
    if case.order_status == "unknown" or len(present) <= 1:
        return "incomplete", "Incomplete", False
    if any(g.kind == "missing_expected_step" for g in gaps):
        missing = next((a for a in canon if a not in set(present)), None)
        return "missing", (f"No {step_label(missing)} step" if missing else "Missing a step"), True
    if any(g.kind == "reconciliation" for g in gaps):
        return "unmatched", "Unmatched record", True
    if canon and canon[-1] not in set(present):
        return "ended", f"Ended at {step_label(present[-1])}", False
    return "usual", "—", False


def _corpus_sources(m) -> list[str]:
    """The distinct systems the corpus was read from, in friendly words."""
    return sorted({_SRC_WORD.get(e.source.split(":")[0], e.source.split(":")[0])
                   for e in m.shaped.entities
                   if e.type != "person" and getattr(e, "source", "")})


def _join_words(ws: list[str]) -> str:
    if len(ws) <= 1:
        return ws[0] if ws else "your systems"
    return ", ".join(ws[:-1]) + " and " + ws[-1]


def _corpus_line(m, items) -> str:
    mf = m.manifest or {}
    srcs = _corpus_sources(m)
    if mf.get("source_kind") == "email":
        return (f"Read from <b>{mf.get('n_messages', '?')} emails</b> in {m.slug}. Nothing was "
                f"entered by hand; the threads and who-did-what were worked out from the "
                f"messages, and every line opens to the message it came from.")
    if mf.get("head"):
        return (f"Read from <b>{m.slug}</b> — {mf.get('n_commits', '?')} records of activity. "
                f"Nothing was entered by hand; it was worked out from your own history, and "
                f"every line opens to the record it came from.")
    if mf.get("source_kind") == "combined" or len(srcs) > 1:
        n = mf.get("n_records") or sum(1 for e in m.shaped.entities if e.type != "person")
        return (f"Read across <b>{_join_words(srcs)}</b> — {n} records, nothing entered by "
                f"hand. The runs, the steps and who did what were worked out from the "
                f"artefacts themselves; different systems, one process. Every step opens to "
                f"the record it came from.")
    n = mf.get("n_rows", len(m.cases))
    sheets = len(mf.get("sheets", []) or [])
    where = f"{sheets} spreadsheet{'' if sheets == 1 else 's'}" if sheets else "your records"
    return (f"Read from <b>{where}</b> — {n} {items}. Nothing was entered by hand; it was "
            f"worked out from your own records, and every line opens to the row it came from.")


def write_html(m: InducedModel, path: str | Path, names: dict | None = None,
               activities: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    view = build_view(m, names, activities)
    path.write_text(_TEMPLATE.replace("/*DATA*/", json.dumps(view, default=str)))
    return path


# ---------------------------------------------------------------------------
# Self-contained page. Light, calm, vanilla JS. Renders entirely from VIEW.
# ---------------------------------------------------------------------------
_TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>How the work runs</title>
<style>
  :root{--ink:#16181d;--ink-2:#575d6b;--ink-3:#8b909d;--paper:#fff;--canvas:#f5f6f8;--rule:#e3e5ea;--rule-2:#d3d7de;
    --read:#2b5c8a;--read-bg:#eaf1f7;--read-ink:#1d4467;--open:#8a6b1f;--open-bg:#f7f1e2;--sel:#eceff3;--attn:#b4601a;
    --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif}
  *{box-sizing:border-box}
  body{margin:0;background:var(--canvas);color:var(--ink);font:15px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
  a{color:var(--read);text-decoration:none}a:hover{text-decoration:underline}
  :focus-visible{outline:2px solid var(--read);outline-offset:2px}
  .wrap{max-width:1240px;margin:0 auto;padding:0 24px}
  header.top{background:var(--paper);border-bottom:1px solid var(--rule)}
  header.top .wrap{padding-top:26px}
  h1{font:400 30px/1.2 var(--serif);letter-spacing:-.01em;margin:0 0 8px}
  .lede{max-width:66ch;color:var(--ink-2);margin:0 0 16px}
  .ai{font-size:12px;color:var(--read);background:var(--read-bg);border-radius:4px;padding:1px 7px;margin-left:6px}
  .stats{display:flex;flex-wrap:wrap;gap:0 18px;align-items:baseline;font-size:13.5px;color:var(--ink-2);border-top:1px solid var(--rule);padding:11px 0 13px}
  .stats b{font-weight:500;color:var(--ink);font-variant-numeric:tabular-nums}
  .stats .disclose{margin-left:auto;color:var(--open);background:none;border:1px solid var(--open);border-radius:6px;padding:2px 9px;font:inherit;cursor:pointer}
  .shell{display:grid;grid-template-columns:238px minmax(0,1fr);gap:24px;padding:22px 0 60px}
  nav.rail{position:sticky;top:16px;align-self:start;font-size:14px}
  .railcap{font-size:12.5px;color:var(--ink-3);padding:0 10px 7px}
  .railcap.mt{padding-top:14px;margin-top:12px;border-top:1px solid var(--rule)}
  .rowbtn{display:flex;width:100%;gap:8px;align-items:baseline;background:none;border:0;font:inherit;color:var(--ink);text-align:left;padding:6px 10px;border-radius:6px;cursor:pointer}
  .rowbtn:hover{background:#eef0f3}.rowbtn[aria-current="true"]{background:var(--sel);font-weight:500}
  .rowbtn .n{margin-left:auto;color:var(--ink-3);font-variant-numeric:tabular-nums;font-weight:400}
  .rowbtn.quiet{color:var(--ink-2)}.rowbtn.quiet .n{color:var(--open)}
  main{min-width:0}
  .card{background:var(--paper);border:1px solid var(--rule);border-radius:10px;padding:22px 24px}
  h2{font:400 22px/1.25 var(--serif);margin:0 0 4px}
  .ptag{display:inline-block;font:500 11.5px/1 var(--sans);color:var(--attn);background:var(--open-bg);border-radius:999px;padding:4px 9px;margin-left:8px;vertical-align:middle}
  .sub{font-size:13.5px;color:var(--ink-2);margin:0 0 2px;font-variant-numeric:tabular-nums}
  .who{font-size:13px;color:var(--ink-3);margin:6px 0 0;font-family:var(--mono);word-break:break-word}
  .why{font-size:13px;color:var(--ink-2);margin:10px 0 0;padding-left:10px;border-left:2px solid var(--rule-2)}
  .band{border-top:1px solid var(--rule);margin-top:20px;padding-top:16px}
  .bandhead{display:flex;align-items:baseline;gap:10px;margin-bottom:10px;flex-wrap:wrap}
  .bandhead h3{font:500 14px/1.3 var(--sans);margin:0}.bandhead .note{font-size:13px;color:var(--ink-3)}
  .flow{display:flex;flex-wrap:wrap;gap:6px;align-items:center}
  .chip{font:inherit;font-size:13px;background:var(--read-bg);color:var(--read-ink);border:1px solid transparent;border-radius:6px;padding:5px 9px;cursor:pointer}
  .chip .u{border-bottom:1px dotted currentColor}.chip:hover{border-color:var(--read)}
  .chip[aria-expanded="true"]{background:var(--read);color:#fff}
  .arw{color:var(--ink-3);font-size:12px}
  .legend{font-size:12.5px;color:var(--ink-3);margin-top:8px}
  .prov{margin-top:12px;border:1px solid var(--rule-2);border-radius:8px;background:#fbfcfd;padding:14px 16px}
  .prov h4{font:500 14px/1.3 var(--sans);margin:0 0 2px}.prov .how{font-size:13px;color:var(--ink-2);margin:0 0 10px}
  .quote{font-family:var(--mono);font-size:12.5px;line-height:1.5;background:var(--paper);border:1px solid var(--rule);border-radius:5px;padding:7px 9px;margin:0 0 6px}
  .provfoot{font-size:12.5px;color:var(--ink-3);margin-top:2px}
  .filters{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}
  .fchip{font:inherit;font-size:12.5px;background:var(--paper);border:1px solid var(--rule-2);border-radius:999px;padding:4px 11px;color:var(--ink-2);cursor:pointer}
  .fchip:hover{border-color:var(--ink-3)}.fchip[aria-pressed="true"]{background:var(--ink);border-color:var(--ink);color:#fff}
  table{width:100%;border-collapse:collapse;font-size:13.5px}
  th{font-weight:400;font-size:12.5px;color:var(--ink-3);text-align:left;padding:0 10px 6px 0;border-bottom:1px solid var(--rule)}
  td{padding:9px 10px 9px 0;border-bottom:1px solid var(--rule);vertical-align:top}
  tr:last-child td{border-bottom:0}
  .runs td:first-child{font-variant-numeric:tabular-nums;color:var(--ink-2);width:44px;white-space:nowrap}
  .path{color:var(--ink)}
  td.path1{max-width:420px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .tag{font-size:12px;color:var(--ink-3);white-space:nowrap}.tag.common{color:var(--read)}
  .search{width:100%;font:inherit;font-size:14px;padding:8px 11px;border:1px solid var(--rule-2);border-radius:8px;margin:0 0 10px}
  .thread{width:100%;text-align:left;background:none;border:0;font:inherit;cursor:pointer;padding:0;color:inherit}
  .thread .t{font-weight:500}
  .thread .id{display:block;font-family:var(--mono);font-size:11.5px;color:var(--ink-3);word-break:break-all;margin-top:2px}
  .tchip{font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--open);background:var(--open-bg);border-radius:3px;padding:1px 6px;margin-left:6px}
  tr.open td{background:#fafbfc}
  .detail{padding:4px 0 14px}
  .framing{font-size:12.5px;color:var(--ink-2);margin:0 0 10px}
  .step{border-left:2px solid var(--rule-2);padding:0 0 12px 12px;margin-left:2px}
  .step:last-child{padding-bottom:0}.step.unread{border-left-style:dotted;opacity:.8}
  .step .sn{font-weight:500;font-size:13.5px}.step .sm{font-size:12.5px;color:var(--ink-3);font-variant-numeric:tabular-nums}
  .ev{display:grid;grid-template-columns:minmax(0,1fr) 92px minmax(0,1.3fr);gap:8px;font-size:12.5px;padding:3px 0}
  .ev .evverb{color:var(--ink-2)}.ev .src{font-family:var(--mono);font-size:11.5px;color:var(--ink-3);word-break:break-all}
  .ev.inf{color:var(--ink-3);font-style:italic}
  .more{font-size:13px;color:var(--ink-3);padding-top:12px}
  .group{border:1px solid var(--rule);border-radius:8px;margin-bottom:8px;overflow:hidden}
  .group summary{cursor:pointer;padding:11px 14px;font-size:14px;display:flex;gap:10px;align-items:baseline}
  .group summary::-webkit-details-marker{display:none}
  .group summary::before{content:"›";color:var(--ink-3);display:inline-block;transition:transform .12s}
  .group[open] summary::before{transform:rotate(90deg)}
  .group summary .n{margin-left:auto;color:var(--ink-3);font-variant-numeric:tabular-nums}
  .group .gbody{padding:0 14px 12px 26px;font-size:13.5px;color:var(--ink-2)}
  .leftover h2{color:var(--open)}
  .said{font-size:14px;color:var(--ink-2);max-width:66ch;margin:8px 0 18px}
  .gl td:nth-child(2){width:70px;font-variant-numeric:tabular-nums;color:var(--ink-2)}
  .gl .unc{color:var(--open)}
  .gl .phs{margin-top:4px}.gl .ph{display:inline-block;font-family:var(--mono);font-size:11.5px;color:var(--ink-2);background:var(--canvas);border-radius:4px;padding:1px 6px;margin:2px 4px 0 0}
  .scope{font-size:12.5px;color:var(--ink-3);margin:24px 0 0;padding-left:18px}
  @media (max-width:880px){.shell{grid-template-columns:1fr;gap:14px}nav.rail{position:static;display:flex;gap:6px;overflow-x:auto;padding-bottom:4px}
    .railcap{display:none}.rowbtn{width:auto;white-space:nowrap;border:1px solid var(--rule-2);background:var(--paper)}.card{padding:18px 16px}.stats .disclose{margin-left:0}}
</style></head>
<body>
<header class="top"><div class="wrap">
  <h1 id="title"></h1>
  <p class="lede" id="source"></p>
  <div class="stats" id="stats"></div>
</div></header>
<div class="wrap shell">
  <nav class="rail" id="rail" aria-label="Processes"></nav>
  <main id="pane" tabindex="-1"></main>
</div>
<script id="data" type="application/json">/*DATA*/</script>
<script>
const V = JSON.parse(document.getElementById('data').textContent);
const esc = s => (s==null?'':String(s)).replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]));
const M = V.meta, items = M.items, item = M.item;
const STEPS = Object.fromEntries((V.vocabulary||[]).map(v=>[v.activity, v]));
const procs = V.processes.filter(p=>!p.leftover && !p.project);
const projects = V.processes.filter(p=>p.project);
const leftover = V.processes.find(p=>p.leftover);
const runsOf = id => V.runs.filter(r=>r.kind_id===id);

document.getElementById('title').textContent = M.title;
document.getElementById('source').innerHTML = M.corpus + (M.ai_named?' <span class="ai">names suggested by AI</span>':'');
document.getElementById('stats').innerHTML =
  `<span><b>${M.n_records}</b> records</span><span><b>${M.n_runs}</b> ${esc(items)}</span>` +
  `<span><b>${M.n_processes}</b> process${M.n_processes===1?'':'es'}</span>` +
  (M.n_projects?`<span><b>${M.n_projects}</b> project${M.n_projects===1?'':'s'}</span>`:'') +
  (leftover?`<button class="disclose" data-go="leftover"><b>${M.n_unplaced}</b> ${esc(items)} not placed ↓</button>`:'');

let current = (location.hash||'').slice(1) || (procs[0]||projects[0]||{id:'glossary'}).id;
const rail = document.getElementById('rail'), pane = document.getElementById('pane');

function chip(name){
  const s = STEPS[name]; const read = s && s.phrases && s.phrases.length;
  return `<button class="chip" data-step="${esc(name)}" aria-expanded="false"><span class="${read?'u':''}">${esc(name)}</span></button>`;
}
const flow = steps => steps.map(chip).join('<span class="arw">→</span>');
const pathText = p => p.length ? p.map(esc).join(' <span class="arw">→</span> ') : '—';

function renderRail(){
  const row = (p,cls='') => `<button class="rowbtn ${cls}" data-go="${esc(p.id)}" aria-current="${current===p.id}">${esc(p.name)}<span class="n">${p.count}</span></button>`;
  let h = `<div class="railcap">Processes</div>` + procs.map(p=>row(p)).join('');
  if(projects.length) h += `<div class="railcap mt">Projects · happened once</div>` + projects.map(p=>row(p)).join('');
  h += `<div class="railcap mt">Not process</div>`;
  if(leftover) h += `<button class="rowbtn quiet" data-go="leftover" aria-current="${current==='leftover'}">Not placed<span class="n">${leftover.count}</span></button>`;
  h += `<button class="rowbtn quiet" data-go="glossary" aria-current="${current==='glossary'}">Step glossary<span class="n">${(V.vocabulary||[]).length}</span></button>`;
  rail.innerHTML = h;
}

// One artefact = one piece of evidence, opening to its source record.
const art = a => `<div class="ev ${a.inferred?'inf':''}">
  <span><span class="evref">${esc(a.artefact)}${a.ref?' '+esc(a.ref):''}</span> <span class="evverb">${esc(a.verb)}${a.who?' · '+esc(a.who):''}</span>${a.note?`<div class="quote" style="margin:4px 0 0">${esc(a.note)}</div>`:''}</span>
  <span class="sm">${esc(a.when)}</span>
  <span class="src">${a.is_url?`<a href="${esc(a.src)}" target="_blank" rel="noopener">open ↗</a>`:esc(a.src||'—')}</span></div>`;

function detail(r){
  const frame = M.ai_steps?`<div class="framing"><b>How to read this:</b> each step is what the model read the message as, quoting the line it read; the artefacts beneath are the records, and each opens to its source. A dotted step kept the source's own verb — the model would not commit.</div>`:'';
  const why = r.why?`<div class="why"><b>Why these are one ${esc(item)} (${esc(r.tier)}):</b> ${esc(r.why)}</div>`:'';
  const steps = r.activities.map(n=>`<div class="step ${n.unread_step?'unread':''}">
      <div class="sn">${esc(n.name)}${n.unread_step?' <span class="tag">— not read into a step</span>':''}</div>
      <div class="sm">${esc(n.when)}${n.sources.length?' · '+n.sources.map(esc).join(' + '):''} · ${n.n} record${n.n===1?'':'s'}</div>
      ${n.arts.map(art).join('')}</div>`).join('');
  const inf = r.inferred.length?`<div class="step unread"><div class="sn">Not in any system</div><div class="sm">inferred — never asserted as fact</div>
      ${r.inferred.map(s=>`<div class="ev inf"><span>${esc(s.name)}</span><span class="sm">${esc(s.when)}</span><span>${esc(s.note||'')}</span></div>`).join('')}</div>`:'';
  return `<div class="detail">${frame}${why}<div style="margin-top:10px">${steps}${inf}</div></div>`;
}

function threadTable(rs, withReason){
  if(!rs.length) return `<p class="more">None.</p>`;
  const rows = rs.map((r,i)=>`<tr>
      <td><button class="thread" data-th="${i}" aria-expanded="false"><span class="t">${esc(r.title)}</span>${r.chip_word?`<span class="tchip">${esc(r.chip_word)}</span>`:''}<span class="id">${esc(r.id)}</span></button></td>
      <td>${esc(r.actor)}</td>
      <td class="path path1" title="${esc(r.path)}">${esc(r.path)}</td>
      ${withReason?`<td class="tag" style="white-space:normal">${esc(r.reason||'')}</td>`:`<td class="tag">${esc(r.dev_label)}</td>`}</tr>
      <tr class="det" data-for="${i}" hidden><td colspan="4">${detail(r)}</td></tr>`).join('');
  return `<input class="search" placeholder="Find one by id, owner, or title…" data-search>
    <table><thead><tr><th>${esc(item[0].toUpperCase()+item.slice(1))}</th><th>Owner</th><th>Path taken</th><th>${withReason?'Why it stayed out':'Differs how'}</th></tr></thead><tbody data-tbody>${rows}</tbody></table>`;
}

function renderNode(p){
  const rs = runsOf(p.id);
  const usual = p.paths.find(x=>x.label==='most common');
  const tags = ['All '+p.paths.length, ...new Set(p.paths.map(x=>x.label).filter(Boolean).map(t=>t[0].toUpperCase()+t.slice(1)))];
  const routes = p.paths.map(x=>`<tr data-tag="${esc(x.label)}"><td>${x.count}×</td>
      <td class="path">${pathText(x.seq)}${x.n_traces>1?` <span class="tag">(${x.n_traces} exact routes)</span>`:''}</td>
      <td class="tag ${x.label==='most common'?'common':''}" style="text-align:right">${esc(x.label)}</td></tr>`).join('');
  return `<div class="card">
    <h2>${esc(p.name)}${p.project?'<span class="ptag">project — happened once</span>':''}</h2>
    <p class="sub">${p.count} ${esc(items)} · ${p.n_records} records · ${p.n_records-p.n_unread} read into a step</p>
    ${p.actors.length?`<p class="who">${p.actors.map(esc).join('  ')}</p>`:''}
    ${p.why?`<div class="why"><b>Why these are one ${p.project?'project':'process'} (${esc(p.tier)}):</b> ${esc(p.why)}</div>`:''}
    <div class="band">
      <div class="bandhead"><h3>The steps, in the order they usually happen</h3></div>
      ${p.flow.length?`<div class="flow">${flow(p.flow)}</div>
      <p class="legend">Dotted underline: the name was read from the records' own words. Click any step for those words.</p>`:`<p class="legend">No step in this ${esc(item)} was read.</p>`}
      <div data-prov></div>
    </div>
    <div class="band">
      <div class="bandhead"><h3>Routes taken</h3>
        <span class="note">${p.no_common?`No usual way — every ${esc(item)} took a different route through those steps`:usual?`${usual.count} of ${p.count} ${esc(items)} follow the usual way`:''}${p.n_unread?` · ${p.n_unread} of ${p.n_records} records not read, counted here rather than drawn as steps`:''}</span></div>
      ${p.paths.length?`<div class="filters" data-rfilters>${tags.map((t,i)=>`<button class="fchip" aria-pressed="${i===0}" data-f="${esc(t)}">${esc(t)}</button>`).join('')}</div>
      <table class="runs"><tbody data-rbody>${routes}</tbody></table>`:`<p class="more">No route was read.</p>`}
    </div>
    <div class="band">
      <div class="bandhead"><h3>${esc(items[0].toUpperCase()+items.slice(1))}</h3><span class="note">every row opens to its records</span></div>
      ${threadTable(rs,false)}
    </div>
    <ul class="scope">${M.scope.map(s=>`<li>${esc(s)}</li>`).join('')}</ul>
  </div>`;
}

function renderLeftover(){
  const p = leftover, rs = runsOf(p.id);
  const groups = {};
  rs.forEach(r=>{ (groups[r.reason||'—'] ||= []).push(r); });
  const orph = V.orphans.length?`<p class="said"><b>${V.orphans.length}</b> record${V.orphans.length===1?'':'s'} matched no ${esc(item)} at all (e.g. <code>${esc(V.orphans[0].src)}</code> — ${esc(V.orphans[0].reason)}) — set aside, not counted.</p>`:'';
  return `<div class="card leftover">
    <h2>Not read into any process — ${p.count} ${esc(items)}</h2>
    <p class="said">Nothing in here is a finding. It is the part of the records that isn't process — and the size of it is how much to trust the processes on the left. Nothing here is guessed: ${p.n_unread} of these ${p.n_records} records kept the source's own verb because the model would not commit to what they did.</p>
    ${p.actors.length?`<p class="who">${p.actors.map(esc).join('  ')}</p>`:''}
    <div class="band"><div class="bandhead"><h3>Why each one stayed out</h3></div>
      ${Object.entries(groups).sort((a,b)=>b[1].length-a[1].length).map(([g,list])=>`<details class="group"><summary>${esc(g)}<span class="n">${list.length}</span></summary><div class="gbody">${list.map(r=>esc(r.title)).join(' · ')}</div></details>`).join('')}
    </div>
    ${orph}
    <div class="band"><div class="bandhead"><h3>${esc(items[0].toUpperCase()+items.slice(1))}</h3><span class="note">every row opens to its records</span></div>${threadTable(rs,true)}</div>
    <ul class="scope">${M.scope.map(s=>`<li>${esc(s)}</li>`).join('')}</ul>
  </div>`;
}

function renderGlossary(){
  const rows = (V.vocabulary||[]).map(v=>`<tr class="${v.unclassified?'unc':''}"><td class="${v.unclassified?'unc':''}">${esc(v.activity)}</td><td>${v.n}</td>
    <td>${esc(v.how)}${v.tier?` <span class="tag">${esc(v.tier)}</span>`:''}${v.phrases&&v.phrases.length?`<div class="phs">${v.phrases.map(q=>`<span class="ph">${esc(q)}</span>`).join('')}</div>`:''}</td></tr>`).join('');
  return `<div class="card"><h2>Step glossary</h2>
    <p class="sub">Where each step's name came from. A step's name is a claim like any other — some are the source's own word, some were grouped and named by a model, some were read out of the record's text and show the words they were read from.</p>
    <div class="band"><table class="gl"><thead><tr><th>Step</th><th>Records</th><th>How it got that name</th></tr></thead><tbody>${rows}</tbody></table></div>
    <ul class="scope">${M.scope.map(s=>`<li>${esc(s)}</li>`).join('')}</ul></div>`;
}

function render(){
  renderRail();
  if(current==='leftover' && leftover) pane.innerHTML = renderLeftover();
  else if(current==='glossary') pane.innerHTML = renderGlossary();
  else { const p = V.processes.find(x=>x.id===current) || procs[0] || projects[0]; if(!p){ current='glossary'; return render(); } pane.innerHTML = renderNode(p); }
  wire();
}
function wire(){
  document.querySelectorAll('[data-go]').forEach(b=>b.onclick=()=>{ current=b.dataset.go; location.hash=current; render(); pane.focus(); });
  const prov = pane.querySelector('[data-prov]');
  pane.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{
    const open = c.getAttribute('aria-expanded')==='true';
    pane.querySelectorAll('.chip').forEach(x=>x.setAttribute('aria-expanded','false'));
    if(open){ prov.innerHTML=''; return; }
    c.setAttribute('aria-expanded','true');
    const s = STEPS[c.dataset.step] || {how:'—', phrases:[], n:0};
    prov.innerHTML = `<div class="prov"><h4>${esc(c.dataset.step)}</h4><p class="how">${esc(s.how)} · ${s.n} record${s.n===1?'':'s'}</p>
      ${(s.phrases||[]).map(q=>`<div class="quote">${esc(q)}</div>`).join('') || '<p class="how">No quoted span — this name was not read from a record.</p>'}
      <p class="provfoot">Every reading opens to the record it came from — expand a ${esc(item)} below.</p></div>`;
  });
  const rf = pane.querySelector('[data-rfilters]');
  if(rf) rf.querySelectorAll('.fchip').forEach(b=>b.onclick=()=>{
    rf.querySelectorAll('.fchip').forEach(x=>x.setAttribute('aria-pressed','false')); b.setAttribute('aria-pressed','true');
    const f=b.dataset.f.toLowerCase();
    pane.querySelectorAll('[data-rbody] tr').forEach(tr=>{ tr.hidden = !(f.startsWith('all') || tr.dataset.tag.toLowerCase()===f); });
  });
  pane.querySelectorAll('.thread').forEach(b=>b.onclick=()=>{
    const i=b.dataset.th, det=pane.querySelector(`tr.det[data-for="${i}"]`);
    det.hidden=!det.hidden; b.setAttribute('aria-expanded', String(!det.hidden)); b.closest('tr').classList.toggle('open', !det.hidden);
  });
  const sb = pane.querySelector('[data-search]');
  if(sb) sb.oninput = e=>{ const q=e.target.value.toLowerCase();
    pane.querySelectorAll('[data-tbody] tr:not(.det)').forEach(tr=>{
      const hit = !q || tr.textContent.toLowerCase().includes(q); tr.hidden=!hit;
      const det = tr.nextElementSibling; if(det && det.classList.contains('det') && !hit) det.hidden=true; }); };
}
window.addEventListener('hashchange',()=>{ const h=location.hash.slice(1); if(h && h!==current){ current=h; render(); } });
render();
</script>
</body></html>"""
