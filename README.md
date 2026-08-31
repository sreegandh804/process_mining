# Induction Engine

Turn a real, messy corpus of *artefacts* into a *believable, traceable process
model* — the common path, the variants, the exceptions, and the steps no system
recorded — with **every claim pointing back to its evidence** and carrying a
**confidence tier**. Inference is rendered as inference, never as fact.

The corpus here is the **git history of [`pallets/flask`](https://github.com/pallets/flask)**
(≈5,400 commits), with the same repo's `CHANGES.rst` as a deliberately *thin*
second source. Nothing is fabricated: no invented actors, timestamps, order, or
costs.

---

## Quickstart

```bash
# 1. clone a corpus (any public repo with a real contribution process)
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1000 https://github.com/pallets/flask data/corpus/flask

# 2. cache it to disk (reproducible, offline from here on)
python ingest.py --repo-path data/corpus/flask --slug pallets/flask

# 3. induce the processes and emit the model + inspector
python run.py
```

…or `make run`, which does all three. Then open **`out/inspector.html`** in a
browser and read **`out/model.json`** for the complete artefact.

**A non-git customer** (a firm with spreadsheets, no git) runs through the *same*
engine — no clone, no network:

```bash
python run_tabular.py                       # samples/finance CSV (invoices + bank)
python run_tabular.py --dir samples/grants  # a grant-making tracker
python run_tabular.py --profile accounting  # friendly names for finance
python run_tabular.py --names llm           # let Claude name the processes/steps
python run_tabular.py --xlsx                # read the .xlsx sheets (needs openpyxl)
```

Run the tests with `pytest` (or `make test`). The core suite is offline and
stdlib-only; the held-out-slice tests activate once `pallets/click` is cached
(`make heldout`).

**No dependencies.** The baseline is pure Python standard library — a deliberate
choice, so a run is reproducible and another engineer can pick it up without a
dependency install. `pytest` is the only dev/test dependency.

---

## What it does (definition of done)

1. Runs end-to-end on a real corpus with one command.
2. Holds up on a slice it did **not** see during development — a *different*
   repo, `pallets/click` (`tests/test_heldout.py`): no crashes, sane output.
3. Every surfaced claim points to its evidence and carries a confidence **tier**;
   inference is rendered as inference.
4. This README states plainly what the engine **cannot** conclude and where the
   build stops.

---

## Why this corpus

The brief rewards *awkward material*, and git history of a real project is full
of it — reverts and re-reverts, reopened work, backport trains, bot commits that
look like a process but aren't, commits that reference nothing, and one activity
recorded against two people. It is also, critically, an honest instance of the
hardest truth in the problem statement:

> **The system of record only sees what touched it.** Git records commits and
> merges. Issues, PR reviews, and discussion happened *off-git*. So anything
> about them must be **inferred from references and gaps** — the engine is
> *confidently incomplete* by construction, which is exactly what a real
> customer's system data is.

The execution environment could only read the corpus repo over anonymous git
(no cross-owner GitHub API), which turned out to *sharpen* this: we genuinely
cannot see the issue/PR timeline, so we have to earn every claim about it from
evidence in the commits. See **"What it cannot conclude"** below.

The **thin source** (`CHANGES.rst`) is the proof of graceful degradation. It
records *what* shipped but not *when* each change was made, *by whom*, or in what
*order* — so it enters as `Observation`s with `actor = None`, `seen_at = None`,
and its runs are flagged `order: unknown`. Where a changelog bullet cites a PR
the git side already has, the two sources correlate on the shared number (thin
enriching thick). This is real data through the full pipeline, not a fixture.

---

## The model (the substrate everything normalises into)

Everything becomes one of three canonical types **before** any mining
(`induction/model.py`):

| type | is | key honesty point |
|---|---|---|
| **Entity** | a thing with identity (commit, pr, issue, person, release) | a commit is *not one event* — its timeline yields many events |
| **Event** | a timed change (authored, committed, merged, reverted) | `timestamp`/`actor`/`case_id` are nullable and never invented |
| **Observation** | a state seen, with no action and maybe no time | how thin data enters — a changelog line has no actor and no order |

**Confidence is an ordinal tier, never a fabricated decimal:**

```
direct    read straight from the source (present as data)
joined    deterministic join on a shared key (commit↔PR number, the git DAG)
heuristic rule-based inference (reference similarity, actor+time proximity)
model     LLM inference — same-work judgement, and reading an activity
          out of a record whose verb carries none (opt-in; embeddings
          remain the unbuilt rung behind the same seam)
```

A claim's tier *is* its confidence. The confidence of a chain of inferences is
its weakest link. If a number were ever needed, it would be *calibrated* against
the golden fixture — never invented.

`Evidence` carries a `locator` (a commit sha, `CHANGES.rst:L120`) that resolves
back to the raw artefact, so any claim can be checked by eye.

### What the pipeline builds on top (`induction/process.py`)

| model | id | holds |
|---|---|---|
| **ProcessKind** — *a process* | `kind_1` … or a profile name (`code_contribution`) | `case_ids`, `variants`, `dfg`, `steps`, `rejected` + reason |
| **Case** — *one run* | `case:pr:15` · `case:email:{slug}:{msgid}` · `case:invoice:INV-1001` | `ordered_event_ids`, `order_status`, `trace_signature`, `anchor`, `confidence` |
| **Variant** — *one observed path* | keyed by its `signature` | `signature`, `frequency`, `case_ids`, `role` = common / exception / one-off |
| **Step** — *an activity, corpus-wide* | `step:{action}` | `name` (display), `action` (raw verb), `member_ids`, `event_ids` |
| **Gap** — *what is absent* | `gap:{kind}:{case}:{action}` | `missing_expected_step` / `reconciliation` / `off_system_review`; always `heuristic` |
| **Orphan** — *what joined nothing* | keyed by `record_id` | the reason it joined nothing |

Two naming layers sit on a step, and they are not the same claim:

- **`Step.action`** — the artefact's own verb (`authored`, `approved`, `sent`),
  tier `direct`, because the source said it.
- **the activity** on `Case.trace_signature` — what the abstraction layer
  concluded the record *realises* (`Approved`), tier `model`, carrying the span
  of text it read. Consecutive records realising one activity fold into a single
  step with the artefacts listed beneath it.

```
Entity ──has──▶ Event ──correlate──▶ Case ──segment──▶ ProcessKind
                  │                    │                  │
              Evidence          trace_signature      variants · gaps
```

Every one of those carries a `Confidence`, and a `Case` reports its **weakest
link** — a run assembled across a `model` join reads `model`, even though each
event in it is `direct`.

---

## The pipeline (steps 0–6) and the decisions inside it

Discipline throughout: **deterministic, inspectable baseline first**; add fuzzy/
embedding/LLM only where the baseline *demonstrably* breaks; any inferred field
carries confidence + evidence.

- **Shape (1)** — `adapters/git_history.py`, `adapters/changelog.py`. Each commit
  expands into its events. A committer who differs from the author becomes a
  separate `committed` event — a real handoff, not noise. Co-authors become
  sibling `authored` events (the same-activity merge feeds off this).

- **Correlate (2)** — *the graded core*, `steps/correlate.py`. **One correlator
  serves every source.** Adapters do not correlate; they declare `Link`s
  (`induction/links.py`) — "this record points at that one, by this method, at
  this tier, for this reason" — and the correlator resolves them. A commit's
  merge topology, a PR's `closes #12`, a spreadsheet's foreign key and a
  changelog's `:pr:` citation all arrive as the same shape, so a case can be
  assembled from several sources at once. The correlator contains no source
  name; `tests/test_github.py` asserts that structurally rather than in prose.

  This was not always true, and the reason it had to change is worth stating: a
  correlator per source can only ever see its own records, which makes
  cross-source joining *structurally impossible* — the single hardest thing the
  brief asks for. Deterministic, layered, per-link-scored joins:
  - **merge-DAG topology** (`joined`): a `Merge pull request #N` commit owns the
    commits it introduced onto the trunk. This uses the real git DAG, so it
    recovers *multi-commit* runs that a text-only join would scatter into
    orphans. The walk stops at other PRs' boundaries; a merge that would absorb
    an implausible number of commits is reclassified as a release/integration
    train (`heuristic`, with its reason) rather than asserted as one 1,000-commit
    "instance".
  - **squash subject `(#N)`**, **issue keywords** (`joined`), **bare `#N`**
    (`heuristic` — a mention is not a proven link).
  - What stays unjoined stays unjoined → the orphan queue. On flask ~1,500
    commits join no run; that is honest, not a bug.
  - **fuzzy, no shared key at all** (`heuristic`) — the second pass, over what
    determinism could not explain. Whole *components* are matched (a mail thread
    joins as a thread, not as its first message) on text overlap **plus**
    actor-or-time proximity, and every link records its score and the exact
    tokens it matched: *"no shared key; text overlap 0.38 on words 'csv',
    'export', 'row', and 4 days apart"*. Two guards keep it surgical: it never
    sees a pair determinism already joined, and it declines components whose
    activity signatures are near-identical — those are two runs of one kind
    (two dependency bumps), not two halves of one run. On the git corpus it
    fires **zero** times, which is the correct answer there and is asserted.

- **Order (3)** — `steps/order.py`. Sort by timestamp; git traces are `ordered`.
  Thin observations with no time are `order: unknown` — surfaced, not guessed.

- **Variants (4)** — `steps/variants.py`. Real observed traces with frequencies,
  labelled common / exception / one-off (a revert is an exception by nature). A
  directly-follows graph is included but explicitly caveated as over-generalising.

- **Label (5)** — `steps/label.py`. An activity is named by its **raw action**
  — the label the source itself gave it (`authored`, and for another source
  `invoice_approved`). No rename table lives in the step. The
  **same-activity-different-people** merge folds a co-authored commit into one
  step with several members, keeping every underlying record.

- **Segment (0)** — `steps/segment.py`. Clusters runs into *kinds* by
  **structure alone** — automated vs human-driven, and their correlation anchor
  — and by default leaves them **unnamed** (`kind_1`, `kind_2`, …) with a
  data-derived rationale. Boundaries are inferred (`heuristic`) and revisable.
  We do not invent domain names for data we know nothing about.

### Profiles — keeping the engine source-agnostic

The two things that are genuinely domain knowledge — *what activities are
called* and *which kinds of process exist* — are **not** hardcoded in the shared
steps. They live in a `Profile` (`induction/profiles.py`) matched to the source:

- **Default (`GENERIC_PROFILE`)** runs on *any* data: activities keep their raw
  action verb, kinds are the unnamed structural clusters above, and the
  "looks like a process, isn't" rule is domain-free (a recurring, fully-automated
  cluster is flagged — it can't prove it "produces nothing" without domain
  knowledge, and says exactly that).
- **`GIT_PROFILE`** (opt in with `python run.py --profile git`) overlays friendly
  names ("Merge pull request", "Dependency bumps") and the git-specific reject
  rule — **without changing any structure** (tests assert the case/orphan/gap
  counts are identical either way).

So a new customer is *an adapter + a profile*, and the engine core never learns a
domain word. This is the fix for the obvious smell — an accounting firm's data
would get coherent, honestly-**unnamed** kinds today, and real names the moment
someone writes a 30-line profile for it.

- **Gaps (6)** — `steps/gaps.py`. Off-system steps inferred from real signals and
  rendered as inference: a merged PR presupposes an off-git open + review; an
  author≠committer split with a multi-day delay is an off-git acceptance that
  left no event. Each names the signal that produced it.

---

## Honesty features (the graded core — `induction/honesty.py`)

- **Orphans** — records that join to no run go to a visible queue with a reason,
  never padded into a rollup or dropped.
- **Reject** — a recurring, machine-driven pattern that moves no product artefact
  is flagged *"looks like a process, isn't"* (dependency bumps), with its reason —
  flagged, not deleted, so a reader can disagree.
- **Unknowns** — missing actor / time / order is marked `unknown`. Absence is a
  finding, not a blank to fill.
- **Divergence hook** — `raw` is kept beside the inferred structure so a later
  owner-validation step can compare belief against data. The loop is *described*
  below, not built.
- **Confidence everywhere** — no node or edge without a tier and evidence.

---

## Output & the thin inspector

`out/model.json` is the complete artefact: process definitions, variants
(traces + frequencies), instances, steps, members, gaps, orphans — every element
carrying `confidence` and `evidence[]`, plus empty-but-present cost/value and
divergence slots.

`out/inspector.html` is a single self-contained file written for a **non-technical
operations lead**, and it is fully data-driven and adaptive — the same page reads
for a grants tracker, an invoice ledger, or a code repo, because it renders
whatever the model induced (no domain words baked in). It shows:

- **The processes found** — each kind as a named flow, with *every* variant path
  and how often each ran; a flagged kind carries its "looks like a process,
  isn't" reason.
- **Every run, not a sample** — a filterable, searchable table of *all* instances;
  the filters ("Missing a step", "Ended early", "Unmatched record", "Automated")
  are just views over the full list. Each row opens to its own timeline, and
  **every step — including an inferred, dashed one we didn't see — resolves to its
  exact source record** (a row, a sha).

Process/step **names** come from the profile, or optionally an LLM naming pass
(`--names llm`, tier `model`, shown as "names suggested by AI"); the LLM only
*names*, never touches structure. Without it, activities read as their raw verbs.

---

## A second customer: spreadsheets (the thin, non-git end)

The brief's hardest case is the customer at the thin end — "one document library
and nothing else" — who most needs the system. To prove the engine is not
git-shaped, `run_tabular.py` runs a small firm's **invoice approval & payment**
process (an Excel/CSV export in `samples/finance/`) through the *same* pipeline —
different domain, no code below the adapter changed.

- **The adapter is spec-driven** (`adapters/tabular.py`): a `TableSpec` maps
  columns to the model — which column is the identity, which are timestamped
  events (and who performed them), which is the status, which are foreign keys.
  A new tracker (tickets, onboarding, cases) is a new *spec*, not new code. CSV
  is stdlib; `.xlsx` uses openpyxl if present.
- **Each invoice row has a multi-date timeline** (raised → submitted → approved
  → paid), so a row yields a real trace — and a *bank-payments* export
  cross-references invoices (`settled`), correlated on the shared id (`joined`),
  exactly like git ↔ changelog.
- **The thin realities are first-class:** a blank date = the step isn't recorded
  (no event, no invented time); a blank actor = unknown (never guessed); a row
  with a status but no dates = an `Observation`, `order: unknown`; a row with no
  id, or a payment for an invoice that doesn't exist, is **surfaced** (orphan /
  unresolved reference), never dropped.
- **It produces findings an operations director would act on**, from real
  discontinuities, all rendered as inference:
  - *invoices paid with no recorded approval* (a control breach) — the generic
    "reached a late step without an earlier one the common path has" detector;
  - *invoices marked paid in the tracker with no matching bank payment*, and *a
    bank payment for an unknown invoice* — cross-source reconciliation;
  - a recurring, zero-value, **system**-actor cluster flagged *"looks like a
    process, isn't"* — by the same generic rule as the git bots.

This is the whole thesis in miniature: **new source = new adapter (+ optional
profile), same engine.** Mail is built too (`adapters/email_mbox.py` — maildir,
mbox or CSV), and `tabular.py` now reads both table shapes: a *wide* tracker
export (a row is a case, a column is a step) and a *long* event log (a row is an
event — the field's standard, and the only shape that can represent a repeated
step). `detect()` proposes which shape a file is and prints what it measured;
`--case-column`/`--activity-column` override it. Calendars and chat remain
unbuilt; each is the same shape of work.

## What it **cannot** conclude

- **Anything about the issue/PR/review timeline directly.** The corpus is git.
  We infer that PRs were opened and reviewed off-git, and mark those as
  inference; we cannot see who reviewed, what was discussed, or how long it took.
- **True process order for thin data.** A changelog gives state, not sequence;
  those runs are `order: unknown`.
- **Whether a correlation is *right*.** We score every join; a `heuristic` join
  can be wrong. The mitigation is that it *reads* as uncertain, not as fact.
- **Cost or value.** No money, effort, or time-per-step figures are produced. The
  slots are exposed and empty. Fabricating them is exactly what the brief forbids.
- **Where one *kind* of process really ends and the next begins.** Segment
  boundaries are inferred and revisable.

---

## Where the build stops, and the upgrade path (deliberately not built)

The baseline is deterministic on purpose. The upgrade path is real and scoped,
step by step — each would be added only where the crude baseline visibly breaks,
and each would enter as a lower, clearly-marked tier:

- ~~**Correlate** — fuzzy joins for records with no shared key.~~ **Built.**
  The GitHub Issues/PR source forced it: an issue and the pull request that
  fixes it routinely share no number, keyword or branch, and there is no DAG to
  fall back on. Text similarity + actor/time proximity, at the *transparent*
  tier — token overlap with corpus-derived rarity weighting, so every join can
  name the words it matched (`induction/text.py`). Embeddings are the next rung
  (`model` tier) and slot in behind the same `similar()` signature; that is
  still unbuilt, deliberately, because the transparent tier is auditable and an
  embedding is not.
- ~~**Segment / Label** — finer kind-boundaries; an LLM to name activities and
  judge "are these two the same step?".~~ **Built**, in two pieces, each gated on
  a measurement rather than a source name:
  - `steps/topics.py` splits a kind by *subject* when structure said nothing —
    fired only where one structural cluster is ≥90% of the corpus (pallets/flask
    partitions 49/24/16/10/1% and is untouched; a mailbox is 100% and is split).
    Deterministic, IDF token overlap, and every kind names the words that made it.
  - `abstraction.py` reads the activity out of the record where the verb is only
    transport — a mailbox records `sent` and nothing else. Gated on median
    records-per-activity ≥ 2.0 (finance 1.00, flask 1.00–1.33, a mailbox 2.00).
    Each reading quotes the span it came from; a record it will not commit to
    keeps its raw verb and is counted as unclassified.

  **The guardrail held:** the model may name and judge equivalence only. It
  labels a record that exists; the deterministic detector finds what is absent.
  Embeddings remain unbuilt behind `similar()` — the transparent tier is
  auditable and an embedding is not.
- **Gaps** — a model to propose *which* off-system step a discontinuity implies,
  beyond the two structural rules built here.

Also stated-but-unbuilt (hooks are in place):

- **The owner-validation loop (belief vs data).** Induce the process from
  evidence *first* (asking up front yields the official version, not the real
  one); show the owner the draft anchored to evidence; let them correct only the
  low-confidence parts; surface disagreements as **divergences**, keeping both
  sides and marking which is belief and which is evidence. Corrections raise a
  claim's tier and are themselves recorded as evidence. `model.json.divergence`
  is the empty hook.
- **Cost / value.** Slots exist per step and per engagement (money + effort;
  revenue + outcomes). Populating them is product work, not build work.

---

## The system around it (the questions we expect)

- **Thick vs thin customer.** The *same* pipeline runs both. The thick customer
  (git) gets timed traces and topological joins; the thin customer (changelog)
  gets states with `order: unknown` and correlation on whatever shared keys
  exist. The thin customer is the one who most needs the honesty machinery, which
  is why the `Observation`/`unknown`/orphan paths are first-class, not fixtures.
- **A customer that isn't git at all.** Built and shown: `run_tabular.py` runs a
  firm's invoice tracker (CSV/Excel) + bank export through the same engine (see
  "A second customer" above), producing real control and reconciliation findings
  with honestly unnamed kinds by default. The core only ever touches
  `Entity`/`Event`/`Observation`; domain vocabulary lives in an adapter + a
  `Profile`. Mail and long-format event logs are built too; calendars and chat
  are not — each the same shape of work as `tabular.py`.
- **What "actionable" means.** Not a prettier map. A finding is actionable when it
  carries the whole chain: *what actually happens → what it costs → where it
  breaks → what to change → who must agree → whether it worked after.* This engine
  delivers the **first link, honestly**: the evidenced process + variant
  frequencies + the exceptions and gaps. Cost/value and the change/sign-off/measure
  links are the product around it. We say plainly the build stops at the first link.
- **Drift.** Because `raw` is kept and every claim is evidenced and timestamped,
  re-running on fresh data and diffing the induced model against the owner-
  confirmed one is how drift would surface — as new divergences.
- **Second and fiftieth customer.** The cost is in the *adapter* (one per source)
  and the correlation heuristics; the model, steps, honesty layer, and inspector
  are source-agnostic. A new customer is a new adapter, not a new engine — proven
  in miniature by running unchanged on a second repo.
- **What we'd measure in production.** Join precision (sampled `heuristic` joins
  a human confirms/rejects), orphan rate and its trend, the share of surfaced
  steps that are inferred vs evidenced, and correction rate in the validation
  loop (how often the owner overrides a low-confidence claim).

---

## Repo layout

```
ingest.py            git corpus -> cached raw JSON (the only step that touches git)
ingest_github.py     GitHub Issues/PR API -> cached raw JSON (needs a token)
run.py               git path:     induce -> out/model.json + out/inspector.html
                     (--with-github adds the Issues/PR corpus to the same run)
run_tabular.py       spreadsheet path (samples/finance): same output, no git
                     (--file DETECTS the shape of any CSV/XLSX and reads it)
run_combined.py      several sources at once (GitHub + a mailbox), + the model tier
run_email.py         a mailbox on its own
samples/finance/     non-git demo corpus (invoices + payments, CSV & XLSX)
samples/grants/      a second non-git corpus (a grant-making tracker)
induction/
  naming.py          OPTIONAL LLM naming of kinds/steps (tier model; --names llm)
  model.py           Entity / Event / Observation, Evidence, Confidence (tiers)
  refs.py            cross-reference extraction (+ the tier each kind earns)
  links.py           the link vocabulary adapters declare into (the correlation seam)
  text.py            token similarity for the fuzzy pass (stdlib, IDF, auditable)
  process.py         induced-model vocabulary (Case, Variant, Step, Gap, Orphan, Kind)
  abstraction.py     artefact verbs -> activities: the verb map, and (where the verb
                     is only transport) reading the record itself
  semantic.py        the model tier's judge/embedder seams (opt-in, --semantic llm)
  profiles.py        where domain vocabulary lives (generic default; git / accounting)
  adapters/          git_history.py, changelog.py, github_api.py, email_mbox.py,
                     tabular.py (wide tracker exports AND long event logs)
  steps/             segment (+ topics: kinds by subject where structure said
                     nothing), order, variants, label; correlate (ONE correlator,
                     every source); gaps + gaps_generic
  honesty.py         orphans, reject, (unknowns/divergence surfaced in emit)
  pipeline.py        induce() (shared core) + thin per-source loaders
  emit.py            InducedModel -> model.json
  inspector.py       InducedModel -> self-contained inspector.html
tests/               golden fixture + ugly-record cases + held-out slice + tabular
```

Start with `model.py` (the substrate) and `steps/correlate.py` (the graded core);
everything else hangs off those two seams.
