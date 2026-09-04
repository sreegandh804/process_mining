# Induction Engine

Turn a messy pile of **artefacts** — git history, GitHub issues/PRs, spreadsheets,
email — into a **believable, traceable process model**: the common path, the
variants, the exceptions, and the steps no system recorded. **Every claim points
back to its evidence and carries a confidence tier.** Inference is shown as
inference, never as fact.

## The idea in one paragraph

An organisation delivers a product or service through a **process** — a series of
tasks. Its systems only ever record **artefacts** (a commit, a ticket, an invoice
row, an email), never the process itself, so the process must be **inferred** from
them. That is process mining — but *classic* process mining reads a clean **event
log** with a ready-made case id; this engine targets the **thin end**, where the
process was never cleanly logged and has to be earned from references, text
overlap, timing and gaps.

> **Not documented at all?** If a process lives only in people's heads, you first
> *task-mine* — a background agent capturing clicks / keystrokes / screenshots — to
> manufacture a log. That log is then just another source into this same engine.
> (Task mining is upstream data collection, deliberately out of scope here.)

## The model — every source normalises to 3 types first (`induction/model.py`)

| type | is | why it's its own type |
|---|---|---|
| **Entity** | a thing with identity (commit, PR, invoice, person) | an artefact is **not one event** — its timeline yields many |
| **Event** | a timed change (authored, approved, paid) | the atom mining consumes; `timestamp`/`actor` nullable, **never invented** |
| **Observation** | a state seen, maybe no time/actor | how thin data enters **without faking an event** |

Everything inferred carries a **Confidence tier** — `direct › joined › heuristic ›
model` — and **Evidence** (a locator that resolves to the raw artefact). A chain is
only as strong as its **weakest link**.

## The pipeline — one engine, every source

A source is just an **adapter** that shapes raw records into those 3 types and
declares *links*. Then the shared core runs:

| step | what it does |
|---|---|
| **Correlate** | group records into **cases** (one run): shared keys / git DAG → fuzzy text + time → an LLM *"same work?"* judge |
| **Order** | sort each case into a trace; no timestamp → `order: unknown` (never guessed) |
| **Label** | name each activity by its raw verb; where the verb is only *transport* (email `sent`), an LLM **reads the activity from the text** |
| **Segment** | cluster runs into process **kinds** by structure, then compute **variants** — real observed paths + frequency, tagged common / exception / one-off |
| **Gaps** | infer off-system steps from discontinuities (dashed, always inference) |
| **Orphans / Reject** | records that joined nothing are surfaced with a reason; a recurring machine-only pattern is flagged *"looks like a process, isn't"* |

A **Case** is one run · a **Variant** is a distinct path shared across runs · a
**Kind** is a whole process.

## Honesty

No node or edge without a tier + evidence. Missing actor / time / order is marked
`unknown`, not filled. Orphans are surfaced, not dropped. The LLM only **names and
groups** records that already exist — it never invents a step; finding what is
*absent* stays deterministic.

## Sources built

git history · GitHub issues/PRs · changelog · spreadsheets (CSV/XLSX — wide
trackers *and* long event logs) · email (maildir / mbox / CSV). **A new source is a
new adapter (+ an optional naming profile), not a new engine.**

## Try it — three bundled samples

Everything below is **real data, in the repo, runnable offline right now**. No
clone, no network, no key needed.

| sample | what the data is | run it |
|---|---|---|
| **`samples/finance/`** | A small firm's **invoice approval & payment**: an invoice tracker (raised → submitted → approved → paid) plus a **bank payments** export that cross-references it on `invoice_id`. Deliberately messy — blank dates, a duplicate row, three date formats, a payment for an invoice that doesn't exist. | `python run_tabular.py` |
| **`samples/grants/`** | A **grant-making tracker** (applied → reviewed → decided → paid → reported). A second, unrelated domain through the same engine — a new `TableSpec`, not new code. | `python run_tabular.py --dir samples/grants` |
| **`samples/enron/`** | **263 real emails** from 3 Enron custodians (`kaminski-v`, `germany-c`, `jones-t`), curated so every subject-thread is a **complete conversation** — 76 threads, all ≥2 messages, no stray singletons (the engine induces 57 runs, 0 orphans). The genuine thin end: real RFC-822 with **no `In-Reply-To` headers at all**, so threads are earned from subject + fuzzy text/time, not a given key. | `python run_email.py --path samples/enron` |

The Enron sample is a subset of the public
**[Enron email dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)**
(the FERC corpus) — real corporate correspondence, not synthetic. It is the
sample that shows why the model tier exists: an email's verb is only *transport*,
so **without a key its steps read `sent → sent`**; with `ANTHROPIC_API_KEY` set,
the record-reading pass turns those into real activities (Requested, Approved, …),
each quoting the line it was read from.

```bash
python run_combined.py --demo          # two sources at once (offline demo, no key)
python run.py                          # a git corpus (after: python ingest.py …)
```

The **LLM tier is on by default** (activity naming + reading; needs
`ANTHROPIC_API_KEY`). With no key it downshifts to the deterministic, offline
baseline and says so on the first line; `--no-llm` forces that baseline. Open
`out/inspector.html`; read `out/model.json`. Run the tests with `pytest`.

## What it can't conclude

The issue/PR/review timeline directly (inferred, marked so) · true order for thin
data · whether a `heuristic` join is *right* (scored; reads as uncertain) ·
cost/value (slots exposed, empty) · exactly where one kind ends and the next
begins (inferred, revisable).

---

**Full design rationale, decisions and trade-offs:** [`docs/DESIGN.md`](docs/DESIGN.md).
