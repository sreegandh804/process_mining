# Induction Engine

Turn a messy pile of **artefacts** — git history, GitHub issues/PRs, spreadsheets,
email — into a **believable, traceable process model**: the common path, the
variants, the exceptions, and the steps no system recorded. **Every claim points
back to its evidence and carries a confidence tier.** Inference is shown as
inference, never as fact.
<img width="1217" height="1296" alt="image" src="https://github.com/user-attachments/assets/2f74e270-f0a4-433f-b881-a3bb532653da" />

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
| **`samples/finance/`** | A small firm's **invoice approval & payment**: an invoice tracker (raised → submitted → approved → paid) plus a **bank payments** export that cross-references it on `invoice_id`. Deliberately messy — blank dates, a duplicate row, three date formats, a payment for an invoice that doesn't exist. | `python3 run_tabular.py` |
| **`samples/grants/`** | A **grant-making tracker** (applied → reviewed → decided → paid → reported). A second, unrelated domain through the same engine — a new `TableSpec`, not new code. | `python3 run_tabular.py --dir samples/grants` |
| **`samples/enron/`** | **263 real emails** from 3 Enron custodians (`kaminski-v`, `germany-c`, `jones-t`), curated so every subject-thread is a **complete conversation** — 76 threads, all ≥2 messages, no stray singletons (the engine induces 57 runs, 0 orphans). The genuine thin end: real RFC-822 with **no `In-Reply-To` headers at all**, so threads are earned from subject + fuzzy text/time, not a given key. | `python3 run_email.py --path samples/enron` |

The Enron sample is a subset of the public
**[Enron email dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)**
(the FERC corpus) — real corporate correspondence, not synthetic. It is the
sample that shows why the model tier exists: an email's verb is only *transport*,
so **without a key its steps read `sent → sent`**; with `ANTHROPIC_API_KEY` set,
the record-reading pass turns those into real activities (Requested, Approved, …),
each quoting the line it was read from.

### The full run, with the model tier on

Export a key, then read the mailbox. This is the run that produces processes and
steps read from the records rather than from the transport verb:

```bash
export ANTHROPIC_API_KEY=sk-ant-...

python3 run_email.py --path samples/enron

wslview out/inspector.html
```

That is ~14 model calls for the reading and naming, plus up to 200 short calls
for the semantic judge. Each seam takes its own model, so the expensive one can
stay sharp while the chatty one stays cheap:

```bash
INDUCTION_ACTIVITY_MODEL=claude-opus-5 \
INDUCTION_NAMING_MODEL=claude-opus-5 \
INDUCTION_SEMANTIC_MODEL=claude-haiku-4-5 \
python3 run_email.py --path samples/enron
```

`INDUCTION_ACTIVITY_MODEL` is the one worth keeping on the strongest model: it
runs the pass that derives the corpus's vocabulary, and everything downstream is
graded against it. `INDUCTION_SEMANTIC_MODEL` is a narrow same-work-or-not
verdict repeated a couple of hundred times, which is why it defaults cheaper.

### The typed tier (optional): Jev in front of Claude

A second, different kind of model can sit in front of the calls above. Jev
(TypeSafe's "System One", via the OpenRouter Decisions API) generates no text at
all: it is handed a `state` and a dict of typed questions, and returns one typed
answer per question, drawn only from options you supplied — a calibrated
probability (`noul`), one label from a set you named plus the full distribution
over it (`choice`), or a place on a rubric you wrote (`score`).

```bash
export OPENROUTER_API_KEY=sk-or-v1-...

python3 run_email.py --path samples/enron           # both tiers
python3 run_email.py --path samples/enron --no-jev  # exactly as before it existed
```

It is opt-in by key and changes nothing about what a run claims, so there is no
downshift note when it is absent — unlike a missing `ANTHROPIC_API_KEY`, which
changes what the run *is* and says so loudly.

**Where it runs, and why only there.** Jev is used at five points, and every one
of them is a question whose answer set already exists:

| Where | Question | Type |
| --- | --- | --- |
| `semantic.py` | same piece of work? and if related, by what? | noul + choice |
| `abstraction.py` gate | does this record perform any step? | noul |
| `abstraction.py` sample | does this record carry the work? | noul |
| `abstraction.py` labels | does this label discriminate — in the corpus, between processes, across runs? | 3 × noul |
| `abstraction.py` assign | which `Process > Step` does this record perform? | choice |
| `honesty.py` | is this cluster a process, where no profile has a rule? | choice |

**Where it deliberately does not run.** Discovery — the pass that invents the
vocabulary — stays on Claude, because a Choice needs its options to exist and on
a fresh corpus they do not. So does every call that has to quote its evidence: a
typed answer is a value, never a sentence, so Jev can produce no `span`, no join
reason and no name. Everywhere this engine shows a claim *with* the text it was
read from, the generative model still writes that text. Jev goes in front of
that call to narrow it, never instead of it. The deterministic spine — parsing,
shared-key joins, the co-occurrence arithmetic in `_detach_lonely_steps` — is
untouched and unscored: a number beside a fact would make the fact look like an
opinion.

**What it adds that was not expressible before.** A `choice` returns the whole
distribution, so a record several steps fit *equally* is now a different finding
from a record nothing fits. Both used to arrive at the same place — dropped, and
counted as declined. They are now two piles with two reasons, and the ambiguous
one is shown with what it was torn between.

**The label screen is one axis at three scopes.** A step label is only useful
because of what the engine computes with it afterwards, so each scope is a
precondition for one of those computations: a label true of every record in the
source leaves the spine with one node; a label fitting every process gives
`_process_of_case` nothing to count; a label fitting one run makes every run its
own variant. No domain is named, which is why the same three questions hold for
a mailbox, an issue tracker, a chat export or a ledger — and where a source's
verbs already discriminate, the reading tier never runs at all. Two of the three
are things the engine later *proves* arithmetically, so the screen's error rate
can be measured rather than tuned.

**Every decision you can see, you can open.** A bare `0.92` beside a step is not
traceability — 0.92 of what, against what else? Four kinds of typed decision are
recorded with the question exactly as the model was asked it, every option it
could have picked, and what the *engine* then did (which is not always what the
answer alone suggests):

| decision | where it shows |
| --- | --- |
| which stage a record performs | chip beside the quoted span |
| why two records are one case | on the join's reason |
| why a proposed step is **absent** from the vocabulary | its own row in the audit table |
| why a cluster is flagged a look-alike | on the flag |

They land in **`model.json` under `typed_decisions`**, not only in the page: the
inspector is a view of that file, and a claim that existed only in HTML would be
one no downstream tool could check. The page reads them back — click any chip and
one popover (never an inline expander, never a hover) shows the question, the
distribution, and the outcome.

Calls that only *narrowed* what was looked at — the record gate, the discovery
sample — assert nothing about any record and are counted, not listed. One row per
gated record would bury the four that matter.

**Are the numbers stable enough to threshold on?** `check_jev.py` answers that,
and nothing else should be trusted until it has:

```bash
python3 check_jev.py --path samples/enron --runs 10
```

It asks each question repeatedly about the same state and reports the spread and,
more usefully, how often the *decision* flipped. A question with a low spread and
a high flip count has its bar in the wrong place, not an unstable answer. It says
nothing about whether an answer is right — for that, compare the screen against
what `_detach_lonely_steps` and `_process_of_case` later prove.

**The rule the tests pin.** The typed tier can only ever *narrow*. A missing key,
an unreachable service, a tripped breaker or a silent answer must leave the
engine doing precisely what it did before this existed — silence is never a "no".
`tests/test_jev.py` asserts that directly, by running the same corpus with a dead
Jev and comparing the readings byte for byte.

**Whether it helps is a question about your corpus, not an opinion.** It changes
which records the expensive pass reads, and that can go either way — Claude reads
a whole thread and reasons about it; Jev picks from a list. `--no-jev` exists so
both paths stay one flag apart: run your corpus twice and compare
`n_unclassified` before trusting either.

### Calls that don't depend on each other now run at once

Two loops used to queue: the record-reading batches (`ceil(records/25)` Opus
calls) and the semantic judge (up to a couple of hundred Haiku calls). Neither
reads the previous call's output, so on the Enron sample that was ~17 minutes of
waiting for about a minute of work.

Both now run concurrently, bounded at three in flight
(`WORKERS` in `induction/concurrency.py`) — well inside a standard rate limit,
and a 429 is retried by the same backoff everything else uses. Same calls, same
tokens, same cost, same output. Raise `WORKERS` only after checking a real run's
log for retry lines.

The judge's decisions stay strictly sequential, because they are order-dependent:
a component already claimed by an earlier pair is out of the running, and which
pair claims it first is decided by shortlist order. So the *judging* fans out and
the *claiming* does not — `in_waves` evaluates a wave, then applies the rule to
the results one at a time, exactly as the serial loop did.
`tests/test_concurrency.py` pins that the joins are identical either way.

Progress streams to stderr as it goes, so a multi-minute run is never a silent
wait — including the line that matters most:

```
· abstraction: verbs are transport, reading 263 records
· abstraction: reading 263 records into 8 activities (…) — 11 batch(es)
· abstraction: and into 7 processes (…) — these become the kinds
· abstraction: read 107 of 263 records (156 declined) · 75 placed in a process
· abstraction: re-segmented on what the records say — 1 kind from the envelope,
    4 from the reading
```

**`156 declined` is the trust number.** A record the model would not commit to
keeps its raw verb and says so in the audit table, rather than being folded into
a step it does not evidence. If that number is most of the corpus, the reading
did not understand this data — and you can see that at a glance instead of
inferring it.

Open `out/inspector.html`; read `out/model.json`.

### The other runners

```bash
python3 run_tabular.py                  # spreadsheets (samples/finance)
python3 run_combined.py --demo          # two sources at once (offline demo, no key)
python3 run.py                          # a git corpus (after: python3 ingest.py …)
```

The **LLM tier is on by default** (activity naming + reading; needs
`ANTHROPIC_API_KEY`). With no key it downshifts to the deterministic, offline
baseline and says so on the first line; `--no-llm` forces that baseline. Run the
tests with `pytest`.

## What it can't conclude

The issue/PR/review timeline directly (inferred, marked so) · true order for thin
data · whether a `heuristic` join is *right* (scored; reads as uncertain) ·
cost/value (slots exposed, empty) · exactly where one kind ends and the next
begins (inferred, revisable).

---

**Full design rationale, decisions and trade-offs:** [`docs/DESIGN.md`](docs/DESIGN.md).
