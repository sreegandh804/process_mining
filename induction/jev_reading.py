"""Jev in the reading tier — the typed decisions around `abstraction.py`.

The reading tier is the expensive half of this engine: a generative model invents
the corpus's process vocabulary from a sample, then reads every record into it.
Both calls are worth what they cost, because both need something only an open-set
model can do — invent a name, and quote the text that justifies a reading.

Neither of them needs to *also* be the thing that decides which records were
worth reading, or whether a proposed label is a label at all. Those are closed
questions: the options exist before the question is asked. This module is the
three places that is true, and nothing else.

  1. `SignalFilter` — per record, does this carry work at all? Three narrow
     Nouls in one call. Used twice: to choose which records the discovery sample
     is built from, and to keep records that perform nothing out of the batches
     the generative classifier is paid to read.

  2. `LabelScreen` — per proposed step label, is this an achievement or is it
     the system's own filing verb in more words? `_enforce_label_form` tests this
     today with a word count, which cannot see that "Correspondence Sharing" is
     exactly what the discovery prompt forbids and is only two words long.

  3. `RecordAssigner` — thread to process, record to step, as Choices over the
     vocabulary that discovery has ALREADY produced. This one never runs first:
     a Choice needs its options to exist, and on a fresh corpus they do not until
     the generative pass has invented them. What it adds is the full distribution
     over those options, which is a distinction this engine could not previously
     draw — a record nothing fits and a record several things fit equally both
     used to arrive at the same place, silently dropped.

What none of them do is produce a span, a name or a reason. The generative tier
still writes every sentence the engine shows a reader. These only narrow what it
is asked and how much of it there is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Questions inside ONE Jev call are already evaluated in parallel server-side;
# `fan_out` is the second axis — many records at a time — and it is what turns a
# serial walk over a corpus into one wide pass.
from induction.concurrency import fan_out as _fan_out


# ---------------------------------------------------------------------------
# 1. Does this record carry work?
# ---------------------------------------------------------------------------

# Decomposed into three, because "is this junk" is three judgements wearing one
# coat: a greeting performs nothing but is not machine noise; a nightly build
# notice is machine noise and performs something; a one-line "approved" performs
# a step and looks like nothing. One broad question averages all three into a
# number that means none of them.
# Each question names the FIELD it turns on, in backticks, rather than saying
# "this record" and leaving the model to work out which part matters. The state
# is already structured — `_record_state` builds it from what the adapters
# parsed — so describing it in prose was asking a model to re-find fields this
# code was holding. Three questions that all read "this record" are three
# questions looking at the same undifferentiated blob; pointed at `record.body`,
# `record.actor` and `record.subject` they are three different questions.
_SIGNAL_QUESTIONS: dict = {
    "performs_action": {
        "type": "noul",
        "instructions": {
            "question": "Does `record.body` show work being DONE?",
            "inspect": "`record.body`",
            "focus": "Something moved forward — not that a message was sent.",
        },
        "criteria": {
            "true": {
                "what": "A decision, approval, delivery, request or change is made here",
                "examples": ["Approving an invoice", "Submitting a document for review"],
            },
            "false": {
                "what": "Nothing is accomplished by this record",
                "not_for": "A short reply that answers a question and settles something",
                "examples": ["Thanks!", "A greeting", "A forward with no comment",
                             "An out-of-office reply"],
            },
        },
    },
    "machine_generated": {
        "type": "noul",
        "instructions": {
            "question": "Was `record` produced by an automated system rather than a person?",
            "inspect": ["`record.actor`", "`record.action`"],
            "focus": "Judge the author in `record.actor`, not the subject matter.",
        },
        "criteria": {
            "true": {"what": "Emitted by a bot, cron job, monitor or pipeline",
                     "examples": ["A nightly build notice", "An automated dependency bump"]},
            "false": {"what": "Written or performed by a person",
                      "not_for": "A person writing about an automated system",
                      "examples": ["An engineer reporting that the nightly build failed"]},
        },
    },
    "carries_work": {
        "type": "noul",
        "instructions": {
            "question": "Would `record` help someone infer what process this corpus is a record of?",
            "inspect": ["`record.subject`", "`record.body`"],
            "focus": "Is the work itself visible in those fields, or only the envelope?",
        },
        "criteria": {
            "true": {"what": "Names the subject matter of the work concretely",
                     "examples": ["A contract's terms being negotiated",
                                  "A candidate being scheduled for an onsite"]},
            "false": {"what": "Could belong to any process, or to none",
                      "examples": ["Please see attached.", "Social chatter"]},
        },
    },
}


@dataclass(frozen=True)
class Signal:
    """One record's reading, as three independent numbers."""

    performs_action: Optional[float] = None
    machine_generated: Optional[float] = None
    carries_work: Optional[float] = None

    @property
    def known(self) -> bool:
        return self.performs_action is not None or self.carries_work is not None

    def worth_reading(self, bar: float) -> bool:
        """Unknown is always worth reading. A filter that cannot run must not
        remove anything — an unreachable service silently emptying the corpus is
        exactly the failure this codebase is built to make impossible."""
        if self.performs_action is None:
            return True
        return self.performs_action >= bar


class SignalFilter:
    """Reads records for signal, so the generative tier is not paid to find none.

    `bar` is deliberately low. This is a gate, not a verdict: it exists to keep
    greetings and pure forwards out of a 16k-token batch, not to decide what a
    record means. Everything it holds back is *gated*, which is its own pile with
    its own count — not declined, not rejected, and never deleted.
    """

    def __init__(self, jev=None, bar: float = 0.25, log=None):
        self._log = log or (lambda m: None)
        if jev is None:
            from induction.jev_call import Jev
            jev = Jev(log=self._log)
        self._jev = jev
        self.bar = bar

    @property
    def available(self) -> bool:
        return self._jev.available

    def read(self, state) -> Signal:
        # Nested under `record` so the questions' backticked paths resolve. A flat
        # state with questions naming `record.body` points at nothing.
        answers = self._jev.ask({"record": state}, _SIGNAL_QUESTIONS)
        if not answers:
            return Signal()

        def p(name):
            a = answers.get(name)
            return a.noul if a is not None else None

        return Signal(performs_action=p("performs_action"),
                      machine_generated=p("machine_generated"),
                      carries_work=p("carries_work"))

    def read_all(self, states: list) -> list[Signal]:
        return [s if s is not None else Signal() for s in _fan_out(self.read, states)]


# ---------------------------------------------------------------------------
# 2. Is this proposed label a step, or the envelope in more words?
# ---------------------------------------------------------------------------

# The rubric the discovery prompt already states in prose and cannot enforce:
# "NEVER return a name that merely restates how a record travelled or was filed."
# A word count cannot test that. An ordered rubric can, and a Score sits it on a
# scale so the engine can require a margin rather than a coin flip.
# One axis, three scopes: DOES THE LABEL DISCRIMINATE, and at what level?
#
# These are not three ways a label can be bad. Each one is a precondition for a
# specific computation the engine performs afterwards, and naming the computation
# is what keeps the test honest rather than a matter of taste:
#
#   corpus scope    -> the spine. A label true of every record in the source gives
#                      every record the same name, and there is no order to find.
#   process scope   -> `_process_of_case`, which places a run by COUNTING its
#                      records' labels. A label every process shares votes for
#                      nothing, and no run can be placed.
#   run scope       -> `variants` and `gaps_generic`, which compare a run against
#                      its kind's common path. A label fitting one run makes every
#                      run its own variant, and no gap is findable.
#
# Stated this way the tests carry no domain and no source. "Transport" was never
# the criterion — it was a symptom of the corpus-scope one. `Sent` is useless
# because it is true of every record whatever was done; on a support desk where
# every record is a reply, `Replied` is useless for exactly the same reason, and
# on a desk that also escalates, refunds and closes, `Replied to customer`
# discriminates and is a real stage. The same three questions hold for an issue
# tracker, a chat export or a ledger, and where a source's verbs already
# discriminate, the reading tier never runs at all.
#
# Three Nouls rather than one rubric, because they are independent — each has a
# label that passes the other two and fails only it — and because a Noul is
# graded on a scale the model calibrates (a probability) instead of levels this
# module invented and then had to guess a bar between. They ride in one call;
# questions over one state are evaluated in parallel.
#
# Two of the three are provable. `_detach_lonely_steps` measures run scope by
# co-occurrence, and `_process_of_case`'s vote counts expose process scope. So
# these are cheap PREDICTIONS of properties the arithmetic later proves, run
# early to stop a vocabulary the engine would have dismantled anyway — and their
# error rate can be measured against that arithmetic rather than tuned by feel.
_LABEL_QUESTIONS: dict = {
    "discriminates_in_corpus": {
        "type": "noul",
        "instructions": {
            "question": "Does `label` distinguish the records it names from the rest "
                        "of this source?",
            "focus": "Ask whether it could be true of every record here, whatever "
                     "was done.",
        },
        "criteria": {
            "true": {
                "what": "Names something only some records in the source did",
                "examples": ["Invoice issued", "Regression reproduced",
                             "Refund authorised"],
            },
            "false": {
                "what": "Could be said of any record in the source",
                "not_for": "A label that happens to use the source's medium but "
                           "still separates records — on a desk that also escalates "
                           "and refunds, 'Replied to customer' separates",
                "examples": ["Sent", "Forwarded", "Posted", "Message logged",
                             "Commented"],
            },
        },
    },
    "discriminates_between_processes": {
        "type": "noul",
        "instructions": {
            "question": "Does `label` belong to `process` in particular, rather than "
                        "fitting every process in `vocabulary` equally?",
            "focus": "A stage of one process should read oddly as a stage of another.",
        },
        "criteria": {
            "true": {
                "what": "Reads as a stage of this process and not of the others",
                "examples": ["Onsite interview in hiring",
                             "Letter of credit drafted in credit support"],
            },
            "false": {
                "what": "A lifecycle position that fits every process listed",
                "examples": ["Requested", "Reviewed", "Approved", "Updated",
                             "Completed"],
            },
        },
    },
    "discriminates_across_runs": {
        "type": "noul",
        "instructions": {
            "question": "Is `label` a stage that many runs of `process` pass through?",
            "focus": "Not whether it is real work — whether two different runs could "
                     "both be at it.",
        },
        "criteria": {
            "true": {
                "what": "A recurring stage; different runs reach it at different times",
                "examples": ["Offer extended", "Imbalance worksheet built"],
            },
            "false": {
                "what": "Fits one particular run, deal, person or counterparty, so no "
                        "two runs can be compared through it",
                "examples": ["Promotion to Managing Director",
                             "Pre-petition exposure worksheets built from allocation "
                             "and imbalance reports"],
            },
        },
    },
}

# A label must clear every scope. The rule lives HERE, in code a reader can see
# and tune per test, rather than inside a rubric's level ordering.
#
# The bar is low on purpose, and lower than the first version's, which was set
# between invented levels and killed nine real stages on the Enron sample
# ("Deal entered in Sitara", "Travel receipts submitted"). A Noul below this is
# the model saying the property more likely does not hold than does; anything
# above it is left to the arithmetic that can actually prove it.
_LABEL_BAR = 0.35


@dataclass(frozen=True)
class Screened:
    """One label's reading, one number per scope.

    `failed` is the scope that sank it, and it is what the log quotes. The first
    version said "it names how a record was filed" whatever the score was, which
    on a label the model had judged merely generic was simply untrue.
    """

    in_corpus: Optional[float] = None
    between_processes: Optional[float] = None
    across_runs: Optional[float] = None

    _WHY = {
        "in_corpus": "it could be said of any record in this source, so it "
                     "separates nothing and leaves no order to find",
        "between_processes": "it fits every process here equally, so counting it "
                             "cannot place a run in one",
        "across_runs": "it fits one run rather than a stage many runs pass "
                       "through, so no two runs can be compared through it",
    }

    def failed(self, bar: float) -> Optional[tuple[str, float, str]]:
        """(scope, score, why) for the first scope the label fails, or None.

        A scope with no answer cannot fail: the screen only ever removes a label
        it actively judged, and silence is not a verdict.
        """
        for scope in ("in_corpus", "between_processes", "across_runs"):
            value = getattr(self, scope)
            if value is not None and value < bar:
                return scope, value, self._WHY[scope]
        return None


class LabelScreen:
    """Screens proposed step labels on the one axis the engine's arithmetic needs.

    Runs AFTER the word-count rule, never instead of it: the length test is free,
    deterministic, and is a proxy for the run-scope question below — now backed by
    it rather than standing in for it. This catches the SHORT label that fails a
    scope, which the length test waves through and which then fragments every
    variant it touches.
    """

    def __init__(self, jev=None, bar: float = _LABEL_BAR, log=None):
        self._log = log or (lambda m: None)
        if jev is None:
            from induction.jev_call import Jev
            jev = Jev(log=self._log)
        self._jev = jev
        self.bar = bar

    @property
    def available(self) -> bool:
        return self._jev.available

    def screen(self, process: str, label: str, vocabulary: Optional[list] = None
               ) -> Screened:
        """All three scopes, in ONE call — they are independent questions about one
        state, so they are evaluated in parallel and cost a single round trip.

        `vocabulary` is the other processes, because the process-scope question is
        literally "does this belong to THIS one rather than those" and cannot be
        asked without them.
        """
        state = {"process": process, "label": label}
        if vocabulary:
            state["vocabulary"] = list(vocabulary)
        answers = self._jev.ask(state, _LABEL_QUESTIONS)

        def p(name):
            a = answers.get(name)
            return a.noul if a is not None else None

        return Screened(in_corpus=p("discriminates_in_corpus"),
                        between_processes=p("discriminates_between_processes"),
                        across_runs=p("discriminates_across_runs"))

    # What a step belonging to no process is called when it is screened. The
    # vocabulary allows loose steps on purpose — "a reading is never lost for want
    # of a home" — and they need screening exactly as much as nested ones do.
    _NO_PROCESS = "(no process)"

    def cull(self, steps_by_process: dict[str, list[str]], loose: Optional[list] = None
             ) -> tuple[dict[str, list[str]], list[str], list[tuple]]:
        """Returns (kept, kept_loose, dropped); dropped entries are
        (process, label, scope, score, why).

        A label Jev could not read is KEPT, and a label is dropped only by a scope
        that actually answered — the screen can only ever remove a label it
        actively judged.
        """
        groups = dict(steps_by_process)
        if loose:
            groups[self._NO_PROCESS] = list(loose)
        processes = [name for name in groups if name != self._NO_PROCESS]
        pairs = [(proc, label) for proc, labels in groups.items() for label in labels]
        read = _fan_out(lambda pl: self.screen(pl[0], pl[1], processes), pairs)
        verdicts = {pl: r for pl, r in zip(pairs, read)}

        kept: dict[str, list[str]] = {}
        kept_loose: list[str] = []
        dropped: list[tuple] = []
        for proc, labels in groups.items():
            survivors = []
            for label in labels:
                screened = verdicts.get((proc, label))
                failure = screened.failed(self.bar) if screened is not None else None
                if failure is None:
                    survivors.append(label)
                else:
                    scope, score, why = failure
                    dropped.append((proc, label, scope, score, why))
            if proc == self._NO_PROCESS:
                kept_loose = survivors
            elif survivors:
                kept[proc] = survivors
        return kept, kept_loose, dropped


# ---------------------------------------------------------------------------
# 3. Which step — one Choice per record, over a vocabulary that exists
# ---------------------------------------------------------------------------
#
# Note what this does NOT ask. The engine's standing rule is that the model NAMES
# and the engine ASSIGNS: a run's process family is decided by `_process_of_case`,
# counting the readings of that run's own records, because answering "what process
# is this thread?" is drawing a boundary and drawing boundaries is structural work.
#
# A Choice over processes would have been exactly that question, so there isn't
# one. Every record is offered the whole `Process > Step` space at once and picks
# the one IT evidences; the run's family is then the plurality of its records'
# picks, computed by the engine as it already is. The record's own thread travels
# in the state as context — a one-line reply means what the question before it
# makes it mean — but context is all it is, and nothing is asked about it.


@dataclass
class Assignment:
    """One record's placement, with the doubt kept.

    `distribution` is the whole point. A peaked distribution is a placement; a
    flat one is the engine being able to say "this record fits several of these
    equally", which it previously could only express by dropping the record and
    counting it with the ones that fitted nothing.
    """

    process: Optional[str] = None
    step: Optional[str] = None
    confidence: Optional[float] = None
    distribution: dict[str, float] = field(default_factory=dict)

    @property
    def placed(self) -> bool:
        return self.step is not None

    def by_process(self) -> dict[str, float]:
        """The distribution summed over each process's own steps.

        Engine arithmetic over labels the model supplied — the same move
        `_process_of_case` makes, and the reason this is not the boundary
        question the engine refuses to ask. Nothing here asks which process the
        record is in; it adds up what the model said about process-qualified
        options and reads the total.
        """
        totals: dict[str, float] = {}
        for option, weight in self.distribution.items():
            process, _ = _read_pair(option)
            if process:
                totals[process] = totals.get(process, 0.0) + weight
        return totals

    @property
    def process_confidence(self) -> Optional[float]:
        totals = self.by_process()
        return max(totals.values()) if totals else None

    @property
    def ambiguous(self) -> bool:
        """True only when the record is torn WITHIN one process.

        The first version compared one flat number against one bar, across every
        process's steps at once. On the Enron sample that is a single Choice over
        28 options spanning 7 processes, and it held back 75 of 184 records —
        most of them not torn at all, merely spread.

        Spread across processes and torn within one are different findings and
        deserve different handling. A record whose mass sits firmly in one
        process, split between two of ITS steps, is exactly the record the
        generative pass should read: it will settle the step and quote the span.
        A record whose mass is spread across several processes has nothing for
        that pass to settle, and is the honest ambiguous case.

        With no distribution to add up — an API that returns a pick and no
        probabilities — this falls back to the pick's own confidence, which is
        the most that can be said from what came back.
        """
        if not self.placed:
            return False
        totals = self.by_process()
        if not totals:
            return (self.confidence or 0.0) < _AMBIGUOUS_BAR
        return max(totals.values()) < _PROCESS_BAR

    def top(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.distribution.items(), key=lambda kv: -kv[1])[:n]


# How much of a record's probability mass must land in ONE process before the
# record is worth reading. Below it the record fits no family in particular, and
# there is nothing for the generative pass to settle.
#
# Deliberately not a bar on the winning STEP: a step bar is a bar on how finely
# the vocabulary was cut, so a corpus with six steps per process would be judged
# more doubtful than one with three for no reason but its own detail.
_PROCESS_BAR = 0.55

# Used only when an answer carries a pick and no distribution to add up.
_AMBIGUOUS_BAR = 0.55

# How a (process, step) pair is spelled as one Choice option, and read back.
_PAIR_SEP = " > "

# Named once, because the ledger has to record the question exactly as the model
# was asked it — a paraphrase in the artefact would be a different question.
_STEP_QUESTION = "Which stage does `record` perform?"


def _pair_options(vocab) -> dict[str, None]:
    """The whole `Process > Step` space as Choice options.

    Flat on the wire, nested in meaning: the separator is what lets the pick be
    read back as a (process, step) pair, and it is why a step can never be
    attached to a process it does not belong to — a combination that is not a
    real pair is not on the list, so it cannot be chosen. That is the same
    guarantee `_clean_readings` enforces after the fact for the generative tier,
    except here it holds by construction.
    """
    options: dict[str, None] = {}
    for process, steps in vocab.steps_by_process.items():
        for step in steps:
            options[f"{process}{_PAIR_SEP}{step}"] = None
    for step in getattr(vocab, "loose", []):
        options[step] = None
    return options


def _read_pair(picked: str) -> tuple[Optional[str], str]:
    if _PAIR_SEP in picked:
        process, step = picked.split(_PAIR_SEP, 1)
        return process, step
    return None, picked


class RecordAssigner:
    """Labels records with steps, over a vocabulary discovery has already produced.

    This narrows what the generative pass is asked; it does not replace it. A
    record placed here still has no span, and a reading without the text that
    justifies it is exactly what `_clean_readings` refuses to keep. What it buys
    is that the expensive pass reads the records that have a place to go, and
    that a record with no clear place says so with numbers instead of vanishing.
    """

    def __init__(self, jev=None, bar: float = _AMBIGUOUS_BAR, log=None):
        self._log = log or (lambda m: None)
        if jev is None:
            from induction.jev_call import Jev
            jev = Jev(log=self._log)
        self._jev = jev
        self.bar = bar

    @property
    def available(self) -> bool:
        return self._jev.available

    def step_of(self, record_state, vocab, context: Optional[list] = None) -> Assignment:
        """Which `Process > Step` does THIS record perform?"""
        options = _pair_options(vocab)
        if not options:
            return Assignment()
        state = {"record": record_state}
        if context:
            # The run around it, so a short reply can be read. Trimmed hard: this
            # is context for one question, not a second record to classify.
            state["thread_context"] = context[:12]
        questions = {
            "step": {
                "type": "choice",
                "instructions": {
                    "question": _STEP_QUESTION,
                    "focus": "The stage the record ACCOMPLISHES, not one it merely "
                             "mentions. Use `thread_context` to read a short reply.",
                },
                "criteria": options,
            },
        }
        a = self._jev.ask(state, questions).get("step")
        if a is None or a.choice is None:
            return Assignment()
        process, step = _read_pair(a.choice)
        return Assignment(process=process, step=step, confidence=a.confidence,
                          distribution=a.probabilities)


# ---------------------------------------------------------------------------
# The facade the reading tier actually holds
# ---------------------------------------------------------------------------

# A record must be at least this sure it carries the work before it is allowed
# into the 150 the vocabulary is discovered from. The bar is a judgement, not a
# measurement — and it applies only when enough records clear it, because a
# sample chosen for signal is worthless if choosing it left too few records to
# see the corpus's range.
_SAMPLE_BAR = 0.50


class JevReading:
    """The three typed passes, bundled, with every count they produce.

    `abstraction.py` holds one of these or None, and calls five methods on it.
    Each one returns the thing it was given, unchanged, when Jev cannot answer —
    so an unreachable service, a missing key or a tripped breaker leaves the
    reading tier exactly as it was before this existed. That is the single
    property worth checking in review: none of these can subtract.
    """

    def __init__(self, jev=None, log=None, signals=True, labels=True, assign=True,
                 ledger=None):
        self._log = log or (lambda m: None)
        if jev is None:
            from induction.jev_call import Jev
            jev = Jev(log=self._log)
        self.jev = jev
        # Where recordable decisions go. The gate is deliberately NOT recorded
        # per record: it asserts nothing about any record, only about what was
        # worth reading, and one row per gated record would bury the decisions a
        # reader can actually dispute. It stays a count.
        self.ledger = ledger
        self.signals = SignalFilter(jev=jev, log=self._log) if signals else None
        self.labels = LabelScreen(jev=jev, log=self._log) if labels else None
        self.assigner = RecordAssigner(jev=jev, log=self._log) if assign else None
        self.n_gated = 0
        self.n_ambiguous = 0
        self.n_labels_dropped = 0

    @property
    def available(self) -> bool:
        return self.jev.available

    # -- 1. gate ------------------------------------------------------------
    def gate(self, abstraction, records: list, events, m, log) -> tuple[list, dict]:
        """Hold back records that perform no step. Returns (kept, signals)."""
        if self.signals is None or not self.signals.available:
            return records, {}
        from induction.abstraction import _record_state

        entities = {e.id: e for e in m.shaped.entities}
        by_event = {e.id: e for e in events}
        states, asked = [], []
        for r in records:
            ev = by_event.get(r["id"])
            if ev is None:
                continue
            states.append(_record_state(ev, m, entities))
            asked.append(r)
        signals = {r["id"]: s for r, s in zip(asked, self.signals.read_all(states))}

        kept, held = [], []
        for r in records:
            signal = signals.get(r["id"], Signal())
            (kept if signal.worth_reading(self.signals.bar) else held).append(r)
        if held:
            abstraction.gated = [
                {"id": r["id"],
                 "reason": "performs no step of any process (a greeting, a bare "
                           "forward, an acknowledgement)",
                 "performs_action": signals[r["id"]].performs_action,
                 "machine_generated": signals[r["id"]].machine_generated}
                for r in held]
            self.n_gated = len(held)
            log(f"abstraction: {len(held)} of {len(records)} records gated as "
                f"performing no step — not read, not deleted, and listed with the "
                f"number that held them back")
        return kept, signals

    # -- 2. sample ----------------------------------------------------------
    def sample_pool(self, distinct: list, signals: dict, log) -> list:
        """Narrow the discovery pool to records that carry the work.

        The pool is narrowed, never reordered: `_spread` then takes its even
        stride across the corpus as it always did. Order is what stops one corner
        of the corpus naming the processes for all of it, and a ranking by score
        would quietly undo that — the two fixes are for different faults and both
        are needed.

        If too few records clear the bar the whole pool is kept. A sample chosen
        for signal that is too small to show the corpus's range is a worse sample,
        not a better one.
        """
        if not signals:
            return distinct
        strong = [r for r in distinct
                  if (signals.get(r["id"], Signal()).carries_work or 0.0) >= _SAMPLE_BAR]
        from induction.abstraction import _DISCOVERY_SAMPLE
        if len(strong) < _DISCOVERY_SAMPLE or len(strong) == len(distinct):
            return distinct
        log(f"abstraction: discovery sample drawn from the {len(strong)} records that "
            f"carry the work, not all {len(distinct)} — same even stride, fewer envelopes")
        return strong

    # -- 3. labels ----------------------------------------------------------
    def screen_labels(self, vocab, log, abstraction=None):
        """Drop proposed steps that name the envelope rather than an achievement."""
        if self.labels is None or not self.labels.available or not vocab:
            return vocab
        from induction.abstraction import ReadVocabulary

        kept, kept_loose, dropped = self.labels.cull(vocab.steps_by_process,
                                                     list(vocab.loose))
        if not dropped:
            return vocab
        self.n_labels_dropped = len(dropped)
        self._record_labels(dropped)
        if abstraction is not None:
            abstraction.dropped_labels = [
                {"process": process, "label": label, "scope": scope,
                 "score": score, "why": why}
                for process, label, scope, score, why in dropped]
        for process, label, scope, score, why in dropped:
            # The line names the scope that failed and says what that costs. A
            # single fixed sentence for every drop, as the first version had, was
            # wrong about most of them.
            log(f"[abstraction] dropped step {label!r} from {process}: "
                f"{scope.replace('_', ' ')} {score:.2f} — {why}")
        return ReadVocabulary(steps_by_process=kept, loose=kept_loose)

    def _record_labels(self, dropped: list) -> None:
        """A dropped label is the hardest decision to notice, so it is recorded.

        A step that survives is visible on the page; a step that was proposed and
        removed leaves no trace at all, and its absence silently reshapes every
        flow the reader is looking at. This is the decision most worth being able
        to argue with.
        """
        if self.ledger is None:
            return
        from induction.decisions import LABEL, Decision

        for process, label, scope, score, why in dropped:
            self.ledger.add(Decision(
                kind=LABEL,
                about=f"{process} > {label}",
                question=_LABEL_QUESTIONS[f"discriminates_{scope}"]
                         ["instructions"]["question"],
                qtype="noul",
                answer=round(score, 4),
                confidence=max(score, 1.0 - score),
                outcome=f"dropped from the vocabulary — {why}",
            ))

    # -- 4. place -----------------------------------------------------------
    def place(self, abstraction, threads: list, vocab, events, m, log) -> tuple[list, dict]:
        """Label each record with a step; hold back the ones nothing clearly fits.

        Returns (threads that still have records, {record id: Assignment}). A
        thread emptied by this is dropped from the batches, not from the corpus:
        its records are in `abstraction.ambiguous`, each with the distribution
        that made it ambiguous.
        """
        if self.assigner is None or not self.assigner.available or not threads:
            return threads, {}
        from induction.abstraction import _record_state

        entities = {e.id: e for e in m.shaped.entities}
        by_event = {e.id: e for e in events}
        placements: dict = {}
        ambiguous: list[dict] = []
        kept_threads: list[dict] = []

        # Every record in the corpus is one independent question given the
        # vocabulary, so they all go out in ONE fan-out rather than a fan-out per
        # thread. Fanning out per thread would serialise the corpus at the thread
        # boundary and spend most of the run waiting on threads of two records.
        # The thread is still what supplies each record's context; it just stops
        # being what paces the calls.
        jobs: list[tuple] = []
        for thread in threads:
            states, asked = [], []
            for r in thread["records"]:
                ev = by_event.get(r["id"])
                if ev is None:
                    continue
                states.append(_record_state(ev, m, entities))
                asked.append(r)
            if not asked:
                kept_threads.append(thread)
                continue
            context = [st.get("subject") or st.get("title") or st.get("body", "")[:120]
                       for st in states][:12]
            jobs.append((thread, asked, states, context))

        flat = [(st, ctx) for _, _, states, ctx in jobs for st in states]
        answered = _fan_out(lambda pair: self.assigner.step_of(pair[0], vocab, pair[1]), flat)
        cursor = 0
        for thread, asked, states, _ in jobs:
            results = [a if a is not None else Assignment()
                       for a in answered[cursor:cursor + len(states)]]
            cursor += len(states)
            kept_records = []
            for r, placed in zip(asked, results):
                placements[r["id"]] = placed
                self._record_step(r["id"], placed)
                if placed.placed and placed.ambiguous:
                    # Answered, and the answer was a tie. This is the only case
                    # that holds a record back.
                    ambiguous.append({"id": r["id"], "top": placed.top(3),
                                      "confidence": placed.confidence})
                else:
                    # Placed confidently, OR not answered at all. An unanswered
                    # record goes to the generative pass exactly as it would have
                    # without this module — silence is not a verdict, and a Jev
                    # that cannot answer must never be able to empty the corpus.
                    kept_records.append(r)
            if kept_records:
                kept_threads.append({"id": thread["id"], "records": kept_records})

        if ambiguous:
            abstraction.ambiguous = ambiguous
            self.n_ambiguous = len(ambiguous)
        n_before = sum(len(t["records"]) for t in threads)
        n_after = sum(len(t["records"]) for t in kept_threads)
        if n_after < n_before:
            log(f"abstraction: the typed tier placed {n_after} of {n_before} records "
                f"confidently; {len(ambiguous)} fit several steps equally and are "
                f"listed with their distribution rather than read")
        return kept_threads, placements

    def _record_step(self, record_id: str, placed: "Assignment") -> None:
        """What the record was offered, what it picked, and what happened next.

        `derived` carries the per-process mass this code adds up afterwards,
        separately from `distribution`, which is what the model actually said.
        The two are different claims and a reader should be able to tell them
        apart: one is an answer, the other is arithmetic over it.
        """
        if self.ledger is None or not placed.placed:
            return
        from induction.decisions import STEP, Decision

        if placed.ambiguous:
            outcome = ("held back as ambiguous — no process held a clear majority "
                       "of the mass, so there was nothing for the reading pass to "
                       "settle; shown with its distribution rather than read")
        else:
            outcome = ("sent to the generative pass, which decides the step and "
                       "quotes the text it read it from")
        self.ledger.add(Decision(
            kind=STEP,
            about=record_id,
            question=_STEP_QUESTION,
            qtype="choice",
            answer=f"{placed.process}{_PAIR_SEP}{placed.step}" if placed.process else placed.step,
            confidence=placed.confidence,
            distribution=dict(placed.distribution),
            derived=placed.by_process(),
            outcome=outcome,
        ))

    # -- 5. report ----------------------------------------------------------
    def report(self, abstraction, log) -> None:
        """One line, so no gate in this module is ever silent."""
        counts = {"gated": self.n_gated, "ambiguous": self.n_ambiguous,
                  "labels_dropped": self.n_labels_dropped,
                  "calls": self.jev.calls, "skipped": self.jev.skipped}
        abstraction.jev_counts = counts
        log(f"jev: {counts['calls']} decision call(s) · {self.n_gated} records gated · "
            f"{self.n_ambiguous} ambiguous · {self.n_labels_dropped} label(s) dropped"
            + (f" · {counts['skipped']} call(s) failed" if counts["skipped"] else ""))
