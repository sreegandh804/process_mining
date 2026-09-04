"""The pipeline — wires steps 0-6 into one induced model.

Order of operations (and why):
  shape      : adapters turn raw -> canonical Entity/Event/Observation + Links.
  correlate  : group records into cases (the graded core).
  order      : sort each case into a trace; flag unknowable order.
  label      : name activities; merge same-activity-different-people.
  segment    : split cases into *kinds*; compute variants per kind.
  reject     : flag look-alike non-processes.
  gaps       : infer off-system steps from discontinuities.
  orphans    : collect everything that joined to nothing.

Everything from `correlate` onward is `induce()`, and it is **the same code for
every source**. A source contributes a `Shaped` bundle and nothing else — no
correlator, no pipeline of its own. `run_pipeline` and `run_tabular_pipeline`
below are thin *loaders*: they know how to find and shape a corpus, then hand it
to the identical core. That is what makes "a new customer is an adapter" true
rather than aspirational; mixing sources is then just concatenating them, which
is exactly how cross-source fuzzy correlation gets its chance to fire.

We keep `shaped` (the substrate, with `raw` intact) beside the induced model so
the divergence hook (belief vs data) has both sides to compare later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from induction.adapters import Shaped
from induction.honesty import apply_reject, collect_orphans
from induction.process import Case, Gap, Orphan, ProcessKind, Step
from induction.steps.correlate import Correlation, CorrelationPolicy, correlate
from induction.steps.order import order, order_observations
from induction.steps.segment import segment


@dataclass
class InducedModel:
    slug: str
    manifest: dict
    shaped: Shaped
    correlation: Correlation
    profile_id: str = "generic"
    kinds: list[ProcessKind] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    merges: list = field(default_factory=list)         # ActivityMerge (same-activity-different-people)
    gaps: list[Gap] = field(default_factory=list)
    orphans: list[Orphan] = field(default_factory=list)

    @property
    def cases(self) -> dict[str, Case]:
        return self.correlation.cases


def induce(shaped: Shaped, slug: str = "model", profile=None, manifest: Optional[dict] = None,
           policy: Optional[CorrelationPolicy] = None,
           gap_detectors: Optional[Iterable[Callable]] = None,
           terminal_action: str = "paid", corroborating_action: str = "settled",
           progress=None) -> InducedModel:
    """Steps 2-6 over an already-shaped corpus. Source-agnostic by construction.

    `shaped` may be one adapter's output or several concatenated — the correlator
    sees one record stream either way, which is the only reason a mail thread can
    join to a ledger row. `progress` (a `Progress`; default silent) reports each
    stage, and at verbose level streams the individual joins/rejects/gaps.
    """
    from induction.profiles import GENERIC_PROFILE, select_profile
    from induction.progress import NULL
    prog = progress or NULL
    if profile is None:
        profile = GENERIC_PROFILE
    elif profile == "auto":
        profile = select_profile(shaped)

    n_records = sum(1 for e in shaped.entities if e.type != "person")
    prog.stage(f"correlate: grouping {n_records} records into cases …")
    corr = correlate(shaped, policy)
    prog.stage(f"correlate: → {len(corr.cases)} cases")
    _stream_inferred_joins(prog, corr)

    order(shaped, corr)
    order_observations(shaped.observations, corr)
    unknown = sum(1 for c in corr.cases.values() if c.order_status == "unknown")
    prog.stage(f"order: {len(corr.cases) - unknown} ordered · {unknown} order-unknown")

    from induction.steps.label import label as label_step
    label_result = label_step(shaped, corr, profile)
    prog.stage(f"label: {len(label_result.steps)} activities · "
               f"{len(label_result.merges)} same-activity merges")

    kinds = segment(shaped, corr, profile)
    apply_reject(kinds, profile)
    prog.stage(f"segment: {len(kinds)} kind(s)")
    for k in kinds:
        if prog.verbose:
            common = next((v for v in k.variants if v.role == "common"), None)
            cp = " → ".join(common.signature) if common and common.signature else "—"
            prog.detail(f"kind {k.name}: {len(k.case_ids)} runs, {len(k.variants)} variants "
                        f"| common: {cp}")
        if k.rejected:
            prog.detail(f"rejected {k.name}: {k.reject_reason}")

    gaps: list[Gap] = []
    for detector in (gap_detectors if gap_detectors is not None
                     else _default_gap_detectors(slug, terminal_action, corroborating_action)):
        gaps.extend(detector(shaped, corr, kinds))
    for g in gaps:
        prog.detail(f"gap {g.kind} in {g.case_id}: {g.description}")

    orphans = collect_orphans(shaped, corr)
    prog.stage(f"gaps: {len(gaps)} inferred · orphans: {len(orphans)}")
    for o in orphans:
        prog.detail(f"orphan {o.record_id}: {o.reason}")
    return InducedModel(
        slug=slug, manifest=manifest or {}, shaped=shaped, correlation=corr,
        profile_id=getattr(profile, "id", "generic"),
        kinds=kinds, steps=label_result.steps, merges=label_result.merges,
        gaps=gaps, orphans=orphans,
    )


def _stream_inferred_joins(prog, corr) -> None:
    """At verbose level, name each case assembled through an *inferred* join — the
    fuzzy/model correlations a reader most wants to watch and audit (deterministic
    joins are the many, obvious ones, so they are not streamed)."""
    if not prog.verbose:
        return
    shown = 0
    for c in corr.cases.values():
        tier = c.confidence.tier.label
        if tier in ("heuristic", "model") and c.confidence.rationale:
            prog.detail(f"correlate [{tier}] {c.id}: {c.confidence.rationale}")
            shown += 1
            if shown >= 100:
                prog.detail("correlate: … (further inferred joins omitted)")
                break


def _default_gap_detectors(slug: str, terminal_action: str, corroborating_action: str):
    """Every detector, every run.

    A detector is an inference *rule*, not a source: each one is a no-op on data
    that lacks its signal (the off-platform-review rule only looks at cases whose
    anchor is a pull request, the reconciliation rule only at terminal states
    that never got corroborated). So they all run and the data decides which
    fire — rather than the caller having to know which source it is holding.
    """
    from induction.steps.gaps import infer_gaps
    from induction.steps.gaps_generic import infer_missing_step_gaps, infer_reconciliation_gaps

    return [
        lambda shaped, corr, kinds: infer_gaps(shaped, corr, slug),
        lambda shaped, corr, kinds: infer_missing_step_gaps(corr, kinds),
        lambda shaped, corr, kinds: infer_reconciliation_gaps(
            shaped, corr, kinds, terminal_action=terminal_action,
            corroborating_action=corroborating_action),
    ]


# ---------------------------------------------------------------------------
# Loaders — the only per-source code, and it only knows how to *read* a corpus
# ---------------------------------------------------------------------------

def run_pipeline(slug: str = "pallets/flask", raw_dir: str = "data/raw",
                 with_thin: bool = True, with_github: bool = False,
                 profile=None, progress=None) -> InducedModel:
    """Git history (+ its changelog, + its GitHub Issues/PR corpus) -> induced model.

    Adding a source here is one `extend` call and nothing else — no branch
    downstream, no correlator, no second pipeline. That is the whole return on
    making adapters declare links instead of correlating.
    """
    # DEFAULT is the generic, source-agnostic profile: unnamed kinds + activities,
    # data-derived rationales. A caller can pass a matching profile (e.g. the git
    # one) purely to make a familiar corpus read nicely — it never changes the
    # structure, only the vocabulary. `profile="auto"` picks one from the source.
    from induction.adapters import git_history
    raw_dir_p = Path(raw_dir)
    key = slug.replace("/", "__")
    manifest_path = raw_dir_p / f"{key}.manifest.json"
    manifest = {}
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text())

    shaped = Shaped()
    if with_github:
        # Loaded FIRST so that its real PR/issue records exist before git's
        # references to them are resolved: the reference then resolves to the
        # record instead of materialising an inferred stub beside it.
        from induction.adapters import github_api
        shaped.extend(github_api.load(raw_dir, slug))
        manifest["with_github"] = True
    shaped.extend(git_history.load(raw_dir, slug))
    if with_thin:
        from induction.adapters import changelog
        shaped.extend(changelog.load(raw_dir, slug))

    return induce(shaped, slug=slug, profile=profile, manifest=manifest, progress=progress)


def run_tabular_pipeline(sources, slug="tabular", profile=None,
                         terminal_action="paid", corroborating_action="settled",
                         max_cases=None, progress=None) -> InducedModel:
    """Spreadsheets -> induced model.

    `sources` : list of (induction.adapters.tabular.TableSpec, path-to-csv-or-xlsx).
    """
    from induction.adapters import tabular

    shaped = Shaped()
    sheets = []
    for spec, path in sources:
        # The two tabular shapes read differently and produce the same records:
        # one row per case with a column per step, or one row per event.
        if isinstance(spec, tabular.EventLogSpec):
            shaped.extend(tabular.load_event_log(spec, path, max_cases=max_cases))
        else:
            shaped.extend(tabular.load(spec, path))
        sheets.append({"source": spec.source, "path": str(path)})

    manifest = {
        "source_kind": "spreadsheet",
        "sheets": sheets,
        "n_rows": len([e for e in shaped.entities if e.type not in ("person", "orphan_row")]),
    }
    return induce(shaped, slug=slug, profile=profile, manifest=manifest,
                  terminal_action=terminal_action,
                  corroborating_action=corroborating_action, progress=progress)
