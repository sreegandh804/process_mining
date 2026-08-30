"""The pipeline — wires steps 0-6 into one induced model.

Order of operations (and why):
  shape      : adapters turn raw -> canonical Entity/Event/Observation.
  correlate  : group events into cases (the graded core).
  order      : sort each case into a trace; flag unknowable order.
  label      : name activities; merge same-activity-different-people.
  segment    : split cases into *kinds*; compute variants per kind.
  reject     : flag look-alike non-processes.
  gaps       : infer off-system steps from discontinuities.
  orphans    : collect everything that joined to nothing.

We keep `shaped` (the substrate, with `raw` intact) beside the induced model so
the divergence hook (belief vs data) has both sides to compare later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from induction.adapters import Shaped, git_history
from induction.honesty import apply_reject, collect_orphans
from induction.process import Case, Gap, Orphan, ProcessKind, Step
from induction.steps.correlate import Correlation, correlate
from induction.steps.order import order
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


def run_pipeline(slug: str = "pallets/flask", raw_dir: str = "data/raw",
                 with_thin: bool = True, profile=None) -> InducedModel:
    # DEFAULT is the generic, source-agnostic profile: unnamed kinds + activities,
    # data-derived rationales. A caller can pass a matching profile (e.g. the git
    # one) purely to make a familiar corpus read nicely — it never changes the
    # structure, only the vocabulary. `profile="auto"` picks one from the source.
    from induction.profiles import GENERIC_PROFILE
    if profile is None:
        profile = GENERIC_PROFILE
    raw_dir_p = Path(raw_dir)
    key = slug.replace("/", "__")
    manifest_path = raw_dir_p / f"{key}.manifest.json"
    manifest = {}
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text())

    # --- shape: thick source (git) ---
    shaped = git_history.load(raw_dir, slug)

    # --- shape: thin source (changelog) — real Observations through the same pipe ---
    if with_thin:
        from induction.adapters import changelog
        thin = changelog.load(raw_dir, slug)
        shaped.extend(thin)

    # `auto` picks a vocabulary from the source now that we have shaped records.
    if profile == "auto":
        from induction.profiles import select_profile
        profile = select_profile(shaped)

    # --- correlate (step 2) ---
    corr = correlate(shaped, slug)
    if with_thin:
        from induction.adapters import changelog as _cl
        _cl.correlate_thin(shaped, corr, slug)

    # --- order (step 3) ---
    order(shaped, corr)
    if with_thin:
        from induction.steps.order import order_observations
        order_observations(shaped.observations, corr)

    # --- label (step 5) ---
    from induction.steps.label import label
    label_result = label(shaped, corr, profile)

    # --- segment (step 0) + variants (step 4) ---
    kinds = segment(shaped, corr, profile)

    # --- reject (§6): flag look-alike non-processes ---
    apply_reject(kinds, profile)

    # --- gaps (step 6): infer off-system steps ---
    from induction.steps.gaps import infer_gaps
    gaps = infer_gaps(shaped, corr, slug)

    # --- orphans (§6) ---
    orphans = collect_orphans(shaped, corr)

    return InducedModel(
        slug=slug, manifest=manifest, shaped=shaped, correlation=corr,
        profile_id=getattr(profile, "id", "generic"),
        kinds=kinds, steps=label_result.steps, merges=label_result.merges,
        gaps=gaps, orphans=orphans,
    )


def run_tabular_pipeline(sources, slug="tabular", profile=None,
                         terminal_action="paid", corroborating_action="settled") -> InducedModel:
    """The non-git path: a list of (TableSpec, path) spreadsheet sources through
    the SAME order/label/segment/reject/orphan machinery, with the generic
    identity/foreign-key correlator and generic gap detectors.

    `sources` : list of (induction.adapters.tabular.TableSpec, path-to-csv-or-xlsx).
    """
    from induction.adapters import tabular
    from induction.profiles import GENERIC_PROFILE
    from induction.steps.correlate_generic import correlate_by_key
    from induction.steps.gaps_generic import infer_missing_step_gaps, infer_reconciliation_gaps
    from induction.steps.label import label as label_step
    from induction.steps.order import order as order_step, order_observations

    if profile is None:
        profile = GENERIC_PROFILE

    shaped = Shaped()
    sheets = []
    for spec, path in sources:
        shaped.extend(tabular.load(spec, path))
        sheets.append({"source": spec.source, "path": str(path)})

    corr = correlate_by_key(shaped)
    order_step(shaped, corr)
    order_observations(shaped.observations, corr)

    label_result = label_step(shaped, corr, profile)
    kinds = segment(shaped, corr, profile)
    apply_reject(kinds, profile)

    gaps = infer_missing_step_gaps(corr, kinds)
    gaps += infer_reconciliation_gaps(shaped, corr, kinds,
                                      terminal_action=terminal_action,
                                      corroborating_action=corroborating_action)
    orphans = collect_orphans(shaped, corr)

    manifest = {
        "source_kind": "spreadsheet",
        "sheets": sheets,
        "n_rows": len([e for e in shaped.entities if e.type not in ("person", "orphan_row")]),
    }
    return InducedModel(
        slug=slug, manifest=manifest, shaped=shaped, correlation=corr,
        profile_id=getattr(profile, "id", "generic"),
        kinds=kinds, steps=label_result.steps, merges=label_result.merges,
        gaps=gaps, orphans=orphans,
    )
