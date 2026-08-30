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
    kinds: list[ProcessKind] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    merges: list = field(default_factory=list)         # ActivityMerge (same-activity-different-people)
    gaps: list[Gap] = field(default_factory=list)
    orphans: list[Orphan] = field(default_factory=list)

    @property
    def cases(self) -> dict[str, Case]:
        return self.correlation.cases


def run_pipeline(slug: str = "pallets/flask", raw_dir: str = "data/raw",
                 with_thin: bool = True) -> InducedModel:
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
    label_result = label(shaped, corr)

    # --- segment (step 0) + variants (step 4) ---
    kinds = segment(shaped, corr)

    # --- reject (§6): flag look-alike non-processes ---
    apply_reject(kinds, shaped, corr)

    # --- gaps (step 6): infer off-system steps ---
    from induction.steps.gaps import infer_gaps
    gaps = infer_gaps(shaped, corr, slug)

    # --- orphans (§6) ---
    orphans = collect_orphans(shaped, corr)

    return InducedModel(
        slug=slug, manifest=manifest, shaped=shaped, correlation=corr,
        kinds=kinds, steps=label_result.steps, merges=label_result.merges,
        gaps=gaps, orphans=orphans,
    )
