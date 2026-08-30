"""Step 0 — Segment: separate the different *kinds* of process.

Hold the line the reviewers will probe:
  - Segment (step 0) = different *kinds* of process (contribution vs release vs
    dependency bumps) — distinct processes.
  - Variant (step 4) = different *runs of one kind*.

Nothing in the data announces where one kind ends and the next begins, so every
boundary here is an **inference**, tier `heuristic`, and correctable. We do not
inflate it: on a single repo the kinds are few and mostly legible from who acts
and what they touch. Embeddings could *propose* finer boundaries — that is the
documented upgrade path, deliberately not built (similar text != same process).

Baseline rules, strongest signal first:
  release_and_backport : the case is a release/integration train.
  dependency_bumps     : commits authored by a dependency bot (dependabot, ...).
  ci_maintenance       : commits authored by a CI bot (pre-commit-ci, ...).
  documentation        : commits that only ever touch docs.
  code_contribution    : everything else — the actual PR/issue contribution flow.
"""

from __future__ import annotations

from collections import defaultdict

from induction.adapters import Shaped
from induction.model import Entity, heuristic
from induction.process import Case, ProcessKind
from induction.steps.correlate import Correlation
from induction.steps.variants import induced_variants

_KIND_NAMES = {
    "code_contribution": "Code contribution (feature / fix)",
    "release_and_backport": "Release & backport",
    "dependency_bumps": "Dependency bumps (bot)",
    "ci_maintenance": "CI / tooling maintenance (bot)",
    "documentation": "Documentation change",
    "release_notes": "Release notes (thin source — order unknown)",
}
_KIND_RATIONALE = {
    "code_contribution": "PR/issue runs whose commits change source or tests and are authored by people.",
    "release_and_backport": "Release-train and backport integration merges (long-lived release branches).",
    "dependency_bumps": "Recurring commits authored by a dependency bot — candidate 'looks like a process, isn't'.",
    "ci_maintenance": "Recurring commits authored by a CI/formatting bot — candidate 'looks like a process, isn't'.",
    "documentation": "Runs whose commits only ever touch documentation files.",
    "release_notes": "Changelog version sections that did not correlate to a git case — thin, order-unknown observations.",
}

_DOC_SUFFIXES = (".rst", ".md")


def _is_doc(path: str) -> bool:
    return path.startswith("docs/") or path.endswith(_DOC_SUFFIXES)


def _classify(case: Case, commit_ents: list[Entity], people: dict[str, Entity]) -> str:
    if case.kind_hint == "notes":
        return "release_notes"
    if case.kind_hint in ("release", "integration"):
        return "release_and_backport"

    if not commit_ents:
        return "code_contribution"

    # Who authored the work, and are they bots?
    bot_domains = defaultdict(int)
    human = 0
    all_files: list[str] = []
    for ce in commit_ents:
        all_files.extend(ce.attrs.get("files", []))
        # the commit's author is recoverable from its raw record
        raw = ce.raw or {}
        name = (raw.get("author", {}) or {}).get("name", "").lower()
        email = (raw.get("author", {}) or {}).get("email", "").lower()
        blob = f"{name} {email}"
        if "dependabot" in blob or "renovate" in blob:
            bot_domains["dependency"] += 1
        elif "pre-commit-ci" in blob or "github-actions" in blob:
            bot_domains["ci"] += 1
        elif "[bot]" in blob:
            bot_domains["other_bot"] += 1
        else:
            human += 1

    total = len(commit_ents)
    if bot_domains.get("dependency", 0) > total / 2:
        return "dependency_bumps"
    if bot_domains.get("ci", 0) > total / 2:
        return "ci_maintenance"

    if all_files and all(_is_doc(f) for f in all_files):
        return "documentation"

    return "code_contribution"


def segment(shaped: Shaped, corr: Correlation) -> list[ProcessKind]:
    entities_by_id = {e.id: e for e in shaped.entities}
    people = {e.id: e for e in shaped.entities if e.type == "person"}

    kind_cases: dict[str, list[str]] = defaultdict(list)
    for case in corr.cases.values():
        commit_ents = [entities_by_id[eid] for eid in case.entity_ids
                       if eid.startswith("commit:") and eid in entities_by_id]
        kind_id = _classify(case, commit_ents, people)
        kind_cases[kind_id].append(case.id)

    kinds: list[ProcessKind] = []
    for kind_id, case_ids in kind_cases.items():
        variants, dfg = induced_variants(case_ids, corr.cases)
        steps_seen = sorted({a for v in variants for a in v.signature})
        kinds.append(ProcessKind(
            id=kind_id,
            name=_KIND_NAMES.get(kind_id, kind_id),
            rationale=_KIND_RATIONALE.get(kind_id, ""),
            confidence=heuristic("segment boundaries are inferred from actor/files/anchor, not read"),
            case_ids=case_ids,
            variants=variants,
            dfg=dfg,
            steps=steps_seen,
        ))
    # Biggest kinds first — the common processes read loudest.
    kinds.sort(key=lambda k: -len(k.case_ids))
    return kinds
