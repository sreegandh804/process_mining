"""Domain profiles — where source-specific *vocabulary* lives.

The point of the three canonical types is that the pipeline downstream of an
adapter is source-agnostic. But two things are genuinely domain knowledge and
were, wrongly, hardcoded into shared steps: **what activities are called**
(Label) and **which kinds of process exist** (Segment). A git repo's
"Merge pull request" and "dependency bump" mean nothing to an accounting firm.

So that vocabulary moves here, into a `Profile` matched to the source. The
shared steps call the profile; they never hardcode a domain word.

- With **no profile** (`GENERIC_PROFILE`), the engine still runs on *any* data:
  activities are named by their raw action, and kinds are discovered by
  structural clustering and left **unnamed** (`kind_1`, `kind_2`, …) with a
  data-derived rationale. That is the honest default — we found distinct
  shapes; we will not invent domain names for them.
- With a **matching profile** (`GIT_PROFILE`), the same clusters get real names
  and the domain's own "looks like a process, isn't" rule. Adding a customer =
  adding an adapter + a profile, not touching the engine.

A profile is just data + small functions; nothing here reaches into the
pipeline's control flow.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional


def _humanize(action: str) -> str:
    return action.replace("_", " ").strip().capitalize() or action


def _generic_reject(kf: dict) -> Optional[str]:
    """Domain-free reject rule: a cluster performed entirely by automated actors
    that recurs is a strong look-alike candidate. We cannot prove it "produces
    nothing" without domain knowledge, so we say exactly that."""
    if kf.get("automated") and kf.get("n_cases", 0) > 1:
        return (f"Every run in this cluster is performed by automated actors and it "
                f"recurs ({kf['n_cases']} runs). Strong 'looks like a process, isn't' "
                f"candidate; a source profile would confirm whether it produces "
                f"anything of value. Flagged, not deleted.")
    return None


@dataclass(frozen=True)
class Profile:
    id: str
    # name an activity (a Step) from its raw action verb. The DEFAULT is the
    # identity: the activity is called what the source's own action says it is
    # ("authored", "invoice_approved"), untouched. Naming is deferred, on purpose.
    label_action: Callable[[str], str] = lambda a: a
    # per-case domain features (subkind, object families, …); default: none
    case_features: Callable[..., dict] = lambda case, entities: {}
    # refine the generic cluster key with domain features; default: keep generic
    cluster_key: Callable[[tuple, dict], tuple] = lambda generic_key, cf: generic_key
    # name a discovered kind from its aggregated features -> a stable id, or None
    name_kind: Callable[[dict], Optional[str]] = lambda kf: None
    # pretty display name for a named kind id
    display_name: Callable[[str], Optional[str]] = lambda kid: None
    # a domain rationale for a kind; default: fall back to the data-derived one
    rationale: Callable[[dict], Optional[str]] = lambda kf: None
    # is this kind a look-alike non-process? return a reason or None
    reject_reason: Callable[[dict], Optional[str]] = _generic_reject


GENERIC_PROFILE = Profile(id="generic")


# ---------------------------------------------------------------------------
# The git/GitHub profile — the vocabulary that used to be hardcoded in steps.
# ---------------------------------------------------------------------------
_GIT_ACTIVITY_NAMES = {
    "authored": "Author change",
    "committed": "Apply / land change",
    "merged": "Merge pull request",
    "reverted": "Revert change",
    "released": "Tag release",
    "reviewed": "Review (off-system)",
    "observed_state": "State recorded",
}
_GIT_KIND_NAMES = {
    "code_contribution": "Code contribution (feature / fix)",
    "release_and_backport": "Release & backport",
    "dependency_bumps": "Dependency bumps (bot)",
    "ci_maintenance": "CI / tooling maintenance (bot)",
    "documentation": "Documentation change",
    "release_notes": "Release notes (thin source — order unknown)",
}
_GIT_KIND_RATIONALE = {
    "code_contribution": "PR/issue runs whose commits change source or tests and are authored by people.",
    "release_and_backport": "Release-train and backport integration merges (long-lived release branches).",
    "dependency_bumps": "Recurring commits authored by a dependency bot — candidate 'looks like a process, isn't'.",
    "ci_maintenance": "Recurring commits authored by a CI/formatting bot — candidate 'looks like a process, isn't'.",
    "documentation": "Runs whose commits only ever touch documentation files.",
    "release_notes": "Changelog version sections that did not correlate to a git case — thin, order-unknown observations.",
}
_DOC_SUFFIXES = (".rst", ".md")


def _object_family(path: str) -> str:
    if path.startswith("src/") or path.startswith("tests/") or "/tests/" in path:
        return "code"
    if path.startswith("docs/") or path.endswith(_DOC_SUFFIXES):
        return "docs"
    if "requirements" in path or path.endswith((".txt", ".cfg", ".toml", ".ini", ".lock")):
        return "deps/config"
    return "other"


def _git_case_features(case, entities: dict) -> dict:
    commit_ents = [entities[eid] for eid in case.entity_ids
                   if eid.startswith("commit:") and eid in entities]
    fams: Counter = Counter()
    dep = ci = human = 0
    all_files: list[str] = []
    for ce in commit_ents:
        for f in ce.attrs.get("files", []):
            fams[_object_family(f)] += 1
            all_files.append(f)
        raw = ce.raw or {}
        a = raw.get("author", {}) or {}
        blob = f"{a.get('name', '')} {a.get('email', '')}".lower()
        if "dependabot" in blob or "renovate" in blob:
            dep += 1
        elif "pre-commit-ci" in blob or "github-actions" in blob:
            ci += 1
        else:
            human += 1
    total = len(commit_ents) or 1

    if case.kind_hint == "notes":
        sub = "release_notes"
    elif case.kind_hint in ("release", "integration"):
        sub = "release_and_backport"
    elif dep > total / 2:
        sub = "dependency_bumps"
    elif ci > total / 2:
        sub = "ci_maintenance"
    elif all_files and all(_object_family(f) == "docs" for f in all_files):
        sub = "documentation"
    else:
        sub = "code_contribution"
    return {"subkind": sub, "object_families": fams}


def _git_reject_reason(kf: dict) -> Optional[str]:
    sub = kf.get("subkind")
    if sub not in ("dependency_bumps", "ci_maintenance"):
        return None
    if kf.get("object_families", {}).get("code", 0) > 0:
        return None  # it sometimes changes real code — not a pure look-alike
    driver = "a dependency bot" if sub == "dependency_bumps" else "a CI/formatting bot"
    return (f"Recurring, machine-driven ({driver}) runs that move no product artefact "
            f"(no change to code) and produce nothing the contribution process exists to "
            f"produce. Looks like a process; isn't. Flagged, not deleted — "
            f"{kf.get('n_cases', 0)} runs remain inspectable.")


GIT_PROFILE = Profile(
    id="git",
    label_action=lambda a: _GIT_ACTIVITY_NAMES.get(a, _humanize(a)),
    case_features=_git_case_features,
    cluster_key=lambda gk, cf: ("git", cf.get("subkind", "code_contribution")),
    name_kind=lambda kf: kf.get("subkind"),
    display_name=lambda kid: _GIT_KIND_NAMES.get(kid),
    rationale=lambda kf: _GIT_KIND_RATIONALE.get(kf.get("subkind")),
    reject_reason=_git_reject_reason,
)

# ---------------------------------------------------------------------------
# A second domain, to prove the profile mechanism is not git-shaped. This one
# ONLY renames (no custom clustering) — the smallest a useful profile can be.
# ---------------------------------------------------------------------------
_ACCT_ACTIVITY = {
    "raised": "Invoice raised", "submitted": "Submitted for approval",
    "approved": "Approved", "paid": "Marked paid", "settled": "Bank payment settled",
}
_ACCT_KIND_NAMES = {
    "invoice_approval": "Invoice approval & payment",
    "system_entries": "System / automated entries",
    "unmatched_payments": "Unmatched bank payments",
}
_ACCT_KIND_RATIONALE = {
    "invoice_approval": "Invoices moving through raise → submit → approve → pay, performed by people.",
    "system_entries": "Recurring zero-value entries performed by a system actor — candidate 'looks like a process, isn't'.",
    "unmatched_payments": "Bank payments that reference an invoice not present in the tracker.",
}


def _acct_name_kind(kf: dict) -> Optional[str]:
    kh = kf.get("kind_hint")
    if kh == "invoice":
        return "system_entries" if kf.get("automated") else "invoice_approval"
    if kh == "payment":
        return "unmatched_payments"
    return None


ACCOUNTING_PROFILE = Profile(
    id="accounting",
    label_action=lambda a: _ACCT_ACTIVITY.get(a, _humanize(a)),
    name_kind=_acct_name_kind,
    display_name=lambda kid: _ACCT_KIND_NAMES.get(kid),
    rationale=lambda kf: _ACCT_KIND_RATIONALE.get(_acct_name_kind(kf)),
    # reject_reason stays the generic rule: a recurring, fully-automated cluster.
)

_PROFILES = {"git": GIT_PROFILE, "generic": GENERIC_PROFILE, "accounting": ACCOUNTING_PROFILE}


def select_profile(shaped) -> Profile:
    """Pick a profile from the data's own sources. Git data gets the git
    vocabulary; anything else falls back to the generic, unnamed baseline —
    which is exactly the point: an unknown source still produces a coherent,
    honestly-unnamed model."""
    for rec in list(shaped.events)[:100] + list(shaped.entities)[:100]:
        if getattr(rec, "source", "").startswith("git:"):
            return GIT_PROFILE
    return GENERIC_PROFILE
