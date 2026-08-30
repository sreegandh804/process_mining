"""Step 2 — Correlate: group events into cases (process instances).

This is the weak claim, so it gets the most care. We form cases with **only
deterministic joins**, layered strongest-first, and we score *every* link:

  1. Merge topology  (tier `joined`) — a "Merge pull request #N" commit owns the
     commits it brought onto the trunk. This uses the real git DAG, so it is a
     structural key, not a guess. It recovers genuine *multi-commit* runs that a
     text-only join would scatter into orphans.
  2. Squash subject  (tier `joined`) — a lone trunk commit ending "(#N)".
  3. Issue keyword   (tier `joined`) / bare mention (tier `heuristic`) — attaches
     an issue to a PR run, or anchors a case when there is no PR.
  4. Branch merge    (tier `joined`) — "Merge branch 'stable'" integration
     commits anchor the release/backport process; they do NOT absorb history.

What remains unjoined stays unjoined: it becomes an orphan (§6), never padded
into a case. A deliberately ambiguous reference (a bare "#N") is joined *with
low confidence*, not silently.

Where the deterministic baseline visibly breaks — a follow-up commit that
belongs to a PR but shares no key, two records that are the same activity with
no shared id — is exactly where the documented upgrade path (fuzzy reference /
actor+time proximity / embeddings) would go. That is deliberately not built;
see README. The point of scoring every link is that a future fuzzy join shows
up as `heuristic`, distinguishable at a glance from the `joined` spine.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from induction.adapters import Shaped
from induction.model import Confidence, Entity, Evidence, Event, Tier, joined, heuristic
from induction.process import Case


@dataclass
class Correlation:
    cases: dict[str, Case] = field(default_factory=dict)
    # events left with case_id=None are orphans; the honesty step collects them.

    def case_of(self, event: Event) -> Optional[Case]:
        return self.cases.get(event.case_id) if event.case_id else None


def correlate(shaped: Shaped, slug: str) -> Correlation:
    source = f"git:{slug}"
    commits = {e.id: e for e in shaped.entities if e.type == "commit"}
    sha_of = {e.id: e.raw["sha"] for e in commits.values()}
    id_by_sha = {sha: eid for eid, sha in sha_of.items()}
    events_by_entity: dict[str, list[Event]] = defaultdict(list)
    for ev in shaped.events:
        events_by_entity[ev.entity_id].append(ev)

    parents_by_sha = {sha_of[eid]: c.attrs.get("parents", []) for eid, c in commits.items()}
    present = set(parents_by_sha)  # shas we actually ingested (shallow boundary)

    corr = Correlation()
    assigned: dict[str, str] = {}  # commit entity id -> case id
    inferred_entities: dict[str, Entity] = {}

    trunk = _first_parent_trunk(commits, id_by_sha, sha_of, parents_by_sha, present)
    # A PR's branch walk must stop when it reaches *another* PR's anchor commit —
    # those commits belong to that PR, not this one. This is what keeps a release
    # branch (which is full of merged PRs) from being swallowed whole.
    pr_anchor_shas = {
        sha_of[eid] for eid, ent in commits.items()
        if any(r["kind"] == "pr" and r["via"] in ("merge-pull-request", "squash-subject")
               for r in ent.attrs.get("references", []))
    }

    # ---- pass 1: merge-pull-request topology (strongest structural join) ----
    # Process merges so that a commit lands in the *nearest* PR that introduced
    # it. Iterating in file order (newest first) and skipping already-assigned
    # commits gives that determinism.
    for eid, ent in commits.items():
        if not ent.attrs.get("is_merge"):
            continue
        pr_refs = [r for r in ent.attrs.get("references", []) if r["kind"] == "pr"
                   and r["via"] == "merge-pull-request"]
        if not pr_refs:
            continue
        number = pr_refs[0]["number"]
        subject = ent.attrs.get("subject", "")
        branch, overflowed = _branch_commits(
            eid, sha_of, id_by_sha, parents_by_sha, present, trunk, assigned, pr_anchor_shas,
        )
        if overflowed:
            # A "PR" that introduces more commits than any single contribution
            # plausibly does is a release/integration train. We say so — as a
            # *heuristic* judgement (a size threshold, not a structural fact) —
            # rather than asserting a 1000-commit "process instance".
            looks_release = "release" in subject.lower() or "release" in (ent.attrs.get("merge_branch") or "").lower()
            case = _ensure_case(
                corr, f"case:release:{number}" if looks_release else f"case:integration:pr:{number}",
                "release" if looks_release else "integration",
                {"type": "release" if looks_release else "integration", "pr": number},
                heuristic(f"merge #{number} introduces >{_ABSORB_CAP} commits — treated as a "
                          f"release/integration train, not one contribution run"),
                Evidence(source, sha_of[eid], subject),
            )
            _place(eid, case, assigned, corr, events_by_entity, commits,
                   heuristic("oversized merge reclassified as integration"), sha_of, source)
            continue
        case = _ensure_case(
            corr, f"case:pr:{number}", "pr", {"type": "pr", "number": number},
            joined("commit is the 'Merge pull request' commit for this PR"),
            Evidence(source, sha_of[eid], subject),
        )
        _materialise_pr(inferred_entities, source, number, sha_of[eid], subject, "joined")
        for member_eid in branch:
            if member_eid in assigned:
                continue
            _place(member_eid, case, assigned, corr, events_by_entity, commits,
                   joined("reachable from the PR merge but not from the trunk it merged into"),
                   sha_of, source)
        # the merge commit itself belongs to the case, as a `joined` structural fact
        _place(eid, case, assigned, corr, events_by_entity, commits,
               joined("this is the merge commit that closed the PR"), sha_of, source)

    # ---- pass 2: squash-subject "(#N)" on still-unassigned trunk commits ----
    for eid, ent in commits.items():
        if eid in assigned:
            continue
        pr_refs = [r for r in ent.attrs.get("references", []) if r["kind"] == "pr"
                   and r["via"] == "squash-subject"]
        if not pr_refs:
            continue
        number = pr_refs[0]["number"]
        case = _ensure_case(
            corr, f"case:pr:{number}", "pr", {"type": "pr", "number": number},
            joined("squash-merge subject carries the PR number"),
            Evidence(source, sha_of[eid], ent.attrs.get("subject")),
        )
        _materialise_pr(inferred_entities, source, number, sha_of[eid], ent.attrs.get("subject"), "joined")
        _place(eid, case, assigned, corr, events_by_entity, commits,
               joined("squash-merge subject '(#%d)'" % number), sha_of, source)

    # ---- pass 3: issue references ----
    # An issue on a commit already in a PR case *enriches* that case (the PR
    # addresses the issue); an issue on an otherwise-unassigned commit *anchors*
    # a case of its own.
    for eid, ent in commits.items():
        issue_refs = [r for r in ent.attrs.get("references", []) if r["kind"] == "issue"]
        if not issue_refs:
            continue
        for r in issue_refs:
            number = r["number"]
            iss_id = f"issue:{slug}:{number}"
            tier_label = r["tier"]
            _materialise_issue(inferred_entities, source, slug, number, sha_of[eid],
                               r["snippet"], tier_label)
            if eid in assigned:
                case = corr.cases[assigned[eid]]
                if iss_id not in case.entity_ids:
                    case.entity_ids.append(iss_id)
                    case.evidence.append(Evidence(source, sha_of[eid], r["snippet"]))
            else:
                conf = joined("issue-closing keyword") if tier_label == "joined" \
                    else heuristic("bare '#%d' mention — a reference, not a proven link" % number)
                case = _ensure_case(
                    corr, f"case:issue:{slug}:{number}", "issue",
                    {"type": "issue", "number": number}, conf,
                    Evidence(source, sha_of[eid], r["snippet"]),
                )
                _place(eid, case, assigned, corr, events_by_entity, commits, conf, sha_of, source)

    # ---- pass 4: "Merge branch 'X'" integration commits (release spine) ----
    for eid, ent in commits.items():
        if eid in assigned:
            continue
        branch_name = ent.attrs.get("merge_branch")
        if not branch_name:
            continue
        # Each integration merge is its OWN run — lumping every "Merge branch
        # 'stable'" into one case would invent a single 15-step "instance" that
        # never happened. One backport = one run.
        case = _ensure_case(
            corr, f"case:integration:{branch_name}:{sha_of[eid][:8]}", "integration",
            {"type": "integration", "branch": branch_name, "sha": sha_of[eid][:8]},
            joined("integration merge of branch '%s'" % branch_name),
            Evidence(source, sha_of[eid], ent.attrs.get("subject")),
        )
        _place(eid, case, assigned, corr, events_by_entity, commits,
               joined("integration merge commit"), sha_of, source)

    shaped.entities.extend(inferred_entities.values())
    return corr


# ---------------------------------------------------------------------------
# DAG helpers
# ---------------------------------------------------------------------------
def _first_parent_trunk(commits, id_by_sha, sha_of, parents_by_sha, present) -> set[str]:
    """The mainline: follow first-parent from HEAD. A PR merge's branch commits
    are precisely those reachable from its 2nd parent that are NOT on this trunk,
    so knowing the trunk lets us bound each branch walk to the branch itself."""
    head_sha = None
    for eid, ent in commits.items():
        if any("HEAD" in r for r in ent.attrs.get("refs", [])):
            head_sha = sha_of[eid]
            break
    if head_sha is None and commits:
        # commits.jsonl is newest-first; the first entity is the tip.
        head_sha = next(iter(commits.values())).raw["sha"]
    trunk: set[str] = set()
    cur = head_sha
    while cur and cur in present and cur not in trunk:
        trunk.add(cur)
        parents = parents_by_sha.get(cur, [])
        cur = parents[0] if parents else None
    return trunk


# A PR branch above this size is not one contribution run; the walk gives up and
# the caller reclassifies it as an integration/release train. Generous on
# purpose — real feature PRs sit well under it.
_ABSORB_CAP = 60


def _branch_commits(merge_eid, sha_of, id_by_sha, parents_by_sha, present, trunk,
                    assigned, pr_anchor_shas) -> tuple[list[str], bool]:
    """Commits introduced by a PR merge = reachable from its 2nd+ parents,
    stopping at: the trunk (where the branch forked), already-assigned commits,
    and *other* PR anchor commits (which belong to those PRs). Bounded to the
    branch, so it stays cheap even over long history.

    Returns ``(commits, overflowed)``. ``overflowed`` is True when the walk blew
    past ``_ABSORB_CAP`` — the signal that this is not a normal PR.
    """
    merge_sha = sha_of[merge_eid]
    parents = parents_by_sha.get(merge_sha, [])
    if len(parents) < 2:
        return [], False
    out: list[str] = []
    seen: set[str] = set()
    queue = deque(p for p in parents[1:] if p in present)
    while queue:
        sha = queue.popleft()
        if sha in seen or sha in trunk:
            continue
        seen.add(sha)
        eid = id_by_sha.get(sha)
        if eid is None:
            continue
        if eid in assigned:
            continue                       # already owned by a nearer PR
        if sha in pr_anchor_shas and sha != merge_sha:
            continue                       # boundary: belongs to another PR
        out.append(eid)
        if len(out) > _ABSORB_CAP:
            return out, True
        for p in parents_by_sha.get(sha, []):
            if p in present and p not in trunk and p not in seen:
                queue.append(p)
    return out, False


# ---------------------------------------------------------------------------
# case assembly
# ---------------------------------------------------------------------------
def _ensure_case(corr, case_id, kind_hint, anchor, confidence, ev) -> Case:
    case = corr.cases.get(case_id)
    if case is None:
        case = Case(id=case_id, kind_hint=kind_hint, anchor=anchor, confidence=confidence)
        corr.cases[case_id] = case
    if ev is not None and ev not in case.evidence:
        case.evidence.append(ev)
    return case


def _place(commit_eid, case, assigned, corr, events_by_entity, commits, link_conf, sha_of, source) -> None:
    """Attach a commit (and all its events) to a case, scoring the case link."""
    assigned[commit_eid] = case.id
    if commit_eid not in case.entity_ids:
        case.entity_ids.append(commit_eid)
    for ev in events_by_entity.get(commit_eid, []):
        ev.case_id = case.id
        ev.case_confidence = link_conf
        if ev.id not in case.event_ids:
            case.event_ids.append(ev.id)


def _materialise_pr(store, source, number, locator, snippet, tier_label) -> None:
    """A PR we never saw directly — known only because a commit references it.
    Recorded as an inferred entity so its very existence is legible as inference,
    and so step 6 can flag its (entirely off-git) timeline as a gap."""
    ent_id = f"pr:{number}"
    if ent_id in store:
        return
    store[ent_id] = Entity(
        id=ent_id, source=source, type="pr",
        attrs={"number": number, "known_via": "reference", "timeline": "off-git"},
        confidence=Confidence(Tier.from_label(tier_label),
                              "existence inferred from a commit reference; no direct PR record in git"),
        evidence=[Evidence(source, locator, snippet)],
    )


def _materialise_issue(store, source, slug, number, locator, snippet, tier_label) -> None:
    ent_id = f"issue:{slug}:{number}"
    if ent_id in store:
        return
    store[ent_id] = Entity(
        id=ent_id, source=source, type="issue",
        attrs={"number": number, "known_via": "reference", "timeline": "off-git"},
        confidence=Confidence(Tier.from_label(tier_label),
                              "existence inferred from a commit reference; no direct issue record in git"),
        evidence=[Evidence(source, locator, snippet)],
    )
