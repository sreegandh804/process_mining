"""Honesty features (brief §6) — the graded core.

These are not decoration. The whole claim of the engine is that it is *honest
about what it cannot see*, so each of these turns an absence or an awkward
record into a visible finding rather than a silent drop:

  - orphans : records that join to no case — surfaced in a queue, never padded
              into a rollup and never dropped.
  - reject  : a recurring pattern that moves no product artefact and produces
              nothing of the process's value — flagged "looks like a process,
              isn't" (e.g. an automated bump), with its reason.
  - unknowns: actor / time / order that is genuinely unavailable is marked
              `unknown`. Absence is a finding, not a blank to fill.

`divergence` (belief vs data) is a *hook*, not a workflow — see emit.py and the
README. We keep `raw` next to inferred structure so a later validation step can
compare what the owner believes against what the data shows.
"""

from __future__ import annotations

from collections import defaultdict

from induction.adapters import Shaped
from induction.model import Evidence
from induction.process import Case, Orphan, ProcessKind
from induction.profiles import GENERIC_PROFILE, Profile
from induction.steps.correlate import Correlation


def collect_orphans(shaped: Shaped, corr: Correlation) -> list[Orphan]:
    """Every event/observation with no case is an orphan. We report one orphan
    per orphaned *entity* (a commit can emit several events) with the reason it
    joined to nothing — the honest counterpart to the joined spine."""
    entities_by_id = {e.id: e for e in shaped.entities}
    orphan_entities: dict[str, list] = defaultdict(list)

    for ev in shaped.events:
        if not ev.case_id:
            orphan_entities[ev.entity_id].append(ev)
    for ob in shaped.observations:
        if not ob.case_id:
            orphan_entities[ob.entity_id].append(ob)

    orphans: list[Orphan] = []
    for ent_id, recs in orphan_entities.items():
        ent = entities_by_id.get(ent_id)
        reason = _orphan_reason(ent)
        rec = recs[0]
        rectype = "observation" if rec.__class__.__name__ == "Observation" else "event"
        orphans.append(Orphan(
            record_id=rec.id,
            record_type=rectype,
            entity_id=ent_id,
            reason=reason,
            evidence=list(rec.evidence),
        ))
    orphans.sort(key=lambda o: o.entity_id)
    return orphans


def _orphan_reason(ent) -> str:
    if ent is None:
        return "record's entity is unknown"
    if ent.attrs.get("reason"):
        return ent.attrs["reason"]
    if ent.type == "commit":
        if ent.attrs.get("is_merge"):
            return ("merge commit with no 'Merge pull request #N' and no branch name "
                    "— cannot be attributed to a specific run")
        return ("commit references no PR or issue and is not reachable from any PR "
                "merge — a direct-to-branch commit with no run to attach to")
    return f"{ent.type} record matched no correlation key"


def apply_reject(kinds: list[ProcessKind], profile: Profile = GENERIC_PROFILE) -> None:
    """Flag look-alike non-processes in place, with a concrete reason.

    The *decision* is domain knowledge, so it lives in the profile: the generic
    default flags any recurring, fully-automated cluster (it cannot prove such a
    cluster "produces nothing" without domain knowledge, and says so); a source
    profile can sharpen it (git un-flags bot runs that actually change code).

    We only *flag* — never delete. The rejected kind stays fully inspectable so a
    reader can disagree; that is the point of showing the rejection and its
    reason rather than quietly filtering it out.
    """
    for kind in kinds:
        reason = profile.reject_reason(kind.features)
        if reason:
            kind.rejected = True
            kind.reject_reason = reason


# ---------------------------------------------------------------------------
# The typed second opinion, where the profile has none
# ---------------------------------------------------------------------------

# Three narrow questions, because "is this a real process" is three judgements.
# The generic rule above can only see that a cluster is automated and recurs; it
# says outright that it "cannot prove it produces nothing without domain
# knowledge". These ask the part it cannot: does anything of value come out.
_TRIAGE_QUESTIONS: dict = {
    "produces_an_artefact": {
        "type": "noul",
        "instructions": {
            "question": "Do the runs of this cluster produce or change something "
                        "the organisation keeps?",
            "focus": "An outcome that outlives the run — not a notification about one.",
        },
        "criteria": {
            "true": {"what": "A document, record, decision or change of state results",
                     "examples": ["A bot that opens a pull request changing code",
                                  "An automated invoice actually being issued"]},
            "false": {"what": "Only a notification, log line or status ping results",
                      "not_for": "An automated run that really does change something",
                      "examples": ["A nightly build notice", "A recurring reminder"]},
        },
    },
    "verdict": {
        "type": "choice",
        "instructions": {
            "question": "Is this cluster a real process, or something that merely "
                        "looks like one?",
            "focus": "Recurring shape is not enough; ask what the runs accomplish.",
        },
        "criteria": {
            "real_process": {"what": "Work the organisation means to do, that produces "
                                     "something of the value the process exists for"},
            "machine_noise": {"what": "A recurring automated pattern that moves no "
                                      "product artefact and produces nothing of value",
                              "examples": ["Dependency bumps that change no code",
                                           "CI or formatting churn"]},
            "ambiguous": {"what": "Genuinely cannot be told apart from the evidence shown"},
        },
    },
}


def triage_unflagged(kinds: list[ProcessKind], jev=None, log=None) -> int:
    """Ask the typed tier about the clusters the profile had no opinion on.

    Strictly additive, in three ways that matter:

      - it never touches a kind the profile already flagged, because a domain
        rule is knowledge and this is an opinion, and knowledge wins;
      - it never un-flags anything, for the same reason;
      - it flags only on `machine_noise` won clearly, and says in the reason
        itself that a model said so, with the number — so a reader can disagree
        with it exactly as they can disagree with the profile's own wording.

    `apply_reject` still only ever FLAGS. Nothing here deletes a kind, and the
    runs stay inspectable, which is the whole reason the reject pile is shown
    rather than filtered away. Returns how many kinds it flagged.
    """
    log = log or (lambda m: None)
    if jev is None:
        from induction.jev_call import Jev
        jev = Jev(log=log)
    if not jev.available:
        return 0

    flagged = 0
    for kind in kinds:
        if kind.rejected:
            continue                      # the profile already knows; leave it alone
        state = {"cluster": {"name": kind.name, "rationale": kind.rationale,
                             "n_runs": len(kind.case_ids),
                             "steps": kind.steps[:12], "features": kind.features}}
        answers = jev.ask(state, _TRIAGE_QUESTIONS)
        verdict = answers.get("verdict")
        produces = answers.get("produces_an_artefact")
        if verdict is None or verdict.choice != "machine_noise":
            continue
        if not verdict.sure(0.70):
            continue                      # a tie is not a finding
        if produces is not None and produces.noul is not None and produces.noul >= 0.5:
            continue                      # it does produce something: not noise
        kind.rejected = True
        kind.reject_reason = (
            f"No source profile has a rule for this cluster, so a model was asked: it "
            f"reads these {len(kind.case_ids)} runs as a recurring pattern that moves no "
            f"product artefact and produces nothing of the value a process exists for "
            f"(confidence {verdict.confidence:.2f}). This is inference, not a rule — "
            f"flagged, not deleted, and the runs remain inspectable.")
        flagged += 1
        log(f"[honesty] flagged {kind.name!r} as a look-alike non-process "
            f"(model, {verdict.confidence:.2f}) — no profile rule covered it")
    return flagged
