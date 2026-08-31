"""The model tier — connecting records that mean the same thing in different
words, the join no key and no shared token can make.

Driven by a scripted, offline stand-in for the LLM (`ScriptedJudge`), so the
engine's *handling* of a model-tier join — that it is opt-in, that it reads as
the weakest inference, that it never overrides a stronger join, that it still
refuses look-alikes and the unrelated — is pinned without a key or a network.
Live, `AnthropicJudge` makes the real call behind the same seam.
"""

import pytest

from induction.adapters import Shaped, email_mbox, github_api
from induction.pipeline import induce
from induction.semantic import ScriptedJudge, SemanticProvider
from induction.steps.correlate import CorrelationPolicy
from tests.combined_fixture import GH_PAYLOAD, MAIL, SLUG, MAIL_SLUG


def _shaped():
    s = Shaped()
    s.extend(github_api.shape(GH_PAYLOAD, SLUG))
    s.extend(email_mbox.shape(MAIL, MAIL_SLUG))
    return s


# A transparent offline judge: a rule fires when a shared concept term is on both
# sides. It judges TEXT, like the real model — the engine holds none of these words.
_JUDGE = ScriptedJudge([
    (["sso", "token", "logged out", "signed out", "log in", "login", "expiry"],
     "both describe the SSO login failure after the token refresh"),
    (["csv", "export", "buffer", "stream", "download", "spins", "times out"],
     "both describe the CSV export timeout and its streaming fix"),
])


@pytest.fixture(scope="module")
def base():
    return induce(_shaped(), slug="northwind")


@pytest.fixture(scope="module")
def sem():
    return induce(_shaped(), slug="northwind",
                  policy=CorrelationPolicy(semantic=SemanticProvider(judge=_JUDGE)))


def _case_of(m, eid):
    return next((c for c in m.cases.values() if eid in c.entity_ids), None)


def test_the_model_pass_is_off_by_default(base):
    """No provider, no model joins: the email thread and the GitHub issue about the
    same SSO bug stay in different cases. The engine's default is offline."""
    mail = _case_of(base, "email:northwind-team:a1")
    gh = _case_of(base, "issue:northwind/portal:14")
    assert mail is not None and gh is not None and mail.id != gh.id


def test_the_model_pass_joins_paraphrased_records_across_sources(sem):
    """The email thread ('signed out after login'), the issue ('SSO login fails')
    and the PR share NO number and too few tokens for the heuristic pass. The model
    connects them into one cross-source case."""
    ids = {"email:northwind-team:a1", "email:northwind-team:a2",
           "issue:northwind/portal:14", "pr:15"}
    case = _case_of(sem, "pr:15")
    assert case is not None and ids <= set(case.entity_ids)
    sources = {("mail" if e.startswith("email:") else "github") for e in case.entity_ids}
    assert sources == {"mail", "github"}, "the case must span both sources"


def test_a_model_join_reads_as_the_weakest_inference_and_says_why(sem):
    """A case assembled across a model bridge reports `model` — the weakest tier —
    however deterministic its parts, and carries the model's own reason. Reporting
    anything stronger would launder the guess."""
    case = _case_of(sem, "pr:15")
    assert case.confidence.tier.label == "model"
    assert "model" in case.confidence.rationale.lower()
    assert "sso" in case.confidence.rationale.lower()


def test_the_model_pass_leaves_lookalikes_and_the_unrelated_alone(sem):
    """The judge is only asked about proximate, differently-shaped pairs, so the
    recurring CI notice is still rejected and the lunch thread joins no work."""
    assert any(k.rejected for k in sem.kinds), "recurring automated notice must still reject"
    lunch = _case_of(sem, "email:northwind-team:d1")
    assert lunch is not None
    assert not any(e.startswith(("issue:", "pr:")) for e in lunch.entity_ids), \
        "the lunch thread must not be pulled into a work case"


def test_the_model_pass_only_extends_never_overrides_a_stronger_join(base, sem):
    """Issue #14 and PR #15 are joined deterministically ('fixes #14'). That join
    holds with OR without the model pass — the model is a second pass over the
    leftovers, so it can widen a case but never break or relabel a firm link."""
    for m in (base, sem):
        c = _case_of(m, "pr:15")
        assert "issue:northwind/portal:14" in c.entity_ids


def test_people_are_still_not_merged_across_sources(sem):
    """The same human is `maria` on GitHub and `maria@…` in mail. The model pass
    connects the work; it does not silently fuse the two identities."""
    people = {e.id for e in sem.shaped.entities if e.type == "person"}
    assert "person:gh:maria" in people
    assert "person:mail:maria@northwind.com" in people


def test_scripted_judge_only_fires_on_a_shared_concept():
    """The stand-in judges text, and says no when there is no shared concept —
    the property the whole pass relies on to not fuse everything."""
    assert _JUDGE.judge("the SSO token refresh logs users out", "login broken by token expiry")
    assert _JUDGE.judge("tacos on friday", "lunch plans") is None
    assert _JUDGE.judge("SSO login issue", "CSV export is slow") is None
