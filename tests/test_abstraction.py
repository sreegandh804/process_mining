"""The process-abstraction layer — artefact verbs folded into the activities a
process is made of, with the artefacts kept as the evidence beneath each one.

AI-first by design: the grouping comes from the model, so these tests drive it
with `ScriptedActivityMapper` — a stand-in that simulates the model's answer — and
assert the engine's *handling* of that answer. Live, `AnthropicActivityMapper`
makes the real call behind the same seam. There is no deterministic naming
fallback: with no mapper the engine claims no abstraction and shows raw artefacts.
"""

from induction.abstraction import (AnthropicActivityMapper, ScriptedActivityMapper,
                                    infer_activities)
from induction.adapters import Shaped, email_mbox, github_api
from induction.inspector import build_view
from induction.naming import infer_names
from induction.pipeline import induce
from induction.semantic import SemanticProvider
from induction.steps.correlate import CorrelationPolicy
from tests.combined_fixture import (GH_PAYLOAD, MAIL, SLUG, MAIL_SLUG,
                                     demo_activity_mapper, demo_judge, demo_namer)


def _model():
    s = Shaped()
    s.extend(github_api.shape(GH_PAYLOAD, SLUG))
    s.extend(email_mbox.shape(MAIL, MAIL_SLUG))
    return induce(s, slug="northwind",
                  policy=CorrelationPolicy(semantic=SemanticProvider(judge=demo_judge())))


def _sso_run(view):
    return next(r for r in view["runs"] if "SSO" in r["title"])


def test_scripted_mapper_only_returns_pairs_actually_present():
    """A stand-in for the model must not smuggle in vocabulary the corpus never
    showed — same filter the real mapper applies to the model's answer."""
    mapper = ScriptedActivityMapper({"issue/opened": "Raised", "widget/frobbed": "Frobbed"})
    got = mapper.map([{"artefact": "issue", "action": "opened", "examples": []}])
    assert got == {"issue/opened": "Raised"}


def test_infer_activities_builds_the_corpus_vocabulary():
    """The verb map is keyed by (artefact-type / verb), covering every distinct pair."""
    acts = infer_activities(_model(), demo_activity_mapper()).by_vocab
    assert acts["issue/opened"] == "Raised"
    assert acts["pr/merged"] == "Shipped"
    assert acts["email/sent"] == "Raised"       # an email report and an issue are one activity


def test_the_spine_is_activities_not_artefact_verbs():
    """The whole point: the run reads as a process, not the systems' event log.
    'sent → sent → opened → opened → …' becomes 'Raised → … → Shipped'."""
    v = build_view(_model(), activities=infer_activities(_model(), demo_activity_mapper()))
    path = _sso_run(v)["path"]
    assert "Raised" in path and "Shipped" in path and "Reviewed" in path
    assert "opened" not in path and "sent" not in path   # no raw artefact verbs on the spine


def test_each_activity_carries_its_artefacts_as_evidence():
    """An activity is the claim; the artefacts under it are the proof — and one
    cross-source activity carries evidence from BOTH systems."""
    v = build_view(_model(), activities=infer_activities(_model(), demo_activity_mapper()))
    raised = next(n for n in _sso_run(v)["activities"] if n["name"] == "Raised")
    assert raised["n"] >= 2 and set(raised["sources"]) == {"GitHub", "email"}
    verbs = {a["verb"] for a in raised["arts"]}
    assert verbs and all(a["src"] for a in raised["arts"])   # every artefact resolves to a source


def test_no_artefact_is_lost_in_the_abstraction():
    """Folding verbs into activities is a view: every underlying event is retained
    as evidence, none dropped to make the spine tidy."""
    m = _model()
    v = build_view(m, activities=infer_activities(m, demo_activity_mapper()))
    sso = next(c for c in m.cases.values() if c.id == "case:pr:15")
    under_activities = sum(n["n"] for n in _sso_run(v)["activities"])
    assert under_activities == len(sso.ordered_event_ids)


def test_ai_first_no_deterministic_naming_fallback():
    """With no mapper the engine claims NO abstraction — it shows the raw artefact
    verbs (de-duplicated), never an invented process. Abstraction is the AI's job
    or it does not happen."""
    m = _model()
    v = build_view(m, activities={})           # no AI map
    path = _sso_run(v)["path"]
    assert "Raised" not in path and "Shipped" not in path
    assert "opened" in path or "sent" in path  # the source's own verbs, untouched


def test_anthropic_mapper_is_a_noop_without_a_key(monkeypatch):
    """The real mapper assumes the API but never breaks a keyless run: no key ->
    empty map -> raw view, not a crash."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert AnthropicActivityMapper().map([{"artefact": "issue", "action": "opened", "examples": []}]) == {}


def test_kinds_are_named_by_the_model_not_left_generic():
    """The induced kinds get real, distinct names from the model (an injected
    stand-in here), not Kind 1..n — the distinct processes / offerings a company
    runs. Each name maps to a real kind id and they do not collide into one label."""
    m = _model()
    names = infer_names(m, namer=demo_namer)
    assert names["_ai"] and names["kinds"]
    assert set(names["kinds"]) == {k.id for k in m.kinds}      # every kind named
    assert "Bug-fix delivery" in names["kinds"].values()
    assert len(set(names["kinds"].values())) >= 2              # distinct, not one label


def test_the_process_cards_show_activities_not_raw_verbs():
    """The 'processes we found' cards read as the process — the same activity
    spine as the runs — not the systems' artefact verbs. Regression: they were
    left showing 'sent -> opened -> labeled -> merged -> closed'."""
    m = _model()
    v = build_view(m, activities=infer_activities(m, demo_activity_mapper()))
    dev = next(p for p in v["processes"] if "Bug-fix" in p["name"] or "delivery" in p["name"].lower()
               or any("Shipped" in s for s in p["flow"]))
    assert "Shipped" in dev["flow"] and "Raised" in dev["flow"]
    assert "merged" not in dev["flow"] and "opened" not in dev["flow"]
    # variants are activity paths too
    assert dev["paths"] and all(isinstance(s, str) for s in dev["paths"][0]["seq"])


def test_every_inference_is_traceable_to_a_source_record():
    """Everything the engine shows resolves back to the artefact it was read from:
    each step's evidence carries a source locator, and a real URL is clickable."""
    m = _model()
    v = build_view(m, activities=infer_activities(m, demo_activity_mapper()))
    sso = _sso_run(v)
    all_arts = [a for n in sso["activities"] for a in n["arts"]]
    assert all_arts and all(a["src"] for a in all_arts)        # nothing without a source
    assert any(a["is_url"] for a in all_arts)                  # GitHub artefacts open ↗
    # the cross-source join itself is traceable — the model's reason is on the run
    assert "model" in sso["tier"] and sso["why"]
