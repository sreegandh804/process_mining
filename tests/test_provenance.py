"""Provenance runs through everything: no claim without a tier and evidence, and
inference is never rendered as fact (brief §6, definition-of-done #3)."""

from induction.emit import build_model


def test_no_event_fabricates_an_actor_or_time_it_does_not_have(mini_shaped):
    shaped, corr, slug = mini_shaped
    released = [e for e in shaped.events if e.action == "released"]
    assert released
    # a tag names no author — the engine must not invent one
    assert all(e.actor is None for e in released)


def test_inferred_entities_carry_confidence_and_evidence(mini_shaped):
    shaped, corr, slug = mini_shaped
    inferred = [e for e in shaped.entities if e.type in ("pr", "issue")]
    assert inferred
    for e in inferred:
        assert e.confidence is not None and e.evidence
    # ...while a directly-read commit is tier `direct`
    commit = next(e for e in shaped.entities if e.type == "commit")
    assert commit.confidence.tier.label == "direct"


def test_gaps_are_always_inference_never_fact(mini_model):
    assert mini_model.gaps
    for g in mini_model.gaps:
        # an off-system step can only ever be heuristic/model — never read as fact
        assert g.confidence.tier.label in {"heuristic", "model"}
        assert g.to_dict()["inferred"] is True
        assert g.evidence, "a gap must name the signal that produced it"


def test_emitted_model_is_wall_to_wall_scored(mini_model):
    """Every surfaced element in model.json carries a confidence tier drawn from
    the fixed vocabulary — no bare claims, no invented decimals."""
    doc = build_model(mini_model)
    allowed = {"direct", "joined", "heuristic", "model"}

    for c in doc["cases"]:
        assert c["confidence"]["tier"] in allowed
    for k in doc["process_definitions"]:
        assert k["confidence"]["tier"] in allowed
    for g in doc["gaps"]:
        assert g["confidence"]["tier"] in allowed
    for e in doc["events"]:
        assert e["confidence"]["tier"] in allowed


def test_cost_and_divergence_are_honest_stubs(mini_model):
    doc = build_model(mini_model)
    assert doc["cost_value"]["status"] == "stub"
    # slots exposed, figures NOT fabricated
    for slot in doc["cost_value"]["per_step"].values():
        assert slot == {"money": None, "effort": None}
    assert doc["divergence"]["status"] == "hook"
    assert doc["divergence"]["items"] == []
