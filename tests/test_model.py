"""The substrate: confidence tiers are ordinal, never fabricated decimals."""

from induction.model import (Confidence, Evidence, Event, Tier, direct,
                             heuristic, joined, model)


def test_tiers_are_ordinal_strongest_to_weakest():
    assert Tier.DIRECT > Tier.JOINED > Tier.HEURISTIC > Tier.MODEL


def test_confidence_of_a_chain_is_its_weakest_link():
    chain = [direct(), joined(), heuristic()]
    assert Confidence.weakest(chain).tier is Tier.HEURISTIC
    assert Confidence.weakest([direct(), joined()]).tier is Tier.JOINED


def test_tier_serialises_as_a_label_not_a_number():
    d = joined("shared key").to_dict()
    assert d["tier"] == "joined"
    assert d["rationale"] == "shared key"
    # never a fabricated 0.83-style score
    assert not isinstance(d["tier"], float)


def test_event_keeps_its_own_confidence_separate_from_the_case_link():
    ev = Event(id="e", entity_id="x", action="authored", source="git",
               confidence=direct(), case_confidence=heuristic("bare mention"))
    d = ev.to_dict()
    assert d["confidence"]["tier"] == "direct"        # the event happened
    assert d["case_confidence"]["tier"] == "heuristic"  # its run is a guess
