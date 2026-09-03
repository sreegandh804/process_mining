"""The model tier is on by default, off honestly — never a silent mislabel.

These pin the contract `induction/model_tier.resolve` exists to guarantee:
  - off / --no-llm            -> deterministic, no providers, no exit;
  - auto with no key          -> DOWNSHIFT: inactive, a printed note, no exit;
  - auto with a key + SDK     -> active, real providers wired;
  - an explicit llm with no key -> INSIST: SystemExit, not a mislabelled run.
"""

import pytest

from induction.model_tier import resolve


def test_no_llm_forces_off(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-present")
    monkeypatch.setattr("induction.model_tier._have_sdk", lambda: True)
    tier = resolve("auto", no_llm=True)
    assert tier.active is False
    assert tier.mapper() is None and tier.classifier() is None and tier.semantic() is None
    assert tier.names_enable() is False
    assert "off" in tier.label


def test_explicit_off_is_off(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-present")
    monkeypatch.setattr("induction.model_tier._have_sdk", lambda: True)
    assert resolve("off").active is False


def test_auto_downshifts_without_a_key(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("induction.model_tier._have_sdk", lambda: True)
    tier = resolve("auto")                       # the default path — must not raise
    assert tier.active is False
    assert tier.mapper() is None and tier.semantic() is None
    note = capsys.readouterr().err
    assert "default" in note and "ANTHROPIC_API_KEY" in note  # it said why, out loud


def test_auto_activates_with_key_and_sdk(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-present")
    monkeypatch.setattr("induction.model_tier._have_sdk", lambda: True)
    tier = resolve("auto")
    assert tier.active is True and tier.names_enable() is True
    # Providers are real objects, constructed lazily (no SDK import at build time).
    from induction.abstraction import AnthropicActivityMapper, AnthropicRecordClassifier
    from induction.semantic import SemanticProvider
    assert isinstance(tier.mapper(), AnthropicActivityMapper)
    assert isinstance(tier.classifier(), AnthropicRecordClassifier)
    assert isinstance(tier.semantic(), SemanticProvider)


def test_hybrid_adds_the_embedder(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-present")
    monkeypatch.setattr("induction.model_tier._have_sdk", lambda: True)
    tier = resolve("hybrid")
    assert tier.active is True and tier.hybrid is True
    assert tier.semantic().embedder is not None


def test_insist_without_a_key_stops(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("induction.model_tier._have_sdk", lambda: True)
    with pytest.raises(SystemExit) as e:
        resolve("llm")                           # asked for it by name -> refuse
    assert e.value.code == 2
    assert "cannot run" in capsys.readouterr().err
