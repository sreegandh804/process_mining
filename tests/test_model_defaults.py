"""The generative calls default to Claude Opus 5.5 and keep `high` effort.

Opus 5.5 lowered its default effort to `medium`; discovery sets the vocabulary
everything downstream is graded against, so it asks for `high` explicitly. Other
models get no `effort` at all, because some (Haiku 4.5) reject it.
"""

from induction.anthropic_call import DEFAULT_OPUS, effort_kwargs


def test_default_is_opus_5_5():
    assert DEFAULT_OPUS == "claude-opus-5-5"


def test_effort_is_pinned_high_only_on_opus_5_5():
    assert effort_kwargs("claude-opus-5-5") == {"output_config": {"effort": "high"}}
    assert effort_kwargs("claude-haiku-4-5") == {}
    assert effort_kwargs("claude-sonnet-5") == {}
