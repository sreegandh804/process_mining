"""Whether the model (LLM) tier runs — now DEFAULT ON, with an honest OFF.

The engine's model tier — naming (`naming.py`), activity abstraction
(`abstraction.py`) and, where a runner supports it, semantic correlation
(`semantic.py`) — used to be strictly opt-in behind `--names llm` / `--semantic
llm`. It is now **on by default**: a runner wires the real Anthropic providers
unless told not to.

Making a network/paid tier the default creates exactly one hazard the rest of
this codebase is built to avoid — a deterministic run emitted while *labelled*
as an AI run, because every AI path returns empty on a missing key rather than
breaking. So this module is the single place that decides, per invocation,
whether the tier can actually run, and it is loud about the answer:

  - **off** (`--no-llm`, or an explicit `off`): the deterministic, offline,
    stdlib baseline. No providers, no label.
  - **auto** (the default): use the model tier *if* a key and the SDK are
    present; otherwise **downshift to deterministic and say so in one line**, so
    the output is never an AI label over a raw-verb run. Never raises — an
    offline machine still gets a run.
  - **insist** (an explicit `llm` / `hybrid`): the user asked for the model by
    name, so a missing key is an error, not a downshift — print why and stop,
    rather than pretend.

The providers themselves are still injected and still no-op safely without a
key; this only decides *whether to inject them* and keeps the reported mode
truthful. The core library defaults are untouched (`CorrelationPolicy.semantic`
is still `None`, `infer_names` still defaults `enable=False`), so the engine
stays deterministic and testable when driven directly — the default flips at the
runner seam, which is where the flag lived.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional


def _have_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _have_sdk() -> bool:
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class ModelTier:
    """The resolved decision, plus lazy builders for the providers a runner needs.

    `active` is the single source of truth: True only when the tier will really
    reach the model. `label` is what a runner prints in its header so the mode is
    never a guess.
    """

    active: bool
    label: str
    hybrid: bool = False

    def names_enable(self) -> bool:
        return self.active

    def mapper(self, log=None):
        """Tier-1 activity mapper (verbs -> activities), or None when off."""
        if not self.active:
            return None
        from induction.abstraction import AnthropicActivityMapper
        return AnthropicActivityMapper(log=log)

    def classifier(self, log=None):
        """Tier-2 record reader (for transport-only verbs), or None when off."""
        if not self.active:
            return None
        from induction.abstraction import AnthropicRecordClassifier
        return AnthropicRecordClassifier(log=log)

    def semantic(self, log=None):
        """Semantic-correlation provider (same-work judge, + optional embedder),
        or None when off. Only runners that pass this into a `CorrelationPolicy`
        use it."""
        if not self.active:
            return None
        from induction.semantic import AnthropicJudge, SemanticProvider, VoyageEmbedder
        if self.hybrid:
            return SemanticProvider(judge=AnthropicJudge(log=log), embedder=VoyageEmbedder())
        return SemanticProvider(judge=AnthropicJudge(log=log))


def resolve(mode: str = "auto", *, no_llm: bool = False, stream=None) -> ModelTier:
    """Decide whether the model tier runs for this invocation.

    `mode` is a runner flag value: "auto" (default, on-if-available),
    "off"/"none", "on"/"llm", or "hybrid". `no_llm=True` (a `--no-llm` switch)
    forces off and wins over `mode`.

    Returns a `ModelTier`. Raises `SystemExit(2)` only when the model was asked
    for *by name* (`llm`/`hybrid`) and cannot run — never for the default path.
    """
    # Resolve the stream at call time (not as a default argument), so a test's
    # patched sys.stderr — or any later reassignment — is the one written to.
    if stream is None:
        stream = sys.stderr
    mode = (mode or "auto").lower()
    if no_llm or mode in ("off", "none", "false"):
        return ModelTier(active=False, label="off (--no-llm)" if no_llm else "off")

    insist = mode in ("llm", "on", "hybrid", "true")
    hybrid = mode == "hybrid"

    problems: list[str] = []
    if not _have_key():
        problems.append("ANTHROPIC_API_KEY is not set")
    if not _have_sdk():
        problems.append("the Anthropic SDK is missing (pip install anthropic)")

    if not problems:
        label = "on (Claude)" + (" + embedding shortlist" if hybrid else "")
        return ModelTier(active=True, label=label, hybrid=hybrid)

    if insist:
        # The model was requested explicitly — do not quietly hand back a
        # deterministic run wearing an 'llm' label. Say why, and stop.
        print("[llm] you asked for the model tier, and it cannot run:", file=stream)
        for p in problems:
            print(f"        - {p}", file=stream)
        print("      Fix the above, or pass --no-llm for the deterministic baseline.",
              file=stream)
        raise SystemExit(2)

    # The default path: downshift honestly rather than fail. One line, so nobody
    # spends an afternoon wondering whether the AI ran (it did not).
    print(f"[llm] model tier is on by default, but {problems[0]} — running the "
          f"deterministic baseline (raw verbs, no AI naming/abstraction). "
          f"Set a key to enable it, or pass --no-llm to silence this note.",
          file=stream)
    return ModelTier(active=False, label=f"off ({problems[0]})")
