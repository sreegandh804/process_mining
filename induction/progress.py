"""Live progress — so a run that takes a few minutes isn't a silent wait.

Three levels, because the honest answer to "what's happening?" has two audiences:

  0  QUIET   — nothing (for scripts / `--quiet`).
  1  STAGE   — one line per pipeline stage, with counts (the default). Answers
               "is it alive, and how far along?" — especially the LLM passes,
               which are the multi-minute part, reported per batch.
  2  VERBOSE — also stream the individual *inferences* (`-v`): each non-obvious
               join with the reason it was made, each rejected kind, each gap.
               Not every record — the counts already cover those — just the
               decisions a reader would want to audit as they happen.

Everything goes to stderr, so it never pollutes `model.json`/`inspector.html` on
stdout redirects. The object is also callable, so it can be passed anywhere a
plain `log(msg)` sink is expected (the LLM seams take exactly that).
"""

from __future__ import annotations

import sys
from typing import Optional


class Progress:
    QUIET = 0
    STAGE = 1
    VERBOSE = 2

    def __init__(self, level: int = STAGE, stream=None):
        self.level = level
        self.stream = stream if stream is not None else sys.stderr

    def stage(self, msg: str) -> None:
        """A pipeline milestone (level >= 1)."""
        if self.level >= self.STAGE:
            print(f"  · {msg}", file=self.stream, flush=True)

    def detail(self, msg: str) -> None:
        """One audited decision (level >= 2)."""
        if self.level >= self.VERBOSE:
            print(f"      → {msg}", file=self.stream, flush=True)

    @property
    def verbose(self) -> bool:
        return self.level >= self.VERBOSE

    # A Progress is a drop-in `log(msg)` sink: the LLM seams call it as a plain
    # function, and those messages count as stage-level.
    def __call__(self, msg: str) -> None:
        self.stage(msg)


# A do-nothing sink — the default everywhere in the library, so `induce()` and the
# providers stay silent (and untouched for tests) unless a runner opts in.
NULL = Progress(level=Progress.QUIET)


def from_flags(quiet: bool = False, verbose: bool = False, stream=None) -> Progress:
    """Build a Progress from a runner's `--quiet` / `-v` flags. Default is STAGE."""
    level = Progress.QUIET if quiet else (Progress.VERBOSE if verbose else Progress.STAGE)
    return Progress(level=level, stream=stream)
