"""The induction engine.

Turns a real, messy corpus of *artefacts* (here: the git history of an
open-source repository) into a *believable, traceable process model* — the
common path, the variants, the exceptions, and the steps no system recorded —
with every claim pointing back to its evidence and carrying a confidence tier.

See README.md for the whole story. The pipeline is steps 0-6 in `steps/`,
wired together by `pipeline.py`. Everything normalises into the three
canonical record types in `model.py` before any mining happens.
"""

__version__ = "0.1.0"
