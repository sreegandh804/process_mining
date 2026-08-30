"""The pipeline — steps 0-6 (brief §4).

Discipline that applies to every step:
  1. Build the deterministic, inspectable baseline first. Only reach for
     embeddings / fuzzy matching / an LLM where the baseline *demonstrably*
     fails — and correlation (step 2) is the likeliest place it will.
  2. Any inferred field carries confidence + evidence. Inference is never fact.
  3. Prefer transparent over clever. If we cannot explain why a claim was made,
     we do not make it.
"""
