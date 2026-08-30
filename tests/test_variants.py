"""Step 4 — Variants: real traces with frequencies; exceptions stay visible.

Asserted on the DEFAULT (generic) model, so it tests the source-agnostic
behaviour, not the git vocabulary.
"""


def _kind_with_action(model, action):
    for k in model.kinds:
        if any(action in v.signature for v in k.variants):
            return k
    raise AssertionError(f"no kind contains a '{action}' variant")


def test_the_revert_run_reads_as_an_exception_not_the_common_path(mini_model):
    k = _kind_with_action(mini_model, "reverted")
    exception = [v for v in k.variants if "reverted" in v.signature]
    assert exception and exception[0].role == "exception"


def test_variants_are_real_traces_with_frequencies(mini_model):
    for k in mini_model.kinds:
        for v in k.variants:
            assert v.frequency >= 1
            assert isinstance(v.signature, tuple)
            assert v.case_ids            # every variant points back to its runs


def test_dfg_is_present_but_caveated(mini_model):
    k = _kind_with_action(mini_model, "merged")
    assert "caveat" in k.dfg
    assert k.dfg["nodes"] and k.dfg["edges"]
