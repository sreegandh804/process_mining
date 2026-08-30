"""Step 4 — Variants: real traces with frequencies; exceptions stay visible."""


def _kind(model, kid):
    k = next((k for k in model.kinds if k.id == kid), None)
    assert k is not None, f"expected kind {kid}"
    return k


def test_contribution_kind_distinguishes_the_exception_trace(mini_model):
    k = _kind(mini_model, "code_contribution")
    assert len(k.variants) == 2
    exception = [v for v in k.variants if v.role == "exception"]
    assert exception, "the revert run should read as an exception, not the common path"
    assert "reverted" in exception[0].signature


def test_variants_are_real_traces_with_frequencies(mini_model):
    k = _kind(mini_model, "code_contribution")
    for v in k.variants:
        assert v.frequency >= 1
        assert isinstance(v.signature, tuple)
        assert v.case_ids  # every variant points back to the runs that made it


def test_dfg_is_present_but_caveated(mini_model):
    k = _kind(mini_model, "code_contribution")
    assert "caveat" in k.dfg
    assert k.dfg["nodes"] and k.dfg["edges"]
