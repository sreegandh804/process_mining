"""Profiles: the DEFAULT is source-agnostic and unnamed; a profile only adds
vocabulary, never structure. This is what makes the engine worth having on data
it has never seen — an accounting firm, not just a git repo.
"""


def test_default_is_generic_unnamed_kinds_and_activities(mini_model):
    assert mini_model.profile_id == "generic"
    # kinds are discovered and left UNNAMED, with data-derived rationales
    for k in mini_model.kinds:
        assert k.id.startswith("kind_")
        assert k.name.startswith("Kind ")
        assert "data-derived" in k.rationale
        assert k.features  # it clustered on real structural features
    # activities are named by the source's own action verb — no rename table
    names = {s.name for s in mini_model.steps}
    assert {"authored", "merged"} <= names        # raw actions, untouched


def test_git_profile_adds_names_without_changing_structure(mini_model, mini_model_git):
    assert mini_model_git.profile_id == "git"
    # friendly names appear
    assert any(k.id == "code_contribution" for k in mini_model_git.kinds)
    assert {"Merge pull request", "Author change"} <= {s.name for s in mini_model_git.steps}

    # ...but the STRUCTURE is identical: same cases, orphans, merges, gaps.
    assert set(mini_model.cases) == set(mini_model_git.cases)
    assert len(mini_model.orphans) == len(mini_model_git.orphans)
    assert len(mini_model.merges) == len(mini_model_git.merges)
    assert len(mini_model.gaps) == len(mini_model_git.gaps)


def test_both_profiles_reject_the_bot_cluster(mini_model, mini_model_git):
    # generic: because it is automated + recurring; git: because it also moves no code
    assert any(k.rejected for k in mini_model.kinds)
    assert any(k.rejected for k in mini_model_git.kinds)
