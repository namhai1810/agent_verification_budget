from agent_verify.analysis.phase2_optimization import boosting_grid, monotone_threshold_profiles


def test_optimization_search_space_is_bounded_and_monotone() -> None:
    grid = boosting_grid()
    profiles = monotone_threshold_profiles()
    assert len(grid) == 45
    assert len(profiles) == 286
    assert all(cp1 >= cp2 >= cp3 for cp1, cp2, cp3 in profiles)
