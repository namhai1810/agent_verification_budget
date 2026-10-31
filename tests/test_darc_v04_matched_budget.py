from pathlib import Path
import inspect

import pytest

from agent_verify.analysis.darc_v04_matched_budget import parse_budgets, run


def test_parse_budgets_is_sorted_unique_and_positive() -> None:
    assert parse_budgets("250000, 100000,150000,100000") == [100000, 150000, 250000]
    with pytest.raises(ValueError):
        parse_budgets("0,100000")


def test_matched_budget_runner_is_validation_only_and_does_not_train() -> None:
    source = inspect.getsource(run)
    assert '_load_groups(frozen_root, "validation")' in source
    assert '_load_groups(frozen_root, "test")' not in source
    assert "train_ensemble" not in source
    assert '"frozen_test_loaded": False' in source
    assert Path("scripts/run_darc_v04_matched_budget.sh").exists()

