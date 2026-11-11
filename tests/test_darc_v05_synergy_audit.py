from __future__ import annotations

from types import SimpleNamespace

from agent_verify.analysis.darc_v05_synergy_audit import (
    SEQUENCES,
    _cheapest_success,
    interaction_rows,
    marginal_rows,
    proper_nonempty_verify_subsets,
    strict_synergy,
)


def _group(outcomes: dict[str, int], costs: dict[str, int] | None = None) -> SimpleNamespace:
    costs = costs or {}
    rows = {
        sequence: {
            "passed": outcomes.get(sequence, 0),
            "teacher_total_tokens": costs.get(sequence, 100 * sequence.count("V")),
        }
        for sequence in SEQUENCES
    }
    task = SimpleNamespace(dataset="synthetic", task_id="0")
    return SimpleNamespace(task=task, rows=rows)


def test_strict_synergy_checks_every_proper_subset() -> None:
    outcomes = {sequence: 0 for sequence in SEQUENCES}
    outcomes["VVV"] = 1
    group = _group(outcomes)
    assert set(proper_nonempty_verify_subsets("VVV")) == {
        "VSS",
        "SVS",
        "SSV",
        "VVS",
        "VSV",
        "SVV",
    }
    assert strict_synergy(group, "VVV")
    outcomes["VVS"] = 1
    assert not strict_synergy(_group(outcomes), "VVV")


def test_strict_pair_synergy_and_conditional_marginal_effects() -> None:
    outcomes = {sequence: 0 for sequence in SEQUENCES}
    outcomes["VVS"] = 1
    group = _group(outcomes)
    assert strict_synergy(group, "VVS")
    marginals = marginal_rows("train", "synthetic::0", group)
    assert len(marginals) == 12
    cp2_after_v = next(
        row
        for row in marginals
        if row["checkpoint"] == 2 and row["prefix"] == "V" and row["future_suffix"] == "S"
    )
    assert cp2_after_v["skip_sequence"] == "VSS"
    assert cp2_after_v["verify_sequence"] == "VVS"
    assert cp2_after_v["effect"] == "helpful"
    interactions = interaction_rows("train", "synthetic::0", group)
    assert any(
        row["checkpoint_pair"] == "CP1×CP2"
        and row["condition_action"] == "S"
        and row["strict_pair_synergy"] == 1
        for row in interactions
    )


def test_cheapest_successful_verify_sequence_uses_actual_tokens() -> None:
    outcomes = {sequence: 0 for sequence in SEQUENCES}
    outcomes.update({"VSS": 1, "VVS": 1, "VVV": 1})
    costs = {"VSS": 500, "VVS": 350, "VVV": 900}
    group = _group(outcomes, costs)
    assert _cheapest_success(group, require_verify=True) == "VVS"
