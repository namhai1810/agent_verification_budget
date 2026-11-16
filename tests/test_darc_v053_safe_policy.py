from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from agent_verify.analysis.darc_v05_env import OfflineObservation
from agent_verify.analysis.darc_v05_model import V05Prediction
from agent_verify.analysis.darc_v053_safe_policy import (
    ABLATION_A,
    SafeImprovementPolicy,
    safe_improved_action,
    safe_policy_grid,
)


def _prediction(*, advantage: float, uncertainty: float, call: float = 100.0) -> V05Prediction:
    return V05Prediction(
        q_success=np.asarray([[0.4, 0.4 + advantage]]),
        q_cost=np.asarray([[0.0, call]]),
        advantage=np.asarray([advantage]),
        advantage_uncertainty=np.asarray([uncertainty]),
        additional_cost=np.asarray([call]),
        additional_cost_uncertainty=np.asarray([0.0]),
        effect_probabilities=np.asarray([[0.01, 0.79, 0.20]]),
        immediate_cost=np.asarray([call]),
        immediate_cost_uncertainty=np.asarray([0.0]),
    )


def _observation(*, allowance: float = 1000.0, remaining: int = 10000) -> OfflineObservation:
    return OfflineObservation("task", "", 1, remaining, allowance, 0)


def test_safe_policy_retains_baseline_when_confidence_interval_crosses_margin() -> None:
    policy = SafeImprovementPolicy((0.0, 0.0, 0.0), 1.0, 0.01, 0.0, 0.0, False)
    action, audit = safe_improved_action(
        _observation(),
        baseline_action="S",
        prediction=_prediction(advantage=0.05, uncertainty=0.1),
        policy=policy,
        reference_budget=4912,
        hard_call_token_cap=8192,
    )
    assert action == "S"
    assert audit["switched"] == 0


def test_safe_policy_switches_only_with_confident_evidence() -> None:
    policy = SafeImprovementPolicy((0.0, 0.0, 0.0), 1.0, 0.01, 0.0, 0.0, False)
    action, audit = safe_improved_action(
        _observation(),
        baseline_action="S",
        prediction=_prediction(advantage=0.2, uncertainty=0.01),
        policy=policy,
        reference_budget=4912,
        hard_call_token_cap=8192,
    )
    assert action == "V"
    assert audit["switch_reason"] == "confident_skip_to_verify"


def test_dynamic_shadow_price_is_more_conservative_at_low_allowance() -> None:
    policy = SafeImprovementPolicy((0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.5, True, 1.0)
    prediction = _prediction(advantage=0.1, uncertainty=0.0, call=1000.0)
    high = policy.interval(prediction, 0, checkpoint=1, allowance=4912, reference_budget=4912)[2]
    low = policy.interval(prediction, 0, checkpoint=1, allowance=500, reference_budget=4912)[2]
    assert low < high


def test_config_forbids_frozen_test_and_preserves_original_checkpoints() -> None:
    config = yaml.safe_load(Path("configs/darc_v053_safe_policy.yaml").read_text())
    assert config["data"]["frozen_test_access"] == "forbidden"
    assert "myopic_targets/seed_{seed}/iteration_0.pt" in config["baseline"]["checkpoint_template"]
    assert config["baseline"]["require_exact_reproduction"] is True
    assert ABLATION_A.startswith("A")
    assert safe_policy_grid(config, dynamic=False)
