"""Exact differentiable objective over all eight terminal trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_verify.analysis.darc_fulltree_data import PREFIX_INDEX, SEQUENCES


@dataclass(frozen=True)
class FullTreeExpectation:
    probabilities: Any
    expected_correctness: Any
    expected_tokens: Any
    expected_calls: Any
    expected_kl: Any
    expected_entropy: Any
    loss: Any


def trajectory_probabilities(verify_probability: Any) -> Any:
    """Return [batch, 8] path probabilities from [batch, 7] state policies."""

    torch = __import__("torch")
    rows = []
    for sequence in SEQUENCES:
        probability = torch.ones(
            verify_probability.shape[0],
            dtype=verify_probability.dtype,
            device=verify_probability.device,
        )
        prefix = ""
        for action in sequence:
            p_verify = verify_probability[:, PREFIX_INDEX[prefix]]
            probability = probability * (p_verify if action == "V" else 1.0 - p_verify)
            prefix += action
        rows.append(probability)
    return torch.stack(rows, dim=1)


def state_reach_probabilities(verify_probability: Any) -> Any:
    """Probability that the policy reaches each of the seven observable prefixes."""

    torch = __import__("torch")
    p = verify_probability
    root = torch.ones_like(p[:, PREFIX_INDEX[""]])
    reach_s = 1.0 - p[:, PREFIX_INDEX[""]]
    reach_v = p[:, PREFIX_INDEX[""]]
    reach_ss = reach_s * (1.0 - p[:, PREFIX_INDEX["S"]])
    reach_sv = reach_s * p[:, PREFIX_INDEX["S"]]
    reach_vs = reach_v * (1.0 - p[:, PREFIX_INDEX["V"]])
    reach_vv = reach_v * p[:, PREFIX_INDEX["V"]]
    return torch.stack((root, reach_s, reach_v, reach_ss, reach_sv, reach_vs, reach_vv), dim=1)


def fulltree_expectation(
    verify_probability: Any,
    outcomes: Any,
    trajectory_tokens: Any,
    *,
    target_tokens: float,
    reference_tokens: float,
    lagrange_multiplier: float,
    anchor_probability: Any | None = None,
    kl_weight: float = 0.0,
    entropy_weight: float = 0.0,
) -> FullTreeExpectation:
    """Compute the exact terminal objective without forced-action labels."""

    torch = __import__("torch")
    eps = 1e-6
    probability = trajectory_probabilities(verify_probability)
    expected_correctness = (probability * outcomes).sum(dim=1)
    expected_tokens = (probability * trajectory_tokens).sum(dim=1)
    calls = torch.as_tensor(
        [sequence.count("V") for sequence in SEQUENCES],
        dtype=probability.dtype,
        device=probability.device,
    )
    expected_calls = (probability * calls.unsqueeze(0)).sum(dim=1)
    reach = state_reach_probabilities(verify_probability)
    p = verify_probability.clamp(eps, 1.0 - eps)
    entropy_per_state = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    expected_entropy = (reach * entropy_per_state).sum(dim=1) / 3.0
    expected_kl = torch.zeros_like(expected_entropy)
    if anchor_probability is not None:
        q = anchor_probability.clamp(eps, 1.0 - eps)
        kl = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
        expected_kl = (reach * kl).sum(dim=1) / 3.0
    normalized_gap = (expected_tokens - float(target_tokens)) / max(float(reference_tokens), 1.0)
    objective = (
        expected_correctness
        - float(lagrange_multiplier) * normalized_gap
        - float(kl_weight) * expected_kl
        + float(entropy_weight) * expected_entropy
    )
    return FullTreeExpectation(
        probabilities=probability,
        expected_correctness=expected_correctness,
        expected_tokens=expected_tokens,
        expected_calls=expected_calls,
        expected_kl=expected_kl,
        expected_entropy=expected_entropy,
        loss=-objective.mean(),
    )


def edge_expected_tokens(verify_probability: Any, edge_tokens: Any) -> Any:
    """Independent cost identity used to detect shared-prefix double counting."""

    return (state_reach_probabilities(verify_probability) * verify_probability * edge_tokens).sum(dim=1)
