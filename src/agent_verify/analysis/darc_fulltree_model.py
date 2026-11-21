"""Budget-conditioned DARC policy network for exact full-tree optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agent_verify.analysis.darc_data import EncodedDARC
from agent_verify.analysis.darc_model import _torch, select_device
from agent_verify.analysis.darc_v04_data import MultiHistoryState
from agent_verify.analysis.darc_v04_model import _base_arrays, _batch


def network_class() -> Any:
    torch = _torch()
    nn = torch.nn

    class DARCFullTreePolicy(nn.Module):
        def __init__(self, embedding_dim: int, metadata_dim: int, params: dict[str, Any]) -> None:
            super().__init__()
            hidden = int(params.get("hidden_dim", 128))
            dropout = float(params.get("dropout", 0.15))
            self.segment_projections = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(embedding_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
                    for _ in range(4)
                ]
            )
            self.state_fusion = nn.Sequential(
                nn.Linear(hidden * 4 + metadata_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.downstream_projection = nn.Sequential(
                nn.Linear(embedding_dim, hidden), nn.LayerNorm(hidden), nn.GELU()
            )
            self.downstream_attention = nn.MultiheadAttention(
                hidden,
                int(params.get("attention_heads", 4)),
                dropout=dropout,
                batch_first=True,
            )
            self.context_gate = nn.Linear(hidden * 2, hidden)
            self.context_norm = nn.LayerNorm(hidden)
            self.trunk = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
            )
            self.action_embedding = nn.Embedding(3, hidden // 4, padding_idx=0)
            self.action_projection = nn.Sequential(nn.Linear(hidden // 2, hidden // 2), nn.GELU())
            self.budget_projection = nn.Sequential(
                nn.Linear(3, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU()
            )
            self.joint = nn.Sequential(
                nn.Linear(3 * (hidden // 2), hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
            )
            self.policy_head = nn.Linear(hidden // 2, 1)

        def forward(self, batch: dict[str, Any]) -> Any:
            projected = [
                layer(batch[name])
                for layer, name in zip(
                    self.segment_projections,
                    ("task", "artifact", "history", "current_role"),
                    strict=True,
                )
            ]
            state = self.state_fusion(torch.cat((*projected, batch["metadata"]), dim=1))
            downstream = self.downstream_projection(batch["downstream"])
            context, _ = self.downstream_attention(
                state.unsqueeze(1),
                downstream,
                downstream,
                key_padding_mask=~batch["downstream_mask"],
                need_weights=False,
            )
            context = context.squeeze(1)
            gate = torch.sigmoid(self.context_gate(torch.cat((state, context), dim=1)))
            semantic = self.trunk(self.context_norm(state + gate * context))
            actions = self.action_projection(self.action_embedding(batch["actions"]).flatten(1))
            budget = self.budget_projection(batch["budget"])
            hidden = self.joint(torch.cat((semantic, actions, budget), dim=1))
            return self.policy_head(hidden).squeeze(1)

    return DARCFullTreePolicy


@dataclass
class FullTreePolicyCheckpoint:
    state_dict: dict[str, Any]
    embedding_dim: int
    metadata_dim: int
    params: dict[str, Any]
    reference_budget: float
    metadata_mean: np.ndarray
    metadata_scale: np.ndarray
    variant: str
    seed: int
    fold: int
    history: tuple[dict[str, Any], ...]
    lagrange_multipliers: dict[str, float]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _torch().save({"schema_version": "darc-fulltree-policy-v1", **self.__dict__}, path)

    @classmethod
    def load(cls, path: Path) -> FullTreePolicyCheckpoint:
        payload = _torch().load(path, map_location="cpu", weights_only=False)
        if payload.pop("schema_version") != "darc-fulltree-policy-v1":
            raise ValueError(f"Unsupported Full-Tree checkpoint: {path}")
        payload["metadata_mean"] = np.asarray(payload["metadata_mean"], dtype=np.float32)
        payload["metadata_scale"] = np.asarray(payload["metadata_scale"], dtype=np.float32)
        payload["history"] = tuple(payload.get("history", ()))
        return cls(**payload)

    def instantiate(self, device_name: str) -> tuple[Any, Any]:
        device = select_device(device_name)
        network = network_class()(self.embedding_dim, self.metadata_dim, self.params).to(device)
        network.load_state_dict(self.state_dict)
        network.eval()
        return network, device


def base_arrays(
    encoded: EncodedDARC,
    states: list[MultiHistoryState],
    metadata_mean: np.ndarray,
    metadata_scale: np.ndarray,
) -> dict[str, np.ndarray]:
    return _base_arrays(encoded, states, metadata_mean, metadata_scale)


def policy_batch(
    arrays: dict[str, np.ndarray],
    state_indices: np.ndarray,
    allowances: np.ndarray,
    *,
    reference_budget: float,
    device: Any,
) -> dict[str, Any]:
    values = allowances.astype(np.float32)
    return _batch(
        arrays,
        state_indices.astype(np.int64),
        values,
        values / max(float(reference_budget), 1.0),
        reference_budget,
        device,
    )
