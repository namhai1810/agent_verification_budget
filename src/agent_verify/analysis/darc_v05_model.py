"""DARC v0.5 Q-success/Q-cost ensemble built on the v0.4 state backbone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agent_verify.analysis.darc_data import EncodedDARC
from agent_verify.analysis.darc_model import _torch, select_device
from agent_verify.analysis.darc_v04_data import MultiHistoryState
from agent_verify.analysis.darc_v04_model import DARCv04Ensemble, _base_arrays, _batch


@dataclass(frozen=True)
class V05Prediction:
    q_success: np.ndarray
    q_cost: np.ndarray
    advantage: np.ndarray
    advantage_uncertainty: np.ndarray
    additional_cost: np.ndarray
    additional_cost_uncertainty: np.ndarray
    effect_probabilities: np.ndarray
    immediate_cost: np.ndarray
    immediate_cost_uncertainty: np.ndarray


@dataclass
class V05Member:
    state_dict: dict[str, Any]
    fold: int
    member: int
    iteration: int
    best_epoch: int
    best_validation_loss: float
    cumulative_cost_log_mean: float
    cumulative_cost_log_scale: float
    immediate_cost_log_mean: float
    immediate_cost_log_scale: float
    history: tuple[dict[str, float | int], ...]


def network_class() -> Any:
    torch = _torch()
    nn = torch.nn

    class DARCv05Network(nn.Module):
        def __init__(self, embedding_dim: int, metadata_dim: int, params: dict[str, Any]) -> None:
            super().__init__()
            hidden = int(params.get("hidden_dim", 128))
            dropout = float(params.get("dropout", 0.15))
            self.segment_projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(embedding_dim, hidden),
                        nn.LayerNorm(hidden),
                        nn.GELU(),
                    )
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
            self.q_success_head = nn.Linear(hidden // 2, 2)
            self.q_cost_head = nn.Linear(hidden // 2, 2)
            self.effect_head = nn.Linear(hidden // 2, 3)
            self.immediate_cost_head = nn.Linear(hidden // 2, 1)

        def forward(self, batch: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
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
            return (
                self.q_success_head(hidden),
                self.q_cost_head(hidden),
                self.effect_head(hidden),
                self.immediate_cost_head(hidden).squeeze(1),
            )

    return DARCv05Network


def transfer_v04(network: Any, source: DARCv04Ensemble, member_index: int) -> int:
    state = source.members[member_index % len(source.members)].state_dict
    current = network.state_dict()
    copied = 0
    aliases = {
        "q_head.weight": "q_success_head.weight",
        "q_head.bias": "q_success_head.bias",
        "cost_head.weight": "immediate_cost_head.weight",
        "cost_head.bias": "immediate_cost_head.bias",
    }
    for source_name, value in state.items():
        target_name = aliases.get(source_name, source_name)
        if target_name in current and tuple(current[target_name].shape) == tuple(value.shape):
            current[target_name] = value.detach().clone()
            copied += int(value.numel())
    network.load_state_dict(current)
    return copied


class DARCv05Ensemble:
    def __init__(
        self,
        *,
        embedding_dim: int,
        metadata_dim: int,
        params: dict[str, Any],
        reference_budget: float,
        metadata_mean: np.ndarray,
        metadata_scale: np.ndarray,
        members: list[V05Member],
    ) -> None:
        if not members:
            raise ValueError("DARC v0.5 ensemble requires at least one member")
        self.embedding_dim = int(embedding_dim)
        self.metadata_dim = int(metadata_dim)
        self.params = dict(params)
        self.reference_budget = float(reference_budget)
        self.metadata_mean = np.asarray(metadata_mean, dtype=np.float32)
        self.metadata_scale = np.asarray(metadata_scale, dtype=np.float32)
        self.members = members

    def members_for_fold(self, fold: int) -> DARCv05Ensemble:
        selected = [member for member in self.members if member.fold == fold]
        return DARCv05Ensemble(
            embedding_dim=self.embedding_dim,
            metadata_dim=self.metadata_dim,
            params=self.params,
            reference_budget=self.reference_budget,
            metadata_mean=self.metadata_mean,
            metadata_scale=self.metadata_scale,
            members=selected,
        )

    def save(self, path: Path) -> None:
        torch = _torch()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": "darc-v0.5-ensemble-v1",
                "embedding_dim": self.embedding_dim,
                "metadata_dim": self.metadata_dim,
                "params": self.params,
                "reference_budget": self.reference_budget,
                "metadata_mean": self.metadata_mean,
                "metadata_scale": self.metadata_scale,
                "members": [
                    {
                        "state_dict": row.state_dict,
                        "fold": row.fold,
                        "member": row.member,
                        "iteration": row.iteration,
                        "best_epoch": row.best_epoch,
                        "best_validation_loss": row.best_validation_loss,
                        "cumulative_cost_log_mean": row.cumulative_cost_log_mean,
                        "cumulative_cost_log_scale": row.cumulative_cost_log_scale,
                        "immediate_cost_log_mean": row.immediate_cost_log_mean,
                        "immediate_cost_log_scale": row.immediate_cost_log_scale,
                        "history": list(row.history),
                    }
                    for row in self.members
                ],
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> DARCv05Ensemble:
        payload = _torch().load(path, map_location="cpu", weights_only=False)
        return cls(
            embedding_dim=payload["embedding_dim"],
            metadata_dim=payload["metadata_dim"],
            params=payload["params"],
            reference_budget=payload["reference_budget"],
            metadata_mean=np.asarray(payload["metadata_mean"]),
            metadata_scale=np.asarray(payload["metadata_scale"]),
            members=[
                V05Member(**{**row, "history": tuple(row.get("history", ()))})
                for row in payload["members"]
            ],
        )

    def predict(
        self,
        encoded: EncodedDARC,
        states: list[MultiHistoryState],
        *,
        allowance: float,
        device_name: str,
    ) -> V05Prediction:
        torch = _torch()
        device = select_device(device_name)
        arrays = _base_arrays(encoded, states, self.metadata_mean, self.metadata_scale)
        count = len(states)
        batch = _batch(
            arrays,
            np.arange(count, dtype=np.int64),
            np.full(count, allowance, dtype=np.float32),
            np.full(count, allowance / max(self.reference_budget, 1.0), dtype=np.float32),
            self.reference_budget,
            device,
        )
        success_rows: list[np.ndarray] = []
        cost_rows: list[np.ndarray] = []
        effect_rows: list[np.ndarray] = []
        immediate_rows: list[np.ndarray] = []
        for member in self.members:
            network = network_class()(self.embedding_dim, self.metadata_dim, self.params).to(device)
            network.load_state_dict(member.state_dict)
            network.eval()
            with torch.inference_mode():
                success_logits, cost_scaled, effect_logits, immediate_scaled = network(batch)
                success_rows.append(torch.sigmoid(success_logits).cpu().numpy())
                cost_rows.append(
                    torch.expm1(
                        cost_scaled * member.cumulative_cost_log_scale
                        + member.cumulative_cost_log_mean
                    ).clamp_min(0).cpu().numpy()
                )
                effect_rows.append(torch.softmax(effect_logits, dim=1).cpu().numpy())
                immediate_rows.append(
                    torch.expm1(
                        immediate_scaled * member.immediate_cost_log_scale
                        + member.immediate_cost_log_mean
                    ).clamp_min(0).cpu().numpy()
                )
        success = np.stack(success_rows)
        cost = np.stack(cost_rows)
        immediate = np.stack(immediate_rows)
        advantages = success[:, :, 1] - success[:, :, 0]
        additional_cost = np.maximum(cost[:, :, 1] - cost[:, :, 0], 0.0)
        return V05Prediction(
            q_success=success.mean(axis=0),
            q_cost=cost.mean(axis=0),
            advantage=advantages.mean(axis=0),
            advantage_uncertainty=advantages.std(axis=0),
            additional_cost=additional_cost.mean(axis=0),
            additional_cost_uncertainty=additional_cost.std(axis=0),
            effect_probabilities=np.stack(effect_rows).mean(axis=0),
            immediate_cost=immediate.mean(axis=0),
            immediate_cost_uncertainty=immediate.std(axis=0),
        )

