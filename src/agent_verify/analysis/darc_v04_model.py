"""Neural multi-history Q ensemble for DARC v0.4."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from agent_verify.analysis.darc_data import EncodedDARC
from agent_verify.analysis.darc_model import (
    DARCEnsemble,
    _effective_class_weights,
    _torch,
    select_device,
    set_determinism,
)
from agent_verify.analysis.darc_v04_data import MultiHistoryState, TargetTable


@dataclass(frozen=True)
class V04Prediction:
    q_skip: np.ndarray
    q_verify: np.ndarray
    q_advantage: np.ndarray
    uncertainty: np.ndarray
    effect_probabilities: np.ndarray
    expected_cost: np.ndarray


@dataclass
class V04Member:
    state_dict: dict[str, Any]
    best_epoch: int
    best_validation_loss: float
    cost_log_mean: float
    cost_log_scale: float
    history: tuple[dict[str, float | int], ...]


def _prefix_tokens(states: list[MultiHistoryState]) -> np.ndarray:
    # 0=padding, 1=Skip, 2=Verify.
    output = np.zeros((len(states), 2), dtype=np.int64)
    for index, state in enumerate(states):
        for position, action in enumerate(state.prefix[:2]):
            output[index, position] = 1 if action == "S" else 2
    return output


def _network_class() -> Any:
    torch = _torch()
    nn = torch.nn

    class DARCv04Network(nn.Module):
        def __init__(
            self,
            embedding_dim: int,
            metadata_dim: int,
            params: dict[str, Any],
            *,
            use_history: bool,
            budget_conditioning: bool,
        ) -> None:
            super().__init__()
            hidden = int(params.get("hidden_dim", 128))
            dropout = float(params.get("dropout", 0.15))
            self.use_history = use_history
            self.budget_conditioning = budget_conditioning
            # Names and shapes intentionally match v0.3 for compatible transfer.
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
            self.action_projection = nn.Sequential(
                nn.Linear(hidden // 2, hidden // 2), nn.GELU()
            )
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
            self.q_head = nn.Linear(hidden // 2, 2)
            self.effect_head = nn.Linear(hidden // 2, 3)
            self.cost_head = nn.Linear(hidden // 2, 1)

        def forward(self, batch: dict[str, Any]) -> tuple[Any, Any, Any]:
            segments = [
                batch["task"],
                batch["artifact"],
                batch["history"],
                batch["current_role"],
            ]
            projected = [
                layer(value)
                for layer, value in zip(self.segment_projections, segments, strict=True)
            ]
            if not self.use_history:
                projected[2] = torch.zeros_like(projected[2])
            state = self.state_fusion(
                torch.cat((*projected, batch["metadata"]), dim=1)
            )
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
            state = self.context_norm(state + gate * context)
            semantic = self.trunk(state)
            actions = self.action_embedding(batch["actions"]).flatten(1)
            actions = self.action_projection(actions)
            if not self.use_history:
                actions = torch.zeros_like(actions)
            budget = self.budget_projection(batch["budget"])
            if not self.budget_conditioning:
                budget = torch.zeros_like(budget)
            hidden = self.joint(torch.cat((semantic, actions, budget), dim=1))
            return self.q_head(hidden), self.effect_head(hidden), self.cost_head(hidden).squeeze(1)

    return DARCv04Network


def _base_arrays(
    encoded: EncodedDARC,
    states: list[MultiHistoryState],
    mean: np.ndarray,
    scale: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "task": encoded.task.astype(np.float32),
        "artifact": encoded.artifact.astype(np.float32),
        "history": encoded.history.astype(np.float32),
        "current_role": encoded.current_role.astype(np.float32),
        "downstream": encoded.downstream.astype(np.float32),
        "downstream_mask": encoded.downstream_mask.astype(bool),
        "metadata": ((encoded.metadata - mean) / scale).astype(np.float32),
        "actions": _prefix_tokens(states),
    }


def _batch(
    arrays: dict[str, np.ndarray],
    state_indices: np.ndarray,
    allowance: np.ndarray,
    allowance_fraction: np.ndarray,
    reference_budget: float,
    device: Any,
) -> dict[str, Any]:
    torch = _torch()
    indices = state_indices.astype(np.int64)
    result: dict[str, Any] = {}
    for key, value in arrays.items():
        dtype = torch.bool if value.dtype == bool else torch.long if value.dtype.kind in "iu" else torch.float32
        result[key] = torch.as_tensor(value[indices], dtype=dtype, device=device)
    budget = np.column_stack(
        (
            allowance_fraction,
            np.log1p(np.maximum(allowance, 0.0)) / math.log1p(max(reference_budget, 1.0)),
            np.minimum(np.maximum(allowance / max(reference_budget, 1.0), 0.0), 2.0),
        )
    ).astype(np.float32)
    result["budget"] = torch.as_tensor(budget, dtype=torch.float32, device=device)
    return result


def _transfer_v03(network: Any, ensemble: DARCEnsemble, member_index: int) -> int:
    source = ensemble.members[member_index % len(ensemble.members)].state_dict
    current = network.state_dict()
    copied = 0
    for name, value in source.items():
        if name in current and tuple(current[name].shape) == tuple(value.shape):
            current[name] = value.detach().clone()
            copied += int(value.numel())
    network.load_state_dict(current)
    return copied


class DARCv04Ensemble:
    def __init__(
        self,
        *,
        embedding_dim: int,
        metadata_dim: int,
        params: dict[str, Any],
        use_history: bool,
        budget_conditioning: bool,
        reference_budget: float,
        metadata_mean: np.ndarray,
        metadata_scale: np.ndarray,
        members: list[V04Member],
    ) -> None:
        self.embedding_dim = embedding_dim
        self.metadata_dim = metadata_dim
        self.params = dict(params)
        self.use_history = use_history
        self.budget_conditioning = budget_conditioning
        self.reference_budget = float(reference_budget)
        self.metadata_mean = metadata_mean.astype(np.float32)
        self.metadata_scale = metadata_scale.astype(np.float32)
        self.members = members

    def save(self, path: Path) -> None:
        torch = _torch()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": "darc-v0.4-ensemble-v1",
                "embedding_dim": self.embedding_dim,
                "metadata_dim": self.metadata_dim,
                "params": self.params,
                "use_history": self.use_history,
                "budget_conditioning": self.budget_conditioning,
                "reference_budget": self.reference_budget,
                "metadata_mean": self.metadata_mean,
                "metadata_scale": self.metadata_scale,
                "members": [
                    {
                        "state_dict": member.state_dict,
                        "best_epoch": member.best_epoch,
                        "best_validation_loss": member.best_validation_loss,
                        "cost_log_mean": member.cost_log_mean,
                        "cost_log_scale": member.cost_log_scale,
                        "history": list(member.history),
                    }
                    for member in self.members
                ],
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "DARCv04Ensemble":
        torch = _torch()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            embedding_dim=int(payload["embedding_dim"]),
            metadata_dim=int(payload["metadata_dim"]),
            params=dict(payload["params"]),
            use_history=bool(payload["use_history"]),
            budget_conditioning=bool(payload["budget_conditioning"]),
            reference_budget=float(payload["reference_budget"]),
            metadata_mean=np.asarray(payload["metadata_mean"]),
            metadata_scale=np.asarray(payload["metadata_scale"]),
            members=[
                V04Member(
                    state_dict=row["state_dict"],
                    best_epoch=int(row["best_epoch"]),
                    best_validation_loss=float(row["best_validation_loss"]),
                    cost_log_mean=float(row["cost_log_mean"]),
                    cost_log_scale=float(row["cost_log_scale"]),
                    history=tuple(row.get("history", ())),
                )
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
    ) -> V04Prediction:
        torch = _torch()
        device = select_device(device_name)
        arrays = _base_arrays(
            encoded, states, self.metadata_mean, self.metadata_scale
        )
        indices = np.arange(len(states), dtype=np.int64)
        batch = _batch(
            arrays,
            indices,
            np.full(len(states), allowance, dtype=np.float32),
            np.full(
                len(states), allowance / max(self.reference_budget, 1.0), dtype=np.float32
            ),
            self.reference_budget,
            device,
        )
        q_members: list[np.ndarray] = []
        effects: list[np.ndarray] = []
        costs: list[np.ndarray] = []
        for member in self.members:
            network = _network_class()(
                self.embedding_dim,
                self.metadata_dim,
                self.params,
                use_history=self.use_history,
                budget_conditioning=self.budget_conditioning,
            ).to(device)
            network.load_state_dict(member.state_dict)
            network.eval()
            with torch.inference_mode():
                q_logits, effect_logits, cost_scaled = network(batch)
                q_members.append(torch.sigmoid(q_logits).cpu().numpy())
                effects.append(torch.softmax(effect_logits, dim=1).cpu().numpy())
                costs.append(
                    torch.expm1(
                        cost_scaled * member.cost_log_scale + member.cost_log_mean
                    ).clamp_min(0.0).cpu().numpy()
                )
        q = np.stack(q_members)
        advantage = q[:, :, 1] - q[:, :, 0]
        mean_q = q.mean(axis=0)
        return V04Prediction(
            q_skip=mean_q[:, 0],
            q_verify=mean_q[:, 1],
            q_advantage=advantage.mean(axis=0),
            uncertainty=advantage.std(axis=0),
            effect_probabilities=np.stack(effects).mean(axis=0),
            expected_cost=np.stack(costs).mean(axis=0),
        )


def train_ensemble(
    train_states: list[MultiHistoryState],
    train_encoded: EncodedDARC,
    train_targets: TargetTable,
    validation_states: list[MultiHistoryState],
    validation_encoded: EncodedDARC,
    validation_targets: TargetTable,
    *,
    params: dict[str, Any],
    use_history: bool,
    budget_conditioning: bool,
    reference_budget: float,
    pretrained: DARCEnsemble | None,
    seed: int,
    deterministic: bool,
    device_name: str,
    description: str,
    progress: dict[str, Any],
) -> DARCv04Ensemble:
    torch = _torch()
    device = select_device(device_name)
    mean = train_encoded.metadata.mean(axis=0)
    scale = train_encoded.metadata.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_arrays = _base_arrays(train_encoded, train_states, mean, scale)
    validation_arrays = _base_arrays(validation_encoded, validation_states, mean, scale)
    members: list[V04Member] = []
    member_bar = tqdm(
        range(int(params.get("ensemble_members", 3))),
        desc=description,
        unit="member",
        dynamic_ncols=True,
        disable=not bool(progress.get("enabled", True)),
    )
    group_names = sorted({state.task_key for state in train_states})
    state_group = np.asarray([state.task_key for state in train_states])
    target_group = state_group[train_targets.state_indices]
    for member_index in member_bar:
        member_seed = seed + member_index * 1009
        set_determinism(member_seed, deterministic)
        rng = np.random.default_rng(member_seed)
        selected_groups = set(
            rng.choice(
                group_names,
                size=max(1, int(round(len(group_names) * float(params.get("ensemble_task_fraction", 0.9))))),
                replace=False,
            ).tolist()
        )
        selected = np.flatnonzero(np.isin(target_group, list(selected_groups)))
        labels = train_targets.effect_label[selected]
        weights = torch.as_tensor(
            _effective_class_weights(labels, float(params.get("class_weight_beta", 0.999))),
            dtype=torch.float32,
            device=device,
        )
        log_cost = np.log1p(train_targets.verify_cost[selected]).astype(np.float32)
        cost_mean = float(log_cost.mean())
        cost_scale = max(float(log_cost.std()), 1e-6)
        network = _network_class()(
            train_encoded.task.shape[1],
            train_encoded.metadata.shape[1],
            params,
            use_history=use_history,
            budget_conditioning=budget_conditioning,
        ).to(device)
        transferred = _transfer_v03(network, pretrained, member_index) if pretrained else 0
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=float(params.get("learning_rate", 1e-3)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
        )
        best_state: dict[str, Any] | None = None
        best_loss = math.inf
        best_epoch = 0
        stale = 0
        history: list[dict[str, float | int]] = []
        epoch_bar = tqdm(
            range(int(params.get("epochs", 80))),
            desc=f"{description} m{member_index + 1}",
            unit="epoch",
            leave=bool(progress.get("leave_epoch_bars", False)),
            dynamic_ncols=True,
            disable=not bool(progress.get("enabled", True)),
            mininterval=float(progress.get("mininterval_seconds", 1.0)),
        )
        for epoch in epoch_bar:
            network.train()
            rng.shuffle(selected)
            losses: list[float] = []
            q_losses: list[float] = []
            effect_losses: list[float] = []
            cost_losses: list[float] = []
            batch_size = int(params.get("batch_size", 512))
            for start in range(0, len(selected), batch_size):
                rows = selected[start : start + batch_size]
                state_indices = train_targets.state_indices[rows]
                batch = _batch(
                    train_arrays,
                    state_indices,
                    train_targets.allowance[rows],
                    train_targets.allowance_fraction[rows],
                    reference_budget,
                    device,
                )
                q_logits, effect_logits, cost_prediction = network(batch)
                q_target = torch.as_tensor(
                    np.column_stack((train_targets.q_skip[rows], train_targets.q_verify[rows])),
                    dtype=torch.float32,
                    device=device,
                )
                mask = torch.as_tensor(
                    np.column_stack((np.ones(len(rows), dtype=bool), train_targets.verify_mask[rows])),
                    dtype=torch.bool,
                    device=device,
                )
                q_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    q_logits, q_target, reduction="none"
                )
                q_loss = q_raw[mask].mean()
                effect_target = torch.as_tensor(
                    train_targets.effect_label[rows], dtype=torch.long, device=device
                )
                effect_raw = torch.nn.functional.cross_entropy(
                    effect_logits, effect_target, weight=weights, reduction="none"
                )
                probability = torch.softmax(effect_logits, dim=1).gather(
                    1, effect_target[:, None]
                ).squeeze(1)
                effect_loss = (
                    (1.0 - probability) ** float(params.get("focal_gamma", 2.0))
                    * effect_raw
                ).mean()
                cost_target = torch.as_tensor(
                    (np.log1p(train_targets.verify_cost[rows]) - cost_mean) / cost_scale,
                    dtype=torch.float32,
                    device=device,
                )
                cost_loss = torch.nn.functional.smooth_l1_loss(cost_prediction, cost_target)
                loss = (
                    float(params.get("q_loss_weight", 1.0)) * q_loss
                    + float(params.get("effect_loss_weight", 0.35)) * effect_loss
                    + float(params.get("cost_loss_weight", 0.2)) * cost_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), float(params.get("gradient_clip_norm", 5.0))
                )
                optimizer.step()
                losses.append(float(loss.detach()))
                q_losses.append(float(q_loss.detach()))
                effect_losses.append(float(effect_loss.detach()))
                cost_losses.append(float(cost_loss.detach()))
            network.eval()
            val_losses: list[float] = []
            with torch.inference_mode():
                for start in range(0, len(validation_targets.state_indices), batch_size):
                    rows = np.arange(start, min(start + batch_size, len(validation_targets.state_indices)))
                    batch = _batch(
                        validation_arrays,
                        validation_targets.state_indices[rows],
                        validation_targets.allowance[rows],
                        validation_targets.allowance_fraction[rows],
                        reference_budget,
                        device,
                    )
                    q_logits, _effect, _cost = network(batch)
                    target = torch.as_tensor(
                        np.column_stack((validation_targets.q_skip[rows], validation_targets.q_verify[rows])),
                        dtype=torch.float32,
                        device=device,
                    )
                    mask = torch.as_tensor(
                        np.column_stack((np.ones(len(rows), dtype=bool), validation_targets.verify_mask[rows])),
                        dtype=torch.bool,
                        device=device,
                    )
                    raw = torch.nn.functional.binary_cross_entropy_with_logits(
                        q_logits, target, reduction="none"
                    )
                    val_losses.append(float(raw[mask].mean()))
            val_loss = float(np.mean(val_losses))
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_epoch = epoch + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            row = {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "q_loss": float(np.mean(q_losses)),
                "effect_loss": float(np.mean(effect_losses)),
                "cost_loss": float(np.mean(cost_losses)),
                "validation_q_loss": val_loss,
                "transferred_parameters": transferred,
            }
            history.append(row)
            epoch_bar.set_postfix(
                loss=f"{row['loss']:.4f}",
                q=f"{row['q_loss']:.4f}",
                effect=f"{row['effect_loss']:.4f}",
                val=f"{val_loss:.4f}",
                patience=f"{stale}/{int(params.get('patience', 12))}",
            )
            if stale >= int(params.get("patience", 12)):
                break
        epoch_bar.close()
        if best_state is None:
            raise RuntimeError("DARC v0.4 training produced no checkpoint")
        members.append(
            V04Member(
                state_dict=best_state,
                best_epoch=best_epoch,
                best_validation_loss=best_loss,
                cost_log_mean=cost_mean,
                cost_log_scale=cost_scale,
                history=tuple(history),
            )
        )
        member_bar.set_postfix(epoch=best_epoch, val=f"{best_loss:.4f}")
    member_bar.close()
    return DARCv04Ensemble(
        embedding_dim=train_encoded.task.shape[1],
        metadata_dim=train_encoded.metadata.shape[1],
        params=params,
        use_history=use_history,
        budget_conditioning=budget_conditioning,
        reference_budget=reference_budget,
        metadata_mean=mean,
        metadata_scale=scale,
        members=members,
    )

