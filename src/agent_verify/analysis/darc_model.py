"""Neural heads, paired counterfactual objectives, calibration, and uncertainty for DARC."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

from agent_verify.analysis.darc_data import DARCExample, EncodedDARC


@dataclass(frozen=True)
class DARCPrediction:
    probabilities: np.ndarray
    advantage: np.ndarray
    advantage_uncertainty: np.ndarray
    expected_total_tokens: np.ndarray
    upper_total_tokens: np.ndarray
    latency_seconds: float


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "DARC needs PyTorch. Install with: pip install -e '.[darc]'"
        ) from error
    return torch


def select_device(name: str) -> Any:
    torch = _torch()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def set_determinism(seed: int, deterministic: bool) -> None:
    torch = _torch()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        # MultiheadAttention may otherwise select memory-efficient SDPA, whose
        # CUDA backward pass is nondeterministic.  The math kernel is slower but
        # makes grouped-CV/model-selection reruns reproducible.
        if torch.cuda.is_available() and hasattr(torch.backends, "cuda"):
            cuda_backend = torch.backends.cuda
            if hasattr(cuda_backend, "enable_flash_sdp"):
                cuda_backend.enable_flash_sdp(False)
            if hasattr(cuda_backend, "enable_mem_efficient_sdp"):
                cuda_backend.enable_mem_efficient_sdp(False)
            if hasattr(cuda_backend, "enable_math_sdp"):
                cuda_backend.enable_math_sdp(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False


def _network_class() -> Any:
    torch = _torch()
    nn = torch.nn

    class DARCNetwork(nn.Module):
        def __init__(
            self,
            embedding_dim: int,
            metadata_dim: int,
            params: dict[str, Any],
            downstream_mode: str,
        ) -> None:
            super().__init__()
            if downstream_mode not in {"none", "checkpoint_only", "semantic_attention"}:
                raise ValueError(f"Unknown downstream mode: {downstream_mode}")
            hidden = int(params.get("hidden_dim", 128))
            dropout = float(params.get("dropout", 0.15))
            self.head_mode = str(params.get("head_mode", "flat"))
            if self.head_mode not in {"flat", "hierarchical"}:
                raise ValueError(f"Unknown DARC outcome head: {self.head_mode}")
            self.downstream_mode = downstream_mode
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
            if downstream_mode == "semantic_attention":
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
            elif downstream_mode == "checkpoint_only":
                self.checkpoint_embedding = nn.Embedding(3, hidden)
                self.context_gate = nn.Linear(hidden * 2, hidden)
                self.context_norm = nn.LayerNorm(hidden)
            self.trunk = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
            )
            if self.head_mode == "flat":
                self.value_head = nn.Linear(hidden // 2, 3)
            else:
                # Factor P(outcome) into P(Change) and P(Helpful | Change).
                # The second logit is supervised only on observed changed pairs.
                self.change_head = nn.Linear(hidden // 2, 1)
                self.direction_head = nn.Linear(hidden // 2, 1)
            self.cost_head = nn.Linear(hidden // 2, 2)

        def forward(self, batch: dict[str, Any]) -> tuple[Any, Any]:
            segments = [
                batch["task"],
                batch["artifact"],
                batch["history"],
                batch["current_role"],
            ]
            projected = [layer(value) for layer, value in zip(self.segment_projections, segments, strict=True)]
            state = self.state_fusion(torch.cat((*projected, batch["metadata"]), dim=1))
            if self.downstream_mode == "semantic_attention":
                downstream = self.downstream_projection(batch["downstream"])
                context, _weights = self.downstream_attention(
                    state.unsqueeze(1),
                    downstream,
                    downstream,
                    key_padding_mask=~batch["downstream_mask"],
                    need_weights=False,
                )
                context = context.squeeze(1)
                gate = torch.sigmoid(self.context_gate(torch.cat((state, context), dim=1)))
                state = self.context_norm(state + gate * context)
            elif self.downstream_mode == "checkpoint_only":
                context = self.checkpoint_embedding(batch["checkpoint"])
                gate = torch.sigmoid(self.context_gate(torch.cat((state, context), dim=1)))
                state = self.context_norm(state + gate * context)
            hidden = self.trunk(state)
            if self.head_mode == "flat":
                outcome = self.value_head(hidden)
            else:
                outcome = torch.cat(
                    (self.change_head(hidden), self.direction_head(hidden)), dim=1
                )
            return outcome, self.cost_head(hidden)

    return DARCNetwork


def _as_tensors(encoded: EncodedDARC, device: Any, mean: np.ndarray, scale: np.ndarray) -> dict[str, Any]:
    torch = _torch()
    metadata = (encoded.metadata - mean) / scale
    return {
        "task": torch.as_tensor(encoded.task, dtype=torch.float32, device=device),
        "artifact": torch.as_tensor(encoded.artifact, dtype=torch.float32, device=device),
        "history": torch.as_tensor(encoded.history, dtype=torch.float32, device=device),
        "current_role": torch.as_tensor(encoded.current_role, dtype=torch.float32, device=device),
        "downstream": torch.as_tensor(encoded.downstream, dtype=torch.float32, device=device),
        "downstream_mask": torch.as_tensor(encoded.downstream_mask, dtype=torch.bool, device=device),
        "metadata": torch.as_tensor(metadata, dtype=torch.float32, device=device),
        "checkpoint": torch.as_tensor(encoded.checkpoint, dtype=torch.long, device=device),
    }


def _subset_batch(batch: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    torch = _torch()
    selected = torch.as_tensor(indices, dtype=torch.long, device=next(iter(batch.values())).device)
    return {name: value.index_select(0, selected) for name, value in batch.items()}


def _effective_class_weights(labels: np.ndarray, beta: float) -> np.ndarray:
    counts = np.bincount(labels, minlength=3).astype(float)
    effective = 1.0 - np.power(beta, counts)
    weights = np.divide(1.0 - beta, effective, out=np.zeros_like(effective), where=effective > 0)
    weights = weights / max(float(weights[weights > 0].mean()), 1e-9)
    return weights


def _effective_binary_weights(labels: np.ndarray, beta: float) -> np.ndarray:
    if len(labels) == 0:
        return np.ones(2, dtype=float)
    counts = np.bincount(labels.astype(int), minlength=2).astype(float)
    effective = 1.0 - np.power(beta, counts)
    weights = np.divide(
        1.0 - beta,
        effective,
        out=np.zeros_like(effective),
        where=effective > 0,
    )
    positive = weights > 0
    normalizer = float(weights[positive].mean()) if np.any(positive) else 1.0
    weights = weights / max(normalizer, 1e-9)
    return weights


def _classification_loss(logits: Any, labels: Any, mode: str, weights: Any, gamma: float) -> Any:
    torch = _torch()
    if mode == "natural_ce":
        return torch.nn.functional.cross_entropy(logits, labels)
    per_row = torch.nn.functional.cross_entropy(logits, labels, weight=weights, reduction="none")
    if mode == "balanced_ce":
        return per_row.mean()
    if mode == "focal":
        probability = torch.softmax(logits, dim=1).gather(1, labels[:, None]).squeeze(1)
        return (((1.0 - probability).clamp_min(0.0) ** gamma) * per_row).mean()
    raise ValueError(f"Unknown DARC value loss: {mode}")


def _binary_loss(
    logits: Any,
    targets: Any,
    mode: str,
    weights: Any,
    gamma: float,
) -> Any:
    """Binary natural/balanced/focal loss with explicit per-class weights."""

    torch = _torch()
    targets = targets.to(dtype=logits.dtype)
    per_row = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    if mode == "natural_ce":
        return per_row.mean()
    class_weight = weights[targets.to(dtype=torch.long)]
    if mode == "balanced_ce":
        return (per_row * class_weight).mean()
    if mode == "focal":
        probability = torch.sigmoid(logits)
        probability_target = torch.where(targets > 0.5, probability, 1.0 - probability)
        focal = (1.0 - probability_target).clamp_min(0.0) ** gamma
        return (per_row * class_weight * focal).mean()
    raise ValueError(f"Unknown DARC hierarchical loss: {mode}")


def _outcome_probabilities(logits: Any, *, head_mode: str, temperature: float) -> Any:
    """Return [Harmful, No-change, Helpful] probabilities for either head."""

    torch = _torch()
    scaled = logits / max(float(temperature), 1e-6)
    if head_mode == "flat":
        return torch.softmax(scaled, dim=1)
    if head_mode != "hierarchical":
        raise ValueError(f"Unknown DARC outcome head: {head_mode}")
    probability_change = torch.sigmoid(scaled[:, 0])
    helpful_given_change = torch.sigmoid(scaled[:, 1])
    harmful = probability_change * (1.0 - helpful_given_change)
    helpful = probability_change * helpful_given_change
    no_change = 1.0 - probability_change
    return torch.stack((harmful, no_change, helpful), dim=1)


def _hierarchical_value_loss(
    logits: Any,
    labels: Any,
    *,
    mode: str,
    change_weights: Any,
    direction_weights: Any,
    gamma: float,
    change_weight: float,
    direction_weight: float,
) -> tuple[Any, Any, Any]:
    """Loss for P(Change) and P(Helpful | Change).

    Label indices follow [Harmful=0, No-change=1, Helpful=2].  Direction is
    deliberately undefined and unsupervised for No-change states.
    """

    torch = _torch()
    changed = labels != 1
    change_targets = changed.to(dtype=torch.float32)
    change_loss = _binary_loss(
        logits[:, 0], change_targets, mode, change_weights, gamma
    )
    if bool(changed.any()):
        direction_targets = (labels[changed] == 2).to(dtype=torch.float32)
        direction_loss = _binary_loss(
            logits[changed, 1],
            direction_targets,
            mode,
            direction_weights,
            gamma,
        )
    else:
        direction_loss = torch.zeros((), device=logits.device)
    total = change_weight * change_loss + direction_weight * direction_loss
    return total, change_loss, direction_loss


def _ranking_pairs(examples: list[DARCExample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(examples):
        grouped.setdefault(row.group_key, []).append(index)
    left: list[int] = []
    right: list[int] = []
    direction: list[float] = []
    for indices in grouped.values():
        for offset, first in enumerate(indices):
            for second in indices[offset + 1 :]:
                difference = examples[first].delta - examples[second].delta
                if difference == 0:
                    continue
                left.append(first)
                right.append(second)
                direction.append(float(np.sign(difference)))
    return np.asarray(left), np.asarray(right), np.asarray(direction, dtype=np.float32)


def _group_subsample_indices(examples: list[DARCExample], seed: int, fraction: float) -> np.ndarray:
    groups = sorted({row.group_key for row in examples})
    rng = np.random.default_rng(seed)
    selected_count = max(1, int(round(len(groups) * fraction)))
    selected_groups = set(rng.choice(groups, size=selected_count, replace=False).tolist())
    return np.asarray([index for index, row in enumerate(examples) if row.group_key in selected_groups])


@dataclass
class DARCMember:
    state_dict: dict[str, Any]
    best_epoch: int
    validation_nll: float
    cost_log_mean: float
    cost_log_scale: float
    history: tuple[dict[str, float | int], ...] = ()


class DARCEnsemble:
    """Small trainable DARC heads over immutable cached semantic embeddings."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        metadata_dim: int,
        params: dict[str, Any],
        downstream_mode: str,
        loss_mode: str,
        members: list[DARCMember],
        metadata_mean: np.ndarray,
        metadata_scale: np.ndarray,
        temperature: float = 1.0,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.metadata_dim = metadata_dim
        self.params = dict(params)
        self.downstream_mode = downstream_mode
        self.loss_mode = loss_mode
        self.members = members
        self.metadata_mean = metadata_mean.astype(np.float32)
        self.metadata_scale = metadata_scale.astype(np.float32)
        self.temperature = float(temperature)

    def _network(self, member: DARCMember, device: Any) -> Any:
        network = _network_class()(
            self.embedding_dim, self.metadata_dim, self.params, self.downstream_mode
        ).to(device)
        network.load_state_dict(member.state_dict)
        network.eval()
        return network

    def predict(self, encoded: EncodedDARC, *, device_name: str = "auto") -> DARCPrediction:
        torch = _torch()
        device = select_device(device_name)
        batch = _as_tensors(encoded, device, self.metadata_mean, self.metadata_scale)
        probabilities: list[np.ndarray] = []
        expected_costs: list[np.ndarray] = []
        upper_costs: list[np.ndarray] = []
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        head_mode = str(self.params.get("head_mode", "flat"))
        for member in self.members:
            network = self._network(member, device)
            with torch.inference_mode():
                logits, cost = network(batch)
                member_probability = _outcome_probabilities(
                    logits,
                    head_mode=head_mode,
                    temperature=self.temperature,
                )
                mean_log = cost[:, 0] * member.cost_log_scale + member.cost_log_mean
                upper_log = cost[:, 1] * member.cost_log_scale + member.cost_log_mean
                expected = torch.expm1(mean_log).clamp_min(0.0)
                upper = torch.maximum(torch.expm1(upper_log).clamp_min(0.0), expected)
            probabilities.append(member_probability.cpu().numpy())
            expected_costs.append(expected.cpu().numpy())
            upper_costs.append(upper.cpu().numpy())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        stacked = np.stack(probabilities)
        advantages = stacked[:, :, 2] - stacked[:, :, 0]
        return DARCPrediction(
            probabilities=stacked.mean(axis=0),
            advantage=advantages.mean(axis=0),
            advantage_uncertainty=advantages.std(axis=0),
            expected_total_tokens=np.stack(expected_costs).mean(axis=0),
            upper_total_tokens=np.stack(upper_costs).mean(axis=0),
            latency_seconds=elapsed,
        )

    def save(self, path: Path) -> None:
        torch = _torch()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "darc-ensemble-v3",
            "embedding_dim": self.embedding_dim,
            "metadata_dim": self.metadata_dim,
            "params": self.params,
            "downstream_mode": self.downstream_mode,
            "loss_mode": self.loss_mode,
            "metadata_mean": self.metadata_mean,
            "metadata_scale": self.metadata_scale,
            "temperature": self.temperature,
            "members": [
                {
                    "state_dict": member.state_dict,
                    "best_epoch": member.best_epoch,
                    "validation_nll": member.validation_nll,
                    "cost_log_mean": member.cost_log_mean,
                    "cost_log_scale": member.cost_log_scale,
                    "history": list(member.history),
                }
                for member in self.members
            ],
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> DARCEnsemble:
        torch = _torch()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        members = [
            DARCMember(
                state_dict=row["state_dict"],
                best_epoch=int(row["best_epoch"]),
                validation_nll=float(row["validation_nll"]),
                cost_log_mean=float(row["cost_log_mean"]),
                cost_log_scale=float(row["cost_log_scale"]),
                history=tuple(row.get("history", ())),
            )
            for row in payload["members"]
        ]
        return cls(
            embedding_dim=int(payload["embedding_dim"]),
            metadata_dim=int(payload["metadata_dim"]),
            params=dict(payload["params"]),
            downstream_mode=str(payload["downstream_mode"]),
            loss_mode=str(payload["loss_mode"]),
            members=members,
            metadata_mean=np.asarray(payload["metadata_mean"]),
            metadata_scale=np.asarray(payload["metadata_scale"]),
            temperature=float(payload.get("temperature", 1.0)),
        )


def train_ensemble(
    train_examples: list[DARCExample],
    train_encoded: EncodedDARC,
    validation_examples: list[DARCExample],
    validation_encoded: EncodedDARC,
    *,
    params: dict[str, Any],
    downstream_mode: str,
    loss_mode: str,
    paired_ranking: bool,
    seed: int,
    device_name: str,
    deterministic: bool,
    description: str,
    progress: dict[str, Any] | None = None,
    early_stopping: bool = True,
) -> DARCEnsemble:
    torch = _torch()
    device = select_device(device_name)
    metadata_mean = train_encoded.metadata.mean(axis=0)
    metadata_scale = train_encoded.metadata.std(axis=0)
    metadata_scale[metadata_scale < 1e-6] = 1.0
    all_train = _as_tensors(train_encoded, device, metadata_mean, metadata_scale)
    validation_batch = _as_tensors(validation_encoded, device, metadata_mean, metadata_scale)
    validation_labels = torch.as_tensor(
        [row.value_label for row in validation_examples], dtype=torch.long, device=device
    )
    head_mode = str(params.get("head_mode", "flat"))
    if head_mode not in {"flat", "hierarchical"}:
        raise ValueError(f"Unknown DARC outcome head: {head_mode}")
    members: list[DARCMember] = []
    ensemble_members = int(params.get("ensemble_members", 3))
    progress = progress or {}
    progress_enabled = bool(progress.get("enabled", True))
    progress_mininterval = float(progress.get("mininterval_seconds", 1.0))
    member_progress = tqdm(
        range(ensemble_members),
        desc=description,
        unit="member",
        dynamic_ncols=True,
        disable=not progress_enabled,
        mininterval=progress_mininterval,
    )
    for member_index in member_progress:
        member_seed = seed + member_index * 1009
        set_determinism(member_seed, deterministic)
        indices = _group_subsample_indices(
            train_examples,
            member_seed,
            float(params.get("ensemble_task_fraction", 0.90)),
        )
        member_examples = [train_examples[int(index)] for index in indices]
        batch = _subset_batch(all_train, indices)
        labels_np = np.asarray([row.value_label for row in member_examples], dtype=int)
        labels = torch.as_tensor(labels_np, dtype=torch.long, device=device)
        log_cost_np = np.log1p([row.teacher_total_tokens for row in member_examples]).astype(np.float32)
        cost_mean = float(log_cost_np.mean())
        cost_scale = max(float(log_cost_np.std()), 1e-6)
        cost_target = torch.as_tensor((log_cost_np - cost_mean) / cost_scale, device=device)
        class_weights = torch.as_tensor(
            _effective_class_weights(labels_np, float(params.get("class_weight_beta", 0.999))),
            dtype=torch.float32,
            device=device,
        )
        change_np = (labels_np != 1).astype(int)
        direction_np = (labels_np[change_np.astype(bool)] == 2).astype(int)
        change_weights = torch.as_tensor(
            _effective_binary_weights(
                change_np, float(params.get("class_weight_beta", 0.999))
            ),
            dtype=torch.float32,
            device=device,
        )
        direction_weights = torch.as_tensor(
            _effective_binary_weights(
                direction_np, float(params.get("class_weight_beta", 0.999))
            ),
            dtype=torch.float32,
            device=device,
        )
        left, right, direction = _ranking_pairs(member_examples)
        left_tensor = torch.as_tensor(left, dtype=torch.long, device=device)
        right_tensor = torch.as_tensor(right, dtype=torch.long, device=device)
        direction_tensor = torch.as_tensor(direction, dtype=torch.float32, device=device)
        network = _network_class()(
            train_encoded.task.shape[1],
            train_encoded.metadata.shape[1],
            params,
            downstream_mode,
        ).to(device)
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=float(params.get("learning_rate", 1e-3)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
        )
        best_state: dict[str, Any] | None = None
        best_nll = math.inf
        best_epoch = 0
        stale = 0
        patience = int(params.get("patience", 20))
        history: list[dict[str, float | int]] = []
        epoch_progress = tqdm(
            range(int(params.get("epochs", 160))),
            desc=f"{description} m{member_index + 1}",
            unit="epoch",
            leave=bool(progress.get("leave_epoch_bars", True)),
            dynamic_ncols=True,
            disable=not progress_enabled,
            mininterval=progress_mininterval,
        )
        for epoch in epoch_progress:
            network.train()
            logits, cost = network(batch)
            change_loss = torch.zeros((), device=device)
            direction_loss = torch.zeros((), device=device)
            if head_mode == "flat":
                value_loss = _classification_loss(
                    logits,
                    labels,
                    loss_mode,
                    class_weights,
                    float(params.get("focal_gamma", 2.0)),
                )
            else:
                value_loss, change_loss, direction_loss = _hierarchical_value_loss(
                    logits,
                    labels,
                    mode=loss_mode,
                    change_weights=change_weights,
                    direction_weights=direction_weights,
                    gamma=float(params.get("focal_gamma", 2.0)),
                    change_weight=float(params.get("change_loss_weight", 1.0)),
                    direction_weight=float(params.get("direction_loss_weight", 1.0)),
                )
            mean_loss = torch.nn.functional.smooth_l1_loss(cost[:, 0], cost_target)
            quantile = float(params.get("cost_quantile", 0.90))
            quantile_error = cost_target - cost[:, 1]
            quantile_loss = torch.maximum(
                quantile * quantile_error,
                (quantile - 1.0) * quantile_error,
            ).mean()
            ranking_loss = torch.zeros((), device=device)
            if paired_ranking and len(left):
                probabilities = _outcome_probabilities(
                    logits, head_mode=head_mode, temperature=1.0
                )
                advantage = probabilities[:, 2] - probabilities[:, 0]
                ordered_margin = direction_tensor * (
                    advantage[left_tensor] - advantage[right_tensor]
                )
                ranking_loss = torch.relu(
                    float(params.get("ranking_margin", 0.10)) - ordered_margin
                ).mean()
            loss = (
                value_loss
                + float(params.get("cost_loss_weight", 0.25)) * mean_loss
                + float(params.get("cost_quantile_weight", 0.20)) * quantile_loss
                + float(params.get("ranking_loss_weight", 0.25)) * ranking_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                network.parameters(), float(params.get("gradient_clip_norm", 5.0))
            )
            optimizer.step()
            network.eval()
            with torch.inference_mode():
                val_logits, _val_cost = network(validation_batch)
                validation_probability = _outcome_probabilities(
                    val_logits, head_mode=head_mode, temperature=1.0
                ).clamp_min(1e-9)
                validation_nll = float(
                    (-torch.log(
                        validation_probability[
                            torch.arange(len(validation_labels), device=device),
                            validation_labels,
                        ]
                    ).mean()).item()
                )
            if not early_stopping:
                # Final refits use a fold-selected epoch count.  Keep the last
                # epoch instead of selecting again on the held-out validation
                # partition (which would leak model-selection information).
                best_nll = validation_nll
                best_epoch = epoch + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                }
                stale = 0
            elif validation_nll < best_nll - 1e-5:
                best_nll = validation_nll
                best_epoch = epoch + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            history.append(
                {
                    "epoch": epoch + 1,
                    "total_loss": float(loss.detach().item()),
                    "value_loss": float(value_loss.detach().item()),
                    "change_loss": float(change_loss.detach().item()),
                    "direction_loss": float(direction_loss.detach().item()),
                    "cost_mean_loss": float(mean_loss.detach().item()),
                    "cost_quantile_loss": float(quantile_loss.detach().item()),
                    "ranking_loss": float(ranking_loss.detach().item()),
                    "validation_nll": validation_nll,
                    "best_validation_nll": best_nll,
                    "stale_epochs": stale,
                }
            )
            epoch_progress.set_postfix(
                loss=f"{history[-1]['total_loss']:.4f}",
                cls=f"{history[-1]['value_loss']:.4f}",
                change=f"{history[-1]['change_loss']:.4f}",
                direction=f"{history[-1]['direction_loss']:.4f}",
                cost=f"{history[-1]['cost_mean_loss']:.4f}",
                qcost=f"{history[-1]['cost_quantile_loss']:.4f}",
                rank=f"{history[-1]['ranking_loss']:.4f}",
                val=f"{validation_nll:.4f}",
                best=f"{best_nll:.4f}",
                patience=f"{stale}/{patience}",
            )
            if early_stopping and stale >= patience:
                break
        epoch_progress.close()
        if best_state is None:
            raise RuntimeError("DARC early stopping failed to capture a model state")
        members.append(
            DARCMember(
                best_state,
                best_epoch,
                best_nll,
                cost_mean,
                cost_scale,
                tuple(history),
            )
        )
        member_progress.set_postfix(
            member=f"{member_index + 1}/{ensemble_members}",
            best_epoch=best_epoch,
            best_nll=f"{best_nll:.4f}",
        )
    member_progress.close()
    return DARCEnsemble(
        embedding_dim=train_encoded.task.shape[1],
        metadata_dim=train_encoded.metadata.shape[1],
        params=params,
        downstream_mode=downstream_mode,
        loss_mode=loss_mode,
        members=members,
        metadata_mean=metadata_mean,
        metadata_scale=metadata_scale,
    )


def calibrate_temperature(
    model: DARCEnsemble,
    encoded: EncodedDARC,
    labels: np.ndarray,
    grid: list[float],
    *,
    device_name: str,
) -> tuple[float, list[dict[str, float]]]:
    original = model.temperature
    rows: list[dict[str, float]] = []
    best: tuple[float, float] | None = None
    selected = 1.0
    for temperature in grid:
        model.temperature = float(temperature)
        prediction = model.predict(encoded, device_name=device_name)
        probabilities = np.clip(prediction.probabilities, 1e-9, 1.0)
        nll = -float(np.mean(np.log(probabilities[np.arange(len(labels)), labels])))
        rows.append({"temperature": float(temperature), "validation_nll": nll})
        key = (nll, abs(float(temperature) - 1.0))
        if best is None or key < best:
            best = key
            selected = float(temperature)
    model.temperature = selected
    if best is None:
        model.temperature = original
        raise ValueError("Temperature grid is empty")
    return selected, rows


def expected_calibration_error(target: np.ndarray, probability: np.ndarray, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        upper_inclusive = index == bins - 1
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1] if upper_inclusive else probability < edges[index + 1]
        )
        if not np.any(selected):
            continue
        total += float(selected.mean()) * abs(
            float(target[selected].mean()) - float(probability[selected].mean())
        )
    return total


def prediction_metrics(
    examples: list[DARCExample], prediction: DARCPrediction, *, bins: int
) -> dict[str, float]:
    labels = np.asarray([row.value_label for row in examples], dtype=int)
    delta = np.asarray([row.delta for row in examples], dtype=float)
    actual_cost = np.asarray([row.teacher_total_tokens for row in examples], dtype=float)
    probabilities = np.clip(prediction.probabilities, 1e-9, 1.0)
    helpful = (labels == 2).astype(int)
    harmful = (labels == 0).astype(int)

    def auprc(target: np.ndarray, score: np.ndarray) -> float:
        return float(average_precision_score(target, score)) if len(np.unique(target)) > 1 else math.nan

    correlation = (
        float(np.corrcoef(delta, prediction.advantage)[0, 1])
        if np.std(delta) > 0 and np.std(prediction.advantage) > 0
        else math.nan
    )
    relative = np.abs(prediction.expected_total_tokens - actual_cost) / np.maximum(actual_cost, 1.0)
    one_hot = np.eye(3)[labels]
    return {
        "value_nll": -float(np.mean(np.log(probabilities[np.arange(len(labels)), labels]))),
        "value_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "helpful_auprc": auprc(helpful, probabilities[:, 2]),
        "harmful_auprc": auprc(harmful, probabilities[:, 0]),
        "helpful_ece": expected_calibration_error(helpful, probabilities[:, 2], bins),
        "harmful_ece": expected_calibration_error(harmful, probabilities[:, 0], bins),
        "gain_pearson": correlation,
        "gain_mae": float(np.mean(np.abs(prediction.advantage - delta))),
        "cost_mae": float(np.mean(np.abs(prediction.expected_total_tokens - actual_cost))),
        "cost_mean_relative_error": float(relative.mean()),
        "cost_upper_coverage": float(np.mean(actual_cost <= prediction.upper_total_tokens)),
        "mean_advantage_uncertainty": float(prediction.advantage_uncertainty.mean()),
        "head_latency_seconds": float(prediction.latency_seconds),
    }


def candidate_score(metrics: dict[str, float]) -> float:
    helpful = metrics["helpful_auprc"] if math.isfinite(metrics["helpful_auprc"]) else 0.0
    harmful = metrics["harmful_auprc"] if math.isfinite(metrics["harmful_auprc"]) else 0.0
    correlation = metrics["gain_pearson"] if math.isfinite(metrics["gain_pearson"]) else 0.0
    return float(
        helpful
        + 0.5 * harmful
        + 0.25 * correlation
        - 0.10 * metrics["value_nll"]
        - 0.20 * (metrics["helpful_ece"] + metrics["harmful_ece"])
    )


def parameter_count(model: DARCEnsemble) -> int:
    network = model._network(model.members[0], select_device("cpu"))
    return int(sum(parameter.numel() for parameter in network.parameters()))
