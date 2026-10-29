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
            use_attention: bool,
        ) -> None:
            super().__init__()
            hidden = int(params.get("hidden_dim", 128))
            dropout = float(params.get("dropout", 0.15))
            self.use_attention = use_attention
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
            if use_attention:
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
            self.value_head = nn.Linear(hidden // 2, 3)
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
            if self.use_attention:
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
            hidden = self.trunk(state)
            return self.value_head(hidden), self.cost_head(hidden)

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


class DARCEnsemble:
    """Small trainable DARC heads over immutable cached semantic embeddings."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        metadata_dim: int,
        params: dict[str, Any],
        use_attention: bool,
        loss_mode: str,
        members: list[DARCMember],
        metadata_mean: np.ndarray,
        metadata_scale: np.ndarray,
        temperature: float = 1.0,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.metadata_dim = metadata_dim
        self.params = dict(params)
        self.use_attention = use_attention
        self.loss_mode = loss_mode
        self.members = members
        self.metadata_mean = metadata_mean.astype(np.float32)
        self.metadata_scale = metadata_scale.astype(np.float32)
        self.temperature = float(temperature)

    def _network(self, member: DARCMember, device: Any) -> Any:
        network = _network_class()(
            self.embedding_dim, self.metadata_dim, self.params, self.use_attention
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
        for member in self.members:
            network = self._network(member, device)
            with torch.inference_mode():
                logits, cost = network(batch)
                member_probability = torch.softmax(logits / self.temperature, dim=1)
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
            "schema_version": "darc-ensemble-v1",
            "embedding_dim": self.embedding_dim,
            "metadata_dim": self.metadata_dim,
            "params": self.params,
            "use_attention": self.use_attention,
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
                }
                for member in self.members
            ],
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> "DARCEnsemble":
        torch = _torch()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        members = [DARCMember(**row) for row in payload["members"]]
        return cls(
            embedding_dim=int(payload["embedding_dim"]),
            metadata_dim=int(payload["metadata_dim"]),
            params=dict(payload["params"]),
            use_attention=bool(payload["use_attention"]),
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
    use_attention: bool,
    loss_mode: str,
    paired_ranking: bool,
    seed: int,
    device_name: str,
    deterministic: bool,
    description: str,
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
    members: list[DARCMember] = []
    ensemble_members = int(params.get("ensemble_members", 3))
    for member_index in tqdm(
        range(ensemble_members),
        desc=description,
        unit="member",
        dynamic_ncols=True,
    ):
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
        left, right, direction = _ranking_pairs(member_examples)
        left_tensor = torch.as_tensor(left, dtype=torch.long, device=device)
        right_tensor = torch.as_tensor(right, dtype=torch.long, device=device)
        direction_tensor = torch.as_tensor(direction, dtype=torch.float32, device=device)
        network = _network_class()(
            train_encoded.task.shape[1],
            train_encoded.metadata.shape[1],
            params,
            use_attention,
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
        epoch_progress = tqdm(
            range(int(params.get("epochs", 160))),
            desc=f"{description} m{member_index + 1}",
            unit="epoch",
            leave=False,
            dynamic_ncols=True,
        )
        for epoch in epoch_progress:
            network.train()
            logits, cost = network(batch)
            value_loss = _classification_loss(
                logits,
                labels,
                loss_mode,
                class_weights,
                float(params.get("focal_gamma", 2.0)),
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
                probabilities = torch.softmax(logits, dim=1)
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
                validation_nll = float(
                    torch.nn.functional.cross_entropy(val_logits, validation_labels).item()
                )
            epoch_progress.set_postfix(nll=f"{validation_nll:.4f}")
            if validation_nll < best_nll - 1e-5:
                best_nll = validation_nll
                best_epoch = epoch + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= int(params.get("patience", 20)):
                break
        epoch_progress.close()
        if best_state is None:
            raise RuntimeError("DARC early stopping failed to capture a model state")
        members.append(
            DARCMember(best_state, best_epoch, best_nll, cost_mean, cost_scale)
        )
    return DARCEnsemble(
        embedding_dim=train_encoded.task.shape[1],
        metadata_dim=train_encoded.metadata.shape[1],
        params=params,
        use_attention=use_attention,
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
