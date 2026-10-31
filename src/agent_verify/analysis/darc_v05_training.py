"""Cross-fitted conservative policy evaluation and training for DARC v0.5."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from agent_verify.analysis.darc_data import EncodedDARC
from agent_verify.analysis.darc_model import _effective_class_weights, _torch, select_device, set_determinism
from agent_verify.analysis.darc_v04_data import MultiHistoryState, verify_cost
from agent_verify.analysis.darc_v04_model import DARCv04Ensemble, _base_arrays, _batch
from agent_verify.analysis.darc_v05_env import (
    Action,
    CBVLevel2OfflineEnv,
    InfeasibleOfflineAction,
    OfflineObservation,
    rollout_forced_action,
)
from agent_verify.analysis.darc_v05_model import (
    DARCv05Ensemble,
    V05Member,
    V05Prediction,
    network_class,
    transfer_v04,
)


@dataclass(frozen=True)
class V05TargetTable:
    state_indices: np.ndarray
    allowance: np.ndarray
    allowance_fraction: np.ndarray
    q_success: np.ndarray
    q_cost: np.ndarray
    action_mask: np.ndarray
    effect_label: np.ndarray
    immediate_verify_cost: np.ndarray


@dataclass(frozen=True)
class ConservativePolicy:
    base_token_price: float
    price_elasticity: float
    advantage_uncertainty_beta: float
    cost_uncertainty_beta: float
    harmful_weight: float
    decision_margin: float
    conservative: bool = True
    dynamic_token_price: bool = True
    uncertainty_penalty: bool = True

    def score(
        self,
        prediction: V05Prediction,
        index: int,
        *,
        allowance: float,
        reference_budget: float,
    ) -> float:
        uncertainty_beta = self.advantage_uncertainty_beta if self.uncertainty_penalty else 0.0
        cost_beta = self.cost_uncertainty_beta if self.uncertainty_penalty else 0.0
        harmful_weight = self.harmful_weight if self.conservative else 0.0
        advantage = float(prediction.advantage[index])
        lower_advantage = advantage - uncertainty_beta * float(
            prediction.advantage_uncertainty[index]
        )
        upper_cost = float(prediction.additional_cost[index]) + cost_beta * float(
            prediction.additional_cost_uncertainty[index]
        )
        pressure = 1.0
        if self.dynamic_token_price:
            pressure = (
                reference_budget / max(float(allowance), 1.0)
            ) ** self.price_elasticity
        token_price = self.base_token_price * pressure
        harmful = float(prediction.effect_probabilities[index, 0])
        return (
            lower_advantage
            - harmful_weight * harmful
            - token_price * upper_cost / max(reference_budget, 1.0)
        )


def assign_task_folds(task_keys: list[str], folds: int, seed: int) -> dict[str, int]:
    if folds < 2:
        raise ValueError("Cross-fitting needs at least two folds")
    rng = np.random.default_rng(seed)
    ordered = sorted(task_keys)
    rng.shuffle(ordered)
    return {key: index % folds for index, key in enumerate(ordered)}


def nearest_prediction(
    predictions: list[V05Prediction], allowances: list[float], allowance: float
) -> V05Prediction:
    return predictions[int(np.argmin(np.abs(np.asarray(allowances) - allowance)))]


def policy_action(
    observation: OfflineObservation,
    *,
    predictions: list[V05Prediction],
    allowances: list[float],
    policy: ConservativePolicy,
    reference_budget: float,
    hard_call_token_cap: int,
) -> Action:
    prediction = nearest_prediction(
        predictions, allowances, observation.per_task_allowance
    )
    index = observation.state_index
    predicted_call = float(prediction.immediate_cost[index]) + float(
        prediction.immediate_cost_uncertainty[index]
    )
    feasible = (
        observation.global_remaining_budget >= hard_call_token_cap
        and predicted_call <= observation.global_remaining_budget
    )
    score = policy.score(
        prediction,
        index,
        allowance=observation.per_task_allowance,
        reference_budget=reference_budget,
    )
    return "V" if feasible and score > policy.decision_margin else "S"


ContinuationFactory = Callable[[MultiHistoryState, int], Callable[[OfflineObservation], Action]]


def generate_policy_targets(
    states: list[MultiHistoryState],
    groups: dict[str, Any],
    state_lookup: dict[tuple[str, str], int],
    *,
    allowances: list[float],
    reference_budget: float,
    hard_call_token_cap: int,
    continuation_factory: ContinuationFactory,
    target_mode: str,
) -> V05TargetTable:
    """Evaluate forced S/V actions under a fixed continuation policy.

    `policy_aware` follows the supplied observable policy. `myopic` forces all
    later actions to Skip. No max over future recorded sequences is performed.
    """

    if target_mode not in {"policy_aware", "myopic"}:
        raise ValueError(f"Unknown target mode {target_mode!r}")
    env = CBVLevel2OfflineEnv(
        groups, state_lookup, hard_call_token_cap=hard_call_token_cap
    )
    state_indices: list[int] = []
    allowance_rows: list[float] = []
    successes: list[tuple[float, float]] = []
    costs: list[tuple[float, float]] = []
    masks: list[tuple[bool, bool]] = []
    effects: list[int] = []
    immediate: list[float] = []
    for allowance in allowances:
        for index, state in enumerate(states):
            # `allowance` is a shadow price/allocation feature, not the global
            # token ledger.  For a positive-budget target we reserve enough
            # global capacity for every remaining call.  Realized recorded costs
            # are observed only after actions and become Q-cost supervision.
            remaining_calls = 3 - len(state.prefix)
            target_global_budget = (
                0
                if allowance <= 0
                else hard_call_token_cap * remaining_calls
            )
            continuation = (
                (lambda _observation: "S")
                if target_mode == "myopic"
                else continuation_factory(state, int(round(allowance)))
            )
            skip = rollout_forced_action(
                env,
                task_key=state.task_key,
                prefix=state.prefix,
                global_budget=target_global_budget,
                per_task_allowance=float(allowance),
                forced_action="S",
                continuation_policy=continuation,
            )
            verify_valid = target_global_budget >= hard_call_token_cap
            verify_success = skip.final_correctness
            verify_tokens = 0
            if verify_valid:
                try:
                    verified = rollout_forced_action(
                        env,
                        task_key=state.task_key,
                        prefix=state.prefix,
                        global_budget=target_global_budget,
                        per_task_allowance=float(allowance),
                        forced_action="V",
                        continuation_policy=continuation,
                    )
                    verify_success = verified.final_correctness
                    verify_tokens = verified.cumulative_teacher_tokens
                except InfeasibleOfflineAction:
                    verify_valid = False
            state_indices.append(index)
            allowance_rows.append(float(allowance))
            successes.append((float(skip.final_correctness), float(verify_success)))
            costs.append((float(skip.cumulative_teacher_tokens), float(verify_tokens)))
            masks.append((True, verify_valid))
            delta = verify_success - skip.final_correctness
            effects.append(2 if delta > 0 else 0 if delta < 0 else 1)
            immediate.append(float(verify_cost(groups[state.task_key], state.prefix)))
    allowance_array = np.asarray(allowance_rows, dtype=np.float32)
    return V05TargetTable(
        state_indices=np.asarray(state_indices, dtype=np.int64),
        allowance=allowance_array,
        allowance_fraction=allowance_array / max(reference_budget, 1.0),
        q_success=np.asarray(successes, dtype=np.float32),
        q_cost=np.asarray(costs, dtype=np.float32),
        action_mask=np.asarray(masks, dtype=bool),
        effect_label=np.asarray(effects, dtype=np.int64),
        immediate_verify_cost=np.asarray(immediate, dtype=np.float32),
    )


def validate_target_coverage(
    targets: V05TargetTable,
    *,
    name: str,
) -> dict[str, Any]:
    """Fail before training if positive-budget Verify supervision is absent."""

    positive = targets.allowance > 0
    verify_valid = targets.action_mask[:, 1]
    positive_rows = int(positive.sum())
    supervised = int((positive & verify_valid).sum())
    diagnostics: dict[str, Any] = {
        "name": name,
        "rows": int(len(targets.state_indices)),
        "positive_budget_rows": positive_rows,
        "verify_supervised_rows": supervised,
        "verify_supervision_rate_positive": supervised / max(positive_rows, 1),
        "allowance_min": float(targets.allowance.min()),
        "allowance_max": float(targets.allowance.max()),
    }
    by_allowance: dict[str, dict[str, float | int]] = {}
    for allowance in sorted(set(float(value) for value in targets.allowance)):
        selected = np.isclose(targets.allowance, allowance)
        valid = int((selected & verify_valid).sum())
        by_allowance[f"{allowance:g}"] = {
            "rows": int(selected.sum()),
            "verify_supervised": valid,
            "coverage": valid / max(int(selected.sum()), 1),
        }
        if allowance > 0 and valid == 0:
            raise RuntimeError(
                f"{name}: all Verify targets are masked at positive allowance "
                f"{allowance:g}; check global-budget versus allowance semantics"
            )
    diagnostics["by_allowance"] = by_allowance
    if positive_rows == 0 or supervised == 0:
        raise RuntimeError(
            f"{name}: no positive-budget Verify supervision; refusing to train"
        )
    return diagnostics


def _transfer_v05(network: Any, source: DARCv05Ensemble, fold: int, member: int) -> int:
    candidates = [row for row in source.members if row.fold == fold and row.member == member]
    if not candidates:
        return 0
    current = network.state_dict()
    copied = 0
    for name, value in candidates[0].state_dict.items():
        if name in current and tuple(current[name].shape) == tuple(value.shape):
            current[name] = value.detach().clone()
            copied += int(value.numel())
    network.load_state_dict(current)
    return copied


def train_crossfit_iteration(
    train_states: list[MultiHistoryState],
    train_encoded: EncodedDARC,
    train_targets: V05TargetTable,
    validation_states: list[MultiHistoryState],
    validation_encoded: EncodedDARC,
    validation_targets: V05TargetTable,
    *,
    task_folds: dict[str, int],
    folds: int,
    members_per_fold: int,
    iteration: int,
    params: dict[str, Any],
    reference_budget: float,
    v04_initialization: DARCv04Ensemble | None,
    previous: DARCv05Ensemble | None,
    seed: int,
    deterministic: bool,
    device_name: str,
    writer: Any,
    variant: str,
    progress: dict[str, Any],
) -> DARCv05Ensemble:
    torch = _torch()
    device = select_device(device_name)
    mean = train_encoded.metadata.mean(axis=0)
    scale = train_encoded.metadata.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_arrays = _base_arrays(train_encoded, train_states, mean, scale)
    validation_arrays = _base_arrays(validation_encoded, validation_states, mean, scale)
    target_groups = np.asarray(
        [train_states[int(index)].task_key for index in train_targets.state_indices]
    )
    members: list[V05Member] = []
    jobs = [(fold, member) for fold in range(folds) for member in range(members_per_fold)]
    jobs_bar = tqdm(
        jobs,
        desc=f"{variant} seed={seed} iteration={iteration}",
        unit="member",
        dynamic_ncols=True,
        disable=not bool(progress.get("tqdm", True)),
    )
    for fold, member_index in jobs_bar:
        member_writer = writer.for_run(
            f"{variant}/seed_{seed}/iteration_{iteration}/"
            f"fold_{fold}_member_{member_index}"
        )
        member_seed = seed + iteration * 100_003 + fold * 10_007 + member_index * 1009
        set_determinism(member_seed, deterministic)
        rng = np.random.default_rng(member_seed)
        selected = np.flatnonzero(
            np.asarray([task_folds[key] != fold for key in target_groups], dtype=bool)
        )
        labels = train_targets.effect_label[selected]
        class_weights = torch.as_tensor(
            _effective_class_weights(labels, float(params.get("class_weight_beta", 0.999))),
            dtype=torch.float32,
            device=device,
        )
        cumulative_log = np.log1p(train_targets.q_cost[selected]).astype(np.float32)
        cumulative_mean = float(cumulative_log.mean())
        cumulative_scale = max(float(cumulative_log.std()), 1e-6)
        immediate_log = np.log1p(train_targets.immediate_verify_cost[selected]).astype(np.float32)
        immediate_mean = float(immediate_log.mean())
        immediate_scale = max(float(immediate_log.std()), 1e-6)
        network = network_class()(
            train_encoded.task.shape[1], train_encoded.metadata.shape[1], params
        ).to(device)
        transferred = (
            _transfer_v05(network, previous, fold, member_index)
            if previous is not None
            else (
                transfer_v04(network, v04_initialization, member_index)
                if v04_initialization is not None
                else 0
            )
        )
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=float(params.get("learning_rate", 7e-4)),
            weight_decay=float(params.get("weight_decay", 1e-4)),
        )
        best_state: dict[str, Any] | None = None
        best_loss = math.inf
        best_epoch = stale = 0
        history: list[dict[str, float | int]] = []
        epoch_bar = tqdm(
            range(int(params.get("epochs", 60))),
            desc=f"s{seed} f{fold} m{member_index} i{iteration}",
            unit="epoch",
            leave=False,
            dynamic_ncols=True,
            disable=not bool(progress.get("tqdm", True)),
            mininterval=float(progress.get("mininterval_seconds", 1.0)),
        )
        for epoch in epoch_bar:
            network.train()
            rng.shuffle(selected)
            totals: list[float] = []
            success_losses: list[float] = []
            cost_losses: list[float] = []
            effect_losses: list[float] = []
            gradient_norms: list[float] = []
            advantages: list[float] = []
            batch_size = int(params.get("batch_size", 512))
            for start in range(0, len(selected), batch_size):
                rows = selected[start : start + batch_size]
                batch = _batch(
                    train_arrays,
                    train_targets.state_indices[rows],
                    train_targets.allowance[rows],
                    train_targets.allowance_fraction[rows],
                    reference_budget,
                    device,
                )
                success_logits, cost_scaled, effect_logits, immediate_scaled = network(batch)
                mask = torch.as_tensor(train_targets.action_mask[rows], dtype=torch.bool, device=device)
                success_target = torch.as_tensor(
                    train_targets.q_success[rows], dtype=torch.float32, device=device
                )
                success_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    success_logits, success_target, reduction="none"
                )
                success_loss = success_raw[mask].mean()
                cost_target = torch.as_tensor(
                    (np.log1p(train_targets.q_cost[rows]) - cumulative_mean) / cumulative_scale,
                    dtype=torch.float32,
                    device=device,
                )
                cost_raw = torch.nn.functional.smooth_l1_loss(
                    cost_scaled, cost_target, reduction="none"
                )
                cost_loss = cost_raw[mask].mean()
                immediate_target = torch.as_tensor(
                    (np.log1p(train_targets.immediate_verify_cost[rows]) - immediate_mean)
                    / immediate_scale,
                    dtype=torch.float32,
                    device=device,
                )
                immediate_loss = torch.nn.functional.smooth_l1_loss(
                    immediate_scaled, immediate_target
                )
                effect_target = torch.as_tensor(
                    train_targets.effect_label[rows], dtype=torch.long, device=device
                )
                effect_raw = torch.nn.functional.cross_entropy(
                    effect_logits, effect_target, weight=class_weights, reduction="none"
                )
                probability = torch.softmax(effect_logits, dim=1).gather(
                    1, effect_target[:, None]
                ).squeeze(1)
                effect_loss = (
                    (1.0 - probability) ** float(params.get("focal_gamma", 2.0))
                    * effect_raw
                ).mean()
                total = (
                    float(params.get("q_success_loss_weight", 1.0)) * success_loss
                    + float(params.get("q_cost_loss_weight", 0.35)) * cost_loss
                    + float(params.get("immediate_cost_loss_weight", 0.15)) * immediate_loss
                    + float(params.get("effect_loss_weight", 0.25)) * effect_loss
                )
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    network.parameters(), float(params.get("gradient_clip_norm", 5.0))
                )
                optimizer.step()
                totals.append(float(total.detach()))
                success_losses.append(float(success_loss.detach()))
                cost_losses.append(float(cost_loss.detach()))
                effect_losses.append(float(effect_loss.detach()))
                gradient_norms.append(float(gradient_norm.detach()))
                advantages.append(
                    float(
                        torch.abs(
                            torch.sigmoid(success_logits[:, 1])
                            - torch.sigmoid(success_logits[:, 0])
                        ).mean().detach()
                    )
                )
            network.eval()
            val_success_rows: list[float] = []
            val_cost_rows: list[float] = []
            with torch.inference_mode():
                all_validation = np.arange(len(validation_targets.state_indices))
                for start in range(0, len(all_validation), batch_size):
                    rows = all_validation[start : start + batch_size]
                    batch = _batch(
                        validation_arrays,
                        validation_targets.state_indices[rows],
                        validation_targets.allowance[rows],
                        validation_targets.allowance_fraction[rows],
                        reference_budget,
                        device,
                    )
                    success_logits, cost_scaled, _effect, _immediate = network(batch)
                    mask = torch.as_tensor(
                        validation_targets.action_mask[rows], dtype=torch.bool, device=device
                    )
                    success_target = torch.as_tensor(
                        validation_targets.q_success[rows], dtype=torch.float32, device=device
                    )
                    success_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                        success_logits, success_target, reduction="none"
                    )
                    val_success_rows.append(float(success_raw[mask].mean()))
                    cost_target = torch.as_tensor(
                        (np.log1p(validation_targets.q_cost[rows]) - cumulative_mean)
                        / cumulative_scale,
                        dtype=torch.float32,
                        device=device,
                    )
                    cost_raw = torch.nn.functional.smooth_l1_loss(
                        cost_scaled, cost_target, reduction="none"
                    )
                    val_cost_rows.append(float(cost_raw[mask].mean()))
            val_success = float(np.mean(val_success_rows))
            val_cost = float(np.mean(val_cost_rows))
            val_total = val_success + float(params.get("q_cost_loss_weight", 0.35)) * val_cost
            row = {
                "seed": seed,
                "variant": variant,
                "fold": fold,
                "member": member_index,
                "iteration": iteration,
                "epoch": epoch + 1,
                "total_loss": float(np.mean(totals)),
                "q_success_loss": float(np.mean(success_losses)),
                "q_cost_loss": float(np.mean(cost_losses)),
                "effect_loss": float(np.mean(effect_losses)),
                "validation_q_success_loss": val_success,
                "validation_q_cost_loss": val_cost,
                "validation_total_loss": val_total,
                "mean_absolute_predicted_advantage": float(np.mean(advantages)),
                "gradient_norm": float(np.mean(gradient_norms)),
                "transferred_parameters": transferred,
            }
            history.append(row)
            # Keep the dashboard compact: experimental dimensions are runs,
            # while only high-value diagnostics become scalar cards.
            member_writer.add_scalar("loss/train_total", row["total_loss"], epoch + 1)
            member_writer.add_scalar(
                "loss/validation_total", row["validation_total_loss"], epoch + 1
            )
            member_writer.add_scalar(
                "loss/validation_q_success",
                row["validation_q_success_loss"],
                epoch + 1,
            )
            member_writer.add_scalar(
                "loss/validation_q_cost",
                row["validation_q_cost_loss"],
                epoch + 1,
            )
            member_writer.add_scalar(
                "diagnostics/mean_absolute_advantage",
                row["mean_absolute_predicted_advantage"],
                epoch + 1,
            )
            member_writer.add_scalar(
                "diagnostics/gradient_norm", row["gradient_norm"], epoch + 1
            )
            if val_total < best_loss - 1e-5:
                best_loss = val_total
                best_epoch = epoch + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            epoch_bar.set_postfix(
                train=f"{row['total_loss']:.4f}",
                val=f"{val_total:.4f}",
                q=f"{val_success:.4f}",
                cost=f"{val_cost:.4f}",
                patience=f"{stale}/{int(params.get('patience', 10))}",
            )
            if stale >= int(params.get("patience", 10)):
                break
        epoch_bar.close()
        if best_state is None:
            raise RuntimeError("DARC v0.5 produced no checkpoint")
        members.append(
            V05Member(
                state_dict=best_state,
                fold=fold,
                member=member_index,
                iteration=iteration,
                best_epoch=best_epoch,
                best_validation_loss=best_loss,
                cumulative_cost_log_mean=cumulative_mean,
                cumulative_cost_log_scale=cumulative_scale,
                immediate_cost_log_mean=immediate_mean,
                immediate_cost_log_scale=immediate_scale,
                history=tuple(history),
            )
        )
        jobs_bar.set_postfix(fold=fold, member=member_index, epoch=best_epoch, val=f"{best_loss:.4f}")
    jobs_bar.close()
    return DARCv05Ensemble(
        embedding_dim=train_encoded.task.shape[1],
        metadata_dim=train_encoded.metadata.shape[1],
        params=params,
        reference_budget=reference_budget,
        metadata_mean=mean,
        metadata_scale=scale,
        members=members,
    )


def policy_grid(config: dict[str, Any], variant: dict[str, Any]) -> list[ConservativePolicy]:
    import itertools

    rows = []
    for values in itertools.product(
        config["base_token_price"],
        config["price_elasticity"],
        config["advantage_uncertainty_beta"],
        config["cost_uncertainty_beta"],
        config["harmful_weight"],
        config["decision_margin"],
    ):
        rows.append(
            ConservativePolicy(
                *map(float, values),
                conservative=bool(variant["conservative"]),
                dynamic_token_price=bool(variant["dynamic_token_price"]),
                uncertainty_penalty=bool(variant["uncertainty_penalty"]),
            )
        )
    # Deduplicate configurations disabled by an ablation.
    unique = {tuple(asdict(row).items()): row for row in rows}
    return list(unique.values())

