"""AdamW training for exact full-tree differentiable policy optimization."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from agent_verify.analysis.darc_data import EncodedDARC
from agent_verify.analysis.darc_fulltree_data import FullTreeDataset
from agent_verify.analysis.darc_fulltree_model import (
    FullTreePolicyCheckpoint,
    base_arrays,
    network_class,
    policy_batch,
)
from agent_verify.analysis.darc_fulltree_objective import (
    edge_expected_tokens,
    fulltree_expectation,
)
from agent_verify.analysis.darc_model import _torch, select_device, set_determinism
from agent_verify.analysis.darc_v04_data import MultiHistoryState


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    _torch().save(payload, temporary)
    temporary.replace(path)


def _anchor_by_state(torch: Any, config: dict[str, Any], device: Any) -> Any:
    cp = [float(value) for value in config["cp_verify_probability"]]
    if len(cp) != 3 or any(not 0.0 < value < 1.0 for value in cp):
        raise ValueError("anchor.cp_verify_probability must contain three probabilities in (0,1)")
    # Prefix order: root; S,V; SS,SV,VS,VV.
    return torch.as_tensor([cp[0], cp[1], cp[1], cp[2], cp[2], cp[2], cp[2]], device=device)


def _metadata_statistics(encoded: EncodedDARC, tree: FullTreeDataset) -> tuple[np.ndarray, np.ndarray]:
    indices = tree.state_indices.reshape(-1)
    values = encoded.metadata[indices]
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def train_fulltree_policy(
    *,
    all_states: list[MultiHistoryState],
    encoded: EncodedDARC,
    train_tree: FullTreeDataset,
    params: dict[str, Any],
    optimization: dict[str, Any],
    anchor: dict[str, Any],
    variant: str,
    reference_budget: float,
    seed: int,
    fold: int,
    device_name: str,
    deterministic: bool,
    checkpoint_path: Path,
    last_path: Path,
    resume: bool,
    progress: dict[str, Any],
) -> FullTreePolicyCheckpoint:
    """Train with fixed epochs; held-out fold outcomes are never used for stopping."""

    torch = _torch()
    set_determinism(seed + 10_007 * (fold + 1), deterministic)
    device = select_device(device_name)
    mean, scale = _metadata_statistics(encoded, train_tree)
    arrays = base_arrays(encoded, all_states, mean, scale)
    network = network_class()(encoded.task.shape[1], encoded.metadata.shape[1], params).to(device)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(params.get("learning_rate", 5e-4)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
    )
    fractions = [float(value) for value in optimization["train_budget_fractions"]]
    if not fractions or any(value < 0 for value in fractions):
        raise ValueError("At least one non-negative training budget is required")
    multipliers = {
        f"{fraction:g}": float(optimization.get("initial_lagrange", 0.05)) for fraction in fractions
    }
    use_anchor = variant == "cp3_anchor"
    if variant not in {"cp3_anchor", "no_anchor"}:
        raise ValueError(f"Unknown Full-Tree variant: {variant}")
    anchor_epochs = int(anchor.get("pretrain_epochs", 0)) if use_anchor else 0
    main_epochs = int(params.get("epochs", 50))
    history: list[dict[str, Any]] = []
    start_stage = "anchor" if anchor_epochs else "main"
    start_epoch = 0
    if resume and last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved.get("variant") != variant or int(saved.get("seed")) != seed:
            raise ValueError(f"Resume checkpoint identity mismatch: {last_path}")
        network.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        multipliers = {str(key): float(value) for key, value in saved["multipliers"].items()}
        history = list(saved.get("history", []))
        start_stage = str(saved["next_stage"])
        start_epoch = int(saved["next_epoch"])

    anchor_state = _anchor_by_state(torch, anchor, device)
    batch_size = int(params.get("task_batch_size", 32))
    clip = float(params.get("gradient_clip_norm", 5.0))
    tqdm_enabled = bool(progress.get("tqdm", True))
    mininterval = float(progress.get("mininterval_seconds", 1.0))
    started = time.perf_counter()

    stages: list[tuple[str, int]] = []
    if anchor_epochs:
        stages.append(("anchor", anchor_epochs))
    stages.append(("main", main_epochs))
    stage_seen = start_stage == "anchor"
    for stage, epochs in stages:
        if not stage_seen:
            if stage == start_stage:
                stage_seen = True
            else:
                continue
        first_epoch = start_epoch if stage == start_stage else 0
        bar = tqdm(
            range(first_epoch, epochs),
            desc=f"FullTree {variant} s{seed} f{fold} {stage}",
            unit="epoch",
            dynamic_ncols=True,
            disable=not tqdm_enabled,
            mininterval=mininterval,
        )
        for epoch in bar:
            network.train()
            rng = np.random.default_rng(
                seed + fold * 100_003 + epoch + (0 if stage == "anchor" else 1_000_000)
            )
            order = rng.permutation(len(train_tree.task_keys))
            losses: list[float] = []
            rewards: list[float] = []
            costs: dict[str, list[float]] = {f"{value:g}": [] for value in fractions}
            calls: list[float] = []
            kls: list[float] = []
            entropies: list[float] = []
            gradient_norms: list[float] = []
            for start in range(0, len(order), batch_size):
                rows = order[start : start + batch_size]
                batch_tree = train_tree.subset(rows)
                for fraction in fractions:
                    target = fraction * reference_budget
                    allowances = batch_tree.remaining_allowance(target)
                    flat_indices = batch_tree.state_indices.reshape(-1)
                    flat_allowances = allowances.reshape(-1)
                    batch = policy_batch(
                        arrays,
                        flat_indices,
                        flat_allowances,
                        reference_budget=reference_budget,
                        device=device,
                    )
                    logits = network(batch).reshape(len(rows), 7)
                    if stage == "anchor":
                        target_probability = anchor_state.unsqueeze(0).expand_as(logits)
                        anchor_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits, target_probability
                        )
                        loss = float(anchor.get("pretrain_weight", 1.0)) * anchor_loss
                        expectation = None
                    else:
                        probability = torch.sigmoid(logits)
                        outcomes = torch.as_tensor(batch_tree.outcomes, dtype=torch.float32, device=device)
                        terminal_cost = torch.as_tensor(
                            batch_tree.trajectory_tokens, dtype=torch.float32, device=device
                        )
                        anchor_probability = (
                            anchor_state.unsqueeze(0).expand_as(probability) if use_anchor else None
                        )
                        expectation = fulltree_expectation(
                            probability,
                            outcomes,
                            terminal_cost,
                            target_tokens=target,
                            reference_tokens=reference_budget,
                            lagrange_multiplier=multipliers[f"{fraction:g}"],
                            anchor_probability=anchor_probability,
                            kl_weight=float(anchor.get("kl_weight", 0.0)) if use_anchor else 0.0,
                            entropy_weight=float(optimization.get("entropy_weight", 0.0)),
                        )
                        # Terminal-cost expectation must equal the sum over visited
                        # Verify edges. This catches accidental shared-prefix duplication.
                        edge_cost = torch.as_tensor(
                            batch_tree.edge_tokens, dtype=torch.float32, device=device
                        )
                        edge_cost_value = edge_expected_tokens(probability, edge_cost)
                        if not torch.allclose(
                            edge_cost_value,
                            expectation.expected_tokens,
                            atol=1e-3,
                            rtol=1e-5,
                        ):
                            raise AssertionError("Expected token identity failed")
                        loss = expectation.loss
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(network.parameters(), clip)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    gradient_norms.append(float(gradient.detach()))
                    if expectation is not None:
                        rewards.append(float(expectation.expected_correctness.mean().detach()))
                        costs[f"{fraction:g}"].append(float(expectation.expected_tokens.mean().detach()))
                        calls.append(float(expectation.expected_calls.mean().detach()))
                        kls.append(float(expectation.expected_kl.mean().detach()))
                        entropies.append(float(expectation.expected_entropy.mean().detach()))
            if stage == "main" and bool(optimization.get("learn_lagrange", True)):
                dual_lr = float(optimization.get("dual_learning_rate", 0.05))
                upper = float(optimization.get("max_lagrange", 20.0))
                for fraction in fractions:
                    key = f"{fraction:g}"
                    observed = float(np.mean(costs[key]))
                    gap = (observed - fraction * reference_budget) / max(reference_budget, 1.0)
                    multipliers[key] = float(np.clip(multipliers[key] + dual_lr * gap, 0.0, upper))
            row = {
                "variant": variant,
                "seed": seed,
                "fold": fold,
                "stage": stage,
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "expected_accuracy": float(np.mean(rewards)) if rewards else float("nan"),
                "expected_tokens": float(np.mean([x for values in costs.values() for x in values]))
                if rewards
                else float("nan"),
                "expected_calls": float(np.mean(calls)) if calls else float("nan"),
                "expected_kl": float(np.mean(kls)) if kls else 0.0,
                "expected_entropy": float(np.mean(entropies)) if entropies else 0.0,
                "gradient_norm": float(np.mean(gradient_norms)),
                "elapsed_seconds": time.perf_counter() - started,
                **{f"lambda_{key}": value for key, value in multipliers.items()},
            }
            history.append(row)
            bar.set_postfix(
                loss=f"{row['loss']:.4f}",
                acc="-" if not rewards else f"{row['expected_accuracy']:.3f}",
                tok="-" if not rewards else f"{row['expected_tokens']:.0f}",
                grad=f"{row['gradient_norm']:.2f}",
            )
            if epoch + 1 < epochs:
                next_stage, next_epoch = stage, epoch + 1
            elif stage == "anchor":
                next_stage, next_epoch = "main", 0
            else:
                next_stage, next_epoch = "done", 0
            _atomic_torch_save(
                {
                    "schema_version": "darc-fulltree-resume-v1",
                    "variant": variant,
                    "seed": seed,
                    "fold": fold,
                    "state_dict": network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "multipliers": multipliers,
                    "history": history,
                    "next_stage": next_stage,
                    "next_epoch": next_epoch,
                },
                last_path,
            )
        bar.close()
        start_epoch = 0

    checkpoint = FullTreePolicyCheckpoint(
        state_dict={name: value.detach().cpu().clone() for name, value in network.state_dict().items()},
        embedding_dim=encoded.task.shape[1],
        metadata_dim=encoded.metadata.shape[1],
        params=dict(params),
        reference_budget=float(reference_budget),
        metadata_mean=mean,
        metadata_scale=scale,
        variant=variant,
        seed=seed,
        fold=fold,
        history=tuple(history),
        lagrange_multipliers=multipliers,
    )
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    checkpoint.save(temporary)
    temporary.replace(checkpoint_path)
    return checkpoint
