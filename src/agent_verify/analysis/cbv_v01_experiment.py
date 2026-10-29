"""Frozen CBV-Bench v0.1 training and budgeted offline evaluation.

This module is deliberately independent from the exploratory HumanEval/LCB loaders.
It consumes only the materialized train/validation/test folders produced by
``freeze_cbv_bench_v01.py``.  Hyperparameters and policy thresholds are selected on
validation; test labels are touched only by the final evaluation stage.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from agent_verify.analysis.c3_controller import (
    FittedSequentialController,
    extract_observations,
)
from agent_verify.analysis.c3_experiment import replay_controller
from agent_verify.analysis.joint_value_cost_data import (
    JointExample,
    VALUE_TO_INDEX,
    value_label,
)
from agent_verify.analysis.joint_value_cost_experiment import prediction_metrics
from agent_verify.analysis.joint_value_cost_models import (
    JointValueCostModel,
    SeparateValueCostModel,
    ValueCostPrediction,
    calibrate_probabilities,
    choose_temperature,
)
from agent_verify.analysis.phase2_data import (
    CHECKPOINT_ROLES,
    PairExample,
    build_observable_features,
)
from agent_verify.analysis.phase2_models import MODEL_GRIDS, THRESHOLD_GRID, TwoHeadPredictor


SPLITS = ("train", "validation", "test")
DATASETS = ("humaneval", "livecodebench", "bigcodebench")
SINGLE_SEQUENCES = {1: "VSS", 2: "SVS", 3: "SSV"}
ALL_SEQUENCES = tuple(
    "".join(bits) for bits in __import__("itertools").product("SV", repeat=3)
)
MODE_DATASETS = {
    "humaneval": {"humaneval"},
    "livecodebench": {"livecodebench"},
    "bigcodebench": {"bigcodebench"},
    "combined": set(DATASETS),
}
ROLES = CHECKPOINT_ROLES


@dataclass(frozen=True)
class FrozenTask:
    task_id: str
    dataset: str
    difficulty: str
    prompt: str
    entry_point: str
    level1_complete: bool
    level2_complete: bool

    @property
    def group_key(self) -> str:
        return f"{self.dataset}::{self.task_id}"


@dataclass(frozen=True)
class TaskTrajectories:
    task: FrozenTask
    rows: dict[str, dict[str, Any]]
    source_level: int


@dataclass(frozen=True)
class ReplayResult:
    policy: str
    protocol: str
    train_mode: str
    scope: str
    budget_fraction: float
    token_budget: int
    task_keys: tuple[str, ...]
    outcomes: np.ndarray
    baseline: np.ndarray
    checkpoints: np.ndarray
    calls: int
    actual_tokens: int
    predicted_tokens: float


class MultiHistoryController:
    """Budget-conditioned fitted-Q approximation over the eight L2 histories.

    At inference, a state contains only the public task, artifacts produced along the
    already observed prefix, checkpoint/history and remaining budget allowance. Future
    outcomes are used solely while constructing train targets.
    """

    def __init__(self, params: dict[str, Any], *, seed: int, continuation: bool) -> None:
        self.params = dict(params)
        self.seed = seed
        self.continuation = continuation
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.scaler: StandardScaler | None = None
        self.q_skip: GradientBoostingRegressor | None = None
        self.q_verify: GradientBoostingRegressor | None = None
        self.cost: GradientBoostingRegressor | None = None
        self.reference_budget = 1.0

    @staticmethod
    def _state(group: TaskTrajectories, prefix: str) -> tuple[str, np.ndarray]:
        checkpoint = len(prefix) + 1
        if checkpoint not in (1, 2, 3):
            raise ValueError(f"Invalid history prefix {prefix!r}")
        skip_sequence = prefix + "S" + "S" * (2 - len(prefix))
        row = group.rows[skip_sequence]
        current = _artifact(row, checkpoint)
        preceding = [_artifact(row, prior) for prior in range(1, checkpoint)]
        text, metadata, _names = build_observable_features(
            prompt=group.task.prompt,
            entry_point=group.task.entry_point,
            checkpoint=checkpoint,
            role=ROLES[checkpoint],
            current_artifact=current,
            preceding_artifacts=preceding,
            local_risk=0.0,
        )
        domain, _domain_names = _domain_metadata(group.task)
        history = [
            float(prefix[index] == "V") if index < len(prefix) else 0.0
            for index in range(2)
        ]
        structured = np.asarray(
            [*metadata, *domain, *history, float(prefix.count("V"))], dtype=float
        )
        return (
            "\n".join(
                (
                    f"[DATASET]\n{group.task.dataset}",
                    f"[DIFFICULTY]\n{group.task.difficulty}",
                    f"[OBSERVED_HISTORY]\n{prefix or 'START'}",
                    text,
                )
            ),
            structured,
        )

    @staticmethod
    def _prefix_cost(group: TaskTrajectories, prefix: str) -> int:
        sequence = prefix + "S" * (3 - len(prefix))
        return int(group.rows[sequence].get("teacher_total_tokens", 0) or 0)

    @classmethod
    def _verify_cost(cls, group: TaskTrajectories, prefix: str) -> int:
        before = cls._prefix_cost(group, prefix)
        after_sequence = prefix + "V" + "S" * (2 - len(prefix))
        after = int(group.rows[after_sequence].get("teacher_total_tokens", 0) or 0)
        return max(0, after - before)

    def _action_target(
        self,
        group: TaskTrajectories,
        prefix: str,
        action: str,
        allowance: float,
    ) -> float:
        direct = prefix + action + "S" * (2 - len(prefix))
        if not self.continuation:
            return float(_passed(group.rows[direct]))
        before = self._prefix_cost(group, prefix)
        candidates = [
            sequence for sequence in ALL_SEQUENCES
            if sequence.startswith(prefix + action)
            and int(group.rows[sequence].get("teacher_total_tokens", 0) or 0) - before
            <= allowance + 1e-9
        ]
        if not candidates:
            return float(_passed(group.rows[direct]))
        # Correctness is the primary utility; lower additional cost breaks ties.
        selected = max(
            candidates,
            key=lambda sequence: (
                _passed(group.rows[sequence]),
                -(int(group.rows[sequence].get("teacher_total_tokens", 0) or 0) - before),
            ),
        )
        return float(_passed(group.rows[selected]))

    def _encode_fit(self, states: list[tuple[str, np.ndarray]]) -> np.ndarray:
        self.vectorizer = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.98,
            max_features=int(self.params.get("max_features", 6000)),
            sublinear_tf=True, token_pattern=r"(?u)\b[\w.]+\b",
        )
        sparse = self.vectorizer.fit_transform([text for text, _ in states])
        components = min(
            int(self.params.get("svd_components", 32)),
            max(1, sparse.shape[0] - 1), max(1, sparse.shape[1] - 1),
        )
        self.svd = TruncatedSVD(n_components=components, random_state=self.seed)
        semantic = self.svd.fit_transform(sparse)
        structured = np.vstack([values for _, values in states])
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(np.column_stack((semantic, structured)))

    def _encode(self, states: list[tuple[str, np.ndarray]]) -> np.ndarray:
        if self.vectorizer is None or self.svd is None or self.scaler is None:
            raise RuntimeError("Multi-history encoder is not fitted")
        semantic = self.svd.transform(self.vectorizer.transform([text for text, _ in states]))
        structured = np.vstack([values for _, values in states])
        return self.scaler.transform(np.column_stack((semantic, structured)))

    def _budget_features(self, base: np.ndarray, allowances: np.ndarray) -> np.ndarray:
        reference = max(self.reference_budget, 1.0)
        extra = np.column_stack(
            (
                np.log1p(np.maximum(allowances, 0.0)),
                np.maximum(allowances, 0.0) / reference,
            )
        )
        return np.column_stack((base, extra))

    def fit(
        self,
        groups: dict[str, TaskTrajectories],
        *,
        budget_levels: list[float],
    ) -> "MultiHistoryController":
        unique_states: list[tuple[str, str, tuple[str, np.ndarray]]] = []
        for key in sorted(groups):
            for checkpoint in (1, 2, 3):
                for bits in __import__("itertools").product("SV", repeat=checkpoint - 1):
                    prefix = "".join(bits)
                    unique_states.append((key, prefix, self._state(groups[key], prefix)))
        base = self._encode_fit([state for _key, _prefix, state in unique_states])
        self.reference_budget = max(float(np.median([self._prefix_cost(group, "VVV") for group in groups.values()])), 1.0)
        matrices: list[np.ndarray] = []
        skip_targets: list[float] = []
        verify_targets: list[float] = []
        cost_targets: list[float] = []
        for allowance in budget_levels:
            matrices.append(
                self._budget_features(base, np.full(len(unique_states), allowance, dtype=float))
            )
            for key, prefix, _state in unique_states:
                group = groups[key]
                skip_targets.append(self._action_target(group, prefix, "S", allowance))
                verify_targets.append(self._action_target(group, prefix, "V", allowance))
                cost_targets.append(math.log1p(self._verify_cost(group, prefix)))
        matrix = np.vstack(matrices)
        params = {
            "loss": "huber",
            "n_estimators": int(self.params.get("n_estimators", 120)),
            "learning_rate": float(self.params.get("learning_rate", 0.03)),
            "max_depth": int(self.params.get("max_depth", 2)),
            "min_samples_leaf": int(self.params.get("min_samples_leaf", 10)),
            "subsample": float(self.params.get("subsample", 0.9)),
        }
        self.q_skip = GradientBoostingRegressor(**params, random_state=self.seed + 11).fit(matrix, np.asarray(skip_targets))
        self.q_verify = GradientBoostingRegressor(**params, random_state=self.seed + 17).fit(matrix, np.asarray(verify_targets))
        self.cost = GradientBoostingRegressor(**params, random_state=self.seed + 23).fit(matrix, np.asarray(cost_targets))
        return self

    def predict(
        self, group: TaskTrajectories, prefix: str, allowance: float
    ) -> tuple[float, float, float]:
        if self.q_skip is None or self.q_verify is None or self.cost is None:
            raise RuntimeError("Multi-history controller is not fitted")
        base = self._encode([self._state(group, prefix)])
        matrix = self._budget_features(base, np.asarray([allowance], dtype=float))
        return (
            float(self.q_skip.predict(matrix)[0]),
            float(self.q_verify.predict(matrix)[0]),
            max(0.0, float(np.expm1(self.cost.predict(matrix)[0]))),
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def _stable_order(values: Iterable[str], seed: int) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )


def load_frozen_partition(root: Path, split: str) -> tuple[dict[str, FrozenTask], list[dict[str, Any]]]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    task_rows = _read_jsonl(root / split / "tasks.jsonl")
    trajectory_rows = _read_jsonl(root / split / "trajectories.jsonl")
    tasks: dict[str, FrozenTask] = {}
    for row in task_rows:
        public = row.get("public_task", {})
        task = FrozenTask(
            task_id=str(row["task_id"]),
            dataset=str(row["dataset"]).lower(),
            difficulty=str(row.get("difficulty", "unknown")).lower(),
            prompt=str(public.get("prompt", "")),
            entry_point=str(public.get("entry_point", "")),
            level1_complete=bool(row.get("level1_complete")),
            level2_complete=bool(row.get("level2_complete")),
        )
        if task.group_key in tasks:
            raise ValueError(f"Duplicate frozen task {task.group_key}")
        tasks[task.group_key] = task
    if any(str(row.get("frozen_split")) != split for row in trajectory_rows):
        raise ValueError(f"Trajectory split annotation mismatch in {split}")
    return tasks, trajectory_rows


def _complete_sequence_map(rows: list[dict[str, Any]], sequences: set[str]) -> dict[str, dict[str, Any]] | None:
    by_sequence: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "completed" or row.get("passed") is None:
            continue
        sequence = str(row.get("action_sequence", ""))
        if sequence in sequences:
            by_sequence.setdefault(sequence, []).append(row)
    if set(by_sequence) != sequences:
        return None
    # Same-level duplicates are not silently merged.  The frozen collection has one
    # rollout per task/level; deterministic sorting is only a defensive fallback.
    return {
        sequence: sorted(candidates, key=lambda row: str(row.get("trajectory_id")))[0]
        for sequence, candidates in by_sequence.items()
    }


def task_trajectories(
    tasks: dict[str, FrozenTask],
    rows: list[dict[str, Any]],
    *,
    level: int,
    datasets: set[str] | None = None,
) -> dict[str, TaskTrajectories]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        dataset = str(row.get("dataset", "")).lower()
        key = f"{dataset}::{row.get('task_id')}"
        if key not in tasks or (datasets is not None and dataset not in datasets):
            continue
        grouped.setdefault(key, {}).setdefault(int(row.get("level", 0)), []).append(row)
    required = set(ALL_SEQUENCES if level == 2 else ("SSS", "VSS", "SVS", "SSV"))
    output: dict[str, TaskTrajectories] = {}
    for key, task in tasks.items():
        if datasets is not None and task.dataset not in datasets:
            continue
        levels = grouped.get(key, {})
        if level == 2:
            mapping = _complete_sequence_map(levels.get(2, []), required)
            # HumanEval/LiveCodeBench recovery collections store the four L1
            # branches under level=1 and only the four new multi branches under
            # level=2.  BigCodeBench L2 stores all eight directly.  Merge only as
            # a compatibility fallback; sequence completeness is still mandatory.
            if mapping is None:
                mapping = _complete_sequence_map(
                    [*levels.get(1, []), *levels.get(2, [])], required
                )
            if mapping is not None:
                output[key] = TaskTrajectories(task, mapping, 2)
            continue
        # Prefer the richer L2 rollout when available, then fall back to L1.  This
        # yields exactly one counterfactual group per task and avoids double counting
        # BigCodeBench L2 tasks that are a subset of its L1 collection.
        mapping = _complete_sequence_map(levels.get(2, []), required)
        source_level = 2
        if mapping is None:
            mapping = _complete_sequence_map(levels.get(1, []), required)
            source_level = 1
        if mapping is not None:
            output[key] = TaskTrajectories(task, mapping, source_level)
    return output


def _passed(row: dict[str, Any]) -> int:
    return int(bool(row.get("passed")))


def _artifact(row: dict[str, Any], checkpoint: int) -> dict[str, Any]:
    snapshots = row.get("artifact_snapshots", [])
    if len(snapshots) < checkpoint or not isinstance(snapshots[checkpoint - 1], dict):
        raise ValueError(f"Missing artifact snapshot CP{checkpoint} in {row.get('trajectory_id')}")
    # A few benchmark prompts intentionally contain escaped/binary-like content.
    # Python's AST parser rejects literal NUL bytes, so sanitize only that control
    # character in the observable copy used for features; raw frozen artifacts stay
    # untouched and their checksums remain reproducible.
    def clean(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace("\x00", "\\x00")
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        return value

    return clean(dict(snapshots[checkpoint - 1]))


def _domain_metadata(task: FrozenTask) -> tuple[tuple[float, ...], tuple[str, ...]]:
    names = (
        "dataset_humaneval", "dataset_livecodebench", "dataset_bigcodebench",
        "difficulty_easy", "difficulty_medium", "difficulty_hard", "difficulty_unknown",
        "local_risk_available",
    )
    difficulty = task.difficulty
    values = (
        float(task.dataset == "humaneval"),
        float(task.dataset == "livecodebench"),
        float(task.dataset == "bigcodebench"),
        float(difficulty == "easy"),
        float(difficulty == "medium"),
        float(difficulty == "hard"),
        float(difficulty not in {"easy", "medium", "hard"}),
        0.0,
    )
    return values, names


def build_level1_examples(groups: dict[str, TaskTrajectories]) -> list[JointExample]:
    examples: list[JointExample] = []
    schema: tuple[str, ...] | None = None
    for key in sorted(groups):
        group = groups[key]
        baseline = group.rows["SSS"]
        y_skip = _passed(baseline)
        for checkpoint, sequence in SINGLE_SEQUENCES.items():
            verified = group.rows[sequence]
            current = _artifact(baseline, checkpoint)
            preceding = [
                _artifact(baseline, prior) for prior in range(1, checkpoint)
            ]
            text, metadata, names = build_observable_features(
                prompt=group.task.prompt,
                entry_point=group.task.entry_point,
                checkpoint=checkpoint,
                role=ROLES[checkpoint],
                current_artifact=current,
                preceding_artifacts=preceding,
                local_risk=0.0,
            )
            domain_values, domain_names = _domain_metadata(group.task)
            names = (*names, *domain_names)
            metadata = (*metadata, *domain_values)
            if schema is None:
                schema = names
            elif schema != names:
                raise ValueError("Frozen Level-1 feature schema changed across rows")
            input_tokens = int(verified.get("teacher_input_tokens", 0) or 0)
            output_tokens = int(verified.get("teacher_output_tokens", 0) or 0)
            total_tokens = int(verified.get("teacher_total_tokens", input_tokens + output_tokens) or 0)
            if total_tokens != input_tokens + output_tokens:
                raise ValueError(f"Teacher token mismatch for {key} {sequence}")
            y_verify = _passed(verified)
            examples.append(
                JointExample(
                    task_id=group.task.task_id,
                    dataset=group.task.dataset,
                    difficulty=group.task.difficulty,
                    checkpoint=checkpoint,
                    role=ROLES[checkpoint],
                    text="\n".join((f"[DATASET]\n{group.task.dataset}", f"[DIFFICULTY]\n{group.task.difficulty}", text)),
                    metadata=metadata,
                    metadata_names=names,
                    y_skip=y_skip,
                    y_verify=y_verify,
                    delta=y_verify - y_skip,
                    value_label=value_label(y_skip, y_verify),
                    local_risk=0.0,
                    teacher_input_tokens=input_tokens,
                    teacher_output_tokens=output_tokens,
                    teacher_total_tokens=total_tokens,
                )
            )
    return examples


def as_pair_examples(examples: list[JointExample]) -> list[PairExample]:
    return [
        PairExample(
            task_id=row.group_key,
            checkpoint=row.checkpoint,
            role=row.role,
            text=row.text,
            metadata=row.metadata,
            metadata_names=row.metadata_names,
            y_skip=row.y_skip,
            y_verify=row.y_verify,
            delta=row.delta,
            local_risk=row.local_risk,
            teacher_changed=False,
            teacher_tokens=row.teacher_total_tokens,
            teacher_latency_seconds=0.0,
            teacher_cost_usd=0.0,
            original_artifact={},
            teacher_explanation="",
        )
        for row in examples
    ]


def subset_examples(examples: list[JointExample], datasets: set[str]) -> list[JointExample]:
    return [row for row in examples if row.dataset in datasets]


def _temperature(model: Any, validation: list[JointExample]) -> float:
    raw = model.raw_value_probabilities(validation)
    labels = np.asarray([row.value_label for row in validation], dtype=int)
    return choose_temperature(raw, labels, [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0])


def _prediction_score(examples: list[JointExample], prediction: ValueCostPrediction) -> float:
    metrics = prediction_metrics(examples, prediction)
    helpful = metrics["helpful_auprc"] if math.isfinite(metrics["helpful_auprc"]) else 0.0
    harmful = metrics["harmful_auprc"] if math.isfinite(metrics["harmful_auprc"]) else 0.0
    corr = metrics["gain_pearson"] if math.isfinite(metrics["gain_pearson"]) else 0.0
    return float(helpful + 0.5 * harmful + 0.25 * corr - 0.1 * metrics["value_log_loss"] - 0.2 * metrics["cost_mean_relative_error"])


def _legacy_costs(train: list[JointExample], evaluation: list[JointExample]) -> np.ndarray:
    medians: dict[tuple[str, int], float] = {}
    global_median = float(np.median([row.teacher_total_tokens for row in train]))
    for dataset in DATASETS:
        for checkpoint in (1, 2, 3):
            values = [row.teacher_total_tokens for row in train if row.dataset == dataset and row.checkpoint == checkpoint]
            medians[(dataset, checkpoint)] = float(np.median(values)) if values else global_median
    return np.asarray([medians[(row.dataset, row.checkpoint)] for row in evaluation], dtype=float)


def _budget_values(train: list[JointExample], task_count: int) -> list[tuple[float, int]]:
    costs = np.asarray([row.teacher_total_tokens for row in train], dtype=float)
    per_task = float(np.median(costs)) if len(costs) else 0.0
    return [(fraction, int(round(fraction * per_task * task_count))) for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)]


def _group_indices(examples: list[JointExample]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(examples):
        grouped.setdefault(row.group_key, []).append(index)
    for indices in grouped.values():
        indices.sort(key=lambda index: examples[index].checkpoint)
    return grouped


def replay_single(
    examples: list[JointExample],
    scores: np.ndarray,
    predicted_costs: np.ndarray,
    *,
    threshold: float,
    token_budget: int,
    seed: int,
    policy: str,
    protocol: str = "A-single",
    train_mode: str = "combined",
    scope: str = "combined",
    budget_fraction: float = 1.0,
    score_per_1k: bool = False,
) -> ReplayResult:
    grouped = _group_indices(examples)
    remaining_predicted = float(token_budget)
    task_keys = _stable_order(grouped, seed)
    outcomes: list[float] = []
    baseline: list[float] = []
    checkpoints: list[int] = []
    actual_tokens = 0
    predicted_tokens = 0.0
    calls = 0
    for key in task_keys:
        chosen: int | None = None
        for index in grouped[key]:
            score = float(scores[index])
            cost = max(float(predicted_costs[index]), 1.0)
            decision_score = 1000.0 * score / cost if score_per_1k else score
            if score > 0.0 and decision_score >= threshold and cost <= remaining_predicted:
                chosen = index
                break
        base = float(examples[grouped[key][0]].y_skip)
        baseline.append(base)
        if chosen is None:
            outcomes.append(base)
            checkpoints.append(0)
            continue
        predicted = max(float(predicted_costs[chosen]), 0.0)
        remaining_predicted -= predicted
        predicted_tokens += predicted
        actual_tokens += examples[chosen].teacher_total_tokens
        outcomes.append(float(examples[chosen].y_verify))
        checkpoints.append(examples[chosen].checkpoint)
        calls += 1
    return ReplayResult(
        policy, protocol, train_mode, scope, budget_fraction, token_budget,
        tuple(task_keys), np.asarray(outcomes), np.asarray(baseline),
        np.asarray(checkpoints), calls, actual_tokens, predicted_tokens,
    )


def replay_single_baseline(
    examples: list[JointExample],
    *,
    policy: str,
    token_budget: int,
    seed: int,
    scope: str,
    budget_fraction: float,
) -> ReplayResult:
    grouped = _group_indices(examples)
    task_keys = _stable_order(grouped, seed)
    remaining = int(token_budget)
    rng = np.random.default_rng(seed)
    outcomes: list[float] = []
    baseline: list[float] = []
    checkpoints: list[int] = []
    tokens = calls = 0
    for key in task_keys:
        indices = grouped[key]
        base = float(examples[indices[0]].y_skip)
        baseline.append(base)
        chosen: int | None = None
        if policy.startswith("Always CP"):
            cp = int(policy[-1])
            chosen = next(i for i in indices if examples[i].checkpoint == cp)
        elif policy == "Random Checkpoint":
            chosen = indices[int(rng.integers(0, 3))]
        elif policy == "Offline Oracle":
            helpful = [i for i in indices if examples[i].delta > 0]
            if helpful:
                chosen = min(helpful, key=lambda i: examples[i].teacher_total_tokens)
        elif policy != "Never Verify":
            raise ValueError(policy)
        if chosen is not None and examples[chosen].teacher_total_tokens <= remaining:
            remaining -= examples[chosen].teacher_total_tokens
            tokens += examples[chosen].teacher_total_tokens
            calls += 1
            outcomes.append(float(examples[chosen].y_verify))
            checkpoints.append(examples[chosen].checkpoint)
        else:
            outcomes.append(base)
            checkpoints.append(0)
    return ReplayResult(
        policy, "A-single", "none", scope, budget_fraction, token_budget,
        tuple(task_keys), np.asarray(outcomes), np.asarray(baseline),
        np.asarray(checkpoints), calls, tokens, float(tokens),
    )


def tune_threshold(
    examples: list[JointExample],
    scores: np.ndarray,
    predicted_costs: np.ndarray,
    train_examples: list[JointExample],
    *,
    thresholds: Iterable[float],
    seed: int,
    score_per_1k: bool,
) -> tuple[float, float]:
    task_count = len(_group_indices(examples))
    best: tuple[float, float, float] | None = None
    selected = 0.0
    for threshold in thresholds:
        replays = [
            replay_single(
                examples, scores, predicted_costs, threshold=float(threshold),
                token_budget=budget, seed=seed, policy="validation",
                budget_fraction=fraction, score_per_1k=score_per_1k,
            )
            for fraction, budget in _budget_values(train_examples, task_count)
        ]
        accuracy = float(np.mean([row.outcomes.mean() for row in replays]))
        calls = float(np.mean([row.calls for row in replays]))
        key = (accuracy, -calls, float(threshold))
        if best is None or key > best:
            best = key
            selected = float(threshold)
    assert best is not None
    return selected, float(best[0])


def train_value_models(
    partitions: dict[str, list[JointExample]],
    output: Path,
    *,
    seed: int,
    resume: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    joint_candidates = [
        {"hidden_dim": 16, "max_features": 4000, "svd_components": 24, "learning_rate": 0.01, "epochs": 300, "batch_size": 64, "lambda_cost": 0.10, "quantile_weight": 0.25, "cost_quantile": 0.9, "huber_delta": 1.0, "l2": 1e-4, "class_balanced": True, "domain_balanced": False},
        {"hidden_dim": 32, "max_features": 6000, "svd_components": 32, "learning_rate": 0.005, "epochs": 400, "batch_size": 64, "lambda_cost": 0.25, "quantile_weight": 0.25, "cost_quantile": 0.9, "huber_delta": 1.0, "l2": 5e-4, "class_balanced": True, "domain_balanced": False},
        {"hidden_dim": 48, "max_features": 8000, "svd_components": 48, "learning_rate": 0.003, "epochs": 450, "batch_size": 64, "lambda_cost": 0.50, "quantile_weight": 0.25, "cost_quantile": 0.9, "huber_delta": 1.0, "l2": 1e-3, "class_balanced": True, "domain_balanced": True},
    ]
    separate_candidates = [
        {"C": 0.1, "max_features": 4000, "svd_components": 24, "huber_epsilon": 1.35, "cost_alpha": 1e-4, "cost_quantile": 0.9, "quantile_estimators": 100, "quantile_depth": 2, "quantile_learning_rate": 0.05, "max_iter": 2000, "class_balanced": True, "domain_balanced": False},
        {"C": 1.0, "max_features": 6000, "svd_components": 32, "huber_epsilon": 1.35, "cost_alpha": 1e-4, "cost_quantile": 0.9, "quantile_estimators": 140, "quantile_depth": 2, "quantile_learning_rate": 0.05, "max_iter": 2000, "class_balanced": True, "domain_balanced": False},
        {"C": 1.0, "max_features": 8000, "svd_components": 48, "huber_epsilon": 1.5, "cost_alpha": 1e-3, "cost_quantile": 0.9, "quantile_estimators": 160, "quantile_depth": 2, "quantile_learning_rate": 0.04, "max_iter": 2000, "class_balanced": True, "domain_balanced": True},
    ]
    registry: dict[str, dict[str, Any]] = {}
    selection_rows: list[dict[str, Any]] = []
    total = len(MODE_DATASETS) * 3
    progress = tqdm(total=total, desc="Value/cost model families", unit="family", dynamic_ncols=True)
    for mode, datasets in MODE_DATASETS.items():
        train = subset_examples(partitions["train"], datasets)
        validation = subset_examples(partitions["validation"], datasets)
        if not train or not validation:
            continue
        mode_dir = output / "models" / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        registry[mode] = {}
        for kind, candidates in (("joint", joint_candidates), ("separate", separate_candidates)):
            model_path = mode_dir / f"{kind}.joblib"
            meta_path = mode_dir / f"{kind}.json"
            if resume and model_path.exists() and meta_path.exists():
                model = joblib.load(model_path)
                meta = json.loads(meta_path.read_text())
                registry[mode][kind] = {"model": model, **meta}
                selection_rows.extend(meta.get("candidate_rows", []))
                progress.update()
                continue
            rows: list[dict[str, Any]] = []
            best: tuple[float, int] | None = None
            chosen: tuple[Any, dict[str, Any], float, ValueCostPrediction] | None = None
            iterator = tqdm(enumerate(candidates), total=len(candidates), desc=f"{mode}/{kind}", unit="candidate", leave=False, dynamic_ncols=True)
            for index, params in iterator:
                cls = JointValueCostModel if kind == "joint" else SeparateValueCostModel
                model = cls(params, seed + index * 1009).fit(train, progress_desc=f"{mode}/{kind}/{index + 1}")
                model.temperature = _temperature(model, validation)
                prediction = model.predict(validation)
                score = _prediction_score(validation, prediction)
                metrics = prediction_metrics(validation, prediction)
                row = {"stage": "value_cost", "train_mode": mode, "model": kind, "candidate": index, "selection_score": score, "temperature": model.temperature, "params_json": json.dumps(params, sort_keys=True), **metrics}
                rows.append(row)
                iterator.set_postfix(score=f"{score:.3f}")
                key = (score, -index)
                if best is None or key > best:
                    best = key
                    chosen = (model, params, score, prediction)
            assert chosen is not None
            model, params, score, prediction = chosen
            threshold, validation_accuracy = tune_threshold(
                validation, prediction.expected_gain, prediction.expected_total_tokens,
                train, thresholds=(-0.05, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1),
                seed=seed, score_per_1k=True,
            )
            joblib.dump(model, model_path, compress=3)
            meta = {"params": params, "selection_score": score, "threshold": threshold, "validation_policy_accuracy": validation_accuracy, "candidate_rows": rows}
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            registry[mode][kind] = {"model": model, **meta}
            selection_rows.extend(rows)
            progress.update()

        # Legacy Compact Boosting: the original two outcome heads plus a conservative
        # checkpoint/domain median cost estimator for online budget admission.
        kind = "legacy_compact_boosting"
        model_path = mode_dir / f"{kind}.joblib"
        meta_path = mode_dir / f"{kind}.json"
        if resume and model_path.exists() and meta_path.exists():
            model = joblib.load(model_path)
            meta = json.loads(meta_path.read_text())
            registry[mode][kind] = {"model": model, **meta}
            selection_rows.extend(meta.get("candidate_rows", []))
            progress.update()
            continue
        train_pair = as_pair_examples(train)
        validation_pair = as_pair_examples(validation)
        rows = []
        best = None
        chosen_legacy = None
        for index, params in tqdm(enumerate(MODEL_GRIDS["semantic_boosting"]), total=len(MODEL_GRIDS["semantic_boosting"]), desc=f"{mode}/legacy", leave=False, dynamic_ncols=True):
            model = TwoHeadPredictor("semantic_boosting", params, seed + index * 1013).fit(train_pair)
            q_skip, q_verify = model.predict(validation_pair)
            delta = q_verify - q_skip
            pred_cost = _legacy_costs(train, validation)
            threshold, validation_accuracy = tune_threshold(
                validation, delta, pred_cost, train, thresholds=THRESHOLD_GRID,
                seed=seed, score_per_1k=False,
            )
            row = {"stage": "value_cost", "train_mode": mode, "model": kind, "candidate": index, "selection_score": validation_accuracy, "temperature": "", "params_json": json.dumps(params, sort_keys=True), "threshold": threshold}
            rows.append(row)
            key = (validation_accuracy, -index)
            if best is None or key > best:
                best = key
                chosen_legacy = (model, params, threshold, validation_accuracy)
        assert chosen_legacy is not None
        model, params, threshold, validation_accuracy = chosen_legacy
        joblib.dump(model, model_path, compress=3)
        meta = {"params": params, "selection_score": validation_accuracy, "threshold": threshold, "validation_policy_accuracy": validation_accuracy, "candidate_rows": rows}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        registry[mode][kind] = {"model": model, **meta}
        selection_rows.extend(rows)
        progress.update()
    progress.close()
    return registry, selection_rows


def train_c3_models(
    partitions: dict[str, list[JointExample]],
    registry: dict[str, dict[str, Any]],
    output: Path,
    *,
    seed: int,
    resume: bool,
) -> list[dict[str, Any]]:
    candidates = [
        {"n_estimators": 80, "learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 12, "subsample": 0.85, "cost_penalty_scale": 0.03},
        {"n_estimators": 120, "learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 10, "subsample": 0.90, "cost_penalty_scale": 0.07},
        {"n_estimators": 160, "learning_rate": 0.02, "max_depth": 3, "min_samples_leaf": 10, "subsample": 0.90, "cost_penalty_scale": 0.10},
    ]
    selection_rows: list[dict[str, Any]] = []
    progress = tqdm(total=len(registry) * 2, desc="C3 controllers", unit="controller", dynamic_ncols=True)
    for mode, families in registry.items():
        datasets = MODE_DATASETS[mode]
        train = subset_examples(partitions["train"], datasets)
        validation = subset_examples(partitions["validation"], datasets)
        joint = families["joint"]["model"]
        train_obs = extract_observations(train, joint)
        val_obs = extract_observations(validation, joint)
        costs = np.asarray([row.teacher_total_tokens for row in train], dtype=float)
        levels = sorted({0.0, *[float(np.quantile(costs, q)) for q in (0.25, 0.5, 0.75, 1.0)]})
        fractions = [0.25, 0.5, 0.75, 1.0]
        mode_dir = output / "models" / mode
        for variant, continuation in (("c3_myopic", False), ("c3_continuation", True)):
            model_path = mode_dir / f"{variant}.joblib"
            meta_path = mode_dir / f"{variant}.json"
            if resume and model_path.exists() and meta_path.exists():
                families[variant] = {"model": joblib.load(model_path), **json.loads(meta_path.read_text())}
                selection_rows.extend(families[variant].get("candidate_rows", []))
                progress.update()
                continue
            best = None
            chosen = None
            rows = []
            for index, params in tqdm(enumerate(candidates), total=len(candidates), desc=f"{mode}/{variant}", unit="candidate", leave=False, dynamic_ncols=True):
                controller = FittedSequentialController(params, seed=seed + index * 2017, use_continuation=continuation).fit(train_obs, budget_levels=levels, remaining_fractions=fractions)
                accuracies = []
                token_fractions = []
                task_count = len(_group_indices(validation))
                for fraction, budget in _budget_values(train, task_count):
                    replay, _ = replay_controller(val_obs, controller, policy_name=variant, token_budget=budget, seed=seed, admission_mode="expected_cost")
                    accuracies.append(float(replay.outcomes.mean()))
                    token_fractions.append(float(replay.actual_total_tokens) / max(budget, 1))
                score = float(np.mean(accuracies) - 0.002 * np.mean(token_fractions))
                rows.append({"stage": "c3", "train_mode": mode, "model": variant, "candidate": index, "selection_score": score, "params_json": json.dumps(params, sort_keys=True)})
                key = (score, -index)
                if best is None or key > best:
                    best = key
                    chosen = (controller, params, score)
            assert chosen is not None
            controller, params, score = chosen
            joblib.dump(controller, model_path, compress=3)
            meta = {"params": params, "selection_score": score, "candidate_rows": rows}
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            families[variant] = {"model": controller, **meta}
            selection_rows.extend(rows)
            progress.update()
    progress.close()
    return selection_rows


def _bootstrap_summary(replay: ReplayResult, *, seed: int, repetitions: int) -> dict[str, Any]:
    gain = replay.outcomes - replay.baseline
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(gain), size=(repetitions, len(gain)))
    accuracy_samples = replay.outcomes[indices].mean(axis=1)
    gain_samples = gain[indices].mean(axis=1) * 100.0
    accuracy_ci = np.percentile(accuracy_samples, [2.5, 97.5])
    gain_ci = np.percentile(gain_samples, [2.5, 97.5])
    return {
        "protocol": replay.protocol, "train_mode": replay.train_mode, "scope": replay.scope,
        "policy": replay.policy, "budget_fraction": replay.budget_fraction,
        "token_budget": replay.token_budget, "tasks": len(replay.task_keys),
        "accuracy": float(replay.outcomes.mean()),
        "accuracy_ci95_low": float(accuracy_ci[0]), "accuracy_ci95_high": float(accuracy_ci[1]),
        "gain_vs_never_pp": float(gain.mean() * 100.0),
        "gain_ci95_low_pp": float(gain_ci[0]), "gain_ci95_high_pp": float(gain_ci[1]),
        "teacher_calls": replay.calls, "actual_teacher_tokens": replay.actual_tokens,
        "predicted_teacher_tokens": replay.predicted_tokens,
        "budget_overshoot_tokens": max(0, replay.actual_tokens - replay.token_budget),
        "cp1_calls": int(np.sum(replay.checkpoints == 1)),
        "cp2_calls": int(np.sum(replay.checkpoints == 2)),
        "cp3_calls": int(np.sum(replay.checkpoints == 3)),
        "helpful_selected": int(np.sum(gain > 0)), "harmful_selected": int(np.sum(gain < 0)),
    }


def evaluate_protocol_a(
    partitions: dict[str, list[JointExample]],
    registry: dict[str, dict[str, Any]],
    *,
    seed: int,
    bootstrap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for mode, families in tqdm(registry.items(), desc="Protocol A evaluation", unit="train-mode", dynamic_ncols=True):
        train = subset_examples(partitions["train"], MODE_DATASETS[mode])
        scopes = DATASETS if mode == "combined" else (mode,)
        for scope in (*scopes, "combined") if mode == "combined" else scopes:
            test = partitions["test"] if scope == "combined" else subset_examples(partitions["test"], {scope})
            if not test:
                continue
            task_count = len(_group_indices(test))
            budgets = _budget_values(train, task_count)
            for family in ("joint", "separate"):
                entry = families[family]
                prediction = entry["model"].predict(test)
                metrics = prediction_metrics(test, prediction)
                prediction_rows.append({"train_mode": mode, "scope": scope, "model": family, **metrics})
                for fraction, budget in budgets:
                    replay = replay_single(
                        test, prediction.expected_gain, prediction.expected_total_tokens,
                        threshold=float(entry["threshold"]), token_budget=budget, seed=seed,
                        policy=f"{family.title()} + Threshold", train_mode=mode, scope=scope,
                        budget_fraction=fraction, score_per_1k=True,
                    )
                    rows.append(_bootstrap_summary(replay, seed=seed, repetitions=bootstrap))
            legacy = families["legacy_compact_boosting"]
            q_skip, q_verify = legacy["model"].predict(as_pair_examples(test))
            legacy_cost = _legacy_costs(train, test)
            for fraction, budget in budgets:
                replay = replay_single(
                    test, q_verify - q_skip, legacy_cost, threshold=float(legacy["threshold"]),
                    token_budget=budget, seed=seed, policy="Legacy Compact Boosting",
                    train_mode=mode, scope=scope, budget_fraction=fraction,
                )
                rows.append(_bootstrap_summary(replay, seed=seed, repetitions=bootstrap))
            joint = families["joint"]["model"]
            observations = extract_observations(test, joint)
            for variant, label in (("c3_myopic", "C3 Myopic"), ("c3_continuation", "C3 Continuation")):
                controller = families[variant]["model"]
                for fraction, budget in budgets:
                    replay_raw, _ = replay_controller(observations, controller, policy_name=label, token_budget=budget, seed=seed, admission_mode="expected_cost")
                    replay = ReplayResult(
                        label, "A-single", mode, scope, fraction, budget,
                        replay_raw.task_ids, replay_raw.outcomes,
                        np.asarray([next(row.y_skip for row in test if row.group_key == key) for key in replay_raw.task_ids], dtype=float),
                        replay_raw.selected_checkpoints, int(np.sum(replay_raw.selected_checkpoints > 0)),
                        replay_raw.actual_total_tokens, replay_raw.predicted_total_tokens,
                    )
                    rows.append(_bootstrap_summary(replay, seed=seed, repetitions=bootstrap))
            if mode == "combined":
                for policy in ("Never Verify", "Always CP1", "Always CP2", "Always CP3", "Random Checkpoint", "Offline Oracle"):
                    for fraction, budget in budgets:
                        replay = replay_single_baseline(test, policy=policy, token_budget=budget, seed=seed, scope=scope, budget_fraction=fraction)
                        rows.append(_bootstrap_summary(replay, seed=seed, repetitions=bootstrap))
    return rows, prediction_rows


def _sequence_replay(
    groups: dict[str, TaskTrajectories],
    sequence: str,
    *,
    budget: int,
    fraction: float,
    scope: str,
    seed: int,
) -> ReplayResult:
    remaining = budget
    outcomes = []
    baseline = []
    checkpoints = []
    tokens = calls = 0
    keys = _stable_order(groups, seed)
    for key in keys:
        base = _passed(groups[key].rows["SSS"])
        row = groups[key].rows[sequence]
        cost = int(row.get("teacher_total_tokens", 0) or 0)
        call_count = int(row.get("teacher_calls", sequence.count("V")) or 0)
        if cost <= remaining:
            outcomes.append(float(_passed(row)))
            remaining -= cost
            tokens += cost
            calls += call_count
            checkpoints.append(sum((index + 1) for index, action in enumerate(sequence) if action == "V"))
        else:
            outcomes.append(float(base))
            checkpoints.append(0)
        baseline.append(float(base))
    return ReplayResult(
        f"Fixed {sequence}", "B-multi", "none", scope, fraction, budget,
        tuple(keys), np.asarray(outcomes), np.asarray(baseline), np.asarray(checkpoints),
        calls, tokens, float(tokens),
    )


def replay_multi_controller(
    groups: dict[str, TaskTrajectories],
    controller: MultiHistoryController,
    *,
    budget: int,
    fraction: float,
    scope: str,
    seed: int,
    label: str,
) -> ReplayResult:
    keys = _stable_order(groups, seed)
    predicted_remaining = float(budget)
    outcomes: list[float] = []
    baseline: list[float] = []
    checkpoint_masks: list[int] = []
    actual_tokens = 0
    predicted_tokens = 0.0
    calls = 0
    for order, key in enumerate(keys):
        group = groups[key]
        prefix = ""
        mask = 0
        for checkpoint in (1, 2, 3):
            tasks_remaining = len(keys) - order
            allowance = max(predicted_remaining, 0.0) / max(tasks_remaining, 1)
            q_skip, q_verify, predicted_cost = controller.predict(group, prefix, allowance)
            verify = q_verify > q_skip and predicted_cost <= max(predicted_remaining, 0.0)
            if verify:
                actual_cost = controller._verify_cost(group, prefix)
                prefix += "V"
                predicted_remaining -= predicted_cost
                predicted_tokens += predicted_cost
                actual_tokens += actual_cost
                calls += 1
                mask |= 1 << (checkpoint - 1)
            else:
                prefix += "S"
        outcomes.append(float(_passed(group.rows[prefix])))
        baseline.append(float(_passed(group.rows["SSS"])))
        checkpoint_masks.append(mask)
    return ReplayResult(
        label, "B-multi", "combined", scope, fraction, budget, tuple(keys),
        np.asarray(outcomes), np.asarray(baseline), np.asarray(checkpoint_masks),
        calls, actual_tokens, predicted_tokens,
    )


def train_multi_controllers(
    frozen: dict[str, tuple[dict[str, FrozenTask], list[dict[str, Any]]]],
    output: Path,
    *,
    seed: int,
    resume: bool,
) -> tuple[dict[str, MultiHistoryController], list[dict[str, Any]]]:
    train_tasks, train_rows = frozen["train"]
    val_tasks, val_rows = frozen["validation"]
    train_groups = task_trajectories(train_tasks, train_rows, level=2)
    val_groups = task_trajectories(val_tasks, val_rows, level=2)
    train_costs = np.asarray(
        [int(group.rows["VVV"].get("teacher_total_tokens", 0) or 0) for group in train_groups.values()],
        dtype=float,
    )
    levels = sorted({0.0, *[float(np.quantile(train_costs, q)) for q in (0.25, 0.5, 0.75, 1.0)]})
    candidates = [
        {"n_estimators": 80, "learning_rate": 0.04, "max_depth": 2, "min_samples_leaf": 12, "subsample": 0.85, "max_features": 5000, "svd_components": 24},
        {"n_estimators": 120, "learning_rate": 0.03, "max_depth": 2, "min_samples_leaf": 10, "subsample": 0.90, "max_features": 6000, "svd_components": 32},
        {"n_estimators": 160, "learning_rate": 0.02, "max_depth": 3, "min_samples_leaf": 10, "subsample": 0.90, "max_features": 8000, "svd_components": 48},
    ]
    model_dir = output / "models" / "combined"
    model_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, MultiHistoryController] = {}
    selection_rows: list[dict[str, Any]] = []
    median_vvv = float(np.median(train_costs))
    for variant, continuation in (("multi_myopic", False), ("multi_continuation", True)):
        model_path = model_dir / f"{variant}.joblib"
        meta_path = model_dir / f"{variant}.json"
        if resume and model_path.exists() and meta_path.exists():
            models[variant] = joblib.load(model_path)
            selection_rows.extend(json.loads(meta_path.read_text()).get("candidate_rows", []))
            tqdm.write(f"[{variant}] resume: existing model retained")
            continue
        best: tuple[float, int] | None = None
        chosen: tuple[MultiHistoryController, dict[str, Any], float] | None = None
        candidate_rows: list[dict[str, Any]] = []
        for index, params in tqdm(
            enumerate(candidates), total=len(candidates), desc=variant,
            unit="candidate", dynamic_ncols=True,
        ):
            controller = MultiHistoryController(
                params, seed=seed + index * 3011, continuation=continuation
            ).fit(train_groups, budget_levels=levels)
            accuracies: list[float] = []
            token_rates: list[float] = []
            for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                budget = int(round(fraction * median_vvv * len(val_groups)))
                replay = replay_multi_controller(
                    val_groups, controller, budget=budget, fraction=fraction,
                    scope="combined", seed=seed, label=variant,
                )
                accuracies.append(float(replay.outcomes.mean()))
                token_rates.append(float(replay.actual_tokens) / max(budget, 1))
            score = float(np.mean(accuracies) - 0.002 * np.mean(token_rates))
            row = {
                "stage": "multi_fitted_q", "train_mode": "combined",
                "model": variant, "candidate": index, "selection_score": score,
                "params_json": json.dumps(params, sort_keys=True),
            }
            candidate_rows.append(row)
            key = (score, -index)
            if best is None or key > best:
                best = key
                chosen = (controller, params, score)
        assert chosen is not None
        controller, params, score = chosen
        joblib.dump(controller, model_path, compress=3)
        meta = {"params": params, "selection_score": score, "candidate_rows": candidate_rows}
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        models[variant] = controller
        selection_rows.extend(candidate_rows)
    return models, selection_rows


def evaluate_protocol_b(
    frozen: dict[str, tuple[dict[str, FrozenTask], list[dict[str, Any]]]],
    controllers: dict[str, MultiHistoryController],
    *,
    seed: int,
    bootstrap: int,
) -> list[dict[str, Any]]:
    train_tasks, train_rows = frozen["train"]
    train_groups = task_trajectories(train_tasks, train_rows, level=2)
    median_vvv = float(np.median([int(group.rows["VVV"].get("teacher_total_tokens", 0) or 0) for group in train_groups.values()]))
    rows: list[dict[str, Any]] = []
    test_tasks, test_rows = frozen["test"]
    for scope in (*DATASETS, "combined"):
        datasets = None if scope == "combined" else {scope}
        groups = task_trajectories(test_tasks, test_rows, level=2, datasets=datasets)
        if not groups:
            continue
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            budget = int(round(fraction * median_vvv * len(groups)))
            for sequence in ALL_SEQUENCES:
                replay = _sequence_replay(groups, sequence, budget=budget, fraction=fraction, scope=scope, seed=seed)
                rows.append(_bootstrap_summary(replay, seed=seed, repetitions=bootstrap))
            for variant, label in (
                ("multi_myopic", "L2 Fitted-Q Myopic"),
                ("multi_continuation", "L2 Fitted-Q Continuation"),
            ):
                replay = replay_multi_controller(
                    groups, controllers[variant], budget=budget, fraction=fraction,
                    scope=scope, seed=seed, label=label,
                )
                rows.append(_bootstrap_summary(replay, seed=seed, repetitions=bootstrap))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_figures(policy_rows: list[dict[str, Any]], output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    protocol_a = [row for row in policy_rows if row["protocol"] == "A-single" and row["scope"] == "combined" and row["train_mode"] in {"combined", "none"}]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    selected_policies = ("Never Verify", "Always CP2", "Always CP3", "Legacy Compact Boosting", "Joint + Threshold", "Separate + Threshold", "C3 Myopic", "C3 Continuation", "Offline Oracle")
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_policies)))
    for color, policy in zip(colors, selected_policies):
        points = sorted([row for row in protocol_a if row["policy"] == policy], key=lambda row: float(row["budget_fraction"]))
        if not points:
            continue
        axes[0].plot([float(row["actual_teacher_tokens"]) / 1000 for row in points], [100 * float(row["accuracy"]) for row in points], marker="o", label=policy, color=color)
        axes[1].plot([100 * float(row["budget_fraction"]) for row in points], [100 * float(row["accuracy"]) for row in points], marker="o", label=policy, color=color)
    axes[0].set(title="Protocol A: accuracy–cost frontier", xlabel="Actual Teacher tokens (thousands)", ylabel="Final accuracy (%)")
    axes[1].set(title="Protocol A: budget-conditioned accuracy", xlabel="Budget level (% of reference)", ylabel="Final accuracy (%)")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("CBV-Bench v0.1 frozen-test budget evaluation", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figures / "protocol_a_accuracy_cost.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "protocol_a_accuracy_cost.pdf", bbox_inches="tight")
    plt.close(fig)

    protocol_b = [row for row in policy_rows if row["protocol"] == "B-multi" and row["scope"] == "combined" and float(row["budget_fraction"]) == 1.0]
    if protocol_b:
        protocol_b.sort(key=lambda row: row["policy"])
        fig, ax = plt.subplots(figsize=(12, 6))
        labels = [str(row["policy"]).replace("Fixed ", "") for row in protocol_b]
        values = [100 * float(row["accuracy"]) for row in protocol_b]
        bars = ax.bar(labels, values, color=["#94A3B8" if label == "SSS" else "#7C3AED" for label in labels])
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set(title="Protocol B fixed histories at full common budget", xlabel="Verification sequence", ylabel="Final accuracy (%)")
        fig.tight_layout()
        fig.savefig(figures / "protocol_b_sequences.png", dpi=220, bbox_inches="tight")
        fig.savefig(figures / "protocol_b_sequences.pdf", bbox_inches="tight")
        plt.close(fig)


def write_report(
    output: Path,
    policy_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    split_counts: list[dict[str, Any]],
) -> Path:
    report = output / "report" / "cbv_v01_all_experiments.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    a_full = [row for row in policy_rows if row["protocol"] == "A-single" and row["scope"] == "combined" and float(row["budget_fraction"]) == 1.0 and row["train_mode"] in {"combined", "none"}]
    a_full.sort(key=lambda row: (-float(row["accuracy"]), int(row["actual_teacher_tokens"])))
    b_full = [row for row in policy_rows if row["protocol"] == "B-multi" and row["scope"] == "combined" and float(row["budget_fraction"]) == 1.0]
    b_full.sort(key=lambda row: (-float(row["accuracy"]), int(row["actual_teacher_tokens"])))
    lines = [
        "# CBV-Bench v0.1 — Frozen Experimental Report", "",
        "> Confirmatory offline evaluation. Model selection uses train/validation only; test is evaluated once by this pipeline.", "",
        "## Data protocol", "",
        "| Split | L1 tasks | L1 pairs | L2 tasks |", "|---|---:|---:|---:|",
    ]
    for row in split_counts:
        lines.append(f"| {row['split']} | {row['l1_tasks']} | {row['l1_pairs']} | {row['l2_tasks']} |")
    lines += [
        "", "Protocol A permits at most one Teacher verification per task and compares learned/static policies at common token budgets. Protocol B evaluates the eight observed multi-verification histories on the matched Level-2 subset.", "",
        "Local-risk scores are not present in CBV-Bench v0.1 frozen records, so no pseudo-risk feature is fabricated; `local_risk_available=0` is recorded explicitly.", "",
        "## Protocol A — full-budget frozen test", "",
        "| Policy | Train | Accuracy (95% CI) | Gain vs Never (pp) | Calls | Actual tokens | Helpful / Harmful |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in a_full:
        lines.append(
            f"| {row['policy']} | {row['train_mode']} | {100*float(row['accuracy']):.2f}% [{100*float(row['accuracy_ci95_low']):.2f}, {100*float(row['accuracy_ci95_high']):.2f}] | {float(row['gain_vs_never_pp']):+.2f} | {row['teacher_calls']} | {int(row['actual_teacher_tokens']):,} | {row['helpful_selected']} / {row['harmful_selected']} |"
        )
    lines += [
        "", "![Protocol A accuracy–cost](../figures/protocol_a_accuracy_cost.png)", "",
        "## Protocol B — fixed histories on matched L2 test", "",
        "| Sequence | Accuracy (95% CI) | Gain vs SSS (pp) | Calls | Actual tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in b_full:
        lines.append(
            f"| {str(row['policy']).replace('Fixed ', '')} | {100*float(row['accuracy']):.2f}% [{100*float(row['accuracy_ci95_low']):.2f}, {100*float(row['accuracy_ci95_high']):.2f}] | {float(row['gain_vs_never_pp']):+.2f} | {row['teacher_calls']} | {int(row['actual_teacher_tokens']):,} |"
        )
    lines += [
        "", "![Protocol B sequences](../figures/protocol_b_sequences.png)", "",
        "## Interpretation guardrails", "",
        "- `Offline Oracle` and fixed-history hard accounting are diagnostic upper bounds, not deployable policies.",
        "- Learned admission uses predicted cost; actual token overshoot is reported separately.",
        "- Level-2 is a subset of Level-1 and is never added as independent tasks.",
        "- Confidence intervals are task-clustered bootstrap intervals.",
        "- This pipeline performs no LLM inference and does not retrain Student or Teacher.", "",
        "## Artifacts", "",
        "- `tables/policy_results.csv`: all budgets, scopes and policies.",
        "- `tables/prediction_metrics.csv`: value/cost predictive metrics.",
        "- `tables/model_selection.csv`: validation-only candidate selection.",
        "- `models/`: fitted model files and hyperparameter manifests.",
        "- `figures/`: PNG and PDF plots.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_all(
    frozen_root: Path,
    output: Path,
    *,
    seed: int = 20261029,
    bootstrap: int = 5000,
    resume: bool = True,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    frozen = {split: load_frozen_partition(frozen_root, split) for split in SPLITS}
    partitions: dict[str, list[JointExample]] = {}
    split_counts: list[dict[str, Any]] = []
    for split in SPLITS:
        tasks, rows = frozen[split]
        l1_groups = task_trajectories(tasks, rows, level=1)
        l2_groups = task_trajectories(tasks, rows, level=2)
        partitions[split] = build_level1_examples(l1_groups)
        split_counts.append({"split": split, "l1_tasks": len(l1_groups), "l1_pairs": len(partitions[split]), "l2_tasks": len(l2_groups)})
    train_keys = {row.group_key for row in partitions["train"]}
    val_keys = {row.group_key for row in partitions["validation"]}
    test_keys = {row.group_key for row in partitions["test"]}
    if train_keys & val_keys or train_keys & test_keys or val_keys & test_keys:
        raise AssertionError("Frozen task leakage detected")
    manifest = {
        "schema_version": "cbv-v01-experiment-v1", "seed": seed,
        "frozen_root": str(frozen_root), "split_counts": split_counts,
        "train_validation_test_disjoint": True,
        "test_use": "one-shot confirmatory evaluation",
        "student_teacher_inference": False,
        "budget_protocols": {"A": "max one verification/task", "B": "fixed multi-history replay"},
    }
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    registry, value_selection = train_value_models(partitions, output, seed=seed, resume=resume)
    c3_selection = train_c3_models(partitions, registry, output, seed=seed, resume=resume)
    multi_controllers, multi_selection = train_multi_controllers(
        frozen, output, seed=seed, resume=resume
    )
    policy_a, prediction_rows = evaluate_protocol_a(partitions, registry, seed=seed, bootstrap=bootstrap)
    policy_b = evaluate_protocol_b(
        frozen, multi_controllers, seed=seed, bootstrap=bootstrap
    )
    policy_rows = [*policy_a, *policy_b]
    write_csv(output / "tables" / "policy_results.csv", policy_rows)
    write_csv(output / "tables" / "prediction_metrics.csv", prediction_rows)
    write_csv(
        output / "tables" / "model_selection.csv",
        [*value_selection, *c3_selection, *multi_selection],
    )
    make_figures(policy_rows, output)
    report = write_report(output, policy_rows, prediction_rows, split_counts)
    summary = {
        "report": str(report), "policy_rows": len(policy_rows),
        "prediction_rows": len(prediction_rows), "split_counts": split_counts,
        "best_protocol_a_full_budget": max(
            (row for row in policy_a if row["scope"] == "combined" and float(row["budget_fraction"]) == 1.0),
            key=lambda row: (float(row["accuracy"]), -int(row["actual_teacher_tokens"])),
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return report

