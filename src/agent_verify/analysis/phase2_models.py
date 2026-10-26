"""Small two-head predictors and leakage-safe nested cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from agent_verify.analysis.phase2_data import PairExample, assert_group_isolation, feature_matrix

MODEL_GRIDS: dict[str, list[dict[str, Any]]] = {
    "metadata_logistic": [
        {"C": 0.1, "class_weight": None},
        {"C": 1.0, "class_weight": None},
        {"C": 10.0, "class_weight": None},
        {"C": 1.0, "class_weight": "balanced"},
    ],
    "semantic_logistic": [
        {"C": 0.1, "class_weight": None, "max_features": 4_000},
        {"C": 1.0, "class_weight": None, "max_features": 4_000},
        {"C": 10.0, "class_weight": None, "max_features": 4_000},
        {"C": 1.0, "class_weight": "balanced", "max_features": 4_000},
    ],
    "semantic_boosting": [
        {"max_leaf_nodes": 7, "l2_regularization": 1.0, "svd_components": 24},
        {"max_leaf_nodes": 15, "l2_regularization": 1.0, "svd_components": 24},
        {"max_leaf_nodes": 7, "l2_regularization": 5.0, "svd_components": 24},
        {"max_leaf_nodes": 15, "l2_regularization": 5.0, "svd_components": 24},
    ],
}

THRESHOLD_GRID = np.asarray(
    [-0.10, -0.05, 0.00, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30],
    dtype=float,
)


@dataclass(frozen=True)
class SelectionResult:
    params: dict[str, Any]
    threshold: float
    inner_accuracy: float
    inner_calls_per_task: float
    inner_brier: float


class FeatureEncoder:
    def __init__(self, model_name: str, params: dict[str, Any], seed: int) -> None:
        self.model_name = model_name
        self.params = dict(params)
        self.seed = seed
        self.vectorizer: TfidfVectorizer | None = None
        self.scaler: StandardScaler | None = None
        self.svd: TruncatedSVD | None = None

    def fit_transform(self, examples: list[PairExample]):
        metadata = feature_matrix(examples)
        if self.model_name == "metadata_logistic":
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(metadata)

        max_features = int(self.params.get("max_features", 2_000))
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=max_features,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w.]+\b",
        )
        text_matrix = self.vectorizer.fit_transform([example.text for example in examples])
        if self.model_name == "semantic_logistic":
            self.scaler = StandardScaler()
            scaled_metadata = self.scaler.fit_transform(metadata)
            return sparse.hstack([text_matrix, sparse.csr_matrix(scaled_metadata)], format="csr")

        components = min(
            int(self.params["svd_components"]),
            max(1, text_matrix.shape[0] - 1),
            max(1, text_matrix.shape[1] - 1),
        )
        self.svd = TruncatedSVD(n_components=components, random_state=self.seed)
        dense_text = self.svd.fit_transform(text_matrix)
        self.scaler = StandardScaler()
        combined = np.column_stack([dense_text, metadata])
        return self.scaler.fit_transform(combined)

    def transform(self, examples: list[PairExample]):
        metadata = feature_matrix(examples)
        if self.model_name == "metadata_logistic":
            assert self.scaler is not None
            return self.scaler.transform(metadata)
        assert self.vectorizer is not None
        text_matrix = self.vectorizer.transform([example.text for example in examples])
        if self.model_name == "semantic_logistic":
            assert self.scaler is not None
            scaled_metadata = self.scaler.transform(metadata)
            return sparse.hstack([text_matrix, sparse.csr_matrix(scaled_metadata)], format="csr")
        assert self.svd is not None and self.scaler is not None
        dense_text = self.svd.transform(text_matrix)
        return self.scaler.transform(np.column_stack([dense_text, metadata]))


class TwoHeadPredictor:
    """Shared pre-treatment feature encoder with separate Q_skip and Q_verify heads."""

    def __init__(self, model_name: str, params: dict[str, Any], seed: int) -> None:
        self.model_name = model_name
        self.params = dict(params)
        self.seed = seed
        self.encoder = FeatureEncoder(model_name, params, seed)
        self.skip_head: BaseEstimator | None = None
        self.verify_head: BaseEstimator | None = None

    def _classifier(self, labels: np.ndarray, seed_offset: int) -> BaseEstimator:
        if len(np.unique(labels)) == 1:
            return DummyClassifier(strategy="constant", constant=int(labels[0]))
        if self.model_name in {"metadata_logistic", "semantic_logistic"}:
            return LogisticRegression(
                C=float(self.params["C"]),
                class_weight=self.params.get("class_weight"),
                solver="liblinear",
                max_iter=2_000,
                random_state=self.seed + seed_offset,
            )
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=140,
            max_leaf_nodes=int(self.params["max_leaf_nodes"]),
            l2_regularization=float(self.params["l2_regularization"]),
            early_stopping=False,
            random_state=self.seed + seed_offset,
        )

    def fit(self, examples: list[PairExample]) -> TwoHeadPredictor:
        matrix = self.encoder.fit_transform(examples)
        y_skip = np.asarray([example.y_skip for example in examples], dtype=int)
        y_verify = np.asarray([example.y_verify for example in examples], dtype=int)
        self.skip_head = self._classifier(y_skip, 11)
        self.verify_head = self._classifier(y_verify, 29)
        self.skip_head.fit(matrix, y_skip)
        self.verify_head.fit(matrix, y_verify)
        return self

    @staticmethod
    def _positive_probability(classifier: BaseEstimator, matrix) -> np.ndarray:
        probabilities = classifier.predict_proba(matrix)
        classes = np.asarray(classifier.classes_)
        if 1 not in classes:
            return np.zeros(matrix.shape[0], dtype=float)
        return probabilities[:, int(np.where(classes == 1)[0][0])]

    def predict(self, examples: list[PairExample]) -> tuple[np.ndarray, np.ndarray]:
        if self.skip_head is None or self.verify_head is None:
            raise RuntimeError("Predictor has not been fitted")
        matrix = self.encoder.transform(examples)
        return (
            self._positive_probability(self.skip_head, matrix),
            self._positive_probability(self.verify_head, matrix),
        )


def group_folds(examples: list[PairExample], n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray([example.task_id for example in examples])
    splitter = GroupKFold(n_splits=n_splits)
    folds = []
    for train_indices, test_indices in splitter.split(np.zeros(len(examples)), groups=groups):
        assert_group_isolation(examples, train_indices, test_indices)
        folds.append((train_indices, test_indices))
    return folds


def simulate_sequential_policy(
    examples: list[PairExample],
    delta_predictions: np.ndarray,
    thresholds: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose the first checkpoint above threshold; never inspect later predictions after selection."""

    if len(examples) != len(delta_predictions):
        raise ValueError("Prediction count does not match examples")
    threshold_array = np.full(len(examples), float(thresholds)) if np.isscalar(thresholds) else thresholds
    if len(threshold_array) != len(examples):
        raise ValueError("Threshold count does not match examples")

    task_to_indices: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        task_to_indices.setdefault(example.task_id, []).append(index)
    outcomes: list[float] = []
    selected_indices: list[int] = []
    selected_checkpoints: list[int] = []
    for indices in task_to_indices.values():
        ordered = sorted(indices, key=lambda index: examples[index].checkpoint)
        baseline_values = {examples[index].y_skip for index in ordered}
        if len(baseline_values) != 1:
            raise ValueError("Y_skip differs within a task")
        chosen: int | None = None
        for index in ordered:
            if float(delta_predictions[index]) > float(threshold_array[index]):
                chosen = index
                break
        if chosen is None:
            outcomes.append(float(examples[ordered[0]].y_skip))
            selected_indices.append(-1)
            selected_checkpoints.append(0)
        else:
            outcomes.append(float(examples[chosen].y_verify))
            selected_indices.append(chosen)
            selected_checkpoints.append(examples[chosen].checkpoint)
    return (
        np.asarray(outcomes, dtype=float),
        np.asarray(selected_indices, dtype=int),
        np.asarray(selected_checkpoints, dtype=int),
    )


def _inner_predictions(
    examples: list[PairExample],
    model_name: str,
    params: dict[str, Any],
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    q_skip = np.empty(len(examples), dtype=float)
    q_verify = np.empty(len(examples), dtype=float)
    for fold, (train_indices, validation_indices) in enumerate(group_folds(examples, n_splits)):
        train = [examples[int(index)] for index in train_indices]
        validation = [examples[int(index)] for index in validation_indices]
        predictor = TwoHeadPredictor(model_name, params, seed + fold * 101).fit(train)
        skip_pred, verify_pred = predictor.predict(validation)
        q_skip[validation_indices] = skip_pred
        q_verify[validation_indices] = verify_pred
    return q_skip, q_verify


def select_hyperparameters(
    examples: list[PairExample],
    model_name: str,
    *,
    inner_splits: int,
    seed: int,
) -> SelectionResult:
    y_skip = np.asarray([example.y_skip for example in examples], dtype=float)
    y_verify = np.asarray([example.y_verify for example in examples], dtype=float)
    best_key: tuple[float, float, float, float] | None = None
    best: SelectionResult | None = None
    for candidate_index, params in enumerate(MODEL_GRIDS[model_name]):
        q_skip, q_verify = _inner_predictions(
            examples,
            model_name,
            params,
            inner_splits,
            seed + candidate_index * 1_003,
        )
        brier = float(np.mean(np.concatenate([(q_skip - y_skip) ** 2, (q_verify - y_verify) ** 2])))
        delta_predictions = q_verify - q_skip
        for threshold in THRESHOLD_GRID:
            outcomes, selected, _ = simulate_sequential_policy(
                examples, delta_predictions, float(threshold)
            )
            accuracy = float(outcomes.mean())
            calls = float(np.mean(selected >= 0))
            # Primary: accuracy. Ties: fewer calls, better calibration, more conservative threshold.
            key = (accuracy, -calls, -brier, float(threshold))
            if best_key is None or key > best_key:
                best_key = key
                best = SelectionResult(
                    params=dict(params),
                    threshold=float(threshold),
                    inner_accuracy=accuracy,
                    inner_calls_per_task=calls,
                    inner_brier=brier,
                )
    assert best is not None
    return best
