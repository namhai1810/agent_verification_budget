"""Lightweight shared and separate value-cost predictors.

The joint model is a small NumPy multi-task network. A shared hidden layer feeds
one three-class value head and one two-output cost head (conditional mean and an
upper quantile in log-token space). It intentionally does not fine-tune or call
any language model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from agent_verify.analysis.joint_value_cost_data import JointExample, VALUE_CLASSES


@dataclass(frozen=True)
class ValueCostPrediction:
    value_probabilities: np.ndarray
    expected_gain: np.ndarray
    expected_total_tokens: np.ndarray
    upper_total_tokens: np.ndarray


class JointFeatureEncoder:
    """Train-fitted TF-IDF/SVD plus structured pre-treatment metadata."""

    def __init__(self, *, max_features: int, svd_components: int, seed: int) -> None:
        self.max_features = max_features
        self.svd_components = svd_components
        self.seed = seed
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.scaler: StandardScaler | None = None

    def fit_transform(self, examples: list[JointExample]) -> np.ndarray:
        if not examples:
            raise ValueError("Cannot fit an encoder on an empty dataset")
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=self.max_features,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w.]+\b",
        )
        sparse_text = self.vectorizer.fit_transform([row.text for row in examples])
        components = min(
            self.svd_components,
            max(1, sparse_text.shape[0] - 1),
            max(1, sparse_text.shape[1] - 1),
        )
        self.svd = TruncatedSVD(n_components=components, random_state=self.seed)
        semantic = self.svd.fit_transform(sparse_text)
        metadata = np.asarray([row.metadata for row in examples], dtype=float)
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(np.column_stack([semantic, metadata]))

    def transform(self, examples: list[JointExample]) -> np.ndarray:
        if self.vectorizer is None or self.svd is None or self.scaler is None:
            raise RuntimeError("Encoder has not been fitted")
        sparse_text = self.vectorizer.transform([row.text for row in examples])
        semantic = self.svd.transform(sparse_text)
        metadata = np.asarray([row.metadata for row in examples], dtype=float)
        return self.scaler.transform(np.column_stack([semantic, metadata]))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -60.0, 60.0))
    return exponent / exponent.sum(axis=1, keepdims=True)


def calibrate_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-9, 1.0)
    return _softmax(np.log(clipped) / float(temperature))


def choose_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    grid: list[float],
    sample_weights: np.ndarray | None = None,
) -> float:
    if not grid:
        raise ValueError("Temperature grid must not be empty")
    weights = np.ones(len(labels), dtype=float) if sample_weights is None else sample_weights
    weights = weights / weights.sum()
    best: tuple[float, float] | None = None
    selected = 1.0
    for temperature in grid:
        calibrated = calibrate_probabilities(probabilities, temperature)
        loss = -float(np.sum(weights * np.log(np.clip(calibrated[np.arange(len(labels)), labels], 1e-9, 1.0))))
        key = (loss, abs(float(temperature) - 1.0))
        if best is None or key < best:
            best = key
            selected = float(temperature)
    return selected


def training_sample_weights(
    examples: list[JointExample],
    *,
    class_balanced: bool,
    domain_balanced: bool,
) -> np.ndarray:
    weights = np.ones(len(examples), dtype=float)
    if class_balanced:
        labels = np.asarray([row.value_label for row in examples], dtype=int)
        counts = np.bincount(labels, minlength=len(VALUE_CLASSES))
        nonzero = counts > 0
        class_weights = np.zeros(len(VALUE_CLASSES), dtype=float)
        class_weights[nonzero] = len(labels) / (nonzero.sum() * counts[nonzero])
        weights *= class_weights[labels]
    if domain_balanced:
        datasets = np.asarray([row.dataset for row in examples])
        unique, counts = np.unique(datasets, return_counts=True)
        domain_weights = {
            dataset: len(datasets) / (len(unique) * count)
            for dataset, count in zip(unique, counts, strict=True)
        }
        weights *= np.asarray([domain_weights[row.dataset] for row in examples])
    return weights / max(float(weights.mean()), 1e-9)


class JointValueCostModel:
    """One shared hidden representation with value and robust cost heads."""

    def __init__(self, params: dict[str, Any], seed: int) -> None:
        self.params = dict(params)
        self.seed = seed
        self.encoder = JointFeatureEncoder(
            max_features=int(params.get("max_features", 4_000)),
            svd_components=int(params.get("svd_components", 32)),
            seed=seed,
        )
        self.temperature = 1.0
        self.cost_log_mean = 0.0
        self.cost_log_scale = 1.0
        self.weights: dict[str, np.ndarray] = {}

    def _initialize(self, input_dim: int) -> None:
        rng = np.random.default_rng(self.seed)
        hidden_dim = int(self.params["hidden_dim"])
        self.weights = {
            "shared_w": rng.normal(0.0, np.sqrt(2.0 / (input_dim + hidden_dim)), (input_dim, hidden_dim)),
            "shared_b": np.zeros(hidden_dim),
            "value_w": rng.normal(0.0, np.sqrt(2.0 / (hidden_dim + 3)), (hidden_dim, 3)),
            "value_b": np.zeros(3),
            "cost_w": rng.normal(0.0, np.sqrt(2.0 / (hidden_dim + 2)), (hidden_dim, 2)),
            "cost_b": np.zeros(2),
        }

    def _forward(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden = np.tanh(matrix @ self.weights["shared_w"] + self.weights["shared_b"])
        probabilities = _softmax(hidden @ self.weights["value_w"] + self.weights["value_b"])
        costs = hidden @ self.weights["cost_w"] + self.weights["cost_b"]
        return hidden, probabilities, costs

    def fit(
        self, examples: list[JointExample], progress_desc: str | None = None
    ) -> JointValueCostModel:
        matrix = self.encoder.fit_transform(examples)
        labels = np.asarray([row.value_label for row in examples], dtype=int)
        log_cost = np.log1p([row.teacher_total_tokens for row in examples]).astype(float)
        self.cost_log_mean = float(log_cost.mean())
        self.cost_log_scale = max(float(log_cost.std()), 1e-6)
        cost_target = (log_cost - self.cost_log_mean) / self.cost_log_scale
        sample_weights = training_sample_weights(
            examples,
            class_balanced=bool(self.params.get("class_balanced", True)),
            domain_balanced=bool(self.params.get("domain_balanced", False)),
        )
        self._initialize(matrix.shape[1])
        learning_rate = float(self.params.get("learning_rate", 0.01))
        epochs = int(self.params.get("epochs", 300))
        batch_size = min(int(self.params.get("batch_size", 64)), len(examples))
        lambda_cost = float(self.params.get("lambda_cost", 0.5))
        quantile_weight = float(self.params.get("quantile_weight", 0.25))
        quantile = float(self.params.get("cost_quantile", 0.9))
        huber_delta = float(self.params.get("huber_delta", 1.0))
        l2 = float(self.params.get("l2", 1e-4))
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        first = {name: np.zeros_like(value) for name, value in self.weights.items()}
        second = {name: np.zeros_like(value) for name, value in self.weights.items()}
        rng = np.random.default_rng(self.seed + 17)
        step = 0
        eye = np.eye(3)
        epoch_iterator = tqdm(
            range(epochs),
            desc=progress_desc or "joint epochs",
            unit="epoch",
            dynamic_ncols=True,
            leave=False,
            disable=progress_desc is None,
        )
        for _epoch in epoch_iterator:
            order = rng.permutation(len(examples))
            for start in range(0, len(examples), batch_size):
                indices = order[start : start + batch_size]
                batch = matrix[indices]
                targets = labels[indices]
                target_cost = cost_target[indices]
                weights = sample_weights[indices]
                weights = weights / max(float(weights.sum()), 1e-9)
                hidden, probabilities, costs = self._forward(batch)

                value_gradient = (probabilities - eye[targets]) * weights[:, None]
                mean_error = costs[:, 0] - target_cost
                mean_gradient = np.where(
                    np.abs(mean_error) <= huber_delta,
                    mean_error,
                    huber_delta * np.sign(mean_error),
                )
                mean_gradient *= weights * lambda_cost
                quantile_error = target_cost - costs[:, 1]
                quantile_gradient = np.where(
                    quantile_error >= 0.0,
                    -quantile,
                    1.0 - quantile,
                )
                quantile_gradient *= weights * lambda_cost * quantile_weight
                cost_gradient = np.column_stack([mean_gradient, quantile_gradient])

                gradients: dict[str, np.ndarray] = {}
                gradients["value_w"] = hidden.T @ value_gradient + l2 * self.weights["value_w"]
                gradients["value_b"] = value_gradient.sum(axis=0)
                gradients["cost_w"] = hidden.T @ cost_gradient + l2 * self.weights["cost_w"]
                gradients["cost_b"] = cost_gradient.sum(axis=0)
                hidden_gradient = (
                    value_gradient @ self.weights["value_w"].T
                    + cost_gradient @ self.weights["cost_w"].T
                ) * (1.0 - hidden**2)
                gradients["shared_w"] = batch.T @ hidden_gradient + l2 * self.weights["shared_w"]
                gradients["shared_b"] = hidden_gradient.sum(axis=0)

                step += 1
                for name, gradient in gradients.items():
                    first[name] = beta1 * first[name] + (1.0 - beta1) * gradient
                    second[name] = beta2 * second[name] + (1.0 - beta2) * gradient**2
                    first_hat = first[name] / (1.0 - beta1**step)
                    second_hat = second[name] / (1.0 - beta2**step)
                    self.weights[name] -= learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        return self

    def predict(self, examples: list[JointExample]) -> ValueCostPrediction:
        matrix = self.encoder.transform(examples)
        _hidden, probabilities, normalized_costs = self._forward(matrix)
        probabilities = calibrate_probabilities(probabilities, self.temperature)
        mean_log = normalized_costs[:, 0] * self.cost_log_scale + self.cost_log_mean
        upper_log = normalized_costs[:, 1] * self.cost_log_scale + self.cost_log_mean
        expected = np.maximum(np.expm1(mean_log), 0.0)
        upper = np.maximum(np.expm1(upper_log), expected)
        expected_gain = probabilities[:, 2] - probabilities[:, 0]
        return ValueCostPrediction(probabilities, expected_gain, expected, upper)

    def raw_value_probabilities(self, examples: list[JointExample]) -> np.ndarray:
        matrix = self.encoder.transform(examples)
        return self._forward(matrix)[1]


class SeparateValueCostModel:
    """Independent value classifier and robust/quantile cost regressors."""

    def __init__(self, params: dict[str, Any], seed: int) -> None:
        self.params = dict(params)
        self.seed = seed
        self.encoder = JointFeatureEncoder(
            max_features=int(params.get("max_features", 4_000)),
            svd_components=int(params.get("svd_components", 32)),
            seed=seed,
        )
        self.value_model: LogisticRegression | None = None
        self.cost_model: HuberRegressor | None = None
        self.quantile_model: GradientBoostingRegressor | None = None
        self.temperature = 1.0

    def fit(
        self, examples: list[JointExample], progress_desc: str | None = None
    ) -> SeparateValueCostModel:
        matrix = self.encoder.fit_transform(examples)
        labels = np.asarray([row.value_label for row in examples], dtype=int)
        log_cost = np.log1p([row.teacher_total_tokens for row in examples]).astype(float)
        sample_weights = training_sample_weights(
            examples,
            class_balanced=bool(self.params.get("class_balanced", True)),
            domain_balanced=bool(self.params.get("domain_balanced", False)),
        )
        progress = tqdm(
            total=3,
            desc=progress_desc or "separate stages",
            unit="stage",
            dynamic_ncols=True,
            leave=False,
            disable=progress_desc is None,
        )
        self.value_model = LogisticRegression(
            C=float(self.params.get("C", 1.0)),
            solver="lbfgs",
            max_iter=int(self.params.get("max_iter", 2_000)),
            random_state=self.seed,
        )
        self.value_model.fit(matrix, labels, sample_weight=sample_weights)
        progress.update()
        self.cost_model = HuberRegressor(
            epsilon=float(self.params.get("huber_epsilon", 1.35)),
            alpha=float(self.params.get("cost_alpha", 1e-4)),
            max_iter=int(self.params.get("max_iter", 2_000)),
        )
        self.cost_model.fit(matrix, log_cost, sample_weight=sample_weights)
        progress.update()
        self.quantile_model = GradientBoostingRegressor(
            loss="quantile",
            alpha=float(self.params.get("cost_quantile", 0.9)),
            n_estimators=int(self.params.get("quantile_estimators", 100)),
            max_depth=int(self.params.get("quantile_depth", 2)),
            learning_rate=float(self.params.get("quantile_learning_rate", 0.05)),
            random_state=self.seed + 29,
        )
        self.quantile_model.fit(matrix, log_cost, sample_weight=sample_weights)
        progress.update()
        progress.close()
        return self

    def raw_value_probabilities(self, examples: list[JointExample]) -> np.ndarray:
        if self.value_model is None:
            raise RuntimeError("Separate model has not been fitted")
        matrix = self.encoder.transform(examples)
        raw = self.value_model.predict_proba(matrix)
        probabilities = np.zeros((len(examples), len(VALUE_CLASSES)), dtype=float)
        for column, label in enumerate(self.value_model.classes_):
            probabilities[:, int(label)] = raw[:, column]
        return probabilities

    def predict(self, examples: list[JointExample]) -> ValueCostPrediction:
        if self.cost_model is None or self.quantile_model is None:
            raise RuntimeError("Separate model has not been fitted")
        matrix = self.encoder.transform(examples)
        probabilities = calibrate_probabilities(
            self.raw_value_probabilities(examples), self.temperature
        )
        expected = np.maximum(np.expm1(self.cost_model.predict(matrix)), 0.0)
        upper = np.maximum(np.expm1(self.quantile_model.predict(matrix)), expected)
        expected_gain = probabilities[:, 2] - probabilities[:, 0]
        return ValueCostPrediction(probabilities, expected_gain, expected, upper)
