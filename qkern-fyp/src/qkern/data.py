from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    name: str


def load_dataset(name: str, test_size: float = 0.2, random_state: int = 42, scale: bool = True,
                 n_samples: int | None = None) -> Dataset:
    name = name.lower()

    if name in {"breast_cancer", "cancer", "bc"}:
        data = datasets.load_breast_cancer()
        X, y = data.data, data.target
    elif name in {"wine"}:
        data = datasets.load_wine()
        X, y = data.data, data.target
        # Make binary for consistency: class 0 vs others
        y = (y == 0).astype(int)
    elif name in {"sonar"}:  # from UCI via sklearn? Not directly; fallback to make_classification
        X, y = datasets.make_classification(n_samples=520, n_features=60, n_informative=20,
                                            n_redundant=10, n_repeated=0, n_classes=2,
                                            class_sep=1.2, random_state=random_state)
    elif name in {"moons", "two_moons"}:
        X, y = datasets.make_moons(n_samples=n_samples or 500, noise=0.2, random_state=random_state)
    elif name in {"circles"}:
        X, y = datasets.make_circles(n_samples=n_samples or 500, factor=0.5, noise=0.08,
                                     random_state=random_state)
    elif name in {"xor"}:
        rng = np.random.default_rng(random_state)
        n = n_samples or 600
        X = rng.uniform(-1, 1, size=(n, 2))
        y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
        X += rng.normal(0, 0.1, size=X.shape)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if n_samples is not None and X.shape[0] > n_samples:
        X, _, y, _ = train_test_split(X, y, train_size=n_samples, stratify=y, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return Dataset(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, name=name)
