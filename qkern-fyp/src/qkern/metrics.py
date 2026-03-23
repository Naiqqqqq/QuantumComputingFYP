from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


def center_kernel(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def kernel_alignment(K: np.ndarray, Y: np.ndarray) -> float:
    # Y should be label kernel: yy^T
    Kc = center_kernel(K)
    Yc = center_kernel(Y)
    num = np.sum(Kc * Yc)
    den = np.sqrt(np.sum(Kc * Kc) * np.sum(Yc * Yc) + 1e-12)
    return float(num / den)


def label_kernel(y: np.ndarray) -> np.ndarray:
    y = y.reshape(-1, 1).astype(float)
    return y @ y.T


def rbf_kernel(X: np.ndarray, gamma: float | None = None) -> np.ndarray:
    return pairwise_kernels(X, metric="rbf", gamma=gamma)
