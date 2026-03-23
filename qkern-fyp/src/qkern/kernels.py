from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit import QuantumCircuit


FeatureMapType = Literal["zz", "iqp"]


def _zz_feature_map(x: np.ndarray, num_qubits: int, depth: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    d = depth
    # Data encoding + entangling ZZ layers
    for _ in range(d):
        for i in range(num_qubits):
            qc.rx(x[i % x.shape[0]], i)
        for i in range(num_qubits):
            qc.rz(x[(i + 1) % x.shape[0]], i)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(2.0 * x[(i + 1) % x.shape[0]] * x[i % x.shape[0]], i + 1)
            qc.cx(i, i + 1)
    return qc


def _iqp_feature_map(x: np.ndarray, num_qubits: int, depth: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for _ in range(depth):
        for i in range(num_qubits):
            qc.h(i)
        for i in range(num_qubits):
            qc.rz(x[i % x.shape[0]], i)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(x[i % x.shape[0]] * x[(i + 1) % x.shape[0]], i + 1)
            qc.cx(i, i + 1)
    return qc


@dataclass
class QuantumKernelConfig:
    feature_map: FeatureMapType = "zz"
    num_qubits: int = 4
    depth: int = 2
    shots: Optional[int] = None  # None => exact statevector inner product


def build_feature_map(x: np.ndarray, cfg: QuantumKernelConfig) -> QuantumCircuit:
    if cfg.feature_map == "zz":
        return _zz_feature_map(x, cfg.num_qubits, cfg.depth)
    elif cfg.feature_map == "iqp":
        return _iqp_feature_map(x, cfg.num_qubits, cfg.depth)
    else:
        raise ValueError(f"Unknown feature map: {cfg.feature_map}")


def kernel_value(x: np.ndarray, y: np.ndarray, cfg: QuantumKernelConfig) -> float:
    """Compute k(x, y) = |<0| U(x)^† U(y) |0>|^2 exactly via statevector.

    For now, shots are ignored and exact probability is returned. This is fast and stable
    for small qubit counts (≤8)."""
    num_qubits = cfg.num_qubits
    Ux = build_feature_map(x, cfg)
    Uy = build_feature_map(y, cfg)

    qc = QuantumCircuit(num_qubits)
    qc.compose(Ux.inverse(), inplace=True)
    qc.compose(Uy, inplace=True)

    sv = Statevector.from_instruction(qc)
    amp0 = sv.data[0]
    return float(np.abs(amp0) ** 2)


def embed_state(x: np.ndarray, cfg: QuantumKernelConfig) -> np.ndarray:
    """Return statevector |phi(x)> = U(x)|0> as a complex numpy array."""
    Ux = build_feature_map(x, cfg)
    sv = Statevector.from_instruction(Ux)
    return sv.data  # shape (2**n,)


def states_matrix(X: np.ndarray, cfg: QuantumKernelConfig) -> np.ndarray:
    """Compute statevectors for all samples in X; returns shape (N, D) complex."""
    return np.stack([embed_state(x, cfg) for x in X], axis=0)


def gram_from_states(V: np.ndarray) -> np.ndarray:
    """Given states V (N, D), return Gram K_ij = |<v_i|v_j>|^2."""
    A = V @ V.conj().T  # (N, N) complex inner products
    return np.abs(A) ** 2


def cross_gram_from_states(Va: np.ndarray, Vb: np.ndarray) -> np.ndarray:
    """Cross Gram for test/train: |Va Va^H| elementwise squared magnitude."""
    A = Va @ Vb.conj().T
    return np.abs(A) ** 2


def gram_matrix(X: np.ndarray, cfg: QuantumKernelConfig) -> np.ndarray:
    n = X.shape[0]
    G = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            kij = kernel_value(X[i], X[j], cfg)
            G[i, j] = kij
            G[j, i] = kij
    return G
