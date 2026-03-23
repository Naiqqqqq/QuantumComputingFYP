from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qkern.data import load_dataset
from qkern.kernels import QuantumKernelConfig, states_matrix, gram_from_states, cross_gram_from_states
from qkern.metrics import label_kernel, kernel_alignment
from qkern.models import evaluate_svc_precomputed, evaluate_svc_rbf, evaluate_svc_poly, evaluate_svc_linear

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RESULTS_DIR / "sweep_results.csv"


def run_one(dataset: str, feature_map: str, qubits: int, depth: int, seed: int, C: float = 1.0):
    ds = load_dataset(dataset, random_state=seed)
    cfg = QuantumKernelConfig(feature_map=feature_map, num_qubits=qubits, depth=depth, shots=None)

    t0 = time.time()
    Xtr = ds.X_train[:, :cfg.num_qubits]
    Xte = ds.X_test[:, :cfg.num_qubits]
    Vtr = states_matrix(Xtr, cfg)
    Vte = states_matrix(Xte, cfg)
    Gtr = gram_from_states(Vtr)
    Gte = cross_gram_from_states(Vte, Vtr)
    qk_time = time.time() - t0

    Ytr = label_kernel(ds.y_train.astype(float) * 2 - 1)
    align = kernel_alignment(Gtr, Ytr)

    qk_res = evaluate_svc_precomputed(Gtr, ds.y_train, Gte, ds.y_test, C=C)
    rbf_res = evaluate_svc_rbf(ds.X_train, ds.y_train, ds.X_test, ds.y_test, C=C)
    poly_res = evaluate_svc_poly(ds.X_train, ds.y_train, ds.X_test, ds.y_test, C=C)
    lin_res = evaluate_svc_linear(ds.X_train, ds.y_train, ds.X_test, ds.y_test, C=C)

    return {
        "dataset": ds.name,
        "feature_map": feature_map,
        "qubits": qubits,
        "depth": depth,
        "C": C,
        "align": align,
        "qk_acc": qk_res.acc, "qk_f1": qk_res.f1, "qk_auc": qk_res.auc,
        "rbf_acc": rbf_res.acc, "rbf_f1": rbf_res.f1, "rbf_auc": rbf_res.auc,
        "poly_acc": poly_res.acc, "poly_f1": poly_res.f1, "poly_auc": poly_res.auc,
        "lin_acc": lin_res.acc, "lin_f1": lin_res.f1, "lin_auc": lin_res.auc,
        "qk_time_s": qk_time,
        "seed": seed,
    }


def main():
    dataset_list = ["breast_cancer", "wine", "moons", "circles", "xor"]
    feature_maps = ["zz", "iqp"]
    qubits_list = [2, 4, 6, 8]
    depths = [1, 2, 3]
    seeds = [42, 123, 456]

    combos = list(itertools.product(dataset_list, feature_maps, qubits_list, depths, seeds))
    total = len(combos)
    print(f"Total runs: {total}")

    # Delete old results to start fresh
    if OUT_PATH.exists():
        OUT_PATH.unlink()

    rows = []
    for i, (dataset, fmap, q, d, seed) in enumerate(combos, 1):
        tag = f"[{i}/{total}] {dataset} {fmap} q={q} d={d} s={seed}"
        try:
            row = run_one(dataset, fmap, q, d, seed)
            rows.append(row)
            print(f"{tag}  qk_acc={row['qk_acc']:.3f}  rbf_acc={row['rbf_acc']:.3f}")
        except Exception as e:
            print(f"{tag}  ERROR: {e}")

        # Flush every 20 rows
        if len(rows) % 20 == 0:
            _flush(rows)

    _flush(rows)
    print(f"\nDone. {len(rows)} results saved to {OUT_PATH}")


def _flush(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    main()
