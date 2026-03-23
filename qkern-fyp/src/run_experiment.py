from __future__ import annotations

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Quantum kernel vs RBF baseline")
    parser.add_argument("--dataset", type=str, default="breast_cancer")
    parser.add_argument("--feature_map", type=str, default="zz", choices=["zz", "iqp"])
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--shots", type=int, default=0, help="0=statevector (exact), >0 finite shots")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    ds = load_dataset(args.dataset, random_state=args.seed)

    cfg = QuantumKernelConfig(
        feature_map=args.feature_map,
        num_qubits=args.qubits,
        depth=args.depth,
        shots=None if args.shots == 0 else args.shots,
    )

    # Build Gram matrices with statevector embeddings
    t0 = time.time()
    Xtr = ds.X_train[:, : cfg.num_qubits]
    Xte = ds.X_test[:, : cfg.num_qubits]

    Vtr = states_matrix(Xtr, cfg)
    Vte = states_matrix(Xte, cfg)

    Gtr = gram_from_states(Vtr)
    Gte = cross_gram_from_states(Vte, Vtr)

    qk_time = time.time() - t0

    # Alignment metric
    Ytr = label_kernel(ds.y_train.astype(float) * 2 - 1)
    align = kernel_alignment(Gtr, Ytr)

    # Train SVM with quantum kernel
    qk_res = evaluate_svc_precomputed(Gtr, ds.y_train, Gte, ds.y_test, C=args.C)

    # Classical baselines
    rbf_res = evaluate_svc_rbf(ds.X_train, ds.y_train, ds.X_test, ds.y_test, C=args.C)
    poly_res = evaluate_svc_poly(ds.X_train, ds.y_train, ds.X_test, ds.y_test, C=args.C)
    lin_res = evaluate_svc_linear(ds.X_train, ds.y_train, ds.X_test, ds.y_test, C=args.C)

    out = {
        "dataset": ds.name,
        "feature_map": args.feature_map,
        "qubits": args.qubits,
        "depth": args.depth,
        "shots": args.shots,
        "C": args.C,
        "align": align,
        "qk_acc": qk_res.acc,
        "qk_f1": qk_res.f1,
        "qk_auc": qk_res.auc,
        "rbf_acc": rbf_res.acc,
        "rbf_f1": rbf_res.f1,
        "rbf_auc": rbf_res.auc,
        "poly_acc": poly_res.acc,
        "poly_f1": poly_res.f1,
        "poly_auc": poly_res.auc,
        "lin_acc": lin_res.acc,
        "lin_f1": lin_res.f1,
        "lin_auc": lin_res.auc,
        "qk_time_s": qk_time,
        "seed": args.seed,
    }

    df = pd.DataFrame([out])
    out_path = RESULTS_DIR / "sweep_results.csv"
    if out_path.exists():
        df_prev = pd.read_csv(out_path)
        df = pd.concat([df_prev, df], ignore_index=True)
    df.to_csv(out_path, index=False)

    print(df.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
