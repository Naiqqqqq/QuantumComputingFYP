from __future__ import annotations

import csv
from time import perf_counter
import os, sys

from qiskit.compiler import transpile

sys.path.insert(0, os.path.dirname(__file__))
from arith import schoolbook_multiplier, schoolbook_multiplier_full, optimised_multiplier_full


def collect_metrics(circ):
    t0 = perf_counter()
    tqc = transpile(circ, basis_gates=["cx", "u1", "u2", "u3"],
                    optimization_level=1)
    t_ms = (perf_counter() - t0) * 1000
    ops = tqc.count_ops()
    cx = int(ops.get("cx", 0))
    u = int(ops.get("u1", 0) + ops.get("u2", 0) + ops.get("u3", 0))
    return {
        "depth": tqc.depth(),
        "cx": cx,
        "u1q": u,
        "qubits": tqc.num_qubits,
        "transpile_ms": round(t_ms, 1),
    }


def main():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "metrics.csv")

    rows = []
    configs = [
        ("lowbits",   schoolbook_multiplier),
        ("baseline",  schoolbook_multiplier_full),
        ("optimised", optimised_multiplier_full),
    ]

    for mode, builder in configs:
        for n in [3, 4, 6, 8]:
            print(f"Collecting metrics: {mode} n={n} ...", end=" ", flush=True)
            mul = builder(n)
            m = collect_metrics(mul)
            rows.append({"mode": mode, "n": n, **m})
            print(f"done  ({m['qubits']} qubits, depth={m['depth']}, cx={m['cx']})")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["mode", "n", "qubits", "depth", "cx", "u1q", "transpile_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote metrics to {out_csv}")


if __name__ == "__main__":
    main()
