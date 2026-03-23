from __future__ import annotations

import csv
import os, sys
from itertools import product
from time import perf_counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

sys.path.insert(0, os.path.dirname(__file__))
from arith import schoolbook_multiplier_full, optimised_multiplier_full


def set_bits(qc: QuantumCircuit, base: int, value: int, n: int):
    for i in range(n):
        if (value >> i) & 1:
            qc.x(base + i)


def build_noise_model(p1: float, p2: float) -> NoiseModel:
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["u1", "u2", "u3"])
    noise.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return noise


def run_noisy(builder, n: int, noise_model, shots: int = 1024):
    mul = builder(n)
    total = mul.num_qubits
    acc_off = 2 * n
    measure_bits = 2 * n

    backend = AerSimulator(method="matrix_product_state", noise_model=noise_model)
    correct = 0
    total_cases = (1 << n) ** 2

    for a, b in product(range(1 << n), range(1 << n)):
        qc = QuantumCircuit(total, measure_bits)
        set_bits(qc, 0, a, n)
        set_bits(qc, n, b, n)
        qc.compose(mul, inplace=True)
        for i in range(measure_bits):
            qc.measure(acc_off + i, i)
        tqc = transpile(qc, basis_gates=["cx", "u1", "u2", "u3", "measure", "barrier"],
                        optimization_level=0)
        result = backend.run(tqc, shots=shots).result()
        counts = result.get_counts()
        expected = (a * b) & ((1 << measure_bits) - 1)
        expected_key = format(expected, f"0{measure_bits}b")
        correct += counts.get(expected_key, 0)

    accuracy = correct / (total_cases * shots)
    return accuracy


def main():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    noise_levels = [
        ("noiseless", 0.0, 0.0),
        ("low",       0.0001, 0.001),
        ("medium",    0.001,  0.01),
        ("high",      0.01,   0.1),
    ]

    configs = [
        ("baseline",  schoolbook_multiplier_full),
        ("optimised", optimised_multiplier_full),
    ]

    rows = []
    for noise_label, p1, p2 in noise_levels:
        noise_model = build_noise_model(p1, p2) if p1 > 0 else None
        for mode, builder in configs:
            for n in [3, 4]:
                print(f"Noise={noise_label} {mode} n={n} ...", end=" ", flush=True)
                start = perf_counter()
                acc = run_noisy(builder, n, noise_model, shots=512)
                elapsed = perf_counter() - start
                rows.append({
                    "noise_level": noise_label,
                    "p1": p1, "p2": p2,
                    "mode": mode, "n": n,
                    "accuracy": round(acc, 6),
                    "time_s": round(elapsed, 1),
                })
                print(f"accuracy={acc:.4f} [{elapsed:.1f}s]")

    out_csv = os.path.join(results_dir, "noise.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["noise_level", "p1", "p2", "mode", "n", "accuracy", "time_s"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote noise results to {out_csv}")


if __name__ == "__main__":
    main()
