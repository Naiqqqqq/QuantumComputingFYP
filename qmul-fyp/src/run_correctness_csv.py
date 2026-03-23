from __future__ import annotations

import csv
from itertools import product
import os, sys
from time import perf_counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

sys.path.insert(0, os.path.dirname(__file__))
from arith import schoolbook_multiplier, schoolbook_multiplier_full, optimised_multiplier_full


def set_bits(qc: QuantumCircuit, base: int, value: int, n: int):
    for i in range(n):
        if (value >> i) & 1:
            qc.x(base + i)


def main():
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    backend = AerSimulator(method="matrix_product_state")
    rows = []

    configs = [
        ("lowbits",   3, schoolbook_multiplier,      False),
        ("baseline",  3, schoolbook_multiplier_full,  True),
        ("baseline",  4, schoolbook_multiplier_full,  True),
        ("baseline",  5, schoolbook_multiplier_full,  True),
        ("optimised", 3, optimised_multiplier_full,   True),
        ("optimised", 4, optimised_multiplier_full,   True),
        ("optimised", 5, optimised_multiplier_full,   True),
    ]

    for mode, n, builder, full in configs:
        mul = builder(n)
        total = mul.num_qubits
        acc_off = 2 * n
        measure_bits = 2 * n if full else n

        ok = True
        start = perf_counter()
        for a, b in product(range(1 << n), range(1 << n)):
            qc = QuantumCircuit(total, measure_bits)
            set_bits(qc, 0, a, n)
            set_bits(qc, n, b, n)
            qc.compose(mul, inplace=True)
            for i in range(measure_bits):
                qc.measure(acc_off + i, i)
            tqc = transpile(qc, basis_gates=["cx", "u1", "u2", "u3", "measure", "barrier"],
                           optimization_level=0)
            result = backend.run(tqc, shots=1).result()
            counts = result.get_counts()
            bitstring = list(counts.keys())[0]
            observed = int(bitstring, 2)
            expected = (a * b) & ((1 << measure_bits) - 1)
            pass_flag = int(observed == expected)
            rows.append({
                "mode": mode, "n": n,
                "a": a, "b": b,
                "expected": expected, "observed": observed,
                "pass": pass_flag,
            })
            ok &= bool(pass_flag)

        elapsed = (perf_counter() - start) * 1000
        total_cases = (1 << n) ** 2
        passed = sum(1 for r in rows[-total_cases:] if r["pass"] == 1)
        rows.append({
            "mode": mode, "n": n,
            "a": "ALL", "b": "ALL",
            "expected": total_cases, "observed": passed,
            "pass": int(ok), "elapsed_ms": round(elapsed, 1),
        })
        print(f"{mode} n={n}: {'PASS' if ok else 'FAIL'} ({passed}/{total_cases}) "
              f"[{round(elapsed/1000,1)}s]")

    out_csv = os.path.join(results_dir, "correctness.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["mode", "n", "a", "b", "expected", "observed", "pass", "elapsed_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote correctness results to {out_csv}")


if __name__ == "__main__":
    main()
