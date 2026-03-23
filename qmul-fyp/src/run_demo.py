from __future__ import annotations

from itertools import product

from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.compiler import transpile

from arith import schoolbook_multiplier


def set_bits(qc: QuantumCircuit, base: int, value: int, n: int):
    for i in range(n):
        if (value >> i) & 1:
            qc.x(base + i)


def read_bits(bitstring: str, total_qubits: int, base: int, n: int) -> int:
    val = 0
    for i in range(n):
        qubit_index = base + i
        char_index = total_qubits - 1 - qubit_index
        bit = int(bitstring[char_index])
        val |= (bit << i)
    return val


def main():
    n = 3
    mul = schoolbook_multiplier(n)
    total = 3 * n + 2
    a_off, b_off, acc_off = 0, n, 2 * n

    backend = Aer.get_backend("aer_simulator")
    ok = True
    tested = 0
    for a, b in product(range(1 << n), range(1 << n)):
        qc = QuantumCircuit(total, n)
        set_bits(qc, a_off, a, n)
        set_bits(qc, b_off, b, n)
        qc.append(mul.to_gate(), list(range(total)))
        for i in range(n):
            qc.measure(acc_off + i, i)
        tqc = transpile(qc, backend)
        result = backend.run(tqc, shots=2048).result()
        counts = result.get_counts()
        bitstring = max(counts, key=counts.get)
        meas = read_bits(bitstring, total, acc_off, n)
        expected = (a * b) & ((1 << n) - 1)
        ok &= (meas == expected)
        tested += 1

    tqc2 = transpile(mul, backend, optimization_level=1)
    print(f"3x3 multiply low-bits correctness: {'PASS' if ok else 'FAIL'} ({tested} cases)")
    print(f"Transpiled depth: {tqc2.depth()}")
    print(f"Gate counts: {dict(tqc2.count_ops())}")


if __name__ == "__main__":
    main()
