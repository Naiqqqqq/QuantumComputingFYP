from __future__ import annotations

from qiskit import QuantumCircuit


def _maj(qc: QuantumCircuit, c: int, b: int, a: int):
    """MAJ (majority) gate: |c,b,a⟩ → |c⊕a, b⊕a, MAJ(a,b,c)⟩."""
    qc.cx(a, b)
    qc.cx(a, c)
    qc.ccx(c, b, a)


def _uma(qc: QuantumCircuit, c: int, b: int, a: int):
    """UMA (un-majority-add) gate: inverse of MAJ, producing sum in b."""
    qc.ccx(c, b, a)
    qc.cx(a, c)
    qc.cx(c, b)


def cuccaro_adder(n: int) -> QuantumCircuit:
    """Return a Cuccaro ripple-carry adder gate (b := a + b) with carry.

    Implements the MAJ/UMA construction from Cuccaro et al. (2004).

    Qubit layout: [a0..a_{n-1}][b0..b_{n-1}][cin][cout]
    Total qubits: 2n + 2

    Operation: b := a + b, with cin as carry-in and cout as carry-out.
    a and cin are restored to their original values.
    """
    a_off = 0
    b_off = n
    cin = 2 * n
    cout = 2 * n + 1

    qc = QuantumCircuit(2 * n + 2, name=f"cuccaro_{n}")

    # Forward pass: propagate carries via MAJ gates
    _maj(qc, cin, b_off + 0, a_off + 0)
    for i in range(1, n):
        _maj(qc, a_off + i - 1, b_off + i, a_off + i)

    # Copy carry-out
    qc.cx(a_off + n - 1, cout)

    # Backward pass: compute sums via UMA gates
    for i in range(n - 1, 0, -1):
        _uma(qc, a_off + i - 1, b_off + i, a_off + i)
    _uma(qc, cin, b_off + 0, a_off + 0)

    return qc.to_gate(label=f"cuccaro_{n}")
