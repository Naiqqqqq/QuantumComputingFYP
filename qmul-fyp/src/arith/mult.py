from __future__ import annotations

from qiskit import QuantumCircuit

from .adder import cuccaro_adder, _maj, _uma


# ---------------------------------------------------------------------------
# Optimised multiplier: partial-product pre-computation
# ---------------------------------------------------------------------------

def optimised_multiplier_full(n: int) -> QuantumCircuit:
    """n-bit multiplier with partial-product pre-computation (full 2n-bit product).

    **Optimisation over baseline:**  Instead of calling n sequential
    controlled adders (each with controlled-MAJ/UMA overhead), this
    design pre-computes all partial-product bits and uses un-controlled
    adders, which are cheaper per gate.

    1. Pre-compute partial-product bits  pp[i][j] = a_i AND b_j  via
       Toffolis (independent, parallelisable).
    2. For each shift j, add pp column j into acc[j:j+n] with an
       un-controlled Cuccaro adder (carry into acc[j+n]).
    3. Un-compute partial-product ancillae.

    Registers: a(n) | b(n) | acc(2n) | cin | pp(n*n)
    Total qubits: 4n + 1 + n*n
    """
    a_off = 0
    b_off = n
    acc_off = 2 * n
    cin = 4 * n           # single carry-in qubit (always starts 0)
    pp_off = 4 * n + 1    # partial-product ancillae

    total_qubits = 4 * n + 1 + n * n
    qc = QuantumCircuit(total_qubits, name=f"opt_mul_full_{n}")

    # --- Step 1: compute partial products pp[i][j] = a_i AND b_j ---
    for i in range(n):
        for j in range(n):
            qc.ccx(a_off + i, b_off + j, pp_off + i * n + j)

    # --- Step 2: for each shift j, add pp-column into accumulator ---
    for j in range(n):
        m = min(n, 2 * n - j)
        if m <= 0:
            continue
        pp_col = [pp_off + i * n + j for i in range(m)]  # a-side
        acc_slice = list(range(acc_off + j, acc_off + j + m))  # b-side

        # Inline Cuccaro adder: pp_col → acc_slice, carry into acc[j+m]
        _maj(qc, cin, acc_slice[0], pp_col[0])
        for k in range(1, m):
            _maj(qc, pp_col[k - 1], acc_slice[k], pp_col[k])

        # Carry-out → next accumulator position (if room)
        if j + m < 2 * n:
            qc.cx(pp_col[m - 1], acc_off + j + m)

        # UMA backward pass
        for k in range(m - 1, 0, -1):
            _uma(qc, pp_col[k - 1], acc_slice[k], pp_col[k])
        _uma(qc, cin, acc_slice[0], pp_col[0])

    # --- Step 3: un-compute partial products ---
    for i in range(n):
        for j in range(n):
            qc.ccx(a_off + i, b_off + j, pp_off + i * n + j)

    return qc


# ---------------------------------------------------------------------------
# Baseline multipliers
# ---------------------------------------------------------------------------

def schoolbook_multiplier_full(n: int) -> QuantumCircuit:
    """n-bit schoolbook (shift-and-add) multiplier returning full 2n-bit product.

    Registers: a (n), b (n), acc (2n), cin
    Computes acc := a * b  (full 2n-bit result).
    Total qubits: a(n) + b(n) + acc(2n) + 1 cin = 4n + 1
    """
    a_off = 0
    b_off = n
    acc_off = 2 * n
    cin = 4 * n

    qc = QuantumCircuit(4 * n + 1, name=f"schoolbook_mul_full_{n}")

    for j in range(n):
        m = min(n, 2 * n - j)
        if m <= 0:
            continue

        a_wires = list(range(a_off, a_off + m))
        acc_wires = list(range(acc_off + j, acc_off + j + m))

        # Controlled-MAJ forward pass (controlled on b[j])
        ctrl = b_off + j
        qc.ccx(ctrl, a_wires[0], acc_wires[0])             # CX a→b controlled
        qc.ccx(ctrl, a_wires[0], cin)                       # CX a→c controlled
        qc.mcx([ctrl, cin, acc_wires[0]], a_wires[0])       # CCX c,b,a controlled

        for k in range(1, m):
            qc.ccx(ctrl, a_wires[k], acc_wires[k])
            qc.ccx(ctrl, a_wires[k], a_wires[k - 1])
            qc.mcx([ctrl, a_wires[k - 1], acc_wires[k]], a_wires[k])

        # Carry → acc[j+m] (controlled on b[j])
        if j + m < 2 * n:
            qc.ccx(ctrl, a_wires[m - 1], acc_off + j + m)

        # Controlled-UMA backward pass
        for k in range(m - 1, 0, -1):
            qc.mcx([ctrl, a_wires[k - 1], acc_wires[k]], a_wires[k])
            qc.ccx(ctrl, a_wires[k], a_wires[k - 1])
            qc.ccx(ctrl, a_wires[k - 1], acc_wires[k])

        qc.mcx([ctrl, cin, acc_wires[0]], a_wires[0])
        qc.ccx(ctrl, a_wires[0], cin)
        qc.ccx(ctrl, cin, acc_wires[0])

    return qc


def schoolbook_multiplier(n: int) -> QuantumCircuit:
    """n-bit schoolbook (shift-and-add) multiplier.

    Registers: a (n), b (n), acc (n), carry_in, carry_out
    Computes acc := acc + a * b in-place; for demo we start acc=0.
    Total qubits: a(n) + b(n) + acc(n) + 2 carries
    """
    a_off = 0
    b_off = n
    acc_off = 2 * n
    c0 = 3 * n
    cN = 3 * n + 1

    qc = QuantumCircuit(3 * n + 2, name=f"schoolbook_mul_{n}")

    # Helper to map adder over slices with control on b[j]
    def add_shifted_controlled(j: int):
        # Active bits this round
        m = n - j
        if m <= 0:
            return
        gate = cuccaro_adder(m).control(1)
        wires = [b_off + j]  # control
        wires += list(range(a_off, a_off + m))  # a[0:m]
        wires += list(range(acc_off + j, acc_off + j + m))  # acc[j:j+m]
        wires += [c0, cN]
        qc.append(gate, wires)

    # For each bit j of b, if b_j == 1, add (a << j) to acc
    for j in range(n):
        add_shifted_controlled(j)

    return qc
