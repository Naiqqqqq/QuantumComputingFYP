# Quantum Multiplication (Beginner Scaffold)

This project implements small, NISQ-friendly quantum arithmetic blocks in Qiskit:
- Ripple-carry adder (Cuccaro MAJ/UMA)
- Schoolbook (shift-add) multiplier

We provide tiny 3–4 bit demos, truth-table tests, and gate/depth metrics.

Quick start
- Python 3.10+
- Install: `pip install -r requirements.txt`
- Run demo: `python -m src.run_demo`

What you’ll see
- Correctness checks for 3×3-bit multiply
- Gate counts and depth from the transpiler
- Optional Aer noise simulation

