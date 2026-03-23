# Quantum Kernel SVM FYP (Option 1)

Goal: Compare quantum kernel SVMs vs classical RBF/poly kernels on small tabular datasets and synthetic tasks. Fully local runs.

Quick start
- Create a Python 3.10+ environment.
- Install requirements: `pip install -r requirements.txt`.
- Run a demo experiment: `python -m src.run_experiment --dataset breast_cancer`.

Structure
- `src/qkern/`: package with datasets, kernels, models, metrics, and utils.
- `src/run_experiment.py`: main script to reproduce results.
- `results/`: CSVs and plots.

Planned experiments
- Datasets: Breast Cancer, Wine, Sonar, Moons, Circles, XOR.
- Quantum feature maps: ZZ, IQP, 2–8 qubits, depth 1–3.
- Baselines: Linear, RBF, Poly SVM; small MLP.
- Metrics: Accuracy/F1/ROC-AUC, kernel alignment (cHSIC), time, confidence intervals.

