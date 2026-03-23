# QuantumComputingFYP

Final Year Project on practical quantum computing, covering:
- **Quantum arithmetic circuits** (Cuccaro adder, schoolbook/optimized multiplication)
- **Quantum kernel machine learning** (ZZ/IQP feature maps with SVM baselines)

## Repository Structure

- `qmul-fyp/` — quantum multiplication circuits, correctness checks, metrics, noise simulation, and plots
- `qkern-fyp/` — quantum kernel experiments, sweeps, metrics, and plots
- `results/appendix_figures/` — appendix circuit images

## Quick Start

### 1) Quantum Multiplication

```bash
cd qmul-fyp
pip install -r requirements.txt
python src/run_correctness_csv.py
python src/run_metrics_csv.py
python src/run_noise_csv.py
python src/plot_results.py
```

### 2) Quantum Kernels

```bash
cd qkern-fyp
pip install -r requirements.txt
python src/sweep.py
python src/plot_results.py
```

## Outputs

- `qmul-fyp/results/` contains correctness, resource, and noise CSVs plus figures
- `qkern-fyp/results/` contains sweep CSVs and kernel-performance figures

## Notes

- This repository is configured to exclude local environments and draft artifacts via `.gitignore`.
- All experiments are designed for reproducibility with fixed seeds and script-based pipelines.
