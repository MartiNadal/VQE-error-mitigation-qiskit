# VQE Mitigation Benchmark: 1D TFIM Analysis
**3rd Year Physics Dissertation Project | Quantum Computing & Error Mitigation**

This repository contains a modular framework for benchmarking Variational Quantum Eigensolver (VQE) performance on the 1D Transverse-Field Ising Model (TFIM) with open boundary conditions. The project systematically evaluates eight error mitigation configurations — all subsets of Readout Correction, Parity Symmetry Verification, and Zero-Noise Extrapolation (ZNE) — simulated on a FakeBrisbane noise model, with exact diagonalisation as the reference benchmark.

## 🔬 Scientific Overview

The benchmark targets the Hamiltonian for a 1D chain of $N$ spins with open boundary conditions:

$$H = -J \sum_{i=1}^{N-1} \sigma_i^z \sigma_{i+1}^z - h \sum_{i=1}^{N} \sigma_i^x$$

Simulations are run across system sizes $N \in \{2, 4, 6, 8, 10\}$, field strengths $h \in \{0.5, 1.0, 2.0\}$ (ordered, critical, disordered phases), and ansatz depths $L \in \{1, 2, 3\}$. Key scientific questions addressed:

- **Phase sensitivity:** How does mitigation performance vary across the quantum phase transition at $h = 1$?
- **Mitigation Pareto frontier:** What is the cost–error trade-off for each mitigation combination?
- **Error scaling:** How does relative error $\epsilon = |E_\text{mit} - E_\text{exact}| / |E_\text{exact}|$ scale with $N$ and $L$?

## 🛠️ Project Structure
```
├── main.py               # Entry point — runs benchmark and generates all plots
├── replotting.py         # Reload saved results and regenerate plots without re-running
├── config.py             # All hyperparameters (system sizes, shots, ZNE scales, etc.)
├── benchmark.py          # Full sweep over (N, h, L) combinations; saves JSON per run
├── energy.py             # Circuit execution, mitigation pipeline, expectation values
├── optimizer.py          # COBYLA optimiser with random restarts
├── hamiltonian.py        # Exact diagonalisation via SparsePauliOp + eigvalsh
├── ansatz.py             # Hardware-efficient ansatz (RY layers + CZ entanglement)
├── plotting.py           # All six publication-quality PDF plots
├── mitigation/
│   ├── readout.py        # Per-qubit readout calibration and matrix inversion
│   ├── parity.py         # Z₂ parity post-selection; returns measured discard fraction
│   ├── zne.py            # Gate folding and Richardson extrapolation
│   └── pec.py            # Probabilistic Error Cancellation (overhead analysis)
└── results/              # Auto-created; one JSON file per (N, h, L) combination
```

## 📊 Key Visualisations

Six plots are generated automatically and saved as high-resolution PDFs in `plots/`:

1. **Relative Error Grid** — $|E_\text{mit} - E_\text{exact}| / |E_\text{exact}|$ vs $N$ for all 8 mitigation configurations, faceted by $h$ and $L$.
2. **Absolute Energy** — Raw energy values with $\pm 1$ SEM error bars vs exact diagonalisation reference.
3. **Convergence Curves** — COBYLA optimisation history normalised by exact energy, across system sizes.
4. **Pareto Frontier** — Cost (effective circuit overhead) vs error for all configurations; Pareto-optimal points labelled.
5. **Error Scaling** — Power-law fits $\epsilon \propto N^\alpha$ per configuration, with fitted exponents $\alpha \pm \sigma_\alpha$.
6. **Parity Discard Heatmap** — Measured parity discard fraction vs $(N, h)$.

To regenerate plots from saved results without re-running the benchmark:
```bash
python replotting.py
```

## 🚀 How to Run

### 1. Requirements

Python 3.10 or later is required. The following versions have been tested:

| Package | Tested version |
|---|---|
| `qiskit` | 1.x |
| `qiskit-aer` | 0.14+ |
| `qiskit-ibm-runtime` | 0.20+ |
| `numpy` | 1.26+ |
| `matplotlib` | 3.8+ |
| `scipy` | 1.11+ |

### 2. Install dependencies
```bash
pip install qiskit qiskit-aer qiskit-ibm-runtime numpy matplotlib scipy
```

> **Note on `FakeBrisbane` import path:** The import `from qiskit_ibm_runtime.fake_provider import FakeBrisbane` is correct for `qiskit-ibm-runtime >= 0.20`. On older versions it may be at `qiskit.providers.fake_provider`. If you get an `ImportError`, update `qiskit-ibm-runtime` first.

### 3. Run the benchmark
```bash
python main.py
```

The script will:
1. Run the full sweep over all $(N, h, L)$ combinations with all 8 mitigation configurations.
2. Save each result to `results/N{N}_h{h:.1f}_L{L}.json` immediately after completion. If the run is interrupted, completed combinations are safe on disk.
3. Generate all six plots to `plots/`.
4. Print a summary table of relative errors to the console.

If results from a previous run already exist on disk, the script will ask whether to reload them or re-run.

### 4. Expected runtime

The full sweep (45 combinations × 8 configurations × 10 repetitions) takes approximately:

| Machine | Estimated time |
|---|---|
| Modern laptop (8 cores) | 2–4 hours |
| Desktop (16 cores) | 1–2 hours |

The benchmark uses multiprocessing automatically. The number of parallel workers is set to `min(cpu_count, n_tasks)` in `benchmark.py`. To limit parallelism, set `max_parallel_threads_aer` in `config.py`.

### 5. Regenerate plots only

If the benchmark has already been run and results are saved in `results/`:
```bash
python replotting.py
```

### 6. Configuration

All parameters are controlled from `config.py`. Key settings:
```python
system_sizes = (2, 4, 6, 8, 10)   # N values
h_fields     = (0.5, 1.0, 2.0)    # transverse field strengths
layers       = (1, 2, 3)           # ansatz depth L
shots_eval   = 8192                # shots for final energy evaluation
n_reps       = 10                  # independent repetitions per evaluation (for SEM)
n_restarts   = 3                   # COBYLA random restarts
zne_scale_factors = (1, 3, 5)      # noise amplification factors for ZNE
```

## 📁 Result Management

Results are saved incrementally: if the run crashes at combination 30/45, the first 29 are safe. To keep multiple independent runs:
```bash
mv results results_run1   # archive previous results
python main.py            # start fresh
```

Or change `results_dir` in `config.py` to a different path before running.

## 📄 License

MIT License — see `LICENSE` for details.
