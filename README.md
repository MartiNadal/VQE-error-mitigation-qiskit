# VQE Mitigation Benchmark: 1D TFIM Analysis
**3rd Year Physics Dissertation Project | Quantum Computing & Error Mitigation**

This repository contains a modular framework for benchmarking Variational Quantum Eigensolver (VQE) performance on the 1D Transverse-Field Ising Model (TFIM) with open boundary conditions. The project systematically evaluates eight error mitigation configurations — all subsets of Readout Mitigation, Parity Symmetry Verification, and Zero-Noise Extrapolation (ZNE) — simulated on a FakeBrisbane noise model, with exact diagonalisation as the reference benchmark.

## 🔬 Scientific Overview

The benchmark targets the Hamiltonian for a 1D chain of $N$ spins with open boundary conditions:

$$H = -J \sum_{i=1}^{N-1} \sigma_i^z \sigma_{i+1}^z - h \sum_{i=1}^{N} \sigma_i^x$$

Simulations are run across system sizes $N \in \{2, 4, 6, 8, 10, 12\}$, field strengths $h \in \{0.5, 1.0, 2.0\}$ (ordered, critical, disordered phases), and ansatz depths $L \in \{1, 2, 3\}$. Key scientific questions addressed:

- **Phase sensitivity:** How does mitigation performance vary across the quantum phase transition at $h = 1$?
- **Mitigation Pareto frontier:** What is the cost–error trade-off for each mitigation combination?
- **Error scaling:** How does relative error $\epsilon = |E_\text{mit} - E_\text{exact}| / |E_\text{exact}|$ scale with $N$ and $L$?
- **ZNE regime sensitivity:** Under what noise model composition does ZNE provide meaningful benefit?

> **Note on backend versatility:**  While the simulations in this project primarily utilize the `FakeBrisbane` noise model, the framework is designed to be backend-agnostic and can be configured to any IBM noise model in `config.py`. 


## 🛠️ Project Structure
```
├── main.py               # Entry point — runs benchmark and generates all plots
├── config.py             # All hyperparameters (system sizes, shots, ZNE scales, etc.)
├── benchmark.py          # Full sweep over (N, h, L) combinations; saves JSON per run
├── energy.py             # Circuit execution, mitigation pipeline, expectation values
├── optimizer.py          # COBYLA optimiser with random restarts
├── hamiltonian.py        # Exact diagonalisation via SparsePauliOp + eigvalsh
├── ansatz.py             # Hardware-efficient ansatz (RY layers + CZ entanglement)
├── plotting.py           # All PDF plots
├── zne_noise_study.py    # Standalone ZNE regime study: parametric noise model comparison
├── hardware_ZNE_scaling  # ZNE noise scaling verification on real ibm_fez QPU.
├── hardware_EM_benchmark # Single (N, h, L) combination benchmark on ibm_fez for validation
├── hardware_errors_info  # Diagnostic script for characterizing the noise profile of IBM QPUs.
├── mitigation/
│   ├── readout.py        # Per-qubit readout calibration and matrix inversion
│   ├── parity.py         # Z₂ parity post-selection; returns measured discard fraction
│   ├── zne.py            # Gate folding and Richardson extrapolation
│   └── pec.py            # Probabilistic Error Cancellation (overhead analysis)
├── results/              # Auto-created; one JSON file per (N, h, L) combination
├── plots/                # Auto-created; one JSON file per (N, h, L) combination
└── results_zne_study/    # Auto-created; one JSON file per (N, h, L) in ZNE study
```

## 📊 Key Visualisations

Six plots are generated automatically by `main.py` and saved as high-resolution PDFs in `plots/`:

1. **Relative Error Grid** — $|E_\text{mit} - E_\text{exact}| / |E_\text{exact}|$ vs $N$ for all 8 mitigation configurations, faceted by $h$ and $L$.
2. **Convergence Curves** — COBYLA optimisation history normalised by exact energy, across system sizes.
3. **Pareto Frontier** — Cost (effective circuit overhead) vs error for all configurations; Pareto-optimal points labelled.
4. **Parity Discard Heatmap** — Measured parity discard fraction vs $(N, h)$ for the `parity` configuration.
5. **Parity Discard Comparison** — Side-by-side heatmaps of parity discard fraction for `parity` vs `ro+parity`, showing how readout correction reduces the effective discard rate.
6. **Parity Discard Delta** — Heatmap of the reduction in discard fraction $(f_\text{parity} - f_\text{ro+parity})$, quantifying where readout correction provides the greatest benefit to parity post-selection.
7. **ZNE Noise Comparison** — Raw vs ZNE relative error across readout-dominated, balanced, and gate-dominated synthetic noise models; demonstrates that ZNE effectiveness depends on the gate-to-readout error ratio.
8. **Continuous noise ratio sweep** — Continuous sweep of ZNE error reduction vs noise ratio $r = p_\text{gate} / p_\text{readout}$ from $r=0.05$ to $r=10$; Shows the relation between ZNE effectiveness and the QPU's noise profile.
9. **Hardware ZNE scaling** — $E(\lambda)$ vs $\lambda=1, 2, 3$ on ibm_fez hardware. Shows the Richardson extrapolation method, yielding $E(0) < E(1)$.
10. **Hardware vs Simulation comparison** — Energy estimates after mitigation on FakeFez and real ibm_fez QPU; suggests that FakeFez is a faithful noise model.

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
3. Generate all plots to `plots/`.
4. Print a summary table of relative errors to the console.

If results from a previous run already exist on disk, the script will ask whether to reload them or re-run.

### 4. Run the ZNE noise study
```bash
python zne_noise_study.py
```

This standalone script investigates ZNE effectiveness across three synthetic noise models (readout-dominated, balanced, and gate-dominated) with independently tunable gate and readout error rates. It runs a reduced sweep ($N \in \{4\}$, $h = 0.5$, $L = 2$) using noiseless VQE parameters for speed, and generates two dedicated plots. Expected runtime: < 5 minutes.

If results from a previous study run already exist in `results_zne_study/`, the script will ask whether to reload them or re-run.


### 5. Run the Hardware studies
> **Note on hardware limitations** Physical hardware validation was conducted on `ibm_fez` rather than `ibm_brisbane` due to specific hardware access limitations after the latter was retired.

```bash
python hardware_zne_scaling.py
```
This script sends the same circuit at scale factors $\lambda = 1, 2, 3$ and executes the richardson extrapolation and generates a plot. Saves results in `results_hardware/` and plot in `plots/`.

Expected runtime: dependent on IBM queue time, once the job is executed < 5 minutes.
```bash
python hardware_EM_benchmark.py
```
This script recycles the optimal parameters from the Fez VQE at a given ($N,h,L$) combination and runs the benchmark measurements on hardware, including all mitigation configurations, and generates a comparison plot. If results from a previous study do not exist in `results/` a fallback VQE is executed to obtain the optimal parameters. Saves results in `results_hardware/` and plot in `plots/`..

Expected runtime: dependent on IBM queue time, once the job is executed < 10 minutes.
### 6. Expected runtime

The full benchmark sweep takes approximately:

| Machine | Estimated time |
|---|----------------|
| Modern laptop (8 cores) | 7–9 hours      |
| Desktop (16 cores) | 4–6 hours      |

The benchmark uses multiprocessing automatically. The number of parallel workers is set to `min(cpu_count, n_tasks)` in `benchmark.py`. To limit parallelism, set `max_parallel_threads_aer` in `config.py`.


### 7. Regenerate plots only

If the benchmark has already been run and results are saved in `results/`:
```bash
python plotting.py
```

### 8. Configuration

All parameters are controlled from `config.py`.