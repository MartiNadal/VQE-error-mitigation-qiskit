# VQE Mitigation Benchmark: 1D TFIM Analysis
**Dissertation Project | Quantum Computing & Error Mitigation**

This repository contains a specialized framework for benchmarking Variational Quantum Eigensolver (VQE) performance on the 1D Transverse Field Ising Model (TFIM). The project evaluates various error mitigation strategies, including Zero-Noise Extrapolation (ZNE) and Readout Mitigation, to analyze their impact on energy accuracy and convergence.

## 🔬 Scientific Overview
The benchmark simulates the Hamiltonian for a 1D chain of $N$ spins:
$$H = -J \sum_{i=1}^{N-1} \sigma_i^z \sigma_{i+1}^z - h \sum_{i=1}^N \sigma_i^x$$

We specifically explore:
* **Phase Transitions:** Benchmarking at $h < 1.0$ (Ordered) and $h = 1.0$ (Critical).
* **Mitigation Pareto Frontier:** Analyzing the trade-off between computational overhead and error reduction.
* **Error Scaling:** Quantifying how relative error $\epsilon$ scales with system size $N$.

## 🛠️ Project Structure
* `main.py`: The primary execution script for the VQE loop.
* `plotting.py`: Visualization module for error grids, energy scales, and Pareto frontiers.
* `config.py`: Centralized hardware and simulation parameters.
* `results/`: (Local only) Storage for CSV data and simulation logs.

## 📊 Key Visualizations
The framework automatically generates several high-fidelity PDF plots suitable for LaTeX inclusion:
1. **Relative Error Grid:** $|E_{mit} - E_{exact}| / |E_{exact}|$ vs $N$.
2. **Pareto Frontier:** Cost-benefit analysis of mitigation overhead.
3. **Convergence Curves:** Optimization history normalized by exact energy.

## 🚀 How to Run
1. Ensure you have the required dependencies:
   ```bash
   pip install qiskit numpy matplotlib scipy
