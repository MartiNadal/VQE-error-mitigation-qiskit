"""
vqe_benchmark_hardware.py
=========================
VQE Error Mitigation Benchmarking — Hybrid Simulator + IBM Hardware.

VERSION 2 — HYBRID WORKFLOW:
    Phase 1 (local): VQE optimisation on ideal (noiseless) AerSimulator.
                     Finds optimal parameters θ* without hardware cost.
    Phase 2 (IBM):   Transfer θ* to real IBM hardware.
                     Evaluate energy with all mitigation strategies.
                     Readout calibration performed on real hardware.

Scientific rationale:
    The assumption that θ*_ideal ≈ θ*_noisy holds for shallow circuits
    (L=1,2) where noise acts as a small perturbation on the energy landscape.
    For L=3, this assumption weakens — results should be interpreted with
    this caveat stated explicitly (see project report).

    This workflow eliminates the large shot cost of VQE optimisation on
    hardware while still producing hardware-validated mitigation benchmarks.
    It is standard practice in the NISQ-era VQE literature.

Hardware budget (IBM free tier: ~10 min/month):
    Phase 2 targets N ∈ {2, 4, 6} at h=1.0 (critical point — most
    physically interesting) for all L values and all mitigation configs.
    Estimated hardware time: 8–14 minutes across two months.
    N=8 is included as an optional flag (comment out if budget is tight).

Author: Marti Nadal
Institution: King's College London — BSc Physics Final Project

SETUP REQUIRED:
    pip install qiskit qiskit-aer qiskit-ibm-runtime
    export QISKIT_IBM_TOKEN="your_token_here"
    OR set IBM_TOKEN variable below.
"""

# ============================================================
# IMPORTS
# ============================================================
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# IBM AUTHENTICATION
# ============================================================
# Option 1: set token here (not recommended for public repos — use env var)
IBM_TOKEN: str = os.environ.get("QISKIT_IBM_TOKEN", "YOUR_TOKEN_HERE")
# Option 2: export QISKIT_IBM_TOKEN=... in your shell before running.

# Target backend name. Change this to match your IBM account's available systems.
# Common free-tier options: "ibm_brisbane", "ibm_sherbrooke", "ibm_kyiv"
IBM_BACKEND_NAME: str = "ibm_brisbane"

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Benchmark hyperparameters. See vqe_benchmark_simulator.py for full docstring.

    Hardware-specific additions:
        hw_system_sizes : N values sent to real IBM hardware.
            Kept small to stay within budget. N=8 commented out — enable
            if you have remaining runtime.
        hw_h_field : single h value used for hardware validation.
            h=1.0 (critical point) chosen: most entangled ground state,
            greatest need for mitigation, most physically interesting.
        shots_hw : shots on real hardware. Higher than simulator default
            to compensate for additional hardware noise variability.
            Contributes most to hardware runtime budget.
    """
    # --- Simulator sweep (Phase 1, local) ---
    system_sizes: tuple[int, ...] = (2, 4, 6, 8, 10)
    h_fields: tuple[float, ...] = (0.5, 1.0, 2.0)
    J: float = 1.0
    layers: tuple[int, ...] = (1, 2, 3)
    shots_eval: int = 8192     # Simulator evaluation shots
    shots_optim: int = 1024    # Optimisation shots (reduced, see rationale)
    n_reps: int = 10
    n_restarts: int = 3
    maxiter: int = 300
    rhobeg: float = np.pi / 4
    zne_scale_factors: tuple[int, ...] = (1, 3, 5)
    seed: int = 42
    results_dir: str = "../MAIN/results"

    # --- Hardware validation (Phase 2, IBM) ---
    hw_system_sizes: tuple[int, ...] = (2, 4, 6, 8)
    # N=8 is included. Monitor your IBM runtime budget carefully:
    # N=8 circuits are deeper and take longer per job than N≤6.
    # If you are running low on budget, remove 8 from this tuple.
    hw_h_field: float = 1.0
    shots_hw: int = 8192
    hw_n_reps: int = 5
    # Fewer reps than simulator: hardware shots are expensive.
    # 5 reps × 8192 shots × 2 bases × n_configs still gives meaningful SEM.


CFG = BenchmarkConfig()

MITIGATION_CONFIGS: dict[str, dict] = {
    "raw":           {"use_readout": False, "use_parity": False, "use_zne": False},
    "readout":       {"use_readout": True,  "use_parity": False, "use_zne": False},
    "parity":        {"use_readout": False, "use_parity": True,  "use_zne": False},
    "zne":           {"use_readout": False, "use_parity": False, "use_zne": True},
    "ro+parity":     {"use_readout": True,  "use_parity": True,  "use_zne": False},
    "ro+zne":        {"use_readout": True,  "use_parity": False, "use_zne": True},
    "parity+zne":    {"use_readout": False, "use_parity": True,  "use_zne": True},
    "ro+parity+zne": {"use_readout": True,  "use_parity": True,  "use_zne": True},
}

PHASE_LABELS = {0.5: "Ordered (h=0.5)", 1.0: "Critical (h=1.0)", 2.0: "Disordered (h=2.0)"}
STYLE: dict[str, dict] = {
    "raw":           {"color": "#e41a1c", "marker": "o", "ls": "-",  "lw": 1.5},
    "readout":       {"color": "#377eb8", "marker": "s", "ls": "-",  "lw": 1.5},
    "parity":        {"color": "#4daf4a", "marker": "^", "ls": "-",  "lw": 1.5},
    "zne":           {"color": "#ff7f00", "marker": "D", "ls": "-",  "lw": 1.5},
    "ro+parity":     {"color": "#984ea3", "marker": "v", "ls": "--", "lw": 1.5},
    "ro+zne":        {"color": "#a65628", "marker": "<", "ls": "--", "lw": 1.5},
    "parity+zne":    {"color": "#f781bf", "marker": ">", "ls": "--", "lw": 1.5},
    "ro+parity+zne": {"color": "#000000", "marker": "*", "ls": "-",  "lw": 2.5},
}

# ============================================================
# BACKEND FACTORIES
# ============================================================

def make_ideal_simulator() -> AerSimulator:
    """
    Noiseless AerSimulator for Phase 1 VQE optimisation.
    No noise model attached — exact statevector sampling.
    """
    return AerSimulator()


def make_noisy_simulator() -> AerSimulator:
    """
    FakeBrisbane-based noisy simulator for local benchmark comparison.
    """
    backend = FakeBrisbane()
    from qiskit_aer.noise import NoiseModel
    noise_model = NoiseModel.from_backend(backend)
    return AerSimulator(noise_model=noise_model)


def connect_ibm_backend(token: str, backend_name: str):
    """
    Connects to IBM Quantum and returns the target backend object.

    Parameters
    ----------
    token : str — IBM Quantum API token.
    backend_name : str — name of the IBM backend (e.g. 'ibm_brisbane').

    Returns
    -------
    IBMBackend : real hardware backend object.

    Raises
    ------
    RuntimeError : if token is invalid or backend not found.
    """
    if token == "YOUR_TOKEN_HERE":
        raise RuntimeError(
            "IBM token not set. Export QISKIT_IBM_TOKEN or set IBM_TOKEN in the script."
        )
    logger.info("Connecting to IBM Quantum (backend: %s)...", backend_name)
    service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    backend = service.backend(backend_name)
    logger.info(
        "Connected. Backend status: %s | Pending jobs: %s",
        backend.status().status_msg,
        backend.status().pending_jobs,
    )
    return backend


# ============================================================
# PHYSICS FUNCTIONS (identical to simulator version)
# ============================================================

def get_exact_energy(N: int, J: float, h: float) -> float:
    """Ground state energy via exact diagonalisation. See simulator script."""
    if N < 2:
        raise ValueError(f"N must be ≥ 2, got {N}.")
    pauli_list = []
    for i in range(N - 1):
        label = ["I"] * N
        label[i], label[i + 1] = "Z", "Z"
        pauli_list.append(("".join(label), -J))
    for i in range(N):
        label = ["I"] * N
        label[i] = "X"
        pauli_list.append(("".join(label), -h))
    H_op = SparsePauliOp.from_list(pauli_list)
    return float(np.linalg.eigvalsh(H_op.to_matrix())[0])


def build_ansatz(N: int, L: int) -> tuple[QuantumCircuit, list[Parameter]]:
    """Hardware-Efficient Ansatz. See simulator script."""
    if N < 2:
        raise ValueError(f"N must be ≥ 2, got {N}.")
    if L < 1:
        raise ValueError(f"L must be ≥ 1, got {L}.")
    qc = QuantumCircuit(N)
    params = [Parameter(f"θ_{l}_{q}") for l in range(L) for q in range(N)]
    p_idx = 0
    for _ in range(L):
        for q in range(N):
            qc.ry(params[p_idx], q)
            p_idx += 1
        for q in range(N - 1):
            qc.cz(q, q + 1)
    return qc, params


def apply_zne_folding(qc: QuantumCircuit, scale: int) -> QuantumCircuit:
    """Gate folding for ZNE. See simulator script."""
    if scale < 1 or scale % 2 == 0:
        raise ValueError(f"ZNE scale must be odd and ≥ 1, got {scale}.")
    if scale == 1:
        return qc.copy()
    n_folds = (scale - 1) // 2
    folded = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        if inst.operation.name in ("barrier", "measure", "reset"):
            folded.append(inst)
            continue
        folded.append(inst)
        try:
            inv_op = inst.operation.inverse()
        except Exception:
            logger.warning("Gate '%s' has no inverse; skipping fold.", inst.operation.name)
            continue
        for _ in range(n_folds):
            folded.append(inv_op, inst.qubits, inst.clbits)
            folded.append(inst)
    return folded


def zne_extrapolate(scale_factors: tuple[int, ...], energies: np.ndarray) -> float:
    """Richardson extrapolation to λ=0. See simulator script."""
    coeffs = np.polyfit(np.array(scale_factors, dtype=float), energies, deg=len(scale_factors) - 1)
    return float(coeffs[-1])


def zne_error_propagation(scale_factors: tuple[int, ...], sems: np.ndarray) -> float:
    """Propagates SEM through Richardson extrapolation. See simulator script."""
    lambdas = np.array(scale_factors, dtype=float)
    n = len(lambdas)
    weights = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                weights[i] *= (0.0 - lambdas[j]) / (lambdas[i] - lambdas[j])
    return float(np.sqrt(np.sum((weights * sems) ** 2)))


def parity_post_selection(probs: dict[str, float]) -> dict[str, float]:
    """Even-parity post-selection. See simulator script."""
    even = {b: p for b, p in probs.items() if b.count("1") % 2 == 0}
    total = sum(even.values())
    if total <= 0.0:
        return probs
    return {b: p / total for b, p in even.items()}


def compute_zz_energy(probs: dict[str, float], N: int, J: float) -> float:
    """Vectorised ZZ expectation value. See simulator script."""
    if not probs:
        return 0.0
    bitstrings = list(probs.keys())
    prob_array = np.array(list(probs.values()), dtype=np.float64)
    bit_matrix = np.array([[int(c) for c in b[::-1]] for b in bitstrings], dtype=np.int8)
    same_spin = (bit_matrix[:, :-1] == bit_matrix[:, 1:]).astype(np.float64)
    zz_eigenvalues = 2.0 * same_spin - 1.0
    return float(-J * np.sum(prob_array[:, np.newaxis] * zz_eigenvalues))


def compute_x_energy(probs: dict[str, float], N: int, h: float) -> float:
    """Vectorised X expectation value. See simulator script."""
    if not probs:
        return 0.0
    bitstrings = list(probs.keys())
    prob_array = np.array(list(probs.values()), dtype=np.float64)
    bit_matrix = np.array([[int(c) for c in b[::-1]] for b in bitstrings], dtype=np.int8)
    x_eigenvalues = 1.0 - 2.0 * bit_matrix.astype(np.float64)
    return float(-h * np.sum(prob_array[:, np.newaxis] * x_eigenvalues))


# ============================================================
# READOUT CALIBRATION — SIMULATOR VERSION
# ============================================================

def get_readout_matrices_sim(
    N: int, sim: AerSimulator, shots: int, seed: int
) -> list[np.ndarray]:
    """Per-qubit readout calibration on the simulator. See simulator script."""
    cal_circuits = []
    for q in range(N):
        c0 = QuantumCircuit(N, 1)
        c0.measure(q, 0)
        c1 = QuantumCircuit(N, 1)
        c1.x(q)
        c1.measure(q, 0)
        cal_circuits.extend([c0, c1])

    result = sim.run(cal_circuits, shots=shots, seed_simulator=seed).result()
    matrices = []
    for q in range(N):
        cnt0 = result.get_counts(2 * q)
        cnt1 = result.get_counts(2 * q + 1)
        A = np.array([
            [cnt0.get("0", 0) / shots, cnt1.get("0", 0) / shots],
            [cnt0.get("1", 0) / shots, cnt1.get("1", shots) / shots],
        ], dtype=np.float64)
        matrices.append(np.linalg.inv(A))
    return matrices


# ============================================================
# READOUT CALIBRATION — HARDWARE VERSION
# ============================================================

def get_readout_matrices_hw(
    N: int,
    backend,
    shots: int,
    qubit_mapping: list[int],
) -> list[np.ndarray]:
    """
    Per-qubit readout calibration on real IBM hardware.

    Uses SamplerV2 primitive — the current IBM Runtime standard for
    hardware circuit execution. Calibration circuits are transpiled
    to the backend's native gate set and connectivity before submission.

    The qubit_mapping list specifies which physical qubits on the backend
    correspond to your logical qubits 0..N-1.  This matters because IBM
    hardware has non-trivial connectivity — choosing well-connected qubits
    avoids SWAP overhead in the entangling gates.

    Parameters
    ----------
    N : int — number of logical qubits.
    backend : IBMBackend — connected real hardware backend.
    shots : int — shots per calibration circuit.
    qubit_mapping : list[int] — physical qubit indices, length N.
        Example: [0, 1, 2, 3] uses physical qubits 0-3.
        Check backend.coupling_map for connected pairs.

    Returns
    -------
    list[np.ndarray] — per-qubit inverse assignment matrices.
    """
    logger.info("Calibrating readout on hardware (N=%d)...", N)

    sampler = SamplerV2(backend)
    cal_circuits = []

    for q_logical in range(N):
        q_physical = qubit_mapping[q_logical]

        c0 = QuantumCircuit(N, 1)
        c0.measure(q_logical, 0)
        c0_t = transpile(c0, backend=backend, initial_layout=qubit_mapping,
                         optimization_level=0)

        c1 = QuantumCircuit(N, 1)
        c1.x(q_logical)
        c1.measure(q_logical, 0)
        c1_t = transpile(c1, backend=backend, initial_layout=qubit_mapping,
                         optimization_level=0)

        cal_circuits.extend([c0_t, c1_t])

    # Submit all calibration circuits in one batch
    pub_list = [(circuit, [], shots) for circuit in cal_circuits]
    job = sampler.run(pub_list)
    logger.info("Calibration job submitted (ID: %s). Waiting...", job.job_id())
    result = job.result()

    matrices = []
    for q in range(N):
        # SamplerV2 returns BitArray results; extract counts as dict
        counts_0 = result[2 * q].data.c.get_counts()
        counts_1 = result[2 * q + 1].data.c.get_counts()

        A = np.array([
            [counts_0.get("0", 0) / shots, counts_1.get("0", 0) / shots],
            [counts_0.get("1", 0) / shots, counts_1.get("1", shots) / shots],
        ], dtype=np.float64)

        cond = np.linalg.cond(A)
        if cond > 1e10:
            logger.warning("HW readout matrix qubit %d near-singular (cond=%.2e).", q, cond)

        matrices.append(np.linalg.inv(A))
        logger.info(
            "  Qubit %d (physical %d): P(0|0)=%.3f  P(1|1)=%.3f",
            q, qubit_mapping[q], A[0, 0], A[1, 1],
        )

    return matrices


# ============================================================
# READOUT MITIGATION (MARGINAL-BASED)
# ============================================================

def apply_readout_mitigation(
    counts: dict[str, int],
    matrices: list[np.ndarray],
    N: int,
    shots: int,
) -> dict[str, float]:
    """Marginal-based readout correction. See simulator script."""
    probs = np.zeros(2 ** N, dtype=np.float64)
    for bitstring, count in counts.items():
        probs[int(bitstring, 2)] = count / shots

    probs = probs.reshape([2] * N)
    for q in range(N):
        probs = np.moveaxis(probs, q, 0)
        shape_rest = probs.shape[1:]
        probs = probs.reshape(2, -1)
        probs = matrices[q] @ probs
        probs = probs.reshape((2,) + shape_rest)
        probs = np.moveaxis(probs, 0, q)
    probs = probs.reshape(-1)

    return {format(i, f"0{N}b"): float(probs[i]) for i in range(2 ** N)}


# ============================================================
# ENERGY EVALUATION — IDEAL SIMULATOR (for optimisation)
# ============================================================

def get_energy_ideal(
    ansatz: QuantumCircuit,
    params: np.ndarray,
    N: int,
    J: float,
    h: float,
    sim_ideal: AerSimulator,
    shots: int,
    seed: int,
) -> float:
    """
    Noiseless energy evaluation for VQE optimisation.

    Uses the ideal (no noise model) AerSimulator.  Reduced shot count
    acceptable here because the noiseless landscape is smooth and the
    optimiser only needs relative ordering of parameter sets.

    Parameters
    ----------
    ansatz, params, N, J, h : standard physics inputs.
    sim_ideal : AerSimulator — noiseless simulator.
    shots, seed : sampling parameters.

    Returns
    -------
    float : ideal (noiseless) ⟨H⟩ estimate.
    """
    qc_z = ansatz.copy()
    qc_z.measure_all()
    qc_x = ansatz.copy()
    qc_x.h(range(N))
    qc_x.measure_all()

    bound_z = qc_z.assign_parameters(params)
    bound_x = qc_x.assign_parameters(params)

    result = sim_ideal.run([bound_z, bound_x], shots=shots, seed_simulator=seed).result()
    counts_z = result.get_counts(0)
    counts_x = result.get_counts(1)

    probs_z = {b: c / shots for b, c in counts_z.items()}
    probs_x = {b: c / shots for b, c in counts_x.items()}

    return compute_zz_energy(probs_z, N, J) + compute_x_energy(probs_x, N, h)


# ============================================================
# VQE OPTIMISATION (IDEAL SIMULATOR, PHASE 1)
# ============================================================

def run_vqe_ideal(
    ansatz: QuantumCircuit,
    N: int,
    J: float,
    h: float,
    sim_ideal: AerSimulator,
    cfg: BenchmarkConfig,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    """
    VQE optimisation on the noiseless simulator.

    Finds θ* by minimising the ideal energy.  These parameters are then
    transferred to real hardware for evaluation — the "classical pre-training"
    or "warm-start VQE" approach.

    Phase 1 uses reduced shots (cfg.shots_optim) for speed.
    Multiple random restarts guard against local minima.

    Parameters
    ----------
    ansatz : parameterised circuit.
    N, J, h : Hamiltonian parameters.
    sim_ideal : noiseless AerSimulator.
    cfg : BenchmarkConfig.
    seed : random seed.

    Returns
    -------
    best_params : np.ndarray — optimal θ* on the ideal landscape.
    convergence_history : list[float] — energy per iteration.
    """
    rng = np.random.default_rng(seed)
    n_params = ansatz.num_parameters
    best_params = None
    best_energy = np.inf
    convergence_history: list[float] = []

    for restart in range(cfg.n_restarts):
        restart_seed = seed + restart * 1000
        x0 = rng.uniform(-np.pi, np.pi, n_params)
        history: list[float] = []

        def objective(p: np.ndarray) -> float:
            e = get_energy_ideal(
                ansatz, p, N, J, h, sim_ideal,
                shots=cfg.shots_optim,
                seed=restart_seed + len(history),
            )
            history.append(e)
            return e

        result = minimize(
            objective, x0,
            method="COBYLA",
            options={"maxiter": cfg.maxiter, "rhobeg": cfg.rhobeg},
        )
        logger.info(
            "  [Ideal VQE] Restart %d/%d: E=%.6f (nfev=%d)",
            restart + 1, cfg.n_restarts, result.fun, result.nfev,
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x
            convergence_history = history

    return best_params, convergence_history


# ============================================================
# HARDWARE ENERGY EVALUATION (SINGLE SHOT)
# ============================================================

def get_energy_hw_single(
    ansatz: QuantumCircuit,
    params: np.ndarray,
    N: int,
    J: float,
    h: float,
    backend,
    shots: int,
    qubit_mapping: list[int],
    readout_matrices: Optional[list[np.ndarray]] = None,
    use_parity: bool = False,
    zne_scale: int = 1,
) -> float:
    """
    Evaluates ⟨H⟩ on real IBM hardware using SamplerV2.

    Circuit transpilation:
        Circuits are transpiled to the backend's native gate set
        (typically ECR + RZ + SX on IBM) and physical qubit layout
        before submission.  optimization_level=1 is used: minimal
        optimisation that preserves circuit structure but maps to
        available connections.  Higher optimisation levels may merge
        or reorder gates in ways that interact unpredictably with
        ZNE gate folding.

    SamplerV2 primitive:
        IBM's standard interface for measurement-based sampling on
        hardware.  Returns BitArray results that are converted to
        count dictionaries for processing.

    Parameters
    ----------
    ansatz : parameterised circuit.
    params : optimal parameters θ* from Phase 1.
    N, J, h : Hamiltonian parameters.
    backend : IBMBackend — real hardware.
    shots : int — measurement shots.
    qubit_mapping : list[int] — physical qubit layout.
    readout_matrices : per-qubit correction matrices (from hardware calibration).
    use_parity : apply parity post-selection.
    zne_scale : ZNE noise amplification factor.

    Returns
    -------
    float : ⟨H⟩ estimate on real hardware.
    """
    # Build and optionally fold circuits
    qc_base = ansatz.copy()
    if zne_scale > 1:
        qc_base = apply_zne_folding(qc_base, zne_scale)

    qc_z = qc_base.copy()
    qc_z.measure_all()
    qc_x = qc_base.copy()
    qc_x.h(range(N))
    qc_x.measure_all()

    # Bind parameters
    bound_z = qc_z.assign_parameters(params)
    bound_x = qc_x.assign_parameters(params)

    # Transpile to hardware native gates + layout
    # optimization_level=1: basic optimisation (no gate cancellation
    # that would alter ZNE folding structure)
    t_z = transpile(bound_z, backend=backend,
                    initial_layout=qubit_mapping, optimization_level=1)
    t_x = transpile(bound_x, backend=backend,
                    initial_layout=qubit_mapping, optimization_level=1)

    # Submit via SamplerV2
    sampler = SamplerV2(backend)
    pub_list = [(t_z, [], shots), (t_x, [], shots)]
    job = sampler.run(pub_list)
    logger.debug("HW job submitted (ID: %s).", job.job_id())
    result = job.result()

    # Extract counts from BitArray
    counts_z = result[0].data.meas.get_counts()
    counts_x = result[1].data.meas.get_counts()

    # Convert to probabilities
    probs_z: dict[str, float] = {b: c / shots for b, c in counts_z.items()}
    probs_x: dict[str, float] = {b: c / shots for b, c in counts_x.items()}

    # Apply readout mitigation
    if readout_matrices is not None:
        probs_z = apply_readout_mitigation(counts_z, readout_matrices, N, shots)
        probs_x = apply_readout_mitigation(counts_x, readout_matrices, N, shots)

    # Apply parity post-selection
    if use_parity:
        probs_z = parity_post_selection(probs_z)
        probs_x = parity_post_selection(probs_x)

    return compute_zz_energy(probs_z, N, J) + compute_x_energy(probs_x, N, h)


# ============================================================
# HARDWARE ENERGY STATISTICS
# ============================================================

def get_energy_hw_statistics(
    ansatz: QuantumCircuit,
    params: np.ndarray,
    N: int,
    J: float,
    h: float,
    backend,
    shots: int,
    qubit_mapping: list[int],
    n_reps: int,
    **kwargs,
) -> tuple[float, float]:
    """
    Mean ± SEM of ⟨H⟩ from n_reps hardware evaluations.

    Each repetition is a separate hardware job submission.
    Between repetitions, hardware drift may add additional variance
    beyond shot noise — this is captured in the SEM and is part of
    what makes hardware results more variable than simulator results.

    Parameters
    ----------
    As in get_energy_hw_single, plus:
    n_reps : int — number of independent repetitions.
    **kwargs : forwarded to get_energy_hw_single.

    Returns
    -------
    (mean_energy, sem) : tuple[float, float]
    """
    energies = []
    for rep in range(n_reps):
        logger.info("    HW rep %d/%d...", rep + 1, n_reps)
        e = get_energy_hw_single(
            ansatz, params, N, J, h, backend,
            shots=shots, qubit_mapping=qubit_mapping, **kwargs,
        )
        energies.append(e)
        logger.info("      E = %.6f", e)

    arr = np.array(energies)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(n_reps))


# ============================================================
# PHASE 1 — SIMULATOR SWEEP
# ============================================================

def run_simulator_benchmark(cfg: BenchmarkConfig) -> list[dict]:
    """
    Runs the full (N, h, L) benchmark on the noisy simulator (FakeBrisbane).

    This is the primary systematic sweep covering all system sizes,
    phases, and depths.  Results form the main body of the analysis.

    Uses sequential (non-parallel) execution here to keep this script
    self-contained and avoid multiprocessing complexity in the presence
    of hardware connections.  For a parallel version, see
    vqe_benchmark_simulator.py.

    Returns
    -------
    list[dict] : all results, one dict per (N, h, L).
    """
    sim_noisy = make_noisy_simulator()
    all_results = []

    for N in cfg.system_sizes:
        readout_mats = get_readout_matrices_sim(N, sim_noisy, cfg.shots_eval, cfg.seed)
        for h in cfg.h_fields:
            exact = get_exact_energy(N, cfg.J, h)
            for L in cfg.layers:
                logger.info("SIM | N=%d  h=%.1f  L=%d", N, h, L)
                seed = cfg.seed + N * 100 + int(h * 10) + L
                ansatz, _ = build_ansatz(N, L)

                # Optimise on noisy simulator (not ideal — for fair sim comparison)
                best_params, convergence = run_vqe_noisy(
                    ansatz, N, cfg.J, h, sim_noisy, cfg, seed, readout_mats
                )

                config_results = evaluate_all_configs_sim(
                    ansatz, best_params, N, cfg.J, h,
                    sim_noisy, readout_mats, cfg, seed, exact,
                )

                result = {
                    "N": N, "h": h, "L": L,
                    "exact": exact,
                    "convergence": convergence,
                    **config_results,
                }
                all_results.append(result)

                os.makedirs(cfg.results_dir, exist_ok=True)
                with open(os.path.join(cfg.results_dir, f"sim_N{N}_h{h:.1f}_L{L}.json"), "w") as f:
                    json.dump(result, f, indent=2)

    return all_results


def run_vqe_noisy(
    ansatz, N, J, h, sim_noisy, cfg, seed, readout_mats
) -> tuple[np.ndarray, list[float]]:
    """VQE optimisation on noisy simulator (raw energy, reduced shots)."""
    rng = np.random.default_rng(seed)
    n_params = ansatz.num_parameters
    best_params, best_energy, convergence = None, np.inf, []

    for restart in range(cfg.n_restarts):
        x0 = rng.uniform(-np.pi, np.pi, n_params)
        history: list[float] = []

        def objective(p):
            qc_z = ansatz.copy(); qc_z.measure_all()
            qc_x = ansatz.copy(); qc_x.h(range(N)); qc_x.measure_all()
            res = sim_noisy.run(
                [qc_z.assign_parameters(p), qc_x.assign_parameters(p)],
                shots=cfg.shots_optim,
                seed_simulator=seed + restart * 1000 + len(history),
            ).result()
            pz = {b: c / cfg.shots_optim for b, c in res.get_counts(0).items()}
            px = {b: c / cfg.shots_optim for b, c in res.get_counts(1).items()}
            e = compute_zz_energy(pz, N, J) + compute_x_energy(px, N, h)
            history.append(e)
            return e

        res = minimize(objective, x0, method="COBYLA",
                       options={"maxiter": cfg.maxiter, "rhobeg": cfg.rhobeg})
        if res.fun < best_energy:
            best_energy, best_params, convergence = res.fun, res.x, history

    return best_params, convergence


def evaluate_all_configs_sim(
    ansatz, best_params, N, J, h, sim_noisy, readout_mats, cfg, seed, exact
) -> dict:
    """Evaluates all 8 mitigation configurations on the noisy simulator."""
    config_results = {}
    for config_name, flags in MITIGATION_CONFIGS.items():
        use_zne = flags["use_zne"]
        if use_zne:
            scale_means, scale_sems = [], []
            for sf in cfg.zne_scale_factors:
                mean_sf, sem_sf = _get_sim_energy_stats(
                    ansatz, best_params, N, J, h, sim_noisy, cfg, seed,
                    readout_matrices=readout_mats if flags["use_readout"] else None,
                    use_parity=flags["use_parity"],
                    zne_scale=sf,
                )
                scale_means.append(mean_sf)
                scale_sems.append(sem_sf)
            mean_e = zne_extrapolate(cfg.zne_scale_factors, np.array(scale_means))
            sem_e = zne_error_propagation(cfg.zne_scale_factors, np.array(scale_sems))
        else:
            mean_e, sem_e = _get_sim_energy_stats(
                ansatz, best_params, N, J, h, sim_noisy, cfg, seed,
                readout_matrices=readout_mats if flags["use_readout"] else None,
                use_parity=flags["use_parity"],
                zne_scale=1,
            )
        rel_err = abs(mean_e - exact) / abs(exact) if exact != 0 else np.nan
        config_results[config_name] = {"mean": mean_e, "sem": sem_e, "rel_err": rel_err}
    return config_results


def _get_sim_energy_stats(
    ansatz, params, N, J, h, sim, cfg, seed,
    readout_matrices=None, use_parity=False, zne_scale=1
) -> tuple[float, float]:
    """Helper: n_reps energy estimates on noisy simulator."""
    energies = []
    for rep in range(cfg.n_reps):
        qc_base = ansatz.copy()
        if zne_scale > 1:
            qc_base = apply_zne_folding(qc_base, zne_scale)
        qc_z = qc_base.copy(); qc_z.measure_all()
        qc_x = qc_base.copy(); qc_x.h(range(N)); qc_x.measure_all()

        res = sim.run(
            [qc_z.assign_parameters(params), qc_x.assign_parameters(params)],
            shots=cfg.shots_eval, seed_simulator=seed + rep,
        ).result()
        cz, cx = res.get_counts(0), res.get_counts(1)
        pz = {b: c / cfg.shots_eval for b, c in cz.items()}
        px = {b: c / cfg.shots_eval for b, c in cx.items()}
        if readout_matrices:
            pz = apply_readout_mitigation(cz, readout_matrices, N, cfg.shots_eval)
            px = apply_readout_mitigation(cx, readout_matrices, N, cfg.shots_eval)
        if use_parity:
            pz = parity_post_selection(pz)
            px = parity_post_selection(px)
        energies.append(compute_zz_energy(pz, N, J) + compute_x_energy(px, N, h))

    arr = np.array(energies)
    return float(np.mean(arr)), float(np.std(arr, ddof=1) / np.sqrt(cfg.n_reps))


# ============================================================
# PHASE 2 — HARDWARE VALIDATION
# ============================================================

def get_qubit_mapping(N: int, backend) -> list[int]:
    """
    Returns a heuristic linear qubit mapping for the target backend.

    Selects N qubits that form a connected linear chain on the backend's
    coupling map. For ibm_brisbane (heavy-hex topology), qubits 0-1-2-3...
    along the hex edge are connected.

    For a more optimal mapping, use Qiskit's mapomatic or sabre routing,
    but for N ≤ 6 a manual linear chain works reliably on most IBM backends.

    Parameters
    ----------
    N : int — number of logical qubits needed.
    backend : IBMBackend.

    Returns
    -------
    list[int] : physical qubit indices, length N.
    """
    # Default linear chain starting at qubit 0.
    # Verify this is valid for your specific backend's coupling map.
    # Run: print(backend.coupling_map) to see available connections.
    mapping = list(range(N))
    logger.info("Using qubit mapping: %s", mapping)
    logger.info(
        "Verify these qubits form a connected chain on %s. "
        "Check: print(backend.coupling_map)", IBM_BACKEND_NAME
    )
    return mapping


def run_hardware_validation(
    cfg: BenchmarkConfig,
    sim_results: list[dict],
    backend,
) -> list[dict]:
    """
    Phase 2: evaluates pre-trained circuits on real IBM hardware.

    For each (N, L) in the hardware subset (h = cfg.hw_h_field only):
        1. Retrieves θ* from the ideal simulator optimisation.
        2. Calibrates readout matrices on real hardware.
        3. Evaluates all 8 mitigation configurations with n_reps repetitions.
        4. Saves results incrementally.

    Parameters
    ----------
    cfg : BenchmarkConfig.
    sim_results : results from Phase 1 (used to retrieve θ* and ideal energies).
    backend : IBMBackend — connected real hardware.

    Returns
    -------
    list[dict] : hardware results, one per (N, L) combination.
    """
    hw_results = []
    ideal_sim = make_ideal_simulator()

    for N in cfg.hw_system_sizes:
        qubit_mapping = get_qubit_mapping(N, backend)

        for L in cfg.layers:
            h = cfg.hw_h_field
            logger.info("HW | N=%d  h=%.1f  L=%d", N, h, L)
            seed = cfg.seed + N * 100 + int(h * 10) + L

            exact = get_exact_energy(N, cfg.J, h)
            ansatz, _ = build_ansatz(N, L)

            # Phase 1: optimise θ* on ideal simulator
            logger.info("  Optimising on ideal simulator...")
            best_params, convergence = run_vqe_ideal(
                ansatz, N, cfg.J, h, ideal_sim, cfg, seed
            )
            ideal_energy = get_energy_ideal(
                ansatz, best_params, N, cfg.J, h, ideal_sim,
                shots=cfg.shots_eval, seed=seed,
            )
            logger.info("  θ* found. Ideal energy = %.6f  Exact = %.6f", ideal_energy, exact)

            # Hardware readout calibration
            logger.info("  Calibrating readout on hardware...")
            hw_readout_mats = get_readout_matrices_hw(
                N, backend, shots=cfg.shots_hw, qubit_mapping=qubit_mapping,
            )

            # Evaluate all mitigation configurations on hardware
            config_results: dict[str, dict] = {}
            for config_name, flags in MITIGATION_CONFIGS.items():
                logger.info("  Evaluating config: %s", config_name)
                use_zne = flags["use_zne"]

                if use_zne:
                    scale_means, scale_sems = [], []
                    for sf in cfg.zne_scale_factors:
                        mean_sf, sem_sf = get_energy_hw_statistics(
                            ansatz, best_params, N, cfg.J, h,
                            backend, cfg.shots_hw, qubit_mapping,
                            n_reps=cfg.hw_n_reps,
                            readout_matrices=hw_readout_mats if flags["use_readout"] else None,
                            use_parity=flags["use_parity"],
                            zne_scale=sf,
                        )
                        scale_means.append(mean_sf)
                        scale_sems.append(sem_sf)
                    mean_e = zne_extrapolate(cfg.zne_scale_factors, np.array(scale_means))
                    sem_e = zne_error_propagation(cfg.zne_scale_factors, np.array(scale_sems))
                else:
                    mean_e, sem_e = get_energy_hw_statistics(
                        ansatz, best_params, N, cfg.J, h,
                        backend, cfg.shots_hw, qubit_mapping,
                        n_reps=cfg.hw_n_reps,
                        readout_matrices=hw_readout_mats if flags["use_readout"] else None,
                        use_parity=flags["use_parity"],
                        zne_scale=1,
                    )

                rel_err = abs(mean_e - exact) / abs(exact) if exact != 0 else np.nan
                config_results[config_name] = {
                    "mean": mean_e, "sem": sem_e, "rel_err": rel_err,
                }
                logger.info(
                    "    [%s] E=%.5f ± %.5f  rel_err=%.4f",
                    config_name, mean_e, sem_e, rel_err,
                )

            hw_result = {
                "N": N, "h": h, "L": L,
                "exact": exact,
                "ideal_sim_energy": ideal_energy,
                "convergence": convergence,
                "qubit_mapping": qubit_mapping,
                "backend": IBM_BACKEND_NAME,
                **config_results,
            }
            hw_results.append(hw_result)

            os.makedirs(cfg.results_dir, exist_ok=True)
            with open(
                os.path.join(cfg.results_dir, f"hw_N{N}_h{h:.1f}_L{L}.json"), "w"
            ) as f:
                json.dump(hw_result, f, indent=2)

    return hw_results


# ============================================================
# PLOTTING — HARDWARE COMPARISON
# ============================================================

def plot_hardware_comparison(
    hw_results: list[dict],
    sim_results: list[dict],
    cfg: BenchmarkConfig,
) -> None:
    """
    Side-by-side comparison: simulator vs hardware, for each (N, L).

    For each mitigation configuration, plots:
        Left panel: relative error on simulator
        Right panel: relative error on hardware (same config, same N, same L)

    Allows direct assessment of how well simulator predictions translate
    to real hardware, and whether mitigation strategies are equally
    effective on hardware.
    """
    h_val = cfg.hw_h_field
    n_layers = len(cfg.layers)

    fig, axes = plt.subplots(
        n_layers, 2,
        figsize=(14, 5 * n_layers),
    )
    fig.suptitle(
        f"Simulator vs Hardware Validation  |  h={h_val}  ({PHASE_LABELS[h_val]})",
        fontsize=14, fontweight="bold",
    )

    for l_idx, L in enumerate(cfg.layers):
        ax_sim = axes[l_idx, 0] if n_layers > 1 else axes[0]
        ax_hw = axes[l_idx, 1] if n_layers > 1 else axes[1]

        # Simulator data (all N)
        sim_subset = sorted(
            [r for r in sim_results if r["h"] == h_val and r["L"] == L],
            key=lambda r: r["N"],
        )
        # Hardware data (subset of N)
        hw_subset = sorted(
            [r for r in hw_results if r["h"] == h_val and r["L"] == L],
            key=lambda r: r["N"],
        )

        for config_name, sty in STYLE.items():
            if sim_subset:
                ns_s = [r["N"] for r in sim_subset]
                errs_s = [r[config_name]["rel_err"] for r in sim_subset]
                sems_s = [r[config_name]["sem"] / abs(r["exact"]) for r in sim_subset]
                ax_sim.semilogy(
                    ns_s, errs_s,
                    color=sty["color"], marker=sty["marker"],
                    linestyle=sty["ls"], linewidth=sty["lw"],
                    label=config_name,
                )
                ax_sim.fill_between(
                    ns_s,
                    [max(1e-6, e - s) for e, s in zip(errs_s, sems_s)],
                    [e + s for e, s in zip(errs_s, sems_s)],
                    color=sty["color"], alpha=0.15,
                )

            if hw_subset:
                ns_h = [r["N"] for r in hw_subset]
                errs_h = [r[config_name]["rel_err"] for r in hw_subset]
                sems_h = [r[config_name]["sem"] / abs(r["exact"]) for r in hw_subset]
                ax_hw.semilogy(
                    ns_h, errs_h,
                    color=sty["color"], marker=sty["marker"],
                    linestyle=sty["ls"], linewidth=sty["lw"],
                    label=config_name,
                )
                ax_hw.fill_between(
                    ns_h,
                    [max(1e-6, e - s) for e, s in zip(errs_h, sems_h)],
                    [e + s for e, s in zip(errs_h, sems_h)],
                    color=sty["color"], alpha=0.15,
                )

        for ax, label in [(ax_sim, "Simulator (FakeBrisbane)"), (ax_hw, f"Real Hardware ({IBM_BACKEND_NAME})")]:
            ax.set_title(f"L={L}  —  {label}", fontsize=11)
            ax.set_xlabel("System Size N (qubits)", fontsize=10)
            ax.set_ylabel("Relative Error", fontsize=10)
            ax.grid(True, which="both", alpha=0.3, linestyle=":")
            if hw_subset:
                ax.set_xticks([r["N"] for r in hw_subset])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9, framealpha=0.9,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig("plot_hardware_comparison.pdf", bbox_inches="tight", dpi=150)
    plt.show()
    logger.info("Saved: plot_hardware_comparison.pdf")


def plot_three_way_comparison(
    hw_results: list[dict],
    cfg: BenchmarkConfig,
) -> None:
    """
    Three-way comparison for each (N, L): exact | ideal sim | hardware (all configs).

    Provides the most complete picture: where does VQE fail (ideal vs exact gap),
    and how much does noise add (hardware vs ideal), and how much does mitigation recover.
    """
    h_val = cfg.hw_h_field
    for L in cfg.layers:
        hw_subset = sorted(
            [r for r in hw_results if r["L"] == L],
            key=lambda r: r["N"],
        )
        if not hw_subset:
            continue

        ns = [r["N"] for r in hw_subset]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(
            f"Three-Way Comparison  |  h={h_val}  L={L}",
            fontsize=13, fontweight="bold",
        )

        # Exact
        ax.plot(ns, [r["exact"] for r in hw_subset],
                "k--", lw=2.5, label="Exact (ED)", zorder=10)
        # Ideal simulator
        ax.plot(ns, [r["ideal_sim_energy"] for r in hw_subset],
                "b--", lw=2, label="Ideal Simulator", zorder=9)
        # Hardware configs
        for config_name, sty in STYLE.items():
            means = [r[config_name]["mean"] for r in hw_subset]
            sems = [r[config_name]["sem"] for r in hw_subset]
            ax.errorbar(
                ns, means, yerr=sems,
                color=sty["color"], marker=sty["marker"],
                linestyle=sty["ls"], linewidth=sty["lw"],
                capsize=3, label=f"HW: {config_name}",
            )

        ax.set_xlabel("System Size N (qubits)", fontsize=11)
        ax.set_ylabel("⟨H⟩  Ground State Energy", fontsize=11)
        ax.set_xticks(ns)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"plot_three_way_L{L}.pdf", bbox_inches="tight", dpi=150)
        plt.show()
        logger.info("Saved: plot_three_way_L%d.pdf", L)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("VQE Error Mitigation Benchmark — Hybrid Simulator + Hardware")
    logger.info("=" * 60)

    # ── Phase 1: Simulator benchmark (noisy, all N/h/L) ──────────
    logger.info("\n── PHASE 1: Simulator Sweep ──")
    sim_results = run_simulator_benchmark(CFG)

    # ── Phase 2: Hardware validation ─────────────────────────────
    logger.info("\n── PHASE 2: IBM Hardware Validation ──")
    try:
        backend = connect_ibm_backend(IBM_TOKEN, IBM_BACKEND_NAME)
        hw_results = run_hardware_validation(CFG, sim_results, backend)
    except RuntimeError as e:
        logger.error("Hardware connection failed: %s", e)
        logger.info("Skipping hardware phase. Run Phase 1 results only.")
        hw_results = []

    # ── Plotting ─────────────────────────────────────────────────
    logger.info("\n── Generating Plots ──")

    # Simulator plots (same as Version 1)
    from vqe_benchmark_simulator import (
        plot_relative_error, plot_absolute_energy,
        plot_convergence, plot_cost_vs_error,
    )
    plot_relative_error(sim_results, CFG)
    plot_absolute_energy(sim_results, CFG)
    plot_convergence(sim_results, CFG)
    plot_cost_vs_error(sim_results, CFG)

    # Hardware-specific plots
    if hw_results:
        plot_hardware_comparison(hw_results, sim_results, CFG)
        plot_three_way_comparison(hw_results, CFG)

    logger.info("Complete.")