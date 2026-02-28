"""
vqe_benchmark_simulator.py
==========================
VQE Error Mitigation Benchmarking on the 1D Transverse-Field Ising Model (TFIM).

VERSION 1 — LOCAL SIMULATOR ONLY (FakeBrisbane noise model).

Physics
-------
Hamiltonian (open boundary conditions):
    H = -J * sum_{i=0}^{N-2} Z_i Z_{i+1}  -  h * sum_{i=0}^{N-1} X_i

The three h values probe distinct phases of the TFIM:
    h < J  →  ordered (ferromagnetic) phase
    h = J  →  quantum critical point
    h > J  →  disordered (paramagnetic) phase

Mitigation strategies benchmarked (all 8 subsets of {RO, Parity, ZNE}):
    raw, readout (RO), parity, zne,
    ro+parity, ro+zne, parity+zne, ro+parity+zne

Author: [Your Name]
Institution: King's College London — BSc Physics Final Project
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
from multiprocessing import Pool
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("vqe_benchmark.log"), # Saves to a file
        logging.StreamHandler()                  # Still prints to console
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION — single source of truth
# ============================================================

@dataclass(frozen=True)
class BenchmarkConfig:
    """
    All hyperparameters for the benchmark sweep.
    frozen=True: immutable after creation — prevents accidental
    mid-run modification.

    Parameters
    ----------
    system_sizes : tuple[int]
        Number of qubits N. Must be even (parity post-selection
        requires the ground state to lie in the even-parity sector,
        which holds for even N in the TFIM with OBC).
    h_fields : tuple[float]
        Transverse field strengths. Values chosen to probe the
        ordered (0.5), critical (1.0), and disordered (2.0) phases.
    J : float
        ZZ coupling strength. Set to 1.0 to fix the energy scale.
    layers : tuple[int]
        Ansatz layer depths L to sweep. L=1: shallow, low noise.
        L=2: moderate expressibility. L=3: high expressibility, high noise.
    shots_eval : int
        Number of shots for final energy evaluation. Higher = lower shot noise.
    shots_optim : int
        Number of shots during COBYLA optimisation. Reduced from shots_eval
        because the optimiser only needs to distinguish better from worse
        parameter sets, not compute precise energies. Reduces total runtime
        by ~(shots_eval / shots_optim) × maxiter, with negligible precision
        loss on the final reported energies (which always use shots_eval).
    n_reps : int
        Independent repetitions of each energy evaluation for statistical
        error analysis. Mean ± SEM reported from these repetitions.
    n_restarts : int
        Number of random-start COBYLA runs. Best result retained.
        Guards against local minima and barren plateaus.
    maxiter : int
        Maximum COBYLA iterations per restart.
    rhobeg : float
        COBYLA initial step size in parameter space (radians).
        π/4 ≈ 0.785 is a standard choice for angular parameters.
    zne_scale_factors : tuple[int]
        Noise amplification factors for ZNE gate folding. Must be odd
        integers ≥ 1. Three points enables quadratic Richardson
        extrapolation, which is more accurate than linear (two points).
    results_dir : str
        Directory for incremental JSON result files.
    seed : int
        Global random seed for reproducibility (numpy + Aer simulator).
    max_parallel_threads_aer : int
        Internal Aer thread count per process. Set to 1 when using
        multiprocessing so OS-level parallelism is controlled externally.
    """
    system_sizes: tuple[int, ...] = (2, 4, 6, 8, 10)
    h_fields: tuple[float, ...] = (0.5, 1.0, 2.0)
    J: float = 1.0
    layers: tuple[int, ...] = (1, 2, 3)
    shots_eval: int = 8192
    shots_optim: int = 1024
    n_reps: int = 10
    n_restarts: int = 3
    maxiter: int = 300
    rhobeg: float = np.pi / 4
    zne_scale_factors: tuple[int, ...] = (1, 3, 5)
    results_dir: str = "../MAIN/results"
    seed: int = 42
    max_parallel_threads_aer: int = 1


CFG = BenchmarkConfig()

# All 8 mitigation configurations (2^3 subsets of {readout, parity, zne})
# 'readout_mats' value 'inject' is a sentinel: replaced with actual matrices
# at runtime inside run_single_combination().
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

# ============================================================
# BACKEND FACTORY
# (returns a fresh simulator; called inside each worker process)
# ============================================================

def make_simulator(threads: int = CFG.max_parallel_threads_aer) -> AerSimulator:
    """
    Creates an AerSimulator with FakeBrisbane's noise model.

    Called inside each worker process so that simulator objects are
    not shared across processes (AerSimulator is not process-safe).

    Parameters
    ----------
    threads : int
        max_parallel_threads for Aer's internal OpenMP pool.
        Set to 1 when using Python multiprocessing (see Config docstring).

    Returns
    -------
    AerSimulator
        Ready-to-run noisy simulator.
    """
    backend = FakeBrisbane()
    noise_model = NoiseModel.from_backend(backend)
    return AerSimulator(
        noise_model=noise_model,
        max_parallel_threads=threads,
    )


# ============================================================
# 1. EXACT DIAGONALISATION
# ============================================================

def get_exact_energy(N: int, J: float, h: float) -> float:
    """
    Ground state energy of the 1D TFIM via exact diagonalisation.

    Builds the 2^N × 2^N Hamiltonian matrix from Pauli strings and
    returns its minimum eigenvalue.  Open boundary conditions (OBC):
    ZZ coupling between sites i and i+1 for i in 0..N-2 only.

    Complexity: O(2^N) memory, O((2^N)^2) time for eigvalsh.
    Feasible for N ≤ 20 on a standard workstation.

    Parameters
    ----------
    N : int  — number of qubits/spins (≥ 2).
    J : float — ZZ coupling strength.
    h : float — transverse field strength.

    Returns
    -------
    float : ground state energy E_0.
    """
    if N < 2:
        raise ValueError(f"N must be ≥ 2, got {N}.")

    pauli_list: list[tuple[str, complex]] = []

    # ZZ coupling terms: -J * Z_i Z_{i+1}  (OBC: N-1 terms)
    for i in range(N - 1):
        label = ["I"] * N
        # Qiskit's SparsePauliOp convention: label[0] is the RIGHTMOST
        # qubit (little-endian). For a symmetric all-equal-J Hamiltonian
        # the ordering does not affect the eigenspectrum, but we keep
        # it consistent for future site-dependent extensions.
        label[i] = "Z"
        label[i + 1] = "Z"
        pauli_list.append(("".join(label), -J))

    # Transverse field terms: -h * X_i  (N terms)
    for i in range(N):
        label = ["I"] * N
        label[i] = "X"
        pauli_list.append(("".join(label), -h))

    H_op = SparsePauliOp.from_list(pauli_list)
    # to_matrix() returns a dense 2^N × 2^N complex128 array.
    H_matrix = H_op.to_matrix()
    # eigvalsh: symmetric/Hermitian eigenvalue solver.
    # Returns sorted real eigenvalues (ascending). O((2^N)^3) but
    # only eigenvalues needed → uses divide-and-conquer, faster in practice.
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    return float(eigenvalues[0])


# ============================================================
# 2. ANSATZ CONSTRUCTION
# ============================================================

def build_ansatz(N: int, L: int) -> tuple[QuantumCircuit, list[Parameter]]:
    """
    Hardware-Efficient Ansatz (HEA): alternating RY and CZ layers.

    Structure per layer:
        [RY(θ) on each qubit]  →  [CZ on adjacent pairs (linear chain)]

    Physical interpretation:
        RY gates rotate each qubit independently on the Bloch sphere,
        controlling the superposition amplitude.  CZ gates entangle
        neighbouring qubits, allowing the circuit to build up the
        many-body correlations present in the TFIM ground state.
        Each additional layer L increases the entanglement depth
        (how far correlations can propagate) and the expressibility
        (the volume of Hilbert space the ansatz can reach).

    Total parameters : N * L  (one per RY gate).
    Total 2-qubit gates : (N-1) * L  CZ gates.
    Circuit depth (Qiskit definition): 2 * L  (RY and CZ parallelise).

    Parameters
    ----------
    N : int — number of qubits (≥ 2).
    L : int — number of layers (≥ 1).

    Returns
    -------
    qc : QuantumCircuit  — parameterised circuit (no measurements).
    params : list[Parameter]  — symbolic parameters, indexed as
        params[layer * N + qubit].
    """
    if N < 2:
        raise ValueError(f"N must be ≥ 2, got {N}.")
    if L < 1:
        raise ValueError(f"L must be ≥ 1, got {L}.")

    qc = QuantumCircuit(N)
    # Create all symbolic parameters up front.
    # Named θ_l_q: layer l, qubit q, for readability in circuit diagrams.
    params: list[Parameter] = [
        Parameter(f"θ_{l}_{q}")
        for l in range(L)
        for q in range(N)
    ]

    p_idx = 0
    for _ in range(L):
        # RY rotation block — product (unentangled) state after this alone
        for q in range(N):
            qc.ry(params[p_idx], q)
            p_idx += 1
        # CZ entangling block — linear chain matches IBM heavy-hex topology
        # without SWAP overhead for neighbouring qubits.
        for q in range(N - 1):
            qc.cz(q, q + 1)

    return qc, params


# ============================================================
# 3. ZNE GATE FOLDING
# ============================================================

def apply_zne_folding(qc: QuantumCircuit, scale: int) -> QuantumCircuit:
    """
    Gate-folding for Zero Noise Extrapolation (ZNE).

    For noise scale factor λ (odd integer ≥ 1), each gate G is replaced by:
        G (G† G)^{(λ-1)/2}

    This multiplies the noise contribution of every gate by λ while
    preserving the logical unitary (G†G = I, so extra pairs cancel).
    Using ALL gates (not just CZ) ensures both single- and two-qubit
    noise is amplified proportionally.

    Physical justification: under a depolarising noise model, the
    expectation value of an observable scales approximately as:
        E(λ) ≈ E_ideal + λ * a + λ^2 * b + ...
    Evaluating E at multiple λ values allows Richardson extrapolation
    back to λ=0 (zero noise limit).

    Parameters
    ----------
    qc : QuantumCircuit  — circuit AFTER parameter binding and BEFORE
        measurement gates.  Must not contain Measure instructions.
    scale : int  — noise amplification factor. Must be odd and ≥ 1.
        scale=1 returns qc unchanged. scale=3 triples noise. scale=5 quintuples.

    Returns
    -------
    QuantumCircuit : folded circuit with same logical unitary, amplified noise.

    Raises
    ------
    ValueError : if scale is even or < 1.
    """
    if scale < 1 or scale % 2 == 0:
        raise ValueError(f"ZNE scale must be an odd integer ≥ 1, got {scale}.")
    if scale == 1:
        return qc.copy()

    n_folds = (scale - 1) // 2  # number of G†G pairs to append per gate

    folded = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        # Skip barrier and measure instructions (non-unitary)
        if inst.operation.name in ("barrier", "measure", "reset"):
            folded.append(inst)
            continue

        # Append original gate G
        folded.append(inst)

        # Append n_folds copies of (G†, G)
        try:
            inv_op = inst.operation.inverse()
        except Exception:
            # If a gate has no defined inverse, skip folding for it.
            # Log a warning so the user knows.
            logger.warning(
                "Gate '%s' has no inverse; skipping fold for this gate.",
                inst.operation.name,
            )
            continue

        for _ in range(n_folds):
            folded.append(inv_op, inst.qubits, inst.clbits)
            folded.append(inst)

    return folded


# ============================================================
# 4. ZNE RICHARDSON EXTRAPOLATION
# ============================================================

def zne_extrapolate(
    scale_factors: tuple[int, ...],
    energies: np.ndarray,
) -> float:
    """
    Richardson extrapolation to the zero-noise limit.

    Fits a polynomial of degree (len(scale_factors) - 1) to the
    (scale, energy) data points and evaluates at scale=0.

    For 2 points: linear fit  E(λ) = E_0 + aλ
    For 3 points: quadratic   E(λ) = E_0 + aλ + bλ²  (more accurate)

    Parameters
    ----------
    scale_factors : tuple[int]  — noise scale factors used (e.g. (1, 3, 5)).
    energies : np.ndarray  — shape (len(scale_factors),). Mean energy at each scale.

    Returns
    -------
    float : extrapolated zero-noise energy estimate.
    """
    lambdas = np.array(scale_factors, dtype=float)
    degree = len(scale_factors) - 1
    # np.polyfit fits a polynomial of given degree to (x, y) data.
    # coeffs[0] is the highest-degree coefficient.
    coeffs = np.polyfit(lambdas, energies, deg=degree)
    # Evaluate polynomial at λ=0: only the constant term (last coefficient) survives.
    return float(coeffs[-1])


def zne_error_propagation(
    scale_factors: tuple[int, ...],
    sems: np.ndarray,
) -> float:
    """
    Propagates statistical uncertainty through Richardson extrapolation.

    For a linear combination E_0 = sum_i w_i * E(λ_i), the uncertainty is:
        σ_{E_0} = sqrt(sum_i w_i^2 * σ_i^2)

    The weights w_i come from the polynomial fit evaluated at λ=0.
    Here we compute them via the Lagrange basis polynomials of the fit.

    Parameters
    ----------
    scale_factors : tuple[int] — same as in zne_extrapolate.
    sems : np.ndarray — shape (len(scale_factors),). SEM at each scale.

    Returns
    -------
    float : propagated SEM on the extrapolated estimate.
    """
    lambdas = np.array(scale_factors, dtype=float)
    n = len(lambdas)
    # Compute Lagrange basis weights: w_i = L_i(0)
    # L_i(x) = product_{j≠i} (x - λ_j) / (λ_i - λ_j)
    weights = np.ones(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                weights[i] *= (0.0 - lambdas[j]) / (lambdas[i] - lambdas[j])
    return float(np.sqrt(np.sum((weights * sems) ** 2)))


# ============================================================
# 5. READOUT ERROR CALIBRATION
# ============================================================

def get_readout_matrices(
    N: int,
    sim: AerSimulator,
    shots: int,
    seed: int,
) -> list[np.ndarray]:
    """
    Calibrates per-qubit readout assignment matrices.

    For each qubit q, runs two calibration circuits:
        c0: prepare |0⟩, measure → estimates P(0|0) and P(1|0)
        c1: prepare |1⟩ (via X gate), measure → estimates P(0|1) and P(1|1)

    Builds the 2×2 assignment matrix A_q where A_q[i,j] = P(measure i | prepared j).
    Returns M_q = A_q^{-1} for each qubit.

    Assumption: readout errors on different qubits are independent.
    This is the standard local readout mitigation assumption. Cross-talk
    (correlated readout errors) is neglected — a known approximation.

    Parameters
    ----------
    N : int — number of qubits.
    sim : AerSimulator — noisy simulator instance.
    shots : int — shots per calibration circuit.
    seed : int — simulator seed for reproducibility.

    Returns
    -------
    list[np.ndarray] : length N. Each element is a (2,2) float64 array M_q = A_q^{-1}.
    """
    calibration_circuits: list[QuantumCircuit] = []

    for q in range(N):
        # c0: measure qubit q in the |0⟩ state
        c0 = QuantumCircuit(N, 1)
        c0.measure(q, 0)

        # c1: flip qubit q to |1⟩ then measure
        c1 = QuantumCircuit(N, 1)
        c1.x(q)
        c1.measure(q, 0)

        calibration_circuits.extend([c0, c1])

    # Batch all 2N calibration circuits in one job
    job = sim.run(calibration_circuits, shots=shots, seed_simulator=seed)
    result = job.result()

    matrices: list[np.ndarray] = []
    for q in range(N):
        counts_0 = result.get_counts(2 * q)      # c0 results for qubit q
        counts_1 = result.get_counts(2 * q + 1)  # c1 results for qubit q

        A = np.array([
            [counts_0.get("0", 0) / shots,  counts_1.get("0", 0) / shots],
            # A[0,0] = P(measure 0 | prepared 0)
            # A[0,1] = P(measure 0 | prepared 1)
            [counts_0.get("1", 0) / shots,  counts_1.get("1", shots) / shots],
            # A[1,0] = P(measure 1 | prepared 0)
            # A[1,1] = P(measure 1 | prepared 1)
            # Default for counts_1["1"] is `shots` (not 1e-10) so that
            # P(1|1) = 1.0 when no bit-flip occurs (perfect preparation).
        ], dtype=np.float64)

        # Guard against singular matrix (can occur if one outcome never appears)
        cond = np.linalg.cond(A)
        if cond > 1e10:
            logger.warning(
                "Readout matrix for qubit %d is near-singular (cond=%.2e). "
                "Mitigation may be unreliable for this qubit.", q, cond
            )

        matrices.append(np.linalg.inv(A))

    return matrices


# ============================================================
# 6. READOUT MITIGATION (MARGINAL-BASED, MEMORY-EFFICIENT)
# ============================================================

def apply_readout_mitigation(
    counts: dict[str, int],
    matrices: list[np.ndarray],
    N: int,
    shots: int,
) -> dict[str, float]:
    """
    Applies local readout error correction qubit-by-qubit.

    Instead of building the full 2^N × 2^N Kronecker-product correction
    matrix (exponential memory), this function applies each qubit's 2×2
    correction matrix via tensor reshaping of the probability vector.

    Algorithm:
        1. Convert bitstring counts to a probability vector p ∈ R^{2^N}.
        2. For each qubit q, reshape p to expose the q-th axis, apply M_q,
           then reshape back.  This is equivalent to the Kronecker product
           M_{N-1} ⊗ ... ⊗ M_0 applied to p, but uses O(2^N) memory
           instead of O(4^N).

    Parameters
    ----------
    counts : dict[str, int] — raw measurement counts from sim.run().
    matrices : list[np.ndarray] — per-qubit inverse assignment matrices M_q.
    N : int — number of qubits.
    shots : int — total shots (used for normalisation).

    Returns
    -------
    dict[str, float] : bitstring → corrected probability (may be slightly
        negative due to statistical noise in calibration — known artifact
        of linear inversion; more advanced methods like M3 add non-negativity
        constraints but are outside this project's scope).
    """
    # Step 1: Build probability vector indexed by integer representation of bitstring.
    # Qiskit bitstrings are little-endian: rightmost character = qubit 0.
    probs = np.zeros(2 ** N, dtype=np.float64)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        probs[idx] = count / shots

    # Step 2: Apply each qubit's 2×2 correction via reshape trick.
    # Reshape probs to (2, 2, ..., 2) — N axes, one per qubit.
    probs = probs.reshape([2] * N)
    # Qiskit's bitstring ordering: index 0 of the reshaped array corresponds
    # to the most significant bit (leftmost character = qubit N-1).
    for q in range(N):
        # Move axis q to position 0, apply M_q as a (2,2) matrix multiply
        # along that axis, then move it back.
        probs = np.moveaxis(probs, q, 0)           # shape: (2, 2, ...) with q first
        shape_rest = probs.shape[1:]
        probs = probs.reshape(2, -1)               # shape: (2, 2^{N-1})
        probs = matrices[q] @ probs                # shape: (2, 2^{N-1})
        probs = probs.reshape((2,) + shape_rest)   # restore (2, ...)
        probs = np.moveaxis(probs, 0, q)           # move axis q back

    probs = probs.reshape(-1)  # flatten back to 2^N vector

    return {
        format(i, f"0{N}b"): float(probs[i])
        for i in range(2 ** N)
    }


# ============================================================
# 7. PARITY POST-SELECTION
# ============================================================

def parity_post_selection(
    probs: dict[str, float],
) -> dict[str, float]:
    """
    Retains only even-parity bitstrings and renormalises.

    Physical motivation: the TFIM ground state (with OBC and even N)
    has Z₂ symmetry and lies in the even-parity sector (even number
    of spin-downs / qubits in |1⟩).  Single-qubit bit-flip errors
    (the dominant noise channel) flip the parity.  Discarding odd-parity
    measurement outcomes removes the bulk of these errors at zero
    additional circuit cost — only a fraction (1 - P_even) of shots
    are wasted.

    Applied consistently to BOTH Z-basis and X-basis measurements
    (see get_energy_single_shot docstring for justification).

    Parameters
    ----------
    probs : dict[str, float] — bitstring → probability (raw or corrected).

    Returns
    -------
    dict[str, float] — renormalised probabilities over even-parity states.
        Returns input unchanged if total even-parity probability is ≤ 0.
    """
    even_parity = {b: p for b, p in probs.items() if b.count("1") % 2 == 0}
    total = sum(even_parity.values())
    if total <= 0.0:
        logger.warning("Parity post-selection: no even-parity outcomes. Returning raw.")
        return probs
    return {b: p / total for b, p in even_parity.items()}


# ============================================================
# 8. VECTORISED EXPECTATION VALUE COMPUTATION
# ============================================================

def compute_zz_energy(
    probs: dict[str, float],
    N: int,
    J: float,
) -> float:
    """
    Computes <H_ZZ> = -J * sum_{i} <Z_i Z_{i+1}> from Z-basis measurement.

    Vectorised implementation: converts all bitstrings to a 2D integer
    matrix and uses NumPy broadcasting to evaluate all ZZ pair eigenvalues
    simultaneously — ~50× faster than Python loops for N=10.

    The ZZ eigenvalue for a pair (i, i+1) in bitstring b is:
        +1 if b_i == b_{i+1}  (parallel spins: both 0 or both 1)
        -1 if b_i != b_{i+1}  (antiparallel spins)

    Note on Qiskit bitstring ordering (little-endian):
        The rightmost character of the bitstring corresponds to qubit 0.
        We reverse each bitstring before indexing so that column j of
        bit_matrix corresponds to qubit j.

    Parameters
    ----------
    probs : dict[str, float] — bitstring → probability (post-mitigation).
    N : int — number of qubits.
    J : float — ZZ coupling constant.

    Returns
    -------
    float : ZZ contribution to <H>.
    """
    if not probs:
        return 0.0

    bitstrings = list(probs.keys())
    prob_array = np.array(list(probs.values()), dtype=np.float64)
    # prob_array shape: (M,) where M = number of unique bitstrings observed

    # Build integer matrix: each row = one bitstring, each col = one qubit.
    # Reverse bitstring to correct for Qiskit's little-endian ordering.
    bit_matrix = np.array(
        [[int(c) for c in b[::-1]] for b in bitstrings],
        dtype=np.int8,
    )
    # bit_matrix shape: (M, N)
    # bit_matrix[i, q] = value of qubit q in bitstring i (0 or 1)

    # Adjacent pair comparison: columns 0..N-2 vs columns 1..N-1
    same_spin = (bit_matrix[:, :-1] == bit_matrix[:, 1:]).astype(np.float64)
    # same_spin shape: (M, N-1)
    # same_spin[i, j] = 1.0 if qubit j and j+1 have same value in bitstring i

    # ZZ eigenvalue: +1 for same spin, -1 for different
    zz_eigenvalues = 2.0 * same_spin - 1.0
    # Broadcasting: prob_array[:, np.newaxis] has shape (M, 1)
    # → weighted has shape (M, N-1)
    weighted = prob_array[:, np.newaxis] * zz_eigenvalues

    return float(-J * np.sum(weighted))


def compute_x_energy(
    probs: dict[str, float],
    N: int,
    h: float,
) -> float:
    """
    Computes <H_X> = -h * sum_{i} <X_i> from X-basis (Hadamard-rotated) measurement.

    After applying H gates to all qubits before measurement:
        measured '0' on qubit i  →  qubit was in |+⟩ state  →  X eigenvalue = +1
        measured '1' on qubit i  →  qubit was in |-⟩ state  →  X eigenvalue = -1

    So X eigenvalue for qubit i = 1 - 2 * bit_i.

    Parameters
    ----------
    probs : dict[str, float] — bitstring → probability (X-basis, post-mitigation).
    N : int — number of qubits.
    h : float — transverse field strength.

    Returns
    -------
    float : X-field contribution to <H>.
    """
    if not probs:
        return 0.0

    bitstrings = list(probs.keys())
    prob_array = np.array(list(probs.values()), dtype=np.float64)

    # Reverse for little-endian correction (same as ZZ case)
    bit_matrix = np.array(
        [[int(c) for c in b[::-1]] for b in bitstrings],
        dtype=np.int8,
    )
    # bit_matrix shape: (M, N)

    # X eigenvalue: 1 - 2*bit  maps {0→+1, 1→-1}
    x_eigenvalues = 1.0 - 2.0 * bit_matrix.astype(np.float64)
    # x_eigenvalues shape: (M, N)

    # Weight by probability and sum over all qubits and all bitstrings
    weighted = prob_array[:, np.newaxis] * x_eigenvalues

    return float(-h * np.sum(weighted))


# ============================================================
# 9. SINGLE-SHOT ENERGY EVALUATION
# ============================================================

def get_energy_single_shot(
    ansatz: QuantumCircuit,
    params: np.ndarray,
    N: int,
    J: float,
    h: float,
    sim: AerSimulator,
    shots: int,
    seed: int,
    readout_matrices: Optional[list[np.ndarray]] = None,
    use_parity: bool = False,
    zne_scale: int = 1,
) -> float:
    """
    Estimates <H> = <H_ZZ> + <H_X> for one set of parameters.

    Measurement strategy:
        Z-basis circuit: ansatz → measure all
            Used for <ZZ> terms (diagonal in Z basis).
        X-basis circuit: ansatz → H⊗N → measure all
            Hadamard rotates X eigenstates to Z basis, so measuring
            in the computational basis after H gives X expectation values.
            |+⟩ →(H)→ |0⟩ (measured 0, X eigenvalue +1)
            |-⟩ →(H)→ |1⟩ (measured 1, X eigenvalue -1)

    Both circuits submitted in a single Aer job (batched) to reduce
    Python-level job-submission overhead.

    Parity post-selection is applied to BOTH bases:
        Z-basis: post-select on even number of |1⟩ outcomes.
        X-basis: post-select on even number of |1⟩ outcomes in the
            rotated basis. After the H rotation, the X-parity structure
            of the ground state also projects onto even parity of the
            measurement outcomes, so the same filter is physically valid.

    Parameters
    ----------
    ansatz : QuantumCircuit — parameterised circuit (no measurements).
    params : np.ndarray — shape (N*L,). Variational parameters.
    N, J, h : physics parameters.
    sim : AerSimulator — noisy simulator.
    shots : int — number of measurement shots.
    seed : int — simulator random seed.
    readout_matrices : list[np.ndarray] or None — per-qubit M_q^{-1}.
    use_parity : bool — apply parity post-selection to both bases.
    zne_scale : int — ZNE noise amplification factor (odd int ≥ 1).

    Returns
    -------
    float : estimated ⟨H⟩.
    """
    # --- Build circuits ---
    qc_base = ansatz.copy()
    if zne_scale > 1:
        qc_base = apply_zne_folding(qc_base, zne_scale)

    qc_z = qc_base.copy()
    qc_z.measure_all()

    qc_x = qc_base.copy()
    qc_x.h(range(N))
    qc_x.measure_all()

    # --- Bind parameters and run both circuits in one batch ---
    bound_z = qc_z.assign_parameters(params)
    bound_x = qc_x.assign_parameters(params)

    job = sim.run([bound_z, bound_x], shots=shots, seed_simulator=seed)
    result = job.result()
    counts_z = result.get_counts(0)
    counts_x = result.get_counts(1)

    # --- Convert to probabilities ---
    probs_z: dict[str, float] = {b: c / shots for b, c in counts_z.items()}
    probs_x: dict[str, float] = {b: c / shots for b, c in counts_x.items()}

    # --- Apply readout mitigation (if requested) ---
    if readout_matrices is not None:
        probs_z = apply_readout_mitigation(counts_z, readout_matrices, N, shots)
        probs_x = apply_readout_mitigation(counts_x, readout_matrices, N, shots)

    # --- Apply parity post-selection to BOTH bases (if requested) ---
    if use_parity:
        probs_z = parity_post_selection(probs_z)
        probs_x = parity_post_selection(probs_x)

    # --- Compute expectation values ---
    e_zz = compute_zz_energy(probs_z, N, J)
    e_x = compute_x_energy(probs_x, N, h)

    return e_zz + e_x


# ============================================================
# 10. STATISTICAL ENERGY ESTIMATION
# ============================================================

def get_energy_statistics(
    ansatz: QuantumCircuit,
    params: np.ndarray,
    N: int,
    J: float,
    h: float,
    sim: AerSimulator,
    shots: int,
    base_seed: int,
    n_reps: int = CFG.n_reps,
    **kwargs,
) -> tuple[float, float]:
    """
    Estimates ⟨H⟩ as mean ± SEM over n_reps independent repetitions.

    Each repetition uses a different random seed (base_seed + rep_index),
    ensuring statistically independent shot samples.  This captures both
    shot noise and stochastic noise-model fluctuations.

    Parameters
    ----------
    ansatz, params, N, J, h, sim : as in get_energy_single_shot.
    shots : int — shots per repetition.
    base_seed : int — base random seed; rep i uses base_seed + i.
    n_reps : int — number of independent repetitions.
    **kwargs : forwarded verbatim to get_energy_single_shot.
        Accepted keys: readout_matrices, use_parity, zne_scale.
        This pattern (keyword argument forwarding) allows
        get_energy_statistics to work transparently for all
        mitigation configurations without needing to know about them.

    Returns
    -------
    mean_energy : float — best estimate of ⟨H⟩.
    sem : float — standard error of the mean = std(ddof=1) / sqrt(n_reps).
        Represents 68% confidence interval on mean_energy.
    """
    energies = np.array([
        get_energy_single_shot(
            ansatz, params, N, J, h, sim,
            shots=shots,
            seed=base_seed + rep,
            **kwargs,
        )
        for rep in range(n_reps)
    ])

    mean_energy = float(np.mean(energies))
    sem = float(np.std(energies, ddof=1) / np.sqrt(n_reps))
    return mean_energy, sem


# ============================================================
# 11. VQE OPTIMISATION
# ============================================================

def run_vqe(
    ansatz: QuantumCircuit,
    N: int,
    J: float,
    h: float,
    sim: AerSimulator,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    """
    Runs VQE optimisation with multiple random restarts.

    Optimises on raw (unmitigated) noisy energy using reduced shot count
    (CFG.shots_optim).  The optimiser only needs to find the correct basin
    in parameter space — it does not need the precision of final evaluation.
    Final reported energies always use CFG.shots_eval shots.

    Uses COBYLA (Constrained Optimisation By Linear Approximation):
        - Derivative-free: no gradient computation required.
        - Robust to noisy objective functions.
        - Standard choice for VQE on noisy simulators and hardware.

    Multiple restarts guard against barren plateaus and local minima.
    Parameters are drawn uniformly from [-π, π] at each restart.

    Parameters
    ----------
    ansatz : QuantumCircuit — parameterised ansatz.
    N, J, h : physics parameters.
    sim : AerSimulator — noisy simulator.
    seed : int — base random seed.

    Returns
    -------
    best_params : np.ndarray — optimal variational parameters θ*.
    convergence_history : list[float] — energy per iteration (last restart only).
    """
    rng = np.random.default_rng(seed)
    n_params = ansatz.num_parameters

    best_result = None
    best_energy = np.inf
    convergence_history: list[float] = []

    for restart in range(CFG.n_restarts):
        restart_seed = seed + restart * 1000
        x0 = rng.uniform(-np.pi, np.pi, n_params)
        history: list[float] = []

        def objective(p: np.ndarray) -> float:
            e = get_energy_single_shot(
                ansatz, p, N, J, h, sim,
                shots=CFG.shots_optim,
                seed=restart_seed + len(history),
            )
            history.append(e)
            return e

        result = minimize(
            objective,
            x0,
            method="COBYLA",
            options={"maxiter": CFG.maxiter, "rhobeg": CFG.rhobeg},
        )

        logger.info(
            "  Restart %d/%d: final energy = %.6f (converged=%s, nfev=%d)",
            restart + 1, CFG.n_restarts, result.fun, result.success, result.nfev,
        )

        if result.fun < best_energy:
            best_energy = result.fun
            best_result = result
            convergence_history = history

    return best_result.x, convergence_history


# ============================================================
# 12. COMPUTATIONAL COST TRACKER
# ============================================================

def count_circuit_executions(use_zne: bool) -> int:
    """
    Returns the number of circuit executions consumed per energy evaluation.

    Accounting:
        Base evaluation: 2 circuits (Z-basis + X-basis).
        ZNE: 2 circuits per additional scale factor.
        Statistical repetitions: multiplied by n_reps.

    Does not count optimisation circuits (treated as overhead shared
    across all mitigation strategies, since optimisation uses raw energy).

    Parameters
    ----------
    use_zne : bool — whether ZNE extrapolation is used.

    Returns
    -------
    int : number of circuit executions for one (mean, SEM) estimate.
    """
    n_scales = len(CFG.zne_scale_factors) if use_zne else 1
    circuits_per_rep = 2 * n_scales  # Z + X for each scale factor
    return circuits_per_rep * CFG.n_reps


# ============================================================
# 13. SINGLE COMBINATION WORKER
# ============================================================

def run_single_combination(args: tuple) -> dict:
    """
    Worker function for one (N, h, L) combination.
    Designed to run independently in a subprocess (multiprocessing).

    Each worker:
        1. Creates its own AerSimulator instance (not shared across processes).
        2. Calibrates readout matrices.
        3. Runs VQE optimisation to find θ*.
        4. Evaluates all 8 mitigation configurations with statistics.
        5. Saves result to disk immediately (crash-safe).
        6. Returns result dictionary.

    Parameters
    ----------
    args : tuple of (N, h, L, J, base_seed, results_dir)

    Returns
    -------
    dict : result for this (N, h, L) with keys:
        N, h, L, exact, convergence,
        and for each config: {config_name: {'mean': float, 'sem': float,
                                             'rel_err': float, 'n_circuits': int}}
    """
    N, h, L, J, base_seed, results_dir = args
    # Each process needs its own simulator (not picklable / not process-safe)
    sim = make_simulator(threads=1)
    seed = base_seed + N * 100 + int(h * 10) + L

    logger.info("START | N=%d  h=%.1f  L=%d", N, h, L)
    t0 = time.perf_counter()

    # --- Exact diagonalisation (classical reference) ---
    exact = get_exact_energy(N, J, h)

    # --- Readout calibration (one-time per combination) ---
    readout_mats = get_readout_matrices(N, sim, shots=CFG.shots_eval, seed=seed)

    # --- Build ansatz ---
    ansatz, _ = build_ansatz(N, L)

    # --- VQE optimisation (raw, reduced shots) ---
    logger.info("  Optimising VQE...")
    best_params, convergence = run_vqe(ansatz, N, J, h, sim, seed=seed)

    # --- Evaluate all mitigation configurations ---
    config_results: dict[str, dict] = {}

    for config_name, cfg_flags in MITIGATION_CONFIGS.items():
        use_readout = cfg_flags["use_readout"]
        use_parity = cfg_flags["use_parity"]
        use_zne = cfg_flags["use_zne"]

        n_circuits = count_circuit_executions(use_zne)

        if use_zne:
            # ZNE: evaluate at each scale factor separately, then extrapolate
            scale_means = []
            scale_sems = []
            for sf in CFG.zne_scale_factors:
                mean_sf, sem_sf = get_energy_statistics(
                    ansatz, best_params, N, J, h, sim,
                    shots=CFG.shots_eval,
                    base_seed=seed,
                    n_reps=CFG.n_reps,
                    readout_matrices=readout_mats if use_readout else None,
                    use_parity=use_parity,
                    zne_scale=sf,
                )
                scale_means.append(mean_sf)
                scale_sems.append(sem_sf)

            mean_e = zne_extrapolate(
                CFG.zne_scale_factors, np.array(scale_means)
            )
            sem_e = zne_error_propagation(
                CFG.zne_scale_factors, np.array(scale_sems)
            )
        else:
            mean_e, sem_e = get_energy_statistics(
                ansatz, best_params, N, J, h, sim,
                shots=CFG.shots_eval,
                base_seed=seed,
                n_reps=CFG.n_reps,
                readout_matrices=readout_mats if use_readout else None,
                use_parity=use_parity,
                zne_scale=1,
            )

        rel_err = abs(mean_e - exact) / abs(exact) if exact != 0 else np.nan

        config_results[config_name] = {
            "mean": mean_e,
            "sem": sem_e,
            "rel_err": rel_err,
            "n_circuits": n_circuits,
        }

        logger.info(
            "  [%s] E=%.5f ± %.5f  |  rel_err=%.4f  |  circuits=%d",
            config_name, mean_e, sem_e, rel_err, n_circuits,
        )

    elapsed = time.perf_counter() - t0
    logger.info("DONE  | N=%d  h=%.1f  L=%d  (%.1f s)", N, h, L, elapsed)

    result = {
        "N": N,
        "h": h,
        "L": L,
        "exact": exact,
        "convergence": convergence,
        "elapsed_s": elapsed,
        **config_results,
    }

    # Incremental save: write result to disk immediately
    # so progress is not lost if the run crashes mid-sweep.
    os.makedirs(results_dir, exist_ok=True)
    fname = os.path.join(results_dir, f"N{N}_h{h:.1f}_L{L}.json")
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================
# 14. MAIN BENCHMARK LOOP (PARALLEL)
# ============================================================

def run_benchmark(cfg: BenchmarkConfig = CFG) -> list[dict]:
    """
    Runs the full parameter sweep in parallel using multiprocessing.

    Each (N, h, L) combination is an independent task dispatched to a
    worker process.  Pool.map collects results in submission order.

    Parameters
    ----------
    cfg : BenchmarkConfig — frozen configuration dataclass.

    Returns
    -------
    list[dict] : all result dictionaries, one per (N, h, L) combination.
    """
    np.random.seed(cfg.seed)

    # Build flat list of all (N, h, L) combinations
    tasks = [
        (N, h, L, cfg.J, cfg.seed, cfg.results_dir)
        for N in cfg.system_sizes
        for h in cfg.h_fields
        for L in cfg.layers
    ]
    n_tasks = len(tasks)
    # Use 1 worker per logical CPU, but cap at number of tasks
    n_workers = min(os.cpu_count() or 1, n_tasks)

    logger.info(
        "Launching benchmark: %d tasks across %d workers.", n_tasks, n_workers
    )
    logger.info("Config: %s", asdict(cfg))

    if n_workers == 1:
        # Single-process fallback (easier to debug, no pickling issues)
        all_results = [run_single_combination(t) for t in tasks]
    else:
        with Pool(processes=n_workers) as pool:
            all_results = pool.map(run_single_combination, tasks)

    return all_results


# ============================================================
# 15. PLOTTING
# ============================================================

# Consistent colour and marker scheme across all plots
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

PHASE_LABELS = {0.5: "Ordered (h=0.5)", 1.0: "Critical (h=1.0)", 2.0: "Disordered (h=2.0)"}


def plot_relative_error(all_results: list[dict], cfg: BenchmarkConfig) -> None:
    """
    Grid plot: relative error |E_mit - E_exact| / |E_exact| vs N.
    Rows = h values (phases), Columns = L (ansatz depth).
    Log-scale y-axis with shaded error bands (±1 SEM propagated).
    """
    fig, axes = plt.subplots(
        len(cfg.h_fields), len(cfg.layers),
        figsize=(6 * len(cfg.layers), 4 * len(cfg.h_fields)),
        sharex=False, sharey=False,
    )
    fig.suptitle(
        "Relative Error vs System Size  |  1D TFIM VQE Mitigation Benchmark",
        fontsize=15, fontweight="bold",
    )

    for h_idx, h_val in enumerate(cfg.h_fields):
        for l_idx, l_val in enumerate(cfg.layers):
            ax = axes[h_idx, l_idx] if len(cfg.h_fields) > 1 else axes[l_idx]

            subset = sorted(
                [r for r in all_results if r["h"] == h_val and r["L"] == l_val],
                key=lambda r: r["N"],
            )
            ns = [r["N"] for r in subset]

            for config_name, sty in STYLE.items():
                rel_errs = [r[config_name]["rel_err"] for r in subset]
                sems = [r[config_name]["sem"] for r in subset]
                exact_vals = [r["exact"] for r in subset]
                # Propagate SEM to relative error: σ_{rel} ≈ σ_E / |E_exact|
                rel_sems = [s / abs(e) for s, e in zip(sems, exact_vals)]

                ax.semilogy(
                    ns, rel_errs,
                    color=sty["color"], marker=sty["marker"],
                    linestyle=sty["ls"], linewidth=sty["lw"],
                    label=config_name,
                )
                ax.fill_between(
                    ns,
                    [max(1e-6, r - s) for r, s in zip(rel_errs, rel_sems)],
                    [r + s for r, s in zip(rel_errs, rel_sems)],
                    color=sty["color"], alpha=0.15,
                )

            ax.set_xticks(ns)
            ax.grid(True, which="both", alpha=0.3, linestyle=":")
            if h_idx == 0:
                ax.set_title(f"Depth  L = {l_val}", fontsize=12)
            if l_idx == 0:
                ax.set_ylabel(f"{PHASE_LABELS[h_val]}\nRelative Error", fontsize=10)
            if h_idx == len(cfg.h_fields) - 1:
                ax.set_xlabel("System Size  N  (qubits)", fontsize=10)

    # Single shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=4,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9, framealpha=0.9,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig("plot_relative_error.pdf", bbox_inches="tight", dpi=150)
    plt.show()
    logger.info("Saved: plot_relative_error.pdf")


def plot_absolute_energy(all_results: list[dict], cfg: BenchmarkConfig) -> None:
    """
    Grid plot: absolute energy <H> vs N for all methods + exact.
    Error bars = ±1 SEM.
    """
    fig, axes = plt.subplots(
        len(cfg.h_fields), len(cfg.layers),
        figsize=(6 * len(cfg.layers), 4 * len(cfg.h_fields)),
        sharex=False, sharey=False,
    )
    fig.suptitle(
        "Absolute Ground-State Energy vs System Size  |  1D TFIM VQE",
        fontsize=15, fontweight="bold",
    )

    for h_idx, h_val in enumerate(cfg.h_fields):
        for l_idx, l_val in enumerate(cfg.layers):
            ax = axes[h_idx, l_idx] if len(cfg.h_fields) > 1 else axes[l_idx]

            subset = sorted(
                [r for r in all_results if r["h"] == h_val and r["L"] == l_val],
                key=lambda r: r["N"],
            )
            ns = [r["N"] for r in subset]
            exact_vals = [r["exact"] for r in subset]

            ax.plot(ns, exact_vals, "k--", linewidth=2, label="Exact (ED)", zorder=10)

            for config_name, sty in STYLE.items():
                means = [r[config_name]["mean"] for r in subset]
                sems = [r[config_name]["sem"] for r in subset]
                ax.errorbar(
                    ns, means, yerr=sems,
                    color=sty["color"], marker=sty["marker"],
                    linestyle=sty["ls"], linewidth=sty["lw"],
                    capsize=3, label=config_name,
                )

            ax.set_xticks(ns)
            ax.grid(True, alpha=0.3, linestyle=":")
            if h_idx == 0:
                ax.set_title(f"Depth  L = {l_val}", fontsize=12)
            if l_idx == 0:
                ax.set_ylabel(f"{PHASE_LABELS[h_val]}\n⟨H⟩", fontsize=10)
            if h_idx == len(cfg.h_fields) - 1:
                ax.set_xlabel("System Size  N  (qubits)", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=5,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9, framealpha=0.9,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig("plot_absolute_energy.pdf", bbox_inches="tight", dpi=150)
    plt.show()
    logger.info("Saved: plot_absolute_energy.pdf")


def plot_convergence(all_results: list[dict], cfg: BenchmarkConfig) -> None:
    """
    Shows VQE optimisation convergence curves for a representative subset:
    all h values at N=4, L=2.
    """
    fig, axes = plt.subplots(1, len(cfg.h_fields), figsize=(6 * len(cfg.h_fields), 4))
    fig.suptitle(
        "VQE Optimisation Convergence  (N=4, L=2)",
        fontsize=14, fontweight="bold",
    )

    target_N, target_L = 4, 2
    for ax, h_val in zip(axes, cfg.h_fields):
        subset = [
            r for r in all_results
            if r["N"] == target_N and r["L"] == target_L and r["h"] == h_val
        ]
        if not subset:
            ax.set_title(f"h={h_val} — no data")
            continue
        r = subset[0]
        history = r["convergence"]
        ax.plot(range(len(history)), history, color="#333333", linewidth=1.5)
        ax.axhline(r["exact"], color="red", linestyle="--", linewidth=1.5, label="Exact E₀")
        ax.set_title(f"{PHASE_LABELS[h_val]}", fontsize=11)
        ax.set_xlabel("Optimiser Iteration", fontsize=10)
        if h_val == cfg.h_fields[0]:
            ax.set_ylabel("Noisy Energy ⟨H⟩", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    plt.savefig("plot_convergence.pdf", bbox_inches="tight", dpi=150)
    plt.show()
    logger.info("Saved: plot_convergence.pdf")


def plot_cost_vs_error(all_results: list[dict], cfg: BenchmarkConfig) -> None:
    """
    Cost-benefit plot: circuit overhead vs relative error improvement over raw.
    One point per mitigation config per (N, h, L).
    Useful for assessing whether added cost is justified by error reduction.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(
        "Mitigation Cost vs Error Reduction  |  All (N, h, L) Combinations",
        fontsize=13, fontweight="bold",
    )

    for config_name, sty in STYLE.items():
        if config_name == "raw":
            continue
        overheads, improvements = [], []
        for r in all_results:
            raw_err = r["raw"]["rel_err"]
            mit_err = r[config_name]["rel_err"]
            if raw_err > 0 and mit_err < raw_err:
                improvement = (raw_err - mit_err) / raw_err  # fractional improvement
                overhead = r[config_name]["n_circuits"] / r["raw"]["n_circuits"]
                overheads.append(overhead)
                improvements.append(improvement)

        if overheads:
            ax.scatter(
                overheads, improvements,
                color=sty["color"], marker=sty["marker"],
                s=60, alpha=0.7, label=config_name,
            )

    ax.set_xlabel("Circuit Execution Overhead  (×  raw)", fontsize=11)
    ax.set_ylabel("Fractional Error Reduction  (raw − mit) / raw", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot_cost_vs_error.pdf", bbox_inches="tight", dpi=150)
    plt.show()
    logger.info("Saved: plot_cost_vs_error.pdf")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Required for multiprocessing on Windows and macOS (spawn context)
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    logger.info("=" * 60)
    logger.info("VQE Error Mitigation Benchmark — Simulator Only")
    logger.info("=" * 60)

    all_results = run_benchmark(CFG)

    logger.info("All combinations complete. Generating plots...")
    plot_relative_error(all_results, CFG)
    plot_absolute_energy(all_results, CFG)
    plot_convergence(all_results, CFG)
    plot_cost_vs_error(all_results, CFG)

    logger.info("Benchmark complete.")