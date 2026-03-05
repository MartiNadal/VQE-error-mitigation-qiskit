"""
energy.py
=========
Quantum energy expectation value computation.

This module handles:
    1. **Expectation values** (``compute_zz_energy``, ``compute_x_energy``)
   Pure functions converting bitstring probability dicts to Hamiltonian
   contributions.  Fully vectorised via NumPy broadcasting.

    2.  **Single-shot evaluation** (``get_energy_single_shot``)
   Runs one circuit pair (Z-basis + X-basis), applies the requested
   mitigation pipeline, and returns energy and measured parity discard.

    3. **Statistical estimation** (``get_energy_statistics``)
   Calls the single-shot evaluator n_reps times with independent seeds,
   returning mean ± SEM and mean parity discard fraction.

   4. **Optimiser interface** (``get_energy_for_optimiser``)
   Thin scalar wrapper for use inside the COBYLA objective function.

Import example:
    from energy import compute_zz_energy, compute_x_energy
    from energy import get_energy_single_shot, get_energy_statistics, get_energy_for_optimiser
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from config import CFG
from mitigation.readout import apply_readout_mitigation
from mitigation.parity  import parity_post_selection
from mitigation.zne     import apply_zne_folding

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Transpile gates to native IBM Fake.Brisbane gate set
# ─────────────────────────────────────────────────────────────────────────────

def _transpile_to_basis(
    qc: QuantumCircuit,
    basis_gates: list[str],
    optimization_level: int,
) -> QuantumCircuit:
    """
    Transpile ``qc`` to ``basis_gates`` without coupling-map constraints.

    Parameters
    ----------
    qc : QuantumCircuit
        Bound (concrete) circuit with no symbolic Parameters.
    basis_gates : list[str]
        Target gate names, e.g. ``CFG._IBM_BASIS_GATES``.
    optimization_level : int
        0 = no gate cancellation (required for folded circuits).
        1 = light optimisation (safe for the pre-fold ansatz).

    Returns
    -------
    QuantumCircuit
        Transpiled circuit using only ``basis_gates``.
    """
    return transpile(
        qc,
        basis_gates=CFG._IBM_BASIS_GATES,
        coupling_map=None,       # no topology constraint: we model noise only
        optimization_level=optimization_level,
        seed_transpiler=42,      # reproducible decomposition choices
    )


# ─────────────────────────────────────────────────────────────────────────────
# Expectation values
# ─────────────────────────────────────────────────────────────────────────────

def compute_zz_energy(
    probs: dict[str, float],
    N: int,
    J: float,
) -> float:
    """
    <H_ZZ> = -J * sum_{i} <Z_i Z_{i+1}> from Z-basis probabilities.

    ZZ eigenvalue for adjacent pair (i, i+1) in bitstring b:
        +1 if b[i] == b[i+1]  (same spin)
        -1 if b[i] != b[i+1]  (opposite spin)

    Bitstrings reversed for Qiskit little-endian convention:
        rightmost character = qubit 0. After b[::-1], column j = qubit j.

    Vectorised: builds (M, N) integer matrix and uses NumPy broadcasting
    instead of Python loops. ~50x faster for N=10.
    """
    if not probs:
        return 0.0

    bitstrings  = list(probs.keys())
    prob_array  = np.array(list(probs.values()), dtype=np.float64)
    bit_matrix  = np.array([[int(c) for c in b[::-1]] for b in bitstrings],
                            dtype=np.int8)

    same_spin      = (bit_matrix[:, :-1] == bit_matrix[:, 1:]).astype(np.float64)
    zz_eigenvalues = 2.0 * same_spin - 1.0
    weighted       = prob_array[:, np.newaxis] * zz_eigenvalues

    return float(-J * np.sum(weighted))


def compute_x_energy(
    probs: dict[str, float],
    N: int,
    h: float,
) -> float:
    """
    <H_X> = -h * sum_{i} <X_i> from X-basis (Hadamard-rotated) probabilities.

    X eigenvalue for qubit i in bitstring b (after reversal):
        +1 if b[i] == 0  (qubit in |+> state before H gate)
        -1 if b[i] == 1  (qubit in |-> state before H gate)

    Formula: 1 - 2 * bit maps {0 -> +1, 1 -> -1}.
    """
    if not probs:
        return 0.0

    bitstrings    = list(probs.keys())
    prob_array    = np.array(list(probs.values()), dtype=np.float64)
    bit_matrix    = np.array([[int(c) for c in b[::-1]] for b in bitstrings],
                              dtype=np.int8)

    x_eigenvalues = 1.0 - 2.0 * bit_matrix.astype(np.float64)
    weighted      = prob_array[:, np.newaxis] * x_eigenvalues

    return float(-h * np.sum(weighted))


# ─────────────────────────────────────────────────────────────────────────────
# Single-shot energy evaluation
# ─────────────────────────────────────────────────────────────────────────────

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
) -> tuple[float, float]:
    """
    Estimate ⟨H⟩ for one fixed parameter vector using one circuit pair.

    Two circuits are required because the TFIM Hamiltonian has two
    non-commuting measurement bases:
        H_ZZ = −J · Σ ZᵢZᵢ₊₁  →  measured in computational (Z) basis
        H_X  = −h · Σ Xᵢ       →  measured after H⊗N rotation (X basis)

    Both circuits are submitted in a single ``sim.run()`` batch call to
    avoid per-job overhead.

    Mitigation is applied in this order (order matters):
        1. ZNE gate folding  -- modifies the circuit before execution
        2. Readout correction -- corrects the measurement apparatus errors
        3. Parity post-selection -- discards symmetry-violating outcomes

    The parity discard fraction is measured directly from the shot data
    and returned alongside the energy so that the benchmark can record the
    true computational cost rather than assuming a fixed 50% discard.

    Parameters
    ----------
    ansatz : QuantumCircuit
        Parameterised circuit without measurement gates.
    params : np.ndarray
        Shape ``(N·L,)``.  Numerical values for all variational parameters,
        bound in the order they were created in ``build_ansatz``.
    N : int
        Number of qubits.
    J : float
        ZZ coupling constant.
    h : float
        Transverse field strength.
    sim : AerSimulator
        Noisy simulator instance.
    shots : int
        Measurement shots.  Use ``CFG.shots_optim`` during optimisation,
        ``CFG.shots_eval`` for final reported energies.
    seed : int
        Simulator random seed.  Different seeds → independent shot samples.
    readout_matrices : list[np.ndarray] or None
        Per-qubit inverse matrices from ``get_readout_matrices``.
        ``None`` → no readout correction applied.
    use_parity : bool
        If ``True``, apply Z₂ parity post-selection to both bases.
    zne_scale : int
        Noise amplification factor.  1 = no folding; 3 = 3× noise; 5 = 5×.

    Returns
    -------
    energy : float
        Estimated ⟨H⟩ = ⟨H_ZZ⟩ + ⟨H_X⟩.
    parity_discard : float
        Mean measured parity discard fraction across Z and X bases.
        Exactly 0.0 if ``use_parity=False``.
    """
    # ── Step 1: bind parameters BEFORE any manipulation ──────────────────────
    # Converts symbolic RY(theta_i) to concrete RY(1.234).
    # assign_parameters must come before both transpile and fold.
    qc_bound = ansatz.assign_parameters(params)

    # ── Step 2: transpile to native basis gates ───────────────────────────────
    # Decomposes ry → rz + sx + rz and cz → cx + rz (approximately).
    # After this step, every gate has a defined error in the noise model.
    # optimization_level=1: merge adjacent single-qubit gates for efficiency.
    # We do NOT re-transpile after folding (Step 3) to avoid G†G cancellation.
    qc_native = _transpile_to_basis(
        qc_bound,
        basis_gates=CFG._IBM_BASIS_GATES,
        optimization_level=1,
    )

    # ── Step 3: ZNE gate folding on the NATIVE circuit ───────────────────────
    # Fold after transpile: the folded G†G pairs are in native gates,
    # so each pair carries real noise from the model.
    # The circuit is already in basis gates — no further transpilation needed,
    # so Aer will not cancel the folded pairs.
    if zne_scale > 1:
        qc_native = apply_zne_folding(qc_native, zne_scale)

    # ── Step 4: add measurement bases and run ────────────────────────────────
    qc_z = qc_native.copy()
    qc_z.measure_all()

    qc_x = qc_native.copy()
    qc_x.h(range(N))
    qc_x.measure_all()

    # optimization_level=0: the circuits are already in native basis gates.
    # Passing 0 prevents Aer from running any further transpilation pass,
    # which is the safest option now that we have handled transpilation above.
    job = sim.run(
        [qc_z, qc_x],
        shots=shots,
        seed_simulator=seed,
        optimization_level=0,
    )
    result = job.result()
    counts_z = result.get_counts(0)
    counts_x = result.get_counts(1)

    # ── Step 5: post-processing ───────────────────────────────────────────────
    probs_z: dict[str, float] = {b: c / shots for b, c in counts_z.items()}
    probs_x: dict[str, float] = {b: c / shots for b, c in counts_x.items()}

    if readout_matrices is not None:
        probs_z = apply_readout_mitigation(counts_z, readout_matrices, N, shots)
        probs_x = apply_readout_mitigation(counts_x, readout_matrices, N, shots)

    parity_discard_z = 0.0
    parity_discard_x = 0.0
    if use_parity:
        probs_z, parity_discard_z = parity_post_selection(probs_z)
        probs_x, parity_discard_x = parity_post_selection(probs_x)

    energy = compute_zz_energy(probs_z, N, J) + compute_x_energy(probs_x, N, h)
    parity_discard = 0.5 * (parity_discard_z + parity_discard_x)

    return energy, parity_discard


# ─────────────────────────────────────────────────────────────────────────────
# Statistical estimation
# ─────────────────────────────────────────────────────────────────────────────

def get_energy_statistics(
    ansatz: QuantumCircuit,
    params: np.ndarray,
    N: int,
    J: float,
    h: float,
    sim: AerSimulator,
    shots: int,
    base_seed: int,
    n_reps: int = 10,
    **kwargs,
) -> tuple[float, float, float]:
    """
    Mean ± SEM over n_reps independent repetitions at fixed params.

    Returns
    -------
    mean_energy : float
    sem : float  (std(ddof=1) / sqrt(n_reps))
    mean_parity_discard : float
        Mean measured parity discard fraction across all reps.
        0.0 if use_parity=False.
    """
    results = [
        get_energy_single_shot(
            ansatz, params, N, J, h, sim,
            shots=shots,
            seed=base_seed + rep,
            **kwargs,
        )
        for rep in range(n_reps)
    ]

    energies  = np.array([r[0] for r in results])
    discards  = np.array([r[1] for r in results])

    mean_energy          = float(np.mean(energies))
    sem                  = float(np.std(energies, ddof=1) / np.sqrt(n_reps))
    mean_parity_discard  = float(np.mean(discards))

    return mean_energy, sem, mean_parity_discard


# ─────────────────────────────────────────────────────────────────────────────
# Optimiser objective (raw only, no mitigation, returns scalar for scipy)
# ─────────────────────────────────────────────────────────────────────────────

def get_energy_for_optimiser(
    ansatz: QuantumCircuit,
    params: np.ndarray,
    N: int,
    J: float,
    h: float,
    sim: AerSimulator,
    shots: int,
    seed: int,
) -> float:
    """
    Scalar energy for COBYLA. No mitigation, no parity discard tracking.
    Called thousands of times during optimisation; must be fast.
    """
    energy, _ = get_energy_single_shot(
        ansatz, params, N, J, h, sim, shots=shots, seed=seed,
        readout_matrices=None, use_parity=False, zne_scale=1,
    )
    return energy