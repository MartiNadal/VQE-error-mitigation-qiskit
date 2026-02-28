"""
energy.py
=========
Quantum energy expectation value computation.

This module handles:
    1. Vectorised <ZZ> and <X> expectation values from measurement counts.
    2. Single-shot energy evaluation (one call to the simulator).
    3. Statistical energy estimation (mean +/- SEM over n_reps repetitions).

Import example:
    from energy import compute_zz_energy, compute_x_energy
    from energy import get_energy_single_shot, get_energy_statistics, get_energy_for_optimiser
"""

from __future__ import annotations
import logging
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from mitigation.readout import apply_readout_mitigation
from mitigation.parity  import parity_post_selection
from mitigation.zne     import apply_zne_folding

logger = logging.getLogger(__name__)


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
    Estimates <H> for one fixed parameter vector.

    Returns
    -------
    energy : float
        Estimated <H> = <H_ZZ> + <H_X>.
    parity_discard : float
        Mean parity discard fraction across Z and X bases.
        0.0 if use_parity=False. Otherwise the MEASURED fraction of
        probability weight discarded (not assumed).
    """
    qc_base = ansatz.copy()
    if zne_scale > 1:
        qc_base = apply_zne_folding(qc_base, zne_scale)

    qc_z = qc_base.copy()
    qc_z.measure_all()

    qc_x = qc_base.copy()
    qc_x.h(range(N))
    qc_x.measure_all()

    bound_z = qc_z.assign_parameters(params)
    bound_x = qc_x.assign_parameters(params)

    job     = sim.run([bound_z, bound_x], shots=shots, seed_simulator=seed)
    result  = job.result()
    counts_z = result.get_counts(0)
    counts_x = result.get_counts(1)

    probs_z: dict[str, float] = {b: c / shots for b, c in counts_z.items()}
    probs_x: dict[str, float] = {b: c / shots for b, c in counts_x.items()}

    if readout_matrices is not None:
        probs_z = apply_readout_mitigation(counts_z, readout_matrices, N, shots)
        probs_x = apply_readout_mitigation(counts_x, readout_matrices, N, shots)

    # Parity post-selection — returns (filtered_probs, measured_discard_fraction)
    parity_discard_z = 0.0
    parity_discard_x = 0.0
    if use_parity:
        probs_z, parity_discard_z = parity_post_selection(probs_z)
        probs_x, parity_discard_x = parity_post_selection(probs_x)

    energy           = compute_zz_energy(probs_z, N, J) + compute_x_energy(probs_x, N, h)
    parity_discard   = 0.5 * (parity_discard_z + parity_discard_x)

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