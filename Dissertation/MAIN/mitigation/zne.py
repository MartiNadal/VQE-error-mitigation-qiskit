"""
mitigation/zne.py
=================
Zero Noise Extrapolation (ZNE) via gate folding and Richardson extrapolation.

How ZNE works:
    1. Gate folding: replace each gate G with G * (G_dagger * G)^k, giving
       the same logical unitary but (2k+1)x the noise. This is the "scale"
       parameter: scale = 2k+1, so scale=1 means no folding, scale=3 means
       each gate is tripled in noise (k=1), scale=5 means 5x noise (k=2).

    2. Evaluate the observable (energy) at several noise levels:
       E(scale=1), E(scale=3), E(scale=5).

    3. Richardson extrapolation: fit a polynomial to (scale, E(scale)) and
       evaluate at scale=0 (zero-noise limit).

Physical assumption:
    The energy depends on the noise level lambda approximately as a polynomial:
    E(lambda) = E_0 + a*lambda + b*lambda^2 + ...
    where E_0 is the ideal (zero-noise) energy. With 3 scale factors (1, 3, 5),
    we fit a quadratic and extract E_0 as the constant term.

Circuit overhead:
    ZNE with 3 scale factors requires 3x as many circuit executions as raw
    (one set of Z+X circuits per scale factor). This is the dominant cost
    of ZNE and is why non-ZNE methods cluster at 1x overhead while ZNE
    methods cluster at 3x.

Import example:
    from mitigation.zne import apply_zne_folding, zne_extrapolate, zne_error_propagation
"""

from __future__ import annotations
import logging
import numpy as np
from qiskit import QuantumCircuit

logger = logging.getLogger(__name__)


def apply_zne_folding(qc: QuantumCircuit, scale: int) -> QuantumCircuit:
    """
    Gate-level noise amplification by folding: G -> G * (G_dagger * G)^k.

    For scale factor lambda (odd integer >= 1):
        - scale=1: no folding, circuit unchanged
        - scale=3: each gate G -> G * G_dagger * G  (3 applications)
        - scale=5: each gate G -> G * G_dagger * G * G_dagger * G  (5 applications)

    The folded sequence has the same logical unitary as the original
    (G_dagger * G = Identity, so the extra pairs cancel exactly). But each
    gate's noise channel is applied once per gate invocation, so the total
    noise is multiplied by `scale`.

    All gates are folded (both RY single-qubit and CZ two-qubit), ensuring
    proportional amplification of all noise sources. Folding only CZ gates
    (as some implementations do) would disproportionately amplify 2-qubit
    errors while leaving single-qubit errors unchanged.

    Parameters
    ----------
    qc : QuantumCircuit
        The circuit AFTER parameter binding. Must not be executed (no measurements
        added yet). Must be a concrete circuit (parameters already substituted).
    scale : int
        Noise amplification factor. Must be an odd integer >= 1.
        scale=1 returns a copy of qc unchanged.

    Returns
    -------
    QuantumCircuit
        Folded circuit with same logical unitary, scale-times the noise.

    Raises
    ------
    ValueError
        If scale is even or less than 1.
    """
    if scale < 1 or scale % 2 == 0:
        raise ValueError(
            f"ZNE scale must be an odd integer >= 1, got {scale}. "
            f"Valid values: 1, 3, 5, 7, ..."
        )
    if scale == 1:
        return qc.copy()  # no-op: return unchanged copy

    # Number of (G_dagger, G) pairs to append after each original gate G.
    # scale = 1 + 2*n_folds  =>  n_folds = (scale - 1) / 2
    n_folds = (scale - 1) // 2
    # Integer division: (3-1)//2=1, (5-1)//2=2, (7-1)//2=3

    folded = qc.copy()
    folded.clear()

    # qc.data: list of CircuitInstruction named tuples.
    # Each has .operation (gate object), .qubits, .clbits.
    for inst in qc.data:
        # Skip non-unitary instructions: measurements, resets, barriers.
        # These do not have unitary inverses and should not be folded.
        if inst.operation.name in ("barrier", "measure", "reset"):
            folded.append(inst)
            continue

        # Append the original gate G
        folded.append(inst)

        # Compute G_dagger (the inverse/Hermitian conjugate of G)
        try:
            inv_op = inst.operation.inverse()
            # For RY(theta): inverse is RY(-theta)
            # For CZ: self-inverse, so CZ_dagger = CZ
        except Exception:
            # Some custom gates may not have a defined inverse.
            # Log a warning and skip folding for this gate.
            logger.warning(
                "Gate '%s' has no defined inverse; skipping fold. "
                "ZNE noise amplification will be incomplete for this gate.",
                inst.operation.name,
            )
            continue

        # Append n_folds copies of (G†, G)
        for _ in range(n_folds):
            folded.append(inv_op, inst.qubits, inst.clbits)
            folded.append(inst)
        # After this loop, the sequence is: G, G_dagger, G, G_dagger, G, ...
        # Total copies = 1 + 2*n_folds = scale  (always odd, as required)

    return folded


def zne_extrapolate(
    scale_factors: tuple[int, ...],
    energies: np.ndarray,
) -> float:
    """
    Richardson extrapolation: fit polynomial to (scale, energy) and evaluate at 0.

    For 2 scale factors: linear fit   E(lambda) = E_0 + a*lambda
    For 3 scale factors: quadratic    E(lambda) = E_0 + a*lambda + b*lambda^2

    The zero-noise energy E_0 is the polynomial evaluated at lambda=0,
    which equals the constant term (coeffs[-1] from np.polyfit).

    Parameters
    ----------
    scale_factors : tuple[int, ...]
        Noise scale factors at which energies were measured, e.g. (1, 3, 5).
    energies : np.ndarray
        Shape (len(scale_factors),). Mean energy at each scale factor.

    Returns
    -------
    float
        Extrapolated zero-noise energy estimate.
    """
    lambdas = np.array(scale_factors, dtype=float)
    degree = len(scale_factors) - 1
    # With n points, fit a polynomial of degree n-1: uniquely determined.
    # coeffs[0] = highest-degree coefficient, coeffs[-1] = constant term.
    coeffs = np.polyfit(lambdas, energies, deg=degree)
    # Evaluating the polynomial at lambda=0: all terms with lambda^k vanish,
    # leaving only the constant term coeffs[-1] = E_0.
    return float(coeffs[-1])


def zne_error_propagation(
    scale_factors: tuple[int, ...],
    sems: np.ndarray,
) -> float:
    """
    Propagates SEM through Richardson extrapolation using Lagrange weights.

    Richardson extrapolation is a linear combination:
        E_0 = sum_i w_i * E(lambda_i)

    The weights w_i are the Lagrange basis polynomials evaluated at lambda=0:
        w_i = L_i(0) = product_{j != i} (0 - lambda_j) / (lambda_i - lambda_j)

    For scale_factors = (1, 3, 5):
        w_0 = (0-3)(0-5) / (1-3)(1-5) = 15/8 = 1.875
        w_1 = (0-1)(0-5) / (3-1)(3-5) = -5/4 = -1.25
        w_2 = (0-1)(0-3) / (5-1)(5-3) =  3/8 = 0.375

    Since E(lambda_i) are independent measurements with SEMs sigma_i:
        sigma_{E_0} = sqrt(sum_i w_i^2 * sigma_i^2)

    Note: |w_0| = 1.875 > 1, so the extrapolated estimate has LARGER
    uncertainty than the individual measurements. ZNE reduces systematic
    bias (noise) at the cost of increased statistical variance. This is
    the fundamental ZNE tradeoff.

    Parameters
    ----------
    scale_factors : tuple[int, ...]
        Same as in zne_extrapolate.
    sems : np.ndarray
        Shape (len(scale_factors),). SEM on mean energy at each scale.

    Returns
    -------
    float
        Propagated SEM on the extrapolated zero-noise estimate.
    """
    lambdas = np.array(scale_factors, dtype=float)
    n = len(lambdas)

    # Compute Lagrange basis weights w_i = L_i(0)
    weights = np.ones(n, dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                weights[i] *= (0.0 - lambdas[j]) / (lambdas[i] - lambdas[j])
    # weights[i] = product over j!=i of (0 - lambda_j)/(lambda_i - lambda_j)

    # Standard error propagation for a linear combination of independent
    # measurements: sigma_total = sqrt(sum_i (w_i * sigma_i)^2)
    return float(np.sqrt(np.sum((weights * sems) ** 2)))