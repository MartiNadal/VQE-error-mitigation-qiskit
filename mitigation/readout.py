"""
mitigation/readout.py
=====================
Readout error calibration and correction.

Every qubit in a real (or noisy-simulated) quantum computer has imperfect
measurement. When you prepare |0> and measure, you occasionally read "1"
(bit-flip error). These per-qubit error rates are captured in a 2x2
assignment matrix A_q:

    A_q[i, j] = P(measure i | prepared j)

So A_q[1, 0] = P(measure 1 | prepared 0) = readout error rate for |0>.

The correction: if your noisy probability vector is p_noisy, the corrected
estimate is p_corrected = M_q^{-1} * p_noisy, applied independently per qubit.

Assumption: readout errors on different qubits are independent. This is
the standard local readout mitigation assumption. Cross-talk (correlated
errors between qubits sharing readout electronics) is neglected.

Circuit overhead note:
    Calibration requires 2N additional circuits (2 per qubit: |0> and |1>).
    These are run ONCE per (N, h, L) combination and the matrices are reused
    across all 8 mitigation configs and all n_reps repetitions.
    Amortised per-evaluation overhead: 2N / (n_configs * n_reps * n_scales)
    For N=10: 20 / (8 * 10 * 1) = 0.25 extra circuits per evaluation ~ 12%.

Import example:
    from mitigation.readout import get_readout_matrices, apply_readout_mitigation
"""

from __future__ import annotations
import logging
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

logger = logging.getLogger(__name__)


def get_readout_matrices(
    N: int,
    sim: AerSimulator,
    shots: int,
    seed: int,
) -> list[np.ndarray]:
    """
    Calibrates per-qubit readout assignment matrices.

    For each qubit q, runs two circuits:
        c0: prepare |0>, measure  ->  estimates P(0|0) and P(1|0)
        c1: apply X gate, measure ->  estimates P(0|1) and P(1|1)

    All 2N circuits submitted in one batch for efficiency.

    Shot count recommendation:
        2048 shots gives ~0.3% precision on error rates (typical ~1-3%).
        This is more than sufficient. 8192 would waste ~6 seconds per N.
        Use CFG.shots_calibration = 2048 (not shots_eval = 8192).

    Parameters
    ----------
    N : int
        Number of qubits.
    sim : AerSimulator
        Noisy simulator instance.
    shots : int
        Shots per calibration circuit. Recommend 2048.
    seed : int
        Simulator random seed for reproducibility.

    Returns
    -------
    list[np.ndarray]
        Length N list. Each element is a (2, 2) float64 matrix M_q = A_q^{-1}.
        Apply to probability vectors to correct readout errors.
    """
    calibration_circuits: list[QuantumCircuit] = []

    for q in range(N):
        # c0: all qubits initialise to |0> -- just measure qubit q directly
        c0 = QuantumCircuit(N, 1)
        c0.measure(q, 0)

        # c1: flip qubit q to |1> using X gate, then measure
        c1 = QuantumCircuit(N, 1)
        c1.x(q)
        c1.measure(q, 0)

        calibration_circuits.extend([c0, c1])

    # Single batch job: all 2N circuits in one sim.run() call.
    # More efficient than 2N separate calls (avoids per-job overhead).
    job = sim.run(calibration_circuits, shots=shots, seed_simulator=seed)
    result = job.result()

    matrices: list[np.ndarray] = []
    for q in range(N):
        counts_0 = result.get_counts(2 * q)       # c0 outcomes for qubit q
        counts_1 = result.get_counts(2 * q + 1)   # c1 outcomes for qubit q

        # Build the 2x2 assignment matrix A.
        # A[i, j] = P(measure i | prepared j)
        # Column 0: outcomes when prepared in |0>
        # Column 1: outcomes when prepared in |1>
        A = np.array([
            [counts_0.get("0", 0) / shots,  counts_1.get("0", 0) / shots],
            [counts_0.get("1", 0) / shots,  counts_1.get("1", shots) / shots],
            # Note: counts_1.get("1", shots) defaults to `shots` (not 1e-10).
            # If no bit-flips occurred, "1" has count=shots and "0" is absent
            # from the dict. Defaulting to shots gives P(1|1)=1.0. Defaulting
            # to 0 or 1e-10 would give P(1|1)~0, making A near-singular.
        ], dtype=np.float64)

        # Check for near-singular matrix (can happen with extreme noise)
        cond = np.linalg.cond(A)
        if cond > 1e10:
            logger.warning(
                "Readout matrix for qubit %d is near-singular (condition number=%.2e). "
                "Calibration data may be unreliable for this qubit.",
                q, cond,
            )

        matrices.append(np.linalg.inv(A))

    return matrices


def apply_readout_mitigation(
    counts: dict[str, int],
    matrices: list[np.ndarray],
    N: int,
    shots: int,
) -> dict[str, float]:
    """
    Applies per-qubit readout correction using the tensor reshape trick.

    Mathematical goal:
        We want p_corrected = (M_{N-1} x M_{N-2} x ... x M_0) * p_noisy
        where x denotes the Kronecker product and p is the 2^N probability vector.

        Building the full 2^N x 2^N Kronecker product matrix would require
        O(4^N) memory (8 MB for N=10, 8 TB for N=20). The reshape trick
        achieves the same result in O(2^N) memory by exploiting the fact that
        each M_q acts independently on its qubit's axis.

    Algorithm:
        1. Convert bitstring counts to probability vector p of shape (2^N,).
        2. Reshape p to (2, 2, ..., 2) -- N axes, one per qubit.
        3. For each qubit q:
           a. Move axis q to position 0.
           b. Flatten remaining axes: shape (2, 2^{N-1}).
           c. Apply M_q: (2,2) @ (2, 2^{N-1}) = (2, 2^{N-1}).
           d. Restore shape and move axis back.
        4. Flatten back to (2^N,) and convert to dictionary.

        This is equivalent to applying M_q to all 2^{N-1} "slices" of the
        probability array that differ only in qubit q's value.

    Parameters
    ----------
    counts : dict[str, int]
        Raw measurement counts {bitstring: count} from sim.run().
    matrices : list[np.ndarray]
        Per-qubit inverse assignment matrices from get_readout_matrices().
    N : int
        Number of qubits.
    shots : int
        Total shots (for normalisation to probabilities).

    Returns
    -------
    dict[str, float]
        Bitstring -> corrected probability. Values may be slightly negative
        (known artifact of linear inversion with statistical noise in calibration).
        More advanced methods (M3, MTHREE) add non-negativity constraints but
        are outside this project's scope.
    """
    # Step 1: convert counts to flat probability vector indexed by int(bitstring)
    probs = np.zeros(2 ** N, dtype=np.float64)
    for bitstring, count in counts.items():
        # int(bitstring, 2): converts binary string "0101" -> integer 5
        # This gives the index into the probability vector.
        probs[int(bitstring, 2)] = count / shots

    # Step 2: reshape to N-dimensional (2, 2, ..., 2) array.
    # probs[i0, i1, ..., i_{N-1}] = P(qubit 0 in state i_{N-1}, ..., qubit N-1 in state i0)
    # (Qiskit's little-endian: rightmost bit = qubit 0 = last axis after reshape)
    probs = probs.reshape([2] * N)

    # Step 3: apply each qubit's correction matrix independently
    for q in range(N):
        # Move axis q to position 0 so we can apply M_q along it
        probs = np.moveaxis(probs, q, 0)           # shape: (2, 2, ...) with q first

        shape_rest = probs.shape[1:]               # shape of remaining N-1 axes
        probs = probs.reshape(2, -1)               # flatten: (2, 2^{N-1})
        # -1 tells NumPy to infer the size: 2^N / 2 = 2^{N-1}

        probs = matrices[q] @ probs                # (2,2) @ (2, 2^{N-1}) = (2, 2^{N-1})
        # This simultaneously corrects qubit q's readout for all 2^{N-1}
        # combinations of the other qubits' values.

        probs = probs.reshape((2,) + shape_rest)   # restore N-dimensional shape
        probs = np.moveaxis(probs, 0, q)           # move axis back to position q

    # Step 4: flatten to (2^N,) and convert to bitstring dictionary
    probs = probs.reshape(-1)

    return {
        format(i, f"0{N}b"): float(probs[i])
        # format(5, "04b") -> "0101": zero-padded binary string of length N
        for i in range(2 ** N)
    }