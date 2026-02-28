"""
mitigation/parity.py
====================
Parity post-selection for the TFIM ground state.

Physical basis:
    The 1D TFIM Hamiltonian commutes with the Z2 parity operator
    P = product_i X_i. For even N with open boundary conditions, the
    ground state at any h > 0 has Z2 parity eigenvalue +1, corresponding
    to an even number of qubits measured in the |1> state (spin-down).

    Gate errors (particularly single-qubit bit-flip errors, the dominant
    noise channel) change the parity by +/-1. Odd-parity measurement
    outcomes are therefore signatures of errors and can be discarded.

    This technique is called "symmetry verification" in the literature:
    you verify that the measurement outcome respects a known symmetry of
    the exact ground state before including it in the expectation value.

Circuit overhead note:
    Parity adds ZERO extra circuits -- it is pure post-processing of
    existing shot data. However, it effectively wastes ~50% of shots
    (the discarded odd-parity outcomes). To compensate, you need roughly
    twice as many total shots to achieve the same statistical precision
    as the raw method. This is captured in the "effective overhead"
    metric in the cost analysis.

Import example:
    from mitigation.parity import parity_post_selection
"""

from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def parity_post_selection(
    probs: dict[str, float],
) -> tuple[dict[str, float], float]:
    """
    Retains only even-parity bitstrings, renormalises, and returns the
    measured discard fraction.

    Even parity: bitstring has an even number of "1" characters.
    Examples: "0000" (0 ones) -> kept
              "0110" (2 ones) -> kept
              "1111" (4 ones) -> kept
              "0001" (1 one)  -> discarded
              "0111" (3 ones) -> discarded

    The discard fraction is NOT assumed -- it is computed directly from
    the measurement data. In the ordered phase (h=0.5) the ground state
    is nearly ferromagnetic and most outcomes have even parity, so the
    discard fraction is small (~5-15%). In the disordered or noisy regime
    more odd-parity outcomes appear and the discard fraction grows.
    This measured value is used in the cost analysis instead of a 50%
    assumption.

    The effective shot multiplier for cost accounting is:
        multiplier = 1 / (1 - discard_fraction)
    e.g. discard_fraction=0.1 -> multiplier=1.11 (11% extra cost)
         discard_fraction=0.5 -> multiplier=2.00 (100% extra cost)

    Applied to BOTH Z-basis and X-basis measurements. The caller is
    responsible for averaging the discard fraction across both bases.

    Parameters
    ----------
    probs : dict[str, float]
        Bitstring -> probability (raw or readout-corrected).

    Returns
    -------
    filtered_probs : dict[str, float]
        Even-parity bitstrings only, renormalised to sum to 1.
        Returns input unchanged if no even-parity outcomes found.
    discard_fraction : float
        Fraction of probability weight discarded: 1 - sum(even_probs).
        Range [0, 1]. This is the MEASURED discard fraction for this
        specific circuit evaluation.
    """
    even_parity: dict[str, float] = {
        b: p
        for b, p in probs.items()
        if b.count("1") % 2 == 0
    }

    total_kept = sum(even_parity.values())
    total_all  = sum(probs.values())

    # Discard fraction: fraction of total probability weight removed.
    # We normalise by total_all (not 1.0) because after readout correction
    # probabilities may not sum exactly to 1.0.
    if total_all <= 0.0:
        discard_fraction = 0.0
    else:
        discard_fraction = float(1.0 - total_kept / total_all)

    if total_kept <= 0.0:
        logger.warning(
            "Parity post-selection: no even-parity outcomes found "
            "(discard_fraction=%.3f). Returning raw probabilities unchanged.",
            discard_fraction,
        )
        return probs, discard_fraction

    normalised = {b: p / total_kept for b, p in even_parity.items()}
    return normalised, discard_fraction