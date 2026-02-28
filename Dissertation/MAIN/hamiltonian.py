"""
hamiltonian.py
==============
Exact diagonalisation of the 1D Transverse-Field Ising Model (TFIM).

Hamiltonian (open boundary conditions):
    H = -J * sum_{i=0}^{N-2} Z_i Z_{i+1}  -  h * sum_{i=0}^{N-1} X_i

Provides the classical reference energies against which all VQE results
are benchmarked.

Import example:
    from hamiltonian import get_exact_energy
"""

from __future__ import annotations
import numpy as np
from qiskit.quantum_info import SparsePauliOp


def get_exact_energy(N: int, J: float, h: float) -> float:
    """
    Ground state energy of the 1D TFIM via exact diagonalisation.

    Builds the 2^N x 2^N Hamiltonian matrix from Pauli strings and
    returns its minimum eigenvalue (ground state energy E_0).

    Open boundary conditions (OBC): ZZ coupling between sites i and
    i+1 for i in 0..N-2 only. No wraparound term.

    Why OBC: matches IBM hardware's linear chain topology without
    requiring SWAP gates. Finite-size effects are stronger than PBC
    but acceptable for benchmarking mitigation strategies.

    Complexity:
        Memory: O(2^N) for the dense matrix.
        Time:   O((2^N)^2) for eigvalsh (LAPACK dsyevd).
        Feasible for N <= 20 on a standard workstation.

    Parameters
    ----------
    N : int
        Number of qubits/spins. Must be >= 2.
    J : float
        ZZ coupling strength. Positive J = ferromagnetic.
    h : float
        Transverse field strength.
        h < J: ordered (ferromagnetic) phase.
        h = J: quantum critical point.
        h > J: disordered (paramagnetic) phase.

    Returns
    -------
    float
        Ground state energy E_0 (minimum eigenvalue of H).

    Raises
    ------
    ValueError
        If N < 2 (need at least one ZZ pair).
    """
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}.")

    pauli_list: list[tuple[str, complex]] = []

    # ZZ coupling terms: -J * Z_i Z_{i+1}  (N-1 terms for OBC)
    # Each term is a Pauli string with Z at positions i and i+1,
    # identity (I) everywhere else.
    # Example for N=4, i=1: label = ["I", "Z", "Z", "I"] -> "IZZI"
    for i in range(N - 1):
        label = ["I"] * N
        label[i] = "Z"
        label[i + 1] = "Z"
        pauli_list.append(("".join(label), -J))

    # Transverse field terms: -h * X_i  (N terms)
    # Each term has X at position i, identity elsewhere.
    for i in range(N):
        label = ["I"] * N
        label[i] = "X"
        pauli_list.append(("".join(label), -h))

    # Build the Hamiltonian as a SparsePauliOp (memory-efficient sparse format)
    # then convert to a dense 2^N x 2^N complex128 matrix for diagonalisation.
    H_op = SparsePauliOp.from_list(pauli_list)
    H_matrix = H_op.to_matrix()

    # eigvalsh: Hermitian eigenvalue solver (symmetric real matrix here).
    # Returns eigenvalues sorted ascending. O((2^N)^3) but fast in practice
    # due to LAPACK's divide-and-conquer algorithm.
    # Only eigenvalues returned (not eigenvectors) -- faster than eigh.
    eigenvalues = np.linalg.eigvalsh(H_matrix)

    return float(eigenvalues[0])