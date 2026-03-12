"""
ansatz.py
=========
Hardware-Efficient Ansatz (HEA) construction for the VQE.

The HEA uses only gates natively supported on IBM hardware, minimising
compilation overhead and SWAP gate insertion.

Why HEA over a physics-motivated ansatz (e.g. HVA):
    - Generic: not tailored to TFIM, making it a harder but more general test
    - Hardware-native: RY and CZ map directly to IBM's native gate set
    - Linear connectivity: CZ chain matches IBM's heavy-hex qubit topology

Import example:
    from ansatz import build_ansatz
"""

from __future__ import annotations
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def build_ansatz(N: int, L: int) -> tuple[QuantumCircuit, list[Parameter]]:
    """
    Builds a Hardware-Efficient Ansatz with N qubits and L layers.

    Structure:
        [RY(theta) on each qubit]  ->  [CZ on adjacent pairs]  repeated L times.

    Physical meaning:
        RY gates: single-qubit rotations on the Bloch sphere. Each RY(theta)
            rotates the qubit by angle theta about the Y axis. At theta=0,
            RY acts as identity (qubit stays in |0>). The RY angles are the
            variational parameters that COBYLA optimises.

        CZ gates: two-qubit entangling gates. CZ|11> = -|11>, all other
            basis states unchanged. Creates quantum correlations (entanglement)
            between neighbouring qubits. After a CZ chain, the state cannot
            be written as a product of individual qubit states -- it is a
            genuine many-body quantum state.

        Layer L=1: one RY block + one CZ chain. Correlations reach 1 step.
        Layer L=2: two RY+CZ cycles. Correlations reach 2 steps.
        Layer L=3: three cycles. Correlations reach 3 steps.
        For the TFIM ground state to be well-approximated, need L >= N/2.

    Parameters
    ----------
    N : int
        Number of qubits. Must be >= 2.
    L : int
        Number of layers. Must be >= 1.
        More layers = more expressible but deeper circuits = more noise.

    Returns
    -------
    qc : QuantumCircuit
        Parameterised circuit with N*L unbound Parameters (symbolic angles).
        No measurement gates -- added later in energy evaluation.
    params : list[Parameter]
        Ordered list of Parameter objects. params[l*N + q] corresponds to
        the RY gate on qubit q in layer l.

    Raises
    ------
    ValueError
        If N < 2 or L < 1.

    Example
    -------
    For N=4, L=2, the circuit has 8 parameters (theta_0_0 ... theta_1_3):

    Layer 0:
        q0: -[RY(t00)]-●-----------
        q1: -[RY(t01)]-●-●---------
        q2: -[RY(t02)]---●-●-------
        q3: -[RY(t03)]-----●-------

    Layer 1:
        q0: -[RY(t10)]-●-----------
        q1: -[RY(t11)]-●-●---------
        q2: -[RY(t12)]---●-●-------
        q3: -[RY(t13)]-----●-------
    """
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}.")
    if L < 1:
        raise ValueError(f"L must be >= 1, got {L}.")

    qc = QuantumCircuit(N)

    # Create N*L Parameter objects with descriptive names.
    # params[l*N + q] = the RY angle on qubit q in layer l.
    # These are SYMBOLIC -- they hold a name (e.g. "theta_0_3") but no
    # numerical value until .assign_parameters() is called.
    params: list[Parameter] = [
        Parameter(f"theta_{l}_{q}")
        for l in range(L)       # outer loop: layers 0, 1, ..., L-1
        for q in range(N)       # inner loop: qubits 0, 1, ..., N-1
    ]
    # The nested loops create params in order:
    # theta_0_0, theta_0_1, ..., theta_0_{N-1},  <- layer 0
    # theta_1_0, theta_1_1, ..., theta_1_{N-1},  <- layer 1
    # ...

    p_idx = 0  # cursor: which parameter from the list to assign next

    for _ in range(L):
        # --- RY rotation block ---
        # One RY gate per qubit, consuming one parameter each.
        # p_idx advances from 0..N-1 in layer 0, N..2N-1 in layer 1, etc.
        for q in range(N):
            qc.ry(params[p_idx], q)
            # qc.ry(angle, qubit): adds RY(angle) gate to qubit q.
            # angle is params[p_idx] -- a symbolic Parameter object.
            p_idx += 1

        # --- CZ entangling block ---
        # Linear chain: connect qubit 0-1, 1-2, ..., (N-2)-(N-1).
        # No parameters -- CZ is a fixed gate with no free angle.
        for q in range(N - 1):
            qc.cz(q, q + 1)

    return qc, params