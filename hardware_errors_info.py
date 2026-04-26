"""
Hardware_errors_info.py
=======================
Diagnostic script for characterizing the noise profile of IBM QPUs.
Calculates the ratio between two-qubit gate error and readout error.

Rationale:
ZNE effectiveness is heavily dependent on the 'Noise Ratio' (r = P_gate / P_readout).
This script helps determine if a specific processor is gate-dominated or
readout-dominated before running the full benchmark.
"""

from qiskit_ibm_runtime.fake_provider import FakeFez, FakeBrisbane

# ── Backend Characterization ──────────────────────────────────────────────────
for name, backend in [("FakeFez", FakeFez()), ("FakeBrisbane", FakeBrisbane())]:
    props = backend.properties()

    # Extract readout errors for the first 12 qubits
    ro_errors = [props.readout_error(q) for q in range(min(12, backend.num_qubits))]

    # Extract 2-qubit gate errors (CZ/ECR/CX)
    cz_errors = [props.gate_error(g.gate, g.qubits)
                 for g in props.gates if g.gate in ('cz', 'ecr', 'cx')]

    # ── Ratio Calculation ─────────────────────────────────────────────────────
    # Computes the median ratio to avoid being skewed by 'outlier' bad qubits.
    median_ro = sorted(ro_errors)[len(ro_errors) // 2]
    median_gate = sorted(cz_errors)[len(cz_errors) // 2]
    median_ratio = median_gate / median_ro

    # ── Output ────────────────────────────────────────────────────────────────
    print(f"\n{name}:")
    print(f"  Median readout error: {median_ro:.4f}")
    print(f"  Median 2Q gate error: {median_gate:.4f}")
    print(f"  Readout/Gate noise ratio: {median_ratio:.4f}")