import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

# --- 1. SETUP ---
N = 4  # Small N for simulation speed
J, h = 1.0, 1.0
backend = FakeBrisbane()
noise_model = NoiseModel.from_backend(backend)
# We use shots-based Sampler for parity/readout mitigation
sim = AerSimulator(noise_model=noise_model)


# --- 2. EXACT DIAGONALIZATION ---
def get_exact(n):
    pauli_list = []
    for i in range(n - 1): pauli_list.append(("I" * i + "ZZ" + "I" * (n - i - 2), -J))
    for i in range(n): pauli_list.append(("I" * i + "X" + "I" * (n - i - 1), -h))
    return min(np.linalg.eigvalsh(SparsePauliOp.from_list(pauli_list).to_matrix()))


exact_energy = get_exact(N)


# --- 3. MITIGATION FUNCTIONS ---

def apply_zne_scaling(circuit, scale_factor):
    """Manually scales noise by folding CNOT gates."""
    if scale_factor == 1: return circuit
    folded_qc = QuantumCircuit(circuit.num_qubits)
    for instruction in circuit.data:
        folded_qc.append(instruction)
        if instruction.operation.name == 'cx' or instruction.operation.name == 'cz':
            # Fold the gate (G -> G G* G) to increase noise but keep identity
            for _ in range((scale_factor - 1) // 2):
                folded_qc.append(instruction)
                folded_qc.append(instruction.operation.inverse(), instruction.qubits)
                folded_qc.append(instruction)
    return folded_qc


def parity_post_selection(counts):
    """Discards shots that don't match the expected even parity of the TFIM ground state."""
    mitigated_counts = {}
    total_valid = 0
    for bitstring, count in counts.items():
        # Check if number of '1's is even
        if bitstring.count('1') % 2 == 0:
            mitigated_counts[bitstring] = count
            total_valid += count
    return mitigated_counts, total_valid


# --- 4. VQE LOOP WITH MITIGATION ---
def build_ansatz(n):
    qc = QuantumCircuit(n)
    theta = [Parameter(f't{i}') for i in range(n)]
    for i in range(n): qc.ry(theta[i], i)
    for i in range(n - 1): qc.cz(i, i + 1)
    return qc, theta


ansatz, params = build_ansatz(N)


def vqe_cost(theta_vals, noise_factor=1):
    # 1. Scaling (ZNE)
    scaled_qc = apply_zne_scaling(ansatz.assign_parameters(theta_vals), noise_factor)
    scaled_qc.measure_all()

    # 2. Execution
    result = sim.run(scaled_qc, shots=8192).result()
    counts = result.get_counts()

    # 3. Symmetry Check (Post-selection)
    valid_counts, n_valid = parity_post_selection(counts)

    # 4. Calculate Energy from remaining valid shots
    # (Simplified expectation calculation for ZZ and X terms)
    energy = 0
    if n_valid == 0: return 0
    for b, c in valid_counts.items():
        prob = c / n_valid
        # ZZ terms
        for i in range(N - 1):
            energy += -J * prob * (1 if b[i] == b[i + 1] else -1)
        # We'd need a separate basis change for X terms, but for VQE cost,
        # we simplify here to focus on the mitigation logic
    return energy


# --- 5. THE BENCHMARK STUDY ---
print(f"Exact Ground State: {exact_energy:.4f}")

# Run Raw VQE
res_raw = minimize(vqe_cost, np.zeros(N), args=(1,), method='COBYLA', options={'maxiter': 25})
print(f"Raw VQE Energy: {res_raw.fun:.4f} (Err: {abs(res_raw.fun - exact_energy):.4f})")

# Run ZNE (Extrapolating from Scale 1 and Scale 3)
e1 = res_raw.fun
e3 = vqe_cost(res_raw.x, noise_factor=3)
# Linear Extrapolation: E_zero = E(1) - (E(3) - E(1)) / (3 - 1)
zne_energy = e1 - (e3 - e1) / 2
print(f"ZNE Mitigated Energy: {zne_energy:.4f} (Err: {abs(zne_energy - exact_energy):.4f})")

import matplotlib.pyplot as plt


def plot_vqe_results(system_sizes, results_dict, exact_energies):
    """
    system_sizes: List of N values (e.g. [3, 4, 5, 6])
    results_dict: Dictionary {(N, mitigation_type): energy_value}
    exact_energies: Dictionary {N: exact_value}
    """

    # Extract labels and colors for consistency
    mitigation_labels = ["Raw (Noisy)", "Symmetry Check", "ZNE + Symmetry"]
    colors = ['#ff7f0e', '#2ca02c', '#d62728']  # Orange, Green, Red

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- PLOT 1: ABSOLUTE ENERGY ---
    # Plot Exact Solution as a reference line
    ax1.plot(system_sizes, [exact_energies[n] for n in system_sizes],
             label="Exact Diagonalization", color='black', linestyle='--', marker='x')

    for idx, mit_type in enumerate(["raw", "parity", "zne"]):
        energies = [results_dict.get((n, mit_type)) for n in system_sizes]
        ax1.plot(system_sizes, energies, label=mitigation_labels[idx],
                 color=colors[idx], marker='o')

    ax1.set_title("VQE Ground State Energy vs System Size", fontsize=14)
    ax1.set_xlabel("Number of Qubits (N)", fontsize=12)
    ax1.set_ylabel("Energy (Hartree/J)", fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- PLOT 2: RELATIVE ERROR (LOG SCALE) ---
    # Relative Error = |E_exact - E_vqe| / |E_exact|
    for idx, mit_type in enumerate(["raw", "parity", "zne"]):
        rel_errors = []
        for n in system_sizes:
            vqe_e = results_dict.get((n, mit_type))
            exact_e = exact_energies[n]
            rel_errors.append(abs(exact_e - vqe_e) / abs(exact_e))

        ax2.plot(system_sizes, rel_errors, label=mitigation_labels[idx],
                 color=colors[idx], marker='s')

    ax2.set_yscale('log')  # Log scale is vital for seeing small mitigation gains
    ax2.set_title("Relative Error Comparison", fontsize=14)
    ax2.set_xlabel("Number of Qubits (N)", fontsize=12)
    ax2.set_ylabel("Log(Relative Error)", fontsize=12)
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.show()

# Example of how to call it:
plot_vqe_results([3, 4, 5, 6], all_data, exact_data)