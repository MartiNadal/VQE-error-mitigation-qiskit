import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

# ==========================================
# 1. CONSTANTS & PARAMETERS
# ==========================================
SYSTEM_SIZES = [2, 4, 6, 8, 10]  # Scaling N
H_FIELDS = [0.5, 1.0, 2.0]  # Disorder, Critical, Order
J = 1.0
LAYERS = [1, 2]  # Scaling Ansatz depth
SHOTS = 8192

# Setup Local Noisy Backend
backend = FakeBrisbane()
noise_model = NoiseModel.from_backend(backend)
sim = AerSimulator(noise_model=noise_model)


# ==========================================
# 2. CORE QC FUNCTIONS
# ==========================================

def get_exact_energy(N, J, h):
    """Calculates ground state energy via Exact Diagonalization (Matrix Method)."""
    pauli_list = []
    # Coupling terms: -J * Z_i Z_{i+1}
    for i in range(N - 1):
        label = ["I"] * N
        label[i], label[i + 1] = "Z", "Z"
        pauli_list.append(("".join(label), -J))
    # Field terms: -h * X_i
    for i in range(N):
        label = ["I"] * N
        label[i] = "X"
        pauli_list.append(("".join(label), -h))

    H_op = SparsePauliOp.from_list(pauli_list)
    H_matrix = H_op.to_matrix()
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    return min(eigenvalues)


def build_ansatz(N, L):
    """Creates a Hardware-Efficient Ansatz (RY + CZ chain)."""
    qc = QuantumCircuit(N)
    params = [Parameter(f'θ_{i}') for i in range(N * L)]
    p_idx = 0
    for _ in range(L):
        for i in range(N):
            qc.ry(params[p_idx], i)
            p_idx += 1
        for i in range(N - 1):
            qc.cz(i, i + 1)
    return qc, params


def get_readout_matrices(N):
    """Calibrates local readout inverse matrices for error correction."""
    matrices = []
    for q in range(N):
        c0, c1 = QuantumCircuit(N, 1), QuantumCircuit(N, 1)
        c1.x(q)
        c0.measure(q, 0)
        c1.measure(q, 0)
        counts = sim.run([c0, c1], shots=SHOTS).result().get_counts()
        A = np.zeros((2, 2))
        A[0, 0] = counts[0].get('0', 0) / SHOTS
        A[1, 0] = counts[0].get('1', 0) / SHOTS
        A[0, 1] = counts[1].get('0', 0) / SHOTS
        A[1, 1] = counts[1].get('1', 1e-10) / SHOTS  # Prevent div by zero
        matrices.append(np.linalg.inv(A))
    return matrices


def apply_readout_mitigation(counts, matrices):
    """Applies local readout correction to counts dictionary."""
    num_qubits = len(matrices)
    probs = np.zeros(2 ** num_qubits)
    for b, c in counts.items():
        probs[int(b, 2)] = c / SHOTS
    full_M = matrices[0]
    for i in range(1, num_qubits):
        full_M = np.kron(matrices[i], full_M)
    mit_probs = full_M @ probs
    return {format(i, f'0{num_qubits}b'): p for i, p in enumerate(mit_probs)}


def parity_post_selection(probs_dict):
    """Discards results with odd parity and renormalizes (Symmetry Mitigation)."""
    mitigated = {b: p for b, p in probs_dict.items() if b.count('1') % 2 == 0}
    total = sum(mitigated.values())
    if total <= 0: return {b: 0 for b in mitigated}
    return {b: p / total for b, p in mitigated.items()}


def apply_zne_folding(qc, scale):
    """Folds CZ gates to amplify noise for Zero Noise Extrapolation."""
    folded = QuantumCircuit(qc.num_qubits)
    for inst in qc.data:
        folded.append(inst)
        if inst.operation.name == 'cz' and scale == 3:
            folded.append(inst)
            folded.append(inst.operation.inverse(), inst.qubits)
            folded.append(inst)
    return folded


def get_energy(ansatz, params, N, J, h, readout_mats=None, use_parity=False, scale=1):
    """Manually calculates <H> = <ZZ> + <X> with optional mitigation."""
    # Part 1: ZZ Expectation (Z-Basis)
    qc_z = ansatz.copy()
    if scale > 1: qc_z = apply_zne_folding(qc_z, scale)
    qc_z.measure_all()
    counts_z = sim.run(qc_z.assign_parameters(params), shots=SHOTS).result().get_counts()
    p_z = {b: c / SHOTS for b, c in counts_z.items()}

    if readout_mats: p_z = apply_readout_mitigation(counts_z, readout_mats)
    if use_parity: p_z = parity_post_selection(p_z)

    e_zz = 0
    for b, p in p_z.items():
        for i in range(N - 1):
            e_zz += -J * p * (1 if b[i] == b[i + 1] else -1)

    # Part 2: X Expectation (X-Basis)
    qc_x = ansatz.copy()
    if scale > 1: qc_x = apply_zne_folding(qc_x, scale)
    qc_x.h(range(N))
    qc_x.measure_all()
    counts_x = sim.run(qc_x.assign_parameters(params), shots=SHOTS).result().get_counts()
    p_x = {b: c / SHOTS for b, c in counts_x.items()}
    if readout_mats: p_x = apply_readout_mitigation(counts_x, readout_mats)

    e_x = 0
    for b, p in p_x.items():
        for i in range(N):
            e_x += -h * p * (1 if b[i] == '0' else -1)

    return e_zz + e_x


# ==========================================
# 3. MAIN BENCHMARKING LOOP
# ==========================================
all_results = []

for N in SYSTEM_SIZES:
    readout_mats = get_readout_matrices(N)
    for h in H_FIELDS:
        exact_val = get_exact_energy(N, J, h)
        for L in LAYERS:
            print(f"Executing: N={N}, h={h}, L={L}")
            ansatz, p_objs = build_ansatz(N, L)

            # Optimization (Run on RAW noise to find best theta)
            res = minimize(lambda p: get_energy(ansatz, p, N, J, h),
                           np.zeros(N * L), method='COBYLA', options={'maxiter': 30})

            # Evaluate Benchmarks independently
            e_raw = get_energy(ansatz, res.x, N, J, h)
            e_ro = get_energy(ansatz, res.x, N, J, h, readout_mats=readout_mats)
            e_parity = get_energy(ansatz, res.x, N, J, h, use_parity=True)

            # ZNE Calculation
            e1 = e_raw
            e3 = get_energy(ansatz, res.x, N, J, h, scale=3)
            e_zne = e1 - (e3 - e1) / 2

            all_results.append({
                'N': N, 'h': h, 'L': L, 'exact': exact_val,
                'raw': e_raw, 'readout': e_ro, 'parity': e_parity, 'zne': e_zne
            })

# ==========================================
# 4. COMPREHENSIVE PLOTTING
# ==========================================

# We will create a grid of plots: one row per h value (Phase Regime)
fig, axes = plt.subplots(len(H_FIELDS), len(LAYERS), figsize=(18, 12), sharex=True)
fig.suptitle("VQE Error Mitigation Benchmarking: Relative Error vs System Size", fontsize=20)

for h_idx, h_val in enumerate(H_FIELDS):
    for l_idx, l_val in enumerate(LAYERS):
        ax = axes[h_idx, l_idx]

        # Filter data for this specific h and L
        subset = [r for r in all_results if r['h'] == h_val and r['L'] == l_val]
        ns = [r['N'] for r in subset]

        # Calculate Relative Errors: |Exact - Mitigated| / |Exact|
        err_raw = [abs(r['raw'] - r['exact']) / abs(r['exact']) for r in subset]
        err_ro = [abs(r['readout'] - r['exact']) / abs(r['exact']) for r in subset]
        err_parity = [abs(r['parity'] - r['exact']) / abs(r['exact']) for r in subset]
        err_zne = [abs(r['zne'] - r['exact']) / abs(r['exact']) for r in subset]

        ax.plot(ns, err_raw, 'ro-', label='Raw Noisy')
        ax.plot(ns, err_ro, 'bo-', label='Readout Mit')
        ax.plot(ns, err_parity, 'go-', label='Parity Mit')
        ax.plot(ns, err_zne, 'yo-', label='ZNE Mit')

        ax.set_yscale('log')
        if h_idx == 0: ax.set_title(f"Depth L={l_val}")
        if l_idx == 0: ax.set_ylabel(f"h={h_val}\nRel. Error")
        if h_idx == len(H_FIELDS) - 1: ax.set_xlabel("N Qubits")
        ax.grid(True, which="both", alpha=0.3)
        if h_idx == 0 and l_idx == 0: ax.legend()
        plt.xticks(ns)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()



#Show absolute energy with the same method
# Create a large figure: Rows = Number of h regimes, Cols = Number of Depths (L)

fig, axes = plt.subplots(len(H_FIELDS), len(LAYERS), figsize=(15, 12), sharex=True)
fig.suptitle("Absolute Ground State Energy Benchmarking", fontsize=20, y=0.95)

# Loop through each phase regime (h)
for h_idx, h_val in enumerate(H_FIELDS):
    # Loop through each ansatz depth (L)
    for l_idx, l_val in enumerate(LAYERS):

        # Select the specific subplot axis
        # If there's only one row or column, axes is 1D; otherwise it's 2D
        if len(H_FIELDS) > 1 and len(LAYERS) > 1:
            ax = axes[h_idx, l_idx]
        else:
            ax = axes[max(h_idx, l_idx)]

        # 1. Filter and sort data
        subset = [r for r in all_results if r['h'] == h_val and r['L'] == l_val]
        subset = sorted(subset, key=lambda x: x['N'])
        ns = [r['N'] for r in subset]

        # 2. Plot the Exact baseline
        ax.plot(ns, [r['exact'] for r in subset], 'k--', label='Exact (ED)', marker='x', alpha=0.7)

        # 3. Plot the Absolute Energies
        ax.plot(ns, [r['raw'] for r in subset], 'ro-', label='Raw')
        ax.plot(ns, [r['readout'] for r in subset], 'bo-', label='Readout')
        ax.plot(ns, [r['parity'] for r in subset], 'go-', label='Parity')
        ax.plot(ns, [r['zne'] for r in subset], 'yo-', label='ZNE', linewidth=2)

        # 4. Labeling and Formatting
        if h_idx == 0:
            ax.set_title(f"Depth L={l_val}", fontsize=14)
        if l_idx == 0:
            ax.set_ylabel(f"h={h_val}\nEnergy <H>", fontsize=12)
        if h_idx == len(H_FIELDS) - 1:
            ax.set_xlabel("N Qubits", fontsize=12)

        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xticks(ns)

# Create a single legend for the whole figure to keep it clean
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.92), ncol=1)

plt.tight_layout(rect=[0, 0.03, 0.9, 0.93])  # Adjust layout to make room for title/legend
plt.show()