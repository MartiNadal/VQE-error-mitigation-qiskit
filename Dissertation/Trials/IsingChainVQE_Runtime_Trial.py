import numpy as np
from scipy.optimize import minimize
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.options import EstimatorOptions

# 1. Setup Service & Backend
service = QiskitRuntimeService()
backend = service.least_busy(simulator=False, operational=True)

# 2. Physics (Your TFIM Model - No changes here)
N = 5
J, h_field = 1.0, 0.8
qc = QuantumCircuit(N)
thetas = [Parameter(f"θ_{i}") for i in range(N)]
for i in range(N):
    qc.ry(thetas[i], i)
for i in range(N - 1):
    qc.cz(i, i+1)

pauli_terms = []
for i in range(N - 1):
    label = ["I"] * N
    label[i], label[i+1] = "Z", "Z"
    pauli_terms.append(("".join(label), -J))
for i in range(N):
    label = ["I"] * N
    label[i] = "X"
    pauli_terms.append(("".join(label), -h_field))
H_ising = SparsePauliOp.from_list(pauli_terms)

# 3. Transpile
pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
isa_qc = pm.run(qc)
isa_h_ising = H_ising.apply_layout(isa_qc.layout)

# 4. Initialize the Estimator OUTSIDE the loop
# We set mode=backend because Sessions are not available on the Open Plan
estimator = Estimator(mode=backend)

# 5. VQE Cost Function
def cost_func(params, estimator, circuit, observable):
    pub = (circuit, observable, params)
    # This will create a NEW individual job for every iteration
    job = estimator.run([pub])
    result = job.result()[0]
    energy = result.data.evs
    print(f"Current Energy: {energy}")
    return energy

# 6. Execution
initial_params = np.random.rand(N)

print(f"Starting VQE on {backend.name} (Job Mode)...")
res = minimize(
    cost_func,
    initial_params,
    args=(estimator, isa_qc, isa_h_ising),
    method="COBYLA",
    options={'maxiter': 20} # Kept small for testing in Job mode
)

print(f"VQE Complete! Minimum Energy: {res.fun}")
