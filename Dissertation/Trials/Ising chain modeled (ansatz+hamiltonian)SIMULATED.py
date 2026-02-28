# ================================
# Core / numerics
# ================================
import numpy as np
import matplotlib.pyplot as plt

# ================================
# Qiskit core
# ================================
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector

# ================================
# Primitives (modern execution)
# ================================
from qiskit.primitives import StatevectorEstimator, StatevectorSampler

# ================================
# Aer simulator (noise + shots)
# ================================
#from qiskit_aer import AerSimulator
#from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

# ================================
# VQE + optimizers
# ================================
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP, COBYLA, SPSA

# ================================
# Ansatz / circuit libraries
# ================================
#from qiskit.circuit.library import EfficientSU2, TwoLocal, RealAmplitudes

# ================================
# Measurement error mitigation
# ================================
#from qiskit.utils import QuantumInstance
#from qiskit.utils.mitigation import CompleteMeasFitter

# ================================
# Optional: IBM hardware / runtime
# (only if using real backend)
# ================================
#from qiskit_ibm_runtime import QiskitRuntimeService, Estimator as IBMEstimator, session




N = 5   #number of qubits
J = 1.0 #spin-spin coupling
h = 0.8 #transverse field

qc = QuantumCircuit(N)

thetas = []
for i in range(N):
    theta = Parameter(f"θ_{i}")
    thetas.append(theta)
    qc.ry(theta, i)

for i in range(N - 1):
    qc.cz(i, i+1)

pauli_terms = []

for i in range(N - 1):  #generate -JIZZI pauli string
    label = ["I"] * N
    label[i] = "Z"
    label[i+1] = "Z"
    pauli_terms.append(("".join(label), -J))

for i in range(N):  #generate -hX pauli string
    label = ["I"] * N
    label[i] = "X"
    pauli_terms.append(("".join(label), -h))


H_ising = SparsePauliOp.from_list(pauli_terms)
print(H_ising)

estimator = StatevectorEstimator()
optimizer = COBYLA(maxiter=300)

vqe = VQE(
    estimator=estimator,
    ansatz=qc,
    optimizer=optimizer
)

result = vqe.compute_minimum_eigenvalue(H_ising)

print(result)