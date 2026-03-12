"""
optimizer.py
============
VQE parameter optimisation using COBYLA with multiple random restarts.

Why COBYLA:
    COBYLA (Constrained Optimisation By Linear Approximation) is a
    derivative-free method: it evaluates the objective function at a
    simplex of points and fits local linear approximations. No gradients
    are ever computed. This makes it appropriate when:
        - The objective is noisy (shot noise makes numerical gradients
          unreliable: df/dx ~ (f(x+h) - f(x-h))/(2h) amplifies noise)
        - The objective is cheap to evaluate (VQE circuits are fast on
          simulators)
        - The parameter space is moderate (~10-30 dimensions)

    Gradient-based methods (Adam, L-BFGS) are faster per iteration but
    require reliable gradient estimates, which quantum noise destroys.

Reduced shots during optimisation:
    The optimiser uses shots_optim (1024) instead of shots_eval (8192).
    Justification: COBYLA only needs to distinguish better from worse
    parameter sets, not compute precise energies. Shot noise at 1024
    shots (~3%) adds some optimisation noise but the algorithm is
    robust to this. Final reported energies always use shots_eval (8192)
    with full statistical analysis.

Import example:
    from optimizer import run_vqe
"""

from __future__ import annotations
import logging
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from config import CFG
from energy import get_energy_single_shot

logger = logging.getLogger(__name__)


def run_vqe(
    ansatz: QuantumCircuit,
    N: int,
    J: float,
    h: float,
    sim: AerSimulator,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    """
    Runs VQE optimisation with multiple random restarts.

    Multiple restarts are needed because:
        1. The energy landscape has local minima (parameter configurations
           that are locally optimal but not globally optimal).
        2. Barren plateaus: large regions where the gradient (and even the
           finite-difference approximation used by COBYLA) is exponentially
           small in N. The optimiser makes no progress in these regions.
        3. Starting from different random initial points increases the
           probability of finding the global minimum.

    The best result across all restarts is returned.

    Parameters
    ----------
    ansatz : QuantumCircuit
        Parameterised ansatz (no measurements).
    N, J, h : int, float, float
        Hamiltonian parameters.
    sim : AerSimulator
        Noisy simulator for objective evaluation.
    seed : int
        Base random seed. Restart i uses seed + i*1000 to ensure
        independent random initial parameters across restarts.

    Returns
    -------
    best_params : np.ndarray
        Shape (N*L,). Optimal variational parameters theta*.
    convergence_history : list[float]
        Energy at each COBYLA function evaluation for the best restart.
        Useful for plotting the optimisation trajectory and checking
        that the optimiser actually converged.
    """
    rng = np.random.default_rng(seed)
    # default_rng: new-style NumPy RNG (PCG64 algorithm).
    # Preferable to np.random.seed() because it creates an isolated
    # RNG object -- other code cannot accidentally affect its state.

    n_params = ansatz.num_parameters

    best_params: Optional[np.ndarray] = None
    best_energy = np.inf
    convergence_history: list[float] = []

    for restart in range(CFG.n_restarts):
        restart_seed = seed + restart * 1000
        # Spacing restarts 1000 apart ensures their RNG sequences don't
        # overlap in the first few thousand draws.

        # Random initial parameters uniformly distributed in [-pi, pi].
        # This range covers all physically distinct RY angles (RY has
        # period 4*pi but the energy landscape has period 2*pi).
        x0 = rng.uniform(-np.pi, np.pi, n_params)

        history: list[float] = []

        # Define the objective function as a closure.
        # A closure is a function that "captures" variables from its
        # enclosing scope (here: ansatz, N, J, h, sim, restart_seed, history).
        # COBYLA calls objective(p) with the current parameter vector p.
        def objective(p: np.ndarray) -> float:
            e,_ = get_energy_single_shot(
                ansatz, p, N, J, h, sim,
                shots=CFG.shots_optim,          # reduced shots for speed
                seed=restart_seed + len(history), # unique seed per eval
            )
            history.append(e)
            return e

        result = minimize(
            objective,
            x0,
            method="COBYLA",
            options={
                "maxiter": CFG.maxiter,
                # maxiter: maximum number of function evaluations.
                # COBYLA may stop earlier if it judges convergence.
                "rhobeg": CFG.rhobeg,
                # rhobeg: initial radius of the simplex in parameter space
                # (in radians here). Larger = wider initial exploration.
                # pi/4 ~ 0.785 rad is standard for angular parameters.
            },
        )

        logger.info(
            "  Restart %d/%d: E=%.6f  converged=%s  nfev=%d",
            restart + 1, CFG.n_restarts, result.fun, result.success, result.nfev,
        )

        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x
            convergence_history = history

    return best_params, convergence_history