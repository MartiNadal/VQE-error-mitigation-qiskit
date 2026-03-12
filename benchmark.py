"""
benchmark.py
============
Main benchmark sweep: evaluates all mitigation configurations across
all (N, h, L) combinations and saves results incrementally to disk.

Run via main.py, not directly.

Import example:
    from benchmark import run_benchmark, run_single_combination
"""

from __future__ import annotations
import json
import logging
import os
import time
from multiprocessing import Pool

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

from config import CFG, MITIGATION_CONFIGS
from hamiltonian import get_exact_energy
from ansatz import build_ansatz
from mitigation.readout import get_readout_matrices
from mitigation.zne import zne_extrapolate, zne_error_propagation
from energy import get_energy_statistics, get_energy_for_optimiser
from optimizer import run_vqe

logger = logging.getLogger(__name__)


def make_simulator(threads: int = CFG.max_parallel_threads_aer) -> AerSimulator:
    backend     = FakeBrisbane()
    noise_model = NoiseModel.from_backend(backend)
    return AerSimulator(noise_model=noise_model, max_parallel_threads=threads)


def count_circuit_executions_detailed(
    config_flags: dict[str, bool],
    N: int,
    measured_discard: float = 0.0,
) -> dict[str, float]:
    """
    Computes per-evaluation circuit overhead and effective shot overhead.

    Parameters
    ----------
    config_flags : dict -- use_readout, use_parity, use_zne flags
    N : int -- system size (for calibration circuit count)
    measured_discard : float
        MEASURED parity discard fraction from this (N, h, L) run.
        Range [0, 1]. Used to compute the actual shot multiplier.
        If 0.0 (parity not used), shot_multiplier = 1.0.

    Cost components
    ---------------
    1. Per-evaluation circuit overhead (relative to raw):
           non-ZNE: 1.0x  |  ZNE (3 scales): 3.0x
    2. Amortised calibration overhead:
           readout requires 2N circuits run once per combination.
           Amortised across all evaluations in the combination.
    3. Effective shot multiplier for parity:
           parity discards `measured_discard` fraction of shots.
           To obtain the same number of useful shots as raw, you need
           1/(1 - measured_discard) times as many total shots.
           This is the MEASURED multiplier -- it can be 1.05 in the
           ordered phase or 1.8 in the disordered phase.
    """
    use_zne     = config_flags["use_zne"]
    use_readout = config_flags["use_readout"]
    use_parity  = config_flags["use_parity"]

    n_scales            = len(CFG.zne_scale_factors) if use_zne else 1
    n_circuits_per_eval = 2 * n_scales * CFG.n_reps
    overhead_circuits   = float(n_scales)

    calibration_circuits = 2 * N if use_readout else 0
    # Amortise over all 8 configs * n_reps * scales evaluations per combination
    total_eval_circuits = sum(
        2 * (len(CFG.zne_scale_factors) if f["use_zne"] else 1) * CFG.n_reps
        for f in MITIGATION_CONFIGS.values()
    )
    overhead_amortised = overhead_circuits + calibration_circuits / total_eval_circuits

    # Shot multiplier: 1 / (1 - discard_fraction)
    # If measured_discard=0 (parity not used), multiplier=1.0 exactly.
    if use_parity and measured_discard > 0.0:
        shot_multiplier = 1.0 / max(1e-6, 1.0 - measured_discard)
    else:
        shot_multiplier = 1.0

    overhead_effective = overhead_amortised * shot_multiplier

    return {
        "n_circuits_per_eval":   n_circuits_per_eval,
        "overhead_circuits":     overhead_circuits,
        "overhead_amortised":    overhead_amortised,
        "shot_multiplier":       shot_multiplier,
        "measured_discard":      measured_discard,
        "overhead_effective":    overhead_effective,
    }


def run_single_combination(args: tuple) -> dict:
    N, h, L, J, base_seed, results_dir = args
    sim  = make_simulator(threads=1)
    seed = base_seed + N * 100 + int(h * 10) + L

    logger.info("START | N=%d  h=%.1f  L=%d", N, h, L)
    t0 = time.perf_counter()

    exact        = get_exact_energy(N, J, h)
    readout_mats = get_readout_matrices(N, sim, shots=CFG.shots_calibration, seed=seed)
    ansatz, _    = build_ansatz(N, L)

    logger.info("  Optimising VQE...")
    best_params, convergence_raw = run_vqe(ansatz, N, J, h, sim, seed=seed)

    # Normalise convergence history as relative error for each N
    # so convergence plots are comparable across system sizes
    convergence_rel = [
        float(abs(e - exact) / abs(exact)) if exact != 0 else float("nan")
        for e in convergence_raw
    ]

    config_results: dict[str, dict] = {}

    for config_name, cfg_flags in MITIGATION_CONFIGS.items():
        use_readout = cfg_flags["use_readout"]
        use_parity  = cfg_flags["use_parity"]
        use_zne     = cfg_flags["use_zne"]

        if use_zne:
            scale_means, scale_sems, scale_discards = [], [], []
            for sf in CFG.zne_scale_factors:
                m, s, d = get_energy_statistics(
                    ansatz, best_params, N, J, h, sim,
                    shots=CFG.shots_eval, base_seed=seed, n_reps=CFG.n_reps,
                    readout_matrices=readout_mats if use_readout else None,
                    use_parity=use_parity, zne_scale=sf,
                )
                scale_means.append(m); scale_sems.append(s); scale_discards.append(d)

            mean_e    = zne_extrapolate(CFG.zne_scale_factors, np.array(scale_means))
            sem_e     = zne_error_propagation(CFG.zne_scale_factors, np.array(scale_sems))
            mean_disc = float(np.mean(scale_discards))
        else:
            mean_e, sem_e, mean_disc = get_energy_statistics(
                ansatz, best_params, N, J, h, sim,
                shots=CFG.shots_eval, base_seed=seed, n_reps=CFG.n_reps,
                readout_matrices=readout_mats if use_readout else None,
                use_parity=use_parity, zne_scale=1,
            )

        rel_err = abs(mean_e - exact) / abs(exact) if exact != 0 else float("nan")
        cost    = count_circuit_executions_detailed(cfg_flags, N, measured_discard=mean_disc)

        config_results[config_name] = {
            "mean":             mean_e,
            "sem":              sem_e,
            "rel_err":          rel_err,
            "parity_discard":   mean_disc,   # actual measured discard, 0.0 if no parity
            "cost":             cost,
        }
        logger.info("  [%s] E=%.4f ±%.4f  rel_err=%.4f  discard=%.3f",
                    config_name, mean_e, sem_e, rel_err, mean_disc)

    elapsed = time.perf_counter() - t0
    logger.info("DONE  | N=%d  h=%.1f  L=%d  (%.1f s)", N, h, L, elapsed)

    result = {
        "N": N, "h": h, "L": L,
        "exact": exact,
        "convergence_raw": convergence_raw,
        "convergence_rel": convergence_rel,   # normalised for plotting
        "elapsed_s": elapsed,
        **config_results,
    }

    os.makedirs(results_dir, exist_ok=True)
    fname = os.path.join(results_dir, f"N{N}_h{h:.1f}_L{L}.json")
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_benchmark(cfg=CFG) -> list[dict]:
    np.random.seed(cfg.seed)
    tasks = [
        (N, h, L, cfg.J, cfg.seed, cfg.results_dir)
        for N in cfg.system_sizes
        for h in cfg.h_fields
        for L in cfg.layers
    ]
    n_workers = min(os.cpu_count() or 1, len(tasks))
    logger.info("Benchmark: %d tasks, %d workers.", len(tasks), n_workers)

    if n_workers == 1:
        return [run_single_combination(t) for t in tasks]
    with Pool(processes=n_workers) as pool:
        return pool.map(run_single_combination, tasks)


def load_results(results_dir: str = CFG.results_dir) -> list[dict]:
    all_results = []
    if not os.path.isdir(results_dir):
        logger.warning("Results directory '%s' not found.", results_dir)
        return all_results
    for fname in os.listdir(results_dir):
        if fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                all_results.append(json.load(f))
    logger.info("Loaded %d result files from '%s'.", len(all_results), results_dir)
    return all_results