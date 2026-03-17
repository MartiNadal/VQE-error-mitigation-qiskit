"""
diagnose_zne.py
===============
Diagnostic sweep for verifying that Zero-Noise Extrapolation (ZNE)
is working correctly across arbitrary (N, h, L) combinations.

Place this file in the same directory as main.py and run:
    python diagnose_zne.py

This script uses the EXACT same functions and workflow as the main benchmark
(energy.py, ansatz.py, hamiltonian.py, mitigation/zne.py) but with heavily
reduced shots/reps/restarts so each combination runs in ~30 seconds.

Configure the sweep by editing the three lists at the top of main():
    DIAG_SYSTEM_SIZES  = [2, 4, 6]
    DIAG_H_FIELDS      = [0.5, 1.0, 2.0]
    DIAG_LAYERS        = [1, 2]

All (N, h, L) combinations are run sequentially. A PASS/FAIL summary
table is printed at the end showing which combinations failed which checks.

What each check verifies
------------------------
1. CIRCUIT STRUCTURE: Gate count after folding must equal scale × base count.
   Failure → Bug in apply_zne_folding (qubit register mismatch or gate skipping).

2. NOISE MONOTONICITY: |E(scale=k) - E_exact| must increase monotonically with k.
   Failure → Folded G†G pairs are being cancelled by Aer's internal transpiler,
   or the gates carry no noise in the noise model.

3. EXTRAPOLATION: ZNE extrapolated energy must have smaller relative error than raw.
   Failure → Noise is outside the polynomial regime, or Check 2 already failed.

4. CONFIG TABLE: Runs all 8 mitigation configs and prints the relative error table
   (same structure as the main benchmark summary table).

Reduced parameters (vs full benchmark)
---------------------------------------
    shots     = 1024    (vs 8192)
    n_reps    = 3       (vs 10)
    n_restarts= 1       (vs 3)
    maxiter   = 100     (vs 300)
"""

from __future__ import annotations
import logging
import sys
import time
import numpy as np

# ── Suppress Qiskit noise (same as main.py) ──────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("qiskit").setLevel(logging.WARNING)
logging.getLogger("qiskit_aer").setLevel(logging.WARNING)

logger = logging.getLogger("diagnose_zne")

# ── Project imports — identical to main.py ────────────────────────────────────
from config import CFG, MITIGATION_CONFIGS
from hamiltonian import get_exact_energy
from ansatz import build_ansatz
from benchmark import make_simulator
from mitigation.readout import get_readout_matrices
from mitigation.zne import apply_zne_folding, zne_extrapolate, zne_error_propagation
from energy import (
    get_energy_single_shot,
    get_energy_statistics,
    _transpile_to_basis,
)

# ── Reduced run parameters ────────────────────────────────────────────────────
DIAG_SHOTS      = 1024   # reduced from CFG.shots_eval = 8192
DIAG_N_REPS     = 3      # reduced from CFG.n_reps = 10
DIAG_N_RESTARTS = 1      # reduced from CFG.n_restarts = 3
DIAG_MAXITER    = 100    # reduced from CFG.maxiter = 300
DIAG_SEED       = 42


# ─────────────────────────────────────────────────────────────────────────────
# Reduced VQE optimiser
# ─────────────────────────────────────────────────────────────────────────────
def _run_vqe_reduced(ansatz, N, L, h, sim):
    """
    Same COBYLA objective as optimizer.py but with reduced n_restarts and maxiter.
    Uses get_energy_single_shot directly — identical to the production optimiser.

    Parameters
    ----------
    ansatz : QuantumCircuit
        Parameterised ansatz from build_ansatz(N, L).
    N : int
        Number of qubits.
    L : int
        Number of ansatz layers. n_params = N * L.
    h : float
        Transverse field strength.
    sim : AerSimulator
        Noisy simulator.

    Returns
    -------
    best_params : np.ndarray
        Best variational parameters found.
    history : list[float]
        Energy at each COBYLA function evaluation (best restart only).
    """
    from scipy.optimize import minimize

    n_params     = N * L
    rng          = np.random.default_rng(DIAG_SEED)
    best_energy  = float("inf")
    best_params  = None
    best_history = []

    for restart in range(DIAG_N_RESTARTS):
        history = []

        def objective(params, _h=h, _restart=restart):
            e, _ = get_energy_single_shot(
                ansatz, params, N, CFG.J, _h, sim,
                shots=DIAG_SHOTS, seed=DIAG_SEED + _restart,
            )
            history.append(e)
            return e

        x0     = rng.uniform(0, 2 * np.pi, size=n_params)
        result = minimize(
            objective, x0,
            method="COBYLA",
            options={"maxiter": DIAG_MAXITER, "rhobeg": float(np.pi / 4)},
        )
        if result.fun < best_energy:
            best_energy  = result.fun
            best_params  = result.x.copy()
            best_history = history

    return best_params, best_history


# ─────────────────────────────────────────────────────────────────────────────
# Check 1: Circuit structure
# ─────────────────────────────────────────────────────────────────────────────
def check_circuit_structure(ansatz, best_params):
    """
    Verifies gate count scales as: gates(scale=k) == k * gates(scale=1).

    Binds and transpiles identically to get_energy_single_shot, then folds
    at each scale factor and counts unitary (foldable) instructions only.
    Measurement, reset, and barrier gates are excluded since they are not folded.
    """
    qc_bound  = ansatz.assign_parameters(best_params)
    qc_native = _transpile_to_basis(
        qc_bound,
        basis_gates=CFG._IBM_BASIS_GATES,
        optimization_level=1,
    )

    def foldable_count(qc):
        return sum(
            1 for inst in qc.data
            if inst.operation.name not in ("barrier", "measure", "reset")
        )

    base_count = foldable_count(qc_native)
    passed     = True
    rows       = []

    for scale in CFG.zne_scale_factors:
        if scale == 1:
            count = base_count
        else:
            qc_folded = apply_zne_folding(qc_native, scale)
            count     = foldable_count(qc_folded)

        expected = scale * base_count
        ok       = (count == expected)
        if not ok:
            passed = False
        rows.append((scale, count, expected, "PASS" if ok else "FAIL"))

    return passed, base_count, rows


# ─────────────────────────────────────────────────────────────────────────────
# Check 2: Noise monotonicity
# ─────────────────────────────────────────────────────────────────────────────
def check_noise_monotonicity(ansatz, best_params, N, h, sim, exact):
    """
    Energy error must increase monotonically with noise scale.

    Calls get_energy_statistics identically to benchmark.py's
    run_single_combination loop for ZNE configurations.
    """
    scale_energies = []
    scale_sems     = []

    for scale in CFG.zne_scale_factors:
        mean_e, sem_e, _ = get_energy_statistics(
            ansatz, best_params,
            N=N, J=CFG.J, h=h,
            sim=sim,
            shots=DIAG_SHOTS,
            base_seed=DIAG_SEED,
            n_reps=DIAG_N_REPS,
            zne_scale=scale,
        )
        scale_energies.append(mean_e)
        scale_sems.append(sem_e)

    errs     = [abs(e - exact) for e in scale_energies]
    monotone = all(errs[i] < errs[i + 1] for i in range(len(errs) - 1))

    return scale_energies, scale_sems, errs, monotone


# ─────────────────────────────────────────────────────────────────────────────
# Check 3: Extrapolation quality
# ─────────────────────────────────────────────────────────────────────────────
def check_extrapolation(scale_energies, scale_sems, exact):
    """
    Richardson extrapolation must improve on the raw (scale=1) measurement.
    Uses zne_extrapolate and zne_error_propagation — same functions as benchmark.py.
    """
    zne_energy  = zne_extrapolate(CFG.zne_scale_factors, np.array(scale_energies))
    zne_sem     = zne_error_propagation(CFG.zne_scale_factors, np.array(scale_sems))
    raw_rel_err = abs(scale_energies[0] - exact) / abs(exact)
    zne_rel_err = abs(zne_energy - exact) / abs(exact)
    improved    = zne_rel_err < raw_rel_err

    return zne_energy, zne_sem, raw_rel_err, zne_rel_err, improved


# ─────────────────────────────────────────────────────────────────────────────
# Check 4: All 8 mitigation configs
# ─────────────────────────────────────────────────────────────────────────────
def check_all_configs(ansatz, best_params, N, h, sim, readout_mats, exact):
    """
    Runs all 8 mitigation configurations at reduced shots/reps.
    Same loop structure as run_single_combination in benchmark.py.
    """
    results = {}

    for config_name, cfg_flags in MITIGATION_CONFIGS.items():
        use_readout = cfg_flags["use_readout"]
        use_parity  = cfg_flags["use_parity"]
        use_zne     = cfg_flags["use_zne"]

        if use_zne:
            scale_means, scale_sems_list, scale_discards = [], [], []
            for sf in CFG.zne_scale_factors:
                m, s, d = get_energy_statistics(
                    ansatz, best_params,
                    N=N, J=CFG.J, h=h,
                    sim=sim,
                    shots=DIAG_SHOTS,
                    base_seed=DIAG_SEED,
                    n_reps=DIAG_N_REPS,
                    readout_matrices=readout_mats if use_readout else None,
                    use_parity=use_parity,
                    zne_scale=sf,
                )
                scale_means.append(m)
                scale_sems_list.append(s)
                scale_discards.append(d)

            mean_e    = zne_extrapolate(CFG.zne_scale_factors, np.array(scale_means))
            sem_e     = zne_error_propagation(
                            CFG.zne_scale_factors, np.array(scale_sems_list))
            mean_disc = float(np.mean(scale_discards))
        else:
            mean_e, sem_e, mean_disc = get_energy_statistics(
                ansatz, best_params,
                N=N, J=CFG.J, h=h,
                sim=sim,
                shots=DIAG_SHOTS,
                base_seed=DIAG_SEED,
                n_reps=DIAG_N_REPS,
                readout_matrices=readout_mats if use_readout else None,
                use_parity=use_parity,
                zne_scale=1,
            )

        rel_err = abs(mean_e - exact) / abs(exact) if exact != 0 else float("nan")
        results[config_name] = {"mean": mean_e, "sem": sem_e, "rel_err": rel_err}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-combination runner
# ─────────────────────────────────────────────────────────────────────────────
def run_combination(N, h, L, sim, combo_index, n_combos):
    """
    Runs all four diagnostic checks for a single (N, h, L) combination.
    Returns a result dict for the final summary table.
    """
    label = f"N={N}  h={h:.1f}  L={L}"
    t0    = time.perf_counter()

    logger.info("")
    logger.info("█" * 64)
    logger.info("  COMBINATION %d/%d  |  %s", combo_index, n_combos, label)
    logger.info("█" * 64)

    # Setup — mirrors run_single_combination in benchmark.py exactly
    exact        = get_exact_energy(N, CFG.J, h)
    readout_mats = get_readout_matrices(
        N, sim, shots=CFG.shots_calibration, seed=DIAG_SEED
    )
    ansatz, _    = build_ansatz(N, L)

    logger.info("  Exact ground state energy: %.6f", exact)
    logger.info("  Optimising VQE (%d restart, %d maxiter)...",
                DIAG_N_RESTARTS, DIAG_MAXITER)

    best_params, _ = _run_vqe_reduced(ansatz, N, L, h, sim)

    noisy_e, _ = get_energy_single_shot(
        ansatz, best_params, N, CFG.J, h,
        sim, shots=DIAG_SHOTS, seed=DIAG_SEED,
    )
    logger.info("  VQE done | noisy E=%.6f | exact E=%.6f | rel_err=%.4f",
                noisy_e, exact, abs(noisy_e - exact) / abs(exact))

    # ── Check 1 ──────────────────────────────────────────────────────────────
    logger.info("\n  ── Check 1: Gate folding structure ──")
    c1_pass, base_count, c1_rows = check_circuit_structure(ansatz, best_params)
    logger.info("  Base foldable gate count: %d", base_count)
    for scale, count, expected, status in c1_rows:
        logger.info("    scale=%d | gates=%d | expected=%d | %s",
                    scale, count, expected, status)
    logger.info("  Check 1: %s", "PASS" if c1_pass else "FAIL")

    # ── Check 2 ──────────────────────────────────────────────────────────────
    logger.info("\n  ── Check 2: Noise monotonicity ──")
    scale_energies, scale_sems, errs, c2_pass = check_noise_monotonicity(
        ansatz, best_params, N, h, sim, exact
    )
    for sf, e, err in zip(CFG.zne_scale_factors, scale_energies, errs):
        logger.info("    scale=%d | E=%.6f | |E-E0|=%.6f | rel_err=%.4f",
                    sf, e, err, err / abs(exact))
    logger.info("  Check 2: %s", "PASS" if c2_pass else "FAIL")
    if not c2_pass:
        logger.warning(
            "  !! Errors NOT monotonically increasing → folds cancelled by Aer.\n"
            "  !! Fix: transpile([qc_z,qc_x], backend=sim, optimization_level=0)\n"
            "  !!      AFTER measurements, BEFORE sim.run() in energy.py."
        )

    # ── Check 3 ──────────────────────────────────────────────────────────────
    logger.info("\n  ── Check 3: Richardson extrapolation ──")
    zne_e, zne_sem, raw_rel_err, zne_rel_err, c3_pass = check_extrapolation(
        scale_energies, scale_sems, exact
    )
    logger.info("    Raw (scale=1): E=%.6f | rel_err=%.4f",
                scale_energies[0], raw_rel_err)
    logger.info("    ZNE extrap:    E=%.6f ± %.6f | rel_err=%.4f",
                zne_e, zne_sem, zne_rel_err)
    logger.info("    Exact:         E=%.6f", exact)
    if c3_pass:
        improvement = 100 * (raw_rel_err - zne_rel_err) / raw_rel_err
        logger.info("    ZNE improved error by %.1f%%", improvement)
    logger.info("  Check 3: %s", "PASS" if c3_pass else "FAIL")
    if not c3_pass:
        logger.warning(
            "  !! ZNE extrapolation is WORSE than raw.\n"
            "  !! If Check 2 passed, this may be physical (noise not polynomial)\n"
            "  !! and is expected for large N or deep circuits."
        )

    # ── Check 4 ──────────────────────────────────────────────────────────────
    logger.info("\n  ── Check 4: All 8 mitigation configs ──")
    config_results = check_all_configs(
        ansatz, best_params, N, h, sim, readout_mats, exact
    )

    print(f"\n  N={N}  h={h:.1f}  L={L}  |  Exact = {exact:.6f}")
    print(f"  {'Config':<16} {'Mean Energy':>14} {'± SEM':>10} {'Rel Error':>12}")
    print("  " + "-" * 54)
    for config_name, r in config_results.items():
        marker = " ◄" if config_name == "zne" else ""
        print(f"  {config_name:<16} {r['mean']:>14.6f} {r['sem']:>10.6f} "
              f"{r['rel_err']:>12.4f}{marker}")
    print(f"  {'exact':.<16} {exact:>14.6f}\n")

    elapsed = time.perf_counter() - t0
    logger.info("  Combination done in %.1f s", elapsed)

    return {
        "label":        label,
        "N": N, "h": h, "L": L,
        "exact":        exact,
        "c1_pass":      c1_pass,
        "c2_pass":      c2_pass,
        "c3_pass":      c3_pass,
        "elapsed":      elapsed,
        "raw_rel_err":  raw_rel_err,
        "zne_rel_err":  zne_rel_err,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main — configure your sweep here
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Edit these three lists to configure the diagnostic sweep ─────────────
    # Tip: start with small N and few combinations.
    # The full benchmark uses N∈{2,4,6,8,10}, h∈{0.5,1.0,2.0}, L∈{1,2,3}.
    DIAG_SYSTEM_SIZES = [2, 4, 6]        # number of qubits
    DIAG_H_FIELDS     = [0.5, 1.0, 2.0]  # transverse field strengths
    DIAG_LAYERS       = [1, 2]            # ansatz depths
    # ─────────────────────────────────────────────────────────────────────────

    combos   = [
        (N, h, L)
        for N in DIAG_SYSTEM_SIZES
        for h in DIAG_H_FIELDS
        for L in DIAG_LAYERS
    ]
    n_combos = len(combos)

    logger.info("=" * 64)
    logger.info("ZNE DIAGNOSTIC SWEEP")
    logger.info("  Combinations : %d  (N∈%s  h∈%s  L∈%s)",
                n_combos, DIAG_SYSTEM_SIZES, DIAG_H_FIELDS, DIAG_LAYERS)
    logger.info("  Shots/rep    : %d  (full benchmark: %d)", DIAG_SHOTS, CFG.shots_eval)
    logger.info("  Reps         : %d  (full benchmark: %d)", DIAG_N_REPS, CFG.n_reps)
    logger.info("  Restarts     : %d  (full benchmark: %d)", DIAG_N_RESTARTS, CFG.n_restarts)
    logger.info("  Maxiter      : %d  (full benchmark: %d)", DIAG_MAXITER, CFG.maxiter)
    logger.info("=" * 64)

    sim         = make_simulator(threads=1)
    all_results = []

    for idx, (N, h, L) in enumerate(combos, start=1):
        result = run_combination(N, h, L, sim, idx, n_combos)
        all_results.append(result)

    # ── Final summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  DIAGNOSTIC SWEEP SUMMARY")
    print("=" * 80)
    print(f"  {'Combination':<22} {'C1':>4} {'C2':>4} {'C3':>4}  "
          f"{'raw_err':>9} {'zne_err':>9} {'time(s)':>8}")
    print("  " + "-" * 64)

    all_passed = True
    for r in all_results:
        c1 = "OK" if r["c1_pass"] else "FAIL"
        c2 = "OK" if r["c2_pass"] else "FAIL"
        c3 = "OK" if r["c3_pass"] else "FAIL"
        if not (r["c1_pass"] and r["c2_pass"] and r["c3_pass"]):
            all_passed = False
        print(f"  {r['label']:<22} {c1:>4} {c2:>4} {c3:>4}  "
              f"{r['raw_rel_err']:>9.4f} {r['zne_rel_err']:>9.4f} "
              f"{r['elapsed']:>8.1f}")

    print("=" * 80)
    print("  C1=gate structure  C2=noise monotonicity  C3=ZNE improves on raw")
    print("=" * 80)

    if all_passed:
        print("\n  ALL CHECKS PASSED across all combinations.")
        print("  ZNE pipeline is working correctly. Safe to run: python main.py\n")
    else:
        print("\n  ONE OR MORE CHECKS FAILED. Fix before running the full benchmark.\n")
        c1_fails = [r["label"] for r in all_results if not r["c1_pass"]]
        c2_fails = [r["label"] for r in all_results if not r["c2_pass"]]
        c3_fails = [r["label"] for r in all_results if not r["c3_pass"]]
        if c1_fails:
            print(f"  C1 failures: {c1_fails}")
            print("    → Fix apply_zne_folding in mitigation/zne.py")
        if c2_fails:
            print(f"  C2 failures: {c2_fails}")
            print("    → Add transpile([qc_z,qc_x], backend=sim, optimization_level=0)")
            print("      after measurements, before sim.run() in energy.py")
        if c3_fails and not c2_fails:
            print(f"  C3 failures (C2 passed): {c3_fails}")
            print("    → Noise outside polynomial regime for these combos.")
            print("      Expected behaviour at large N or deep circuits — not a code bug.")
        print()


if __name__ == "__main__":
    main()