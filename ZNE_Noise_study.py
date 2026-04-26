"""
zne_noise_study.py
==================
Standalone script demonstrating that ZNE effectiveness depends critically on
the composition of the device noise model — specifically, the ratio of gate
error to readout error.

Scientific motivation
---------------------
ZNE via gate folding amplifies gate errors (which scale with the number of
gate invocations) but is blind to readout errors (which occur at measurement
and are independent of circuit depth). In a readout-dominated regime — which
characterises FakeFez — ZNE provides little benefit because the dominant
noise source is unaffected by folding. In a gate-dominated regime, ZNE can
substantially recover the zero-noise energy.

This script runs two complementary studies:

Study A — Three discrete noise models
--------------------------------------
Three synthetic noise models with fixed total error but varying gate/readout
composition. Produces:
    - plot_zne_lambda_scaling.pdf   : E(lambda) vs lambda for each model
    - plot_zne_noise_comparison.pdf : raw vs ZNE error as grouped bar chart

    Noise models:
        "readout_dom" : p_gate_2q=0.003, p_readout=0.030  (r ≈ 0.1)
        "balanced"    : p_gate_2q=0.015, p_readout=0.015  (r = 1)
        "gate_dom"    : p_gate_2q=0.030, p_readout=0.003  (r ≈ 10)

Study B — Continuous ratio sweep
---------------------------------
10 noise models spanning r = p_gate_2q / p_readout from ~0.05 to ~20 at
fixed total noise p_total = 0.033. Produces:
    - plot_zne_improvement_vs_ratio.pdf : ZNE improvement fraction vs r (curve)

    This identifies r*, the critical ratio above which ZNE is beneficial.

VQE strategy
------------
Parameters are obtained from a *noiseless* statevector simulation. This
decouples parameter quality from noise model choice and allows a direct,
controlled comparison of ZNE behaviour across noise regimes.

Runtime estimate
----------------
Study A: N in {4}, h=1.0, L=2, n_reps=5, shots=4096 → <5 min
Study B: same sweep params, 10 noise models              → <5 min
Total: ~40-65 minutes on a modern laptop.

Outputs
-------
results_zne_study/zne_study_{N}_{h}_{L}.json         — Study A per-(N,h,L)
results_zne_study/zne_ratio_sweep_N{N}_h{h}_L{L}.json — Study B per-(N,h,L)
plots/plot_zne_lambda_scaling.pdf
plots/plot_zne_noise_comparison.pdf
plots/plot_zne_improvement_vs_ratio.pdf

Usage
-----
    python zne_noise_study.py
"""

from __future__ import annotations
import json
import logging
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    ReadoutError,
)

# ── Reuse all existing project modules unchanged ──────────────────────────────
from config import CFG, STYLE, PHASE_LABELS
from hamiltonian import get_exact_energy
from ansatz import build_ansatz
from energy import get_energy_statistics
from mitigation.zne import zne_extrapolate, zne_error_propagation
from optimizer import run_vqe

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("qiskit").setLevel(logging.WARNING)
logging.getLogger("qiskit_aer").setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Study configuration
# Intentionally reduced relative to CFG to keep runtime short.
# ─────────────────────────────────────────────────────────────────────────────

STUDY_SYSTEM_SIZES:  tuple[int, ...]   = (4, )
STUDY_H_FIELDS:      tuple[float, ...] = (1.0,)
STUDY_LAYERS:        tuple[int, ...]   = (2,)
STUDY_N_REPS:        int               = 5
STUDY_SHOTS_EVAL:    int               = 4096
STUDY_SCALE_FACTORS: tuple[int, ...]   = (1, 3, 5)
STUDY_RESULTS_DIR:   str               = "results_zne_study"
STUDY_SEED:          int               = 99


# ─────────────────────────────────────────────────────────────────────────────
# Study A — Three discrete noise models
# ─────────────────────────────────────────────────────────────────────────────

NOISE_MODELS: dict[str, dict] = {
    "readout_dom": {
        "p_gate_1q": 0.001,
        "p_gate_2q": 0.003,
        "p_readout": 0.030,
        "label":     r"Readout-dominated  ($r \approx 0.1$)",
        "color":     "#e41a1c",
    },
    "balanced": {
        "p_gate_1q": 0.003,
        "p_gate_2q": 0.015,
        "p_readout": 0.015,
        "label":     r"Balanced  ($r = 1$)",
        "color":     "#ff7f00",
    },
    "gate_dom": {
        "p_gate_1q": 0.005,
        "p_gate_2q": 0.030,
        "p_readout": 0.003,
        "label":     r"Gate-dominated  ($r \approx 10$)",
        "color":     "#4daf4a",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Study B — Continuous ratio sweep
# ─────────────────────────────────────────────────────────────────────────────

_P_TOTAL:  float = 0.033   # fixed total noise budget: p_gate_2q + p_readout
_N_RATIOS: int   = 10      # number of r values

_ratio_values = [float(x) for x in np.logspace(-1.3, 1.3, _N_RATIOS)]

RATIO_SWEEP: dict[str, dict] = {
    f"r{i:02d}": {
        "r":         r,
        "p_gate_1q": 0.001,
        "p_gate_2q": _P_TOTAL * r / (1.0 + r),
        "p_readout": _P_TOTAL * 1.0 / (1.0 + r),
        "label":     rf"$r={r:.2f}$",
        "color":     cm.RdYlGn(i / (_N_RATIOS - 1)),
    }
    for i, r in enumerate(_ratio_values)
}


# ─────────────────────────────────────────────────────────────────────────────
# Noise model construction  (shared by both studies)
# ─────────────────────────────────────────────────────────────────────────────

def make_parametric_noise_model(
    p_gate_1q: float,
    p_gate_2q: float,
    p_readout: float,
    n_qubits:  int,
) -> NoiseModel:
    """
    Build a synthetic depolarising + readout noise model.

    p_gate_1q : single-qubit depolarising probability (per SX, RZ, X gate)
    p_gate_2q : two-qubit depolarising probability (per CX gate)
    p_readout : symmetric bit-flip probability at measurement (per qubit)

    The separability of gate and readout errors is the key property that
    makes this model useful for studying ZNE: gate folding amplifies
    p_gate_2q and p_gate_1q, while p_readout is fixed at measurement.
    """
    noise_model = NoiseModel()

    err_1q = depolarizing_error(p_gate_1q, 1)
    noise_model.add_all_qubit_quantum_error(err_1q, ["sx", "x", "id"])
    noise_model.add_all_qubit_quantum_error(err_1q, ["rz"])

    err_2q = depolarizing_error(p_gate_2q, 2)
    noise_model.add_all_qubit_quantum_error(err_2q, ["cz"])

    ro_matrix = [[1.0 - p_readout, p_readout],
                 [p_readout,       1.0 - p_readout]]
    ro_err = ReadoutError(ro_matrix)
    for q in range(n_qubits):
        noise_model.add_readout_error(ro_err, [q])

    return noise_model


def make_simulator_from_noise_model(noise_model: NoiseModel) -> AerSimulator:
    return AerSimulator(
        noise_model=noise_model,
        max_parallel_threads=CFG.max_parallel_threads_aer,
    )


def make_ideal_simulator() -> AerSimulator:
    return AerSimulator(method="statevector", max_parallel_threads=1)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation  (shared by both studies)
# ─────────────────────────────────────────────────────────────────────────────

def run_zne_scaling_for_noise_model(
    noise_model_key: str,
    noise_model:     NoiseModel,
    ansatz,
    best_params:     np.ndarray,
    N:               int,
    J:               float,
    h:               float,
    seed:            int,
) -> dict:
    """
    Evaluate E(lambda) at each scale factor for one noise model.

    Returns dict with keys:
        scale_means  : list[float] — mean energy at each lambda
        scale_sems   : list[float] — SEM at each lambda
        extrapolated : float       — Richardson-extrapolated E(lambda=0)
        extrap_sem   : float       — propagated SEM on extrapolated value
        raw_energy   : float       — E(lambda=1) for convenience
    """
    sim = make_simulator_from_noise_model(noise_model)
    scale_means, scale_sems = [], []

    for sf in STUDY_SCALE_FACTORS:
        m, s, _ = get_energy_statistics(
            ansatz, best_params, N, J, h, sim,
            shots=STUDY_SHOTS_EVAL,
            base_seed=seed,
            n_reps=STUDY_N_REPS,
            readout_matrices=None,
            use_parity=False,
            zne_scale=sf,
        )
        scale_means.append(m)
        scale_sems.append(s)
        logger.info(
            "    [%s] lambda=%d  E=%.5f ± %.5f",
            noise_model_key, sf, m, s,
        )

    extrap     = zne_extrapolate(STUDY_SCALE_FACTORS, np.array(scale_means))
    extrap_sem = zne_error_propagation(STUDY_SCALE_FACTORS, np.array(scale_sems))

    return {
        "noise_model_key": noise_model_key,
        "scale_means":     [float(x) for x in scale_means],
        "scale_sems":      [float(x) for x in scale_sems],
        "scale_factors":   list(STUDY_SCALE_FACTORS),
        "extrapolated":    float(extrap),
        "extrap_sem":      float(extrap_sem),
        "raw_energy":      float(scale_means[0]),
    }


def _get_noiseless_params(
    N: int, h: float, L: int, seed: int
) -> tuple:
    """Run noiseless VQE, return (ansatz, best_params, exact)."""
    ideal_sim  = make_ideal_simulator()
    exact      = get_exact_energy(N, CFG.J, h)
    ansatz, _  = build_ansatz(N, L)
    logger.info("  Running noiseless VQE...")
    t0 = time.perf_counter()
    best_params, _ = run_vqe(ansatz, N, CFG.J, h, ideal_sim, seed=seed)
    logger.info("  Noiseless VQE done in %.1f s", time.perf_counter() - t0)
    return ansatz, best_params, exact


# ─────────────────────────────────────────────────────────────────────────────
# Study A — run / load
# ─────────────────────────────────────────────────────────────────────────────

def run_zne_noise_study() -> list[dict]:
    """
    Study A: evaluate ZNE scaling under the three discrete noise models.
    Saves to results_zne_study/zne_study_N{N}_h{h:.1f}_L{L}.json.
    """
    os.makedirs(STUDY_RESULTS_DIR, exist_ok=True)
    np.random.seed(STUDY_SEED)
    all_results = []

    for N in STUDY_SYSTEM_SIZES:
        for h in STUDY_H_FIELDS:
            for L in STUDY_LAYERS:
                seed = STUDY_SEED + N * 100 + int(h * 10) + L
                logger.info("=" * 60)
                logger.info("STUDY A | N=%d  h=%.1f  L=%d", N, h, L)

                ansatz, best_params, exact = _get_noiseless_params(N, h, L, seed)

                nm_results = {}
                for nm_key, nm_cfg in NOISE_MODELS.items():
                    logger.info("  Noise model: %s", nm_key)
                    nm = make_parametric_noise_model(
                        p_gate_1q=nm_cfg["p_gate_1q"],
                        p_gate_2q=nm_cfg["p_gate_2q"],
                        p_readout=nm_cfg["p_readout"],
                        n_qubits=N,
                    )
                    nm_results[nm_key] = run_zne_scaling_for_noise_model(
                        nm_key, nm, ansatz, best_params, N, CFG.J, h, seed,
                    )

                result = {"N": N, "h": h, "L": L, "exact": float(exact),
                          **nm_results}

                fname = os.path.join(
                    STUDY_RESULTS_DIR, f"zne_study_N{N}_h{h:.1f}_L{L}.json"
                )
                with open(fname, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info("  Saved: %s", fname)
                all_results.append(result)

    return all_results


def load_zne_study_results(
    results_dir: str = STUDY_RESULTS_DIR,
) -> list[dict]:
    """Load all Study A JSON files."""
    results = []
    if not os.path.isdir(results_dir):
        logger.warning("Directory '%s' not found.", results_dir)
        return results
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("zne_study_") and fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                results.append(json.load(f))
    logger.info("Loaded %d Study A result files.", len(results))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Study B — run / load
# ─────────────────────────────────────────────────────────────────────────────

def run_zne_ratio_sweep() -> list[dict]:
    """
    Study B: sweep the gate-to-readout error ratio r at fixed total noise
    p_total = p_gate_2q + p_readout = 0.033. VQE parameters are obtained
    once per (N, h, L) from a noiseless simulation and reused across all
    ratio values.

    Saves to results_zne_study/zne_ratio_sweep_N{N}_h{h:.1f}_L{L}.json.
    """
    os.makedirs(STUDY_RESULTS_DIR, exist_ok=True)
    np.random.seed(STUDY_SEED + 1)
    all_results = []

    for N in STUDY_SYSTEM_SIZES:
        for h in STUDY_H_FIELDS:
            for L in STUDY_LAYERS:
                seed = STUDY_SEED + 1 + N * 100 + int(h * 10) + L
                logger.info("=" * 60)
                logger.info("STUDY B | N=%d  h=%.1f  L=%d", N, h, L)

                ansatz, best_params, exact = _get_noiseless_params(N, h, L, seed)

                ratio_results = {}
                for rk, rcfg in RATIO_SWEEP.items():
                    logger.info(
                        "  r=%.3f  (p_gate_2q=%.4f, p_readout=%.4f)",
                        rcfg["r"], rcfg["p_gate_2q"], rcfg["p_readout"],
                    )
                    nm = make_parametric_noise_model(
                        p_gate_1q=rcfg["p_gate_1q"],
                        p_gate_2q=rcfg["p_gate_2q"],
                        p_readout=rcfg["p_readout"],
                        n_qubits=N,
                    )
                    entry = run_zne_scaling_for_noise_model(
                        rk, nm, ansatz, best_params, N, CFG.J, h, seed,
                    )
                    entry["r"] = rcfg["r"]   # store ratio value in JSON
                    ratio_results[rk] = entry

                result = {"N": N, "h": h, "L": L, "exact": float(exact),
                          **ratio_results}

                fname = os.path.join(
                    STUDY_RESULTS_DIR,
                    f"zne_ratio_sweep_N{N}_h{h:.1f}_L{L}.json",
                )
                with open(fname, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info("  Saved: %s", fname)
                all_results.append(result)

    return all_results


def load_zne_ratio_sweep_results(
    results_dir: str = STUDY_RESULTS_DIR,
) -> list[dict]:
    """Load all Study B JSON files."""
    results = []
    if not os.path.isdir(results_dir):
        logger.warning("Directory '%s' not found.", results_dir)
        return results
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("zne_ratio_sweep_") and fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                results.append(json.load(f))
    logger.info("Loaded %d Study B result files.", len(results))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — E(λ) vs λ  (Study A)
# ─────────────────────────────────────────────────────────────────────────────

def plot_zne_lambda_scaling(
    study_results: list[dict],
    N:     int   = 6,
    h:     float = 1.0,
    L:     int   = 2,
    fname: str   = "plot_zne_lambda_scaling.pdf",
) -> None:
    """
    Three-panel figure showing E(lambda) vs lambda for each noise model.

    Each panel shows measured points ±1 SEM, Richardson quadratic fit,
    extrapolated value at lambda=0, and exact diagonalisation reference.
    Demonstrates that ZNE mechanically works and that its effectiveness
    depends on the gate/readout noise ratio.
    """
    result = next(
        (r for r in study_results if r["N"] == N and r["h"] == h and r["L"] == L),
        None,
    )
    if result is None:
        logger.error(
            "No Study A result for N=%d, h=%.1f, L=%d. Run run_zne_noise_study() first.",
            N, h, L,
        )
        return

    exact    = result["exact"]
    lambdas  = np.array(STUDY_SCALE_FACTORS, dtype=float)
    n_models = len(NOISE_MODELS)

    fig, axes = plt.subplots(1, n_models, figsize=(5.0 * n_models, 5), sharey=False)
    fig.suptitle(
        rf"ZNE Scaling: $E(\lambda)$ vs Noise Scale Factor  |  N={N},  $h$={h},  L={L}",
        fontsize=13, fontweight="bold",
    )

    lambda_dense = np.linspace(0.0, max(lambdas) + 0.5, 200)

    for ax, (nm_key, nm_cfg) in zip(axes, NOISE_MODELS.items()):
        nm_result  = result[nm_key]
        means      = np.array(nm_result["scale_means"])
        sems       = np.array(nm_result["scale_sems"])
        extrap     = nm_result["extrapolated"]
        extrap_sem = nm_result["extrap_sem"]
        raw_energy = nm_result["raw_energy"]
        col        = nm_cfg["color"]

        # Richardson polynomial fit
        degree    = len(lambdas) - 1
        coeffs    = np.polyfit(lambdas, means, deg=degree)
        fit_curve = np.polyval(coeffs, lambda_dense)
        ax.plot(lambda_dense, fit_curve,
                color=col, linestyle="-", linewidth=1.8, alpha=0.7,
                label="Richardson fit", zorder=2)

        # Measured E(lambda) points
        ax.errorbar(lambdas, means, yerr=sems,
                    fmt="o", color=col, markersize=7, linewidth=1.5,
                    capsize=4, capthick=1.5,
                    label=r"$E(\lambda)$ ± 1 SEM", zorder=4)

        # Extrapolated value at lambda=0
        ax.errorbar([0.0], [extrap], yerr=[extrap_sem],
                    fmt="*", color=col, markersize=14, linewidth=1.5,
                    capsize=4, capthick=1.5,
                    label=rf"$E_\mathrm{{ZNE}}$ = {extrap:.4f}", zorder=5)

        # Exact reference
        ax.axhline(exact, color="black", linestyle="--", linewidth=1.5,
                   label=rf"Exact: {exact:.4f}", zorder=3)

        # Relative error annotations
        rel_err_raw = abs(raw_energy - exact) / abs(exact)
        rel_err_zne = abs(extrap     - exact) / abs(exact)
        ax.text(
            0.97, 0.97,
            rf"$\epsilon_\mathrm{{raw}}$ = {rel_err_raw:.3f}"
            "\n"
            rf"$\epsilon_\mathrm{{ZNE}}$ = {rel_err_zne:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.set_xlabel(r"Noise scale factor  $\lambda$", fontsize=11)
        ax.set_ylabel(r"$\langle H \rangle$  (energy)", fontsize=11)
        ax.set_title(nm_cfg["label"], fontsize=11)
        ax.set_xticks(list(lambdas) + [0])
        ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()
    os.makedirs(CFG.plots_dir, exist_ok=True)
    out = os.path.join(CFG.plots_dir, fname)
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.show()
    logger.info("Saved: %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Grouped bar chart  (Study A)
# ─────────────────────────────────────────────────────────────────────────────

def plot_zne_noise_comparison(
    study_results: list[dict],
    fname: str = "plot_zne_noise_comparison.pdf",
) -> None:
    """
    Grouped bar chart comparing raw vs ZNE relative error for each noise
    model, with one group per (N, h, L) combination.

    Error bars show ±1 SEM from get_energy_statistics (statistical shot
    noise), propagated through Richardson extrapolation for ZNE bars.
    Solid bars = raw, hatched bars = ZNE.
    """
    if not study_results:
        logger.error("No Study A results to plot.")
        return

    nm_keys  = list(NOISE_MODELS.keys())
    n_models = len(nm_keys)

    study_results_sorted = sorted(
        study_results, key=lambda r: (r["N"], r["h"], r["L"])
    )
    n_combos = len(study_results_sorted)
    x_labels = [
        f"N={r['N']}, h={r['h']}, L={r['L']}"
        for r in study_results_sorted
    ]

    n_bars_per_group = 2 * n_models
    bar_width        = 0.8 / n_bars_per_group
    x                = np.arange(n_combos)

    fig, ax = plt.subplots(figsize=(max(8, 3.5 * n_combos), 5.5))
    fig.suptitle(
        "ZNE Benefit vs Noise Model Composition\n"
        r"($\epsilon = |E - E_\mathrm{exact}| / |E_\mathrm{exact}|$, "
        "error bars = ±1 SEM from shot statistics)",
        fontsize=12, fontweight="bold",
    )

    legend_handles = []

    for nm_idx, nm_key in enumerate(nm_keys):
        nm_cfg = NOISE_MODELS[nm_key]
        col    = nm_cfg["color"]

        raw_vals, raw_errs = [], []
        zne_vals, zne_errs = [], []

        for r in study_results_sorted:
            exact   = r["exact"]
            nm_r    = r[nm_key]
            raw_e   = nm_r["raw_energy"]
            raw_sem = nm_r["scale_sems"][0]
            zne_e   = nm_r["extrapolated"]
            zne_sem = nm_r["extrap_sem"]

            raw_vals.append(abs(raw_e - exact) / abs(exact))
            raw_errs.append(raw_sem / abs(exact))
            zne_vals.append(abs(zne_e - exact) / abs(exact))
            zne_errs.append(zne_sem / abs(exact))

        offset_raw = (2 * nm_idx - n_models + 0.5) * bar_width
        offset_zne = (2 * nm_idx - n_models + 1.5) * bar_width

        b_raw = ax.bar(
            x + offset_raw, raw_vals, bar_width,
            yerr=raw_errs, color=col, alpha=0.85, hatch="",
            capsize=3, error_kw={"linewidth": 1.2, "ecolor": "black"},
            label=f"{nm_cfg['label']} — raw",
        )
        b_zne = ax.bar(
            x + offset_zne, zne_vals, bar_width,
            yerr=zne_errs, color=col, alpha=0.55, hatch="////",
            capsize=3, error_kw={"linewidth": 1.2, "ecolor": "black"},
            label=f"{nm_cfg['label']} — ZNE",
        )
        legend_handles.extend([b_raw, b_zne])

        for i, (rv, zv) in enumerate(zip(raw_vals, zne_vals)):
            if rv > 0:
                improvement = (rv - zv) / rv * 100
                sign = "+" if improvement >= 0 else ""
                ax.text(
                    x[i] + offset_zne,
                    zv + zne_errs[i] + 0.003,
                    f"{sign}{improvement:.0f}%",
                    ha="center", va="bottom",
                    fontsize=7, color=col, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel(r"Relative error  $\epsilon$", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.9,
              ncol=n_models, loc="upper right")
    ax.text(0.01, 0.98, "Solid = Raw  |  Hatched = ZNE",
            transform=ax.transAxes, fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    os.makedirs(CFG.plots_dir, exist_ok=True)
    out = os.path.join(CFG.plots_dir, fname)
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.show()
    logger.info("Saved: %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — ZNE improvement vs ratio curve  (Study B)
# ─────────────────────────────────────────────────────────────────────────────

def plot_zne_improvement_vs_ratio(
    sweep_results: list[dict],
    fname: str = "plot_zne_improvement_vs_ratio.pdf",
) -> None:
    """
    Curve plot of ZNE improvement fraction vs gate-to-readout ratio r.

    For each ratio value:
        improvement(r) = (epsilon_raw - epsilon_ZNE) / epsilon_raw

    Positive = ZNE reduces error. Negative = ZNE overshoots (readout bias
    not removed by extrapolation). The crossover r* identifies the critical
    ratio above which ZNE is beneficial at this noise budget.

    One curve per (N, h, L) combination, coloured by N.
    """
    if not sweep_results:
        logger.error("No Study B results to plot.")
        return

    all_Ns    = sorted(set(r["N"] for r in sweep_results))
    n_colors  = max(len(all_Ns), 2)
    color_map = {
        N: cm.Blues(0.4 + 0.5 * i / (n_colors - 1))
        for i, N in enumerate(all_Ns)
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        r"ZNE Improvement vs Gate-to-Readout Error Ratio  "
        r"$r = p_\mathrm{gate} / p_\mathrm{readout}$"
        "\n"
        rf"Fixed total noise budget  $p_\mathrm{{total}} = {_P_TOTAL}$",
        fontsize=12, fontweight="bold",
    )

    for result in sweep_results:
        N     = result["N"]
        h     = result["h"]
        L     = result["L"]
        exact = result["exact"]

        points = []
        for rk in RATIO_SWEEP:
            if rk not in result:
                continue
            rr      = result[rk]["r"]
            raw_e   = result[rk]["raw_energy"]
            zne_e   = result[rk]["extrapolated"]
            raw_err = abs(raw_e - exact) / abs(exact)
            zne_err = abs(zne_e - exact) / abs(exact)
            improvement = (raw_err - zne_err) / raw_err if raw_err > 0 else 0.0
            points.append((rr, improvement))

        if not points:
            continue

        points.sort(key=lambda p: p[0])
        rs           = [p[0] for p in points]
        improvements = [p[1] for p in points]

        ax.plot(
            rs, improvements,
            color=color_map[N], marker="o", markersize=5, linewidth=1.8,
            label=f"N={N}, h={h}, L={L}",
        )

    # Break-even line
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2,
               label="Break-even (ZNE = raw)", zorder=3)

    # Shaded regions — set after plotting so ylim is set correctly
    ax.set_ylim(-0.05, 1.0)
    ax.axhspan(0.0, 1.0, alpha=0.06, color="green",
               label="ZNE beneficial")

    # Crossover annotation
    ax.axvline(1.0, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(1.05, ax.get_ylim()[0] + 0.05,
            r"$r = 1$", fontsize=8, color="grey", style="italic")

    ax.set_xscale("log")
    ax.set_xlabel(
        r"Gate-to-readout error ratio  $r = p_\mathrm{gate\_2q} \,/\, p_\mathrm{readout}$",
        fontsize=11,
    )
    ax.set_ylabel(
        r"ZNE improvement fraction  "
        r"$(\epsilon_\mathrm{raw} - \epsilon_\mathrm{ZNE})\,/\,\epsilon_\mathrm{raw}$",
        fontsize=11,
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3, linestyle=":")

    plt.tight_layout()
    os.makedirs(CFG.plots_dir, exist_ok=True)
    out = os.path.join(CFG.plots_dir, fname)
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.show()
    logger.info("Saved: %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Study A: three discrete noise models ─────────────────────────────────
    existing_study = [
        f for f in os.listdir(STUDY_RESULTS_DIR)
        if f.startswith("zne_study_") and f.endswith(".json")
    ] if os.path.isdir(STUDY_RESULTS_DIR) else []

    if existing_study:
        ans = input(
            f"Found {len(existing_study)} saved Study A results. Load? [y/n]: "
        ).strip().lower()
        study_results = (
            load_zne_study_results() if ans == "r"
            else run_zne_noise_study()
        )
    else:
        study_results = run_zne_noise_study()

    plot_zne_lambda_scaling(
        study_results,
        N=STUDY_SYSTEM_SIZES[-1],
        h=STUDY_H_FIELDS[0],
        L=STUDY_LAYERS[0],
    )
    plot_zne_noise_comparison(study_results)

    # ── Study B: continuous ratio sweep ──────────────────────────────────────
    existing_sweep = [
        f for f in os.listdir(STUDY_RESULTS_DIR)
        if f.startswith("zne_ratio_sweep_") and f.endswith(".json")
    ] if os.path.isdir(STUDY_RESULTS_DIR) else []

    if existing_sweep:
        ans = input(
            f"Found {len(existing_sweep)} saved Study B results. Load? [y/n]: "
        ).strip().lower()
        sweep_results = (
            load_zne_ratio_sweep_results() if ans == "r"
            else run_zne_ratio_sweep()
        )
    else:
        sweep_results = run_zne_ratio_sweep()

    plot_zne_improvement_vs_ratio(sweep_results)