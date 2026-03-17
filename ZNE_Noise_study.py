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
characterises FakeBrisbane — ZNE provides little benefit because the dominant
noise source is unaffected by folding. In a gate-dominated regime, ZNE can
substantially recover the zero-noise energy.

This script constructs three synthetic noise models with fixed total error
probability but varying gate/readout composition, then measures the ZNE
scaling function E(lambda) at lambda in {1, 3, 5} for each model.

Noise model variants (all depolarising + independent readout errors)
---------------------------------------------------------------------
    "readout_dom"  : p_gate_2q=0.003, p_readout=0.030  — Brisbane-like (r≈0.1)
    "balanced"     : p_gate_2q=0.015, p_readout=0.015  — equal contributions (r=1)
    "gate_dom"     : p_gate_2q=0.030, p_readout=0.003  — gate-dominated (r≈10)

where r = p_gate_2q / p_readout is the gate-to-readout error ratio.

VQE strategy
------------
Parameters are obtained from a *noiseless* statevector simulation. This
decouples parameter quality from noise model choice and allows a direct,
controlled comparison of ZNE behaviour across noise regimes. The scientific
claim is about ZNE scaling (E(lambda) vs lambda), not VQE accuracy; noiseless
parameters are sufficient and simplify the experiment considerably.

Runtime estimate
----------------
N in {4, 6}, h=1.0, L=2, n_reps=5, shots_eval=4096.
Approximately 15-30 minutes on a modern laptop.

Outputs
-------
results_zne_study/zne_study_{N}_{h}_{L}.json — per-(N,h,L) data
plots/plot_zne_lambda_scaling.pdf            — E(lambda) vs lambda figure
plots/plot_zne_noise_comparison.pdf          — rel_err comparison across models

Usage
-----
    python zne_noise_study.py

Import example (for replotting without rerunning):
    from zne_noise_study import load_zne_study_results, plot_zne_lambda_scaling
"""

from __future__ import annotations
import json
import logging
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
# These are intentionally reduced relative to CFG to keep runtime short.
# ─────────────────────────────────────────────────────────────────────────────

STUDY_SYSTEM_SIZES: tuple[int, ...] = (4,)      # representative sizes
STUDY_H_FIELDS:     tuple[float, ...] = (1.0,)    # critical point only (most interesting)
STUDY_LAYERS:       tuple[int, ...] = (2,)         # L=2: enough noise, enough expressibility
STUDY_N_REPS:       int = 5                        # reduced from 10 for speed
STUDY_SHOTS_EVAL:   int = 4096                     # reduced from 8192
STUDY_SCALE_FACTORS: tuple[int, ...] = (1, 3, 5)  # must match CFG.zne_scale_factors
STUDY_RESULTS_DIR:  str = "results_zne_study"
STUDY_SEED:         int = 99                       # separate seed from main benchmark

# Noise model definitions: name -> (p_gate_1q, p_gate_2q, p_readout)
# Total two-qubit error kept fixed at ~0.033 across all models.
# The gate-to-readout ratio r = p_gate_2q / p_readout is the control variable.
NOISE_MODELS: dict[str, dict] = {
    "readout_dom": {
        "p_gate_1q": 0.001,
        "p_gate_2q": 0.003,
        "p_readout": 0.030,
        "label": r"Readout-dominated  ($r \approx 0.1$)",
        "color": "#e41a1c",   # red
    },
    "balanced": {
        "p_gate_1q": 0.003,
        "p_gate_2q": 0.015,
        "p_readout": 0.015,
        "label": r"Balanced  ($r = 1$)",
        "color": "#ff7f00",   # orange
    },
    "gate_dom": {
        "p_gate_1q": 0.005,
        "p_gate_2q": 0.030,
        "p_readout": 0.003,
        "label": r"Gate-dominated  ($r \approx 10$)",
        "color": "#4daf4a",   # green
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Noise model construction
# ─────────────────────────────────────────────────────────────────────────────

def make_parametric_noise_model(
    p_gate_1q: float,
    p_gate_2q: float,
    p_readout: float,
    n_qubits: int,
) -> NoiseModel:
    """
    Build a synthetic depolarising + readout noise model.

    Each parameter independently controls one noise channel:
        p_gate_1q : single-qubit depolarising probability (per SX, RZ, X gate)
        p_gate_2q : two-qubit depolarising probability (per CX gate)
        p_readout : symmetric bit-flip probability at measurement (per qubit)

    This separability is the key property that makes this noise model useful
    for studying ZNE: gate folding amplifies p_gate_2q and p_gate_1q (each
    additional gate invocation adds one error), while p_readout is fixed
    regardless of circuit length.

    Parameters
    ----------
    p_gate_1q : float
        Single-qubit depolarising error probability. Range (0, 1/3).
    p_gate_2q : float
        Two-qubit (CX) depolarising error probability. Range (0, 3/4).
    p_readout : float
        Symmetric readout bit-flip probability. Range (0, 0.5).
    n_qubits : int
        Number of qubits in the circuit.

    Returns
    -------
    NoiseModel
        Qiskit Aer NoiseModel with depolarising gate errors and readout errors.
    """
    noise_model = NoiseModel()

    # Single-qubit gate depolarising error
    # Applied to all single-qubit basis gates: sx, rz, x, id
    err_1q = depolarizing_error(p_gate_1q, 1)
    noise_model.add_all_qubit_quantum_error(err_1q, ["sx", "x", "id"])
    # rz is a virtual gate (zero duration) on real IBM hardware; including it
    # here is conservative and keeps the model self-contained.
    noise_model.add_all_qubit_quantum_error(err_1q, ["rz"])

    # Two-qubit gate depolarising error
    err_2q = depolarizing_error(p_gate_2q, 2)
    noise_model.add_all_qubit_quantum_error(err_2q, ["cx"])

    # Per-qubit symmetric readout error
    # [[P(0|0), P(1|0)], [P(0|1), P(1|1)]]
    ro_matrix = [[1.0 - p_readout, p_readout],
                 [p_readout,       1.0 - p_readout]]
    ro_err = ReadoutError(ro_matrix)
    for q in range(n_qubits):
        noise_model.add_readout_error(ro_err, [q])

    return noise_model


def make_simulator_from_noise_model(noise_model: NoiseModel) -> AerSimulator:
    """
    Wrap a NoiseModel in an AerSimulator.
    Uses max_parallel_threads=1 for reproducibility (same as main benchmark).
    """
    return AerSimulator(
        noise_model=noise_model,
        max_parallel_threads=CFG.max_parallel_threads_aer,
    )


def make_ideal_simulator() -> AerSimulator:
    """
    Noiseless AerSimulator for VQE parameter optimisation.
    method='statevector' gives exact results with no shot noise when shots=None,
    but we use finite shots here so COBYLA sees realistic gradients.
    """
    return AerSimulator(method="statevector", max_parallel_threads=1)


# ─────────────────────────────────────────────────────────────────────────────
# Core study function
# ─────────────────────────────────────────────────────────────────────────────

def run_zne_scaling_for_noise_model(
    noise_model_key: str,
    noise_model: NoiseModel,
    ansatz,
    best_params: np.ndarray,
    N: int,
    J: float,
    h: float,
    seed: int,
) -> dict:
    """
    Evaluate E(lambda) at each scale factor for one noise model.

    Parameters
    ----------
    noise_model_key : str
        Key into NOISE_MODELS dict. Used for logging and saving.
    noise_model : NoiseModel
        The Qiskit Aer noise model to use.
    ansatz : QuantumCircuit
        Parameterised ansatz (from build_ansatz).
    best_params : np.ndarray
        Fixed variational parameters (from noiseless VQE).
    N, J, h : int, float, float
        System parameters.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        "scale_means"  : list[float] — mean energy at each lambda
        "scale_sems"   : list[float] — SEM at each lambda
        "extrapolated" : float       — Richardson-extrapolated E(lambda=0)
        "extrap_sem"   : float       — Propagated SEM on extrapolated value
        "raw_energy"   : float       — E(lambda=1) for convenience
    """
    sim = make_simulator_from_noise_model(noise_model)
    scale_means, scale_sems = [], []

    for sf in STUDY_SCALE_FACTORS:
        m, s, _ = get_energy_statistics(
            ansatz, best_params, N, J, h, sim,
            shots=STUDY_SHOTS_EVAL,
            base_seed=seed,
            n_reps=STUDY_N_REPS,
            readout_matrices=None,  # no readout correction — we want raw ZNE only
            use_parity=False,
            zne_scale=sf,
        )
        scale_means.append(m)
        scale_sems.append(s)
        logger.info("    [%s] lambda=%d  E=%.5f ± %.5f", noise_model_key, sf, m, s)

    extrap = zne_extrapolate(STUDY_SCALE_FACTORS, np.array(scale_means))
    extrap_sem = zne_error_propagation(STUDY_SCALE_FACTORS, np.array(scale_sems))

    return {
        "noise_model_key": noise_model_key,
        "scale_means":     [float(x) for x in scale_means],
        "scale_sems":      [float(x) for x in scale_sems],
        "scale_factors":   list(STUDY_SCALE_FACTORS),
        "extrapolated":    float(extrap),
        "extrap_sem":      float(extrap_sem),
        "raw_energy":      float(scale_means[0]),  # lambda=1 is the raw noisy energy
    }


def run_zne_noise_study() -> list[dict]:
    """
    Full study: for each (N, h, L) combination, find noiseless VQE parameters,
    then evaluate ZNE scaling under all three noise models.

    Saves one JSON per (N, h, L) to STUDY_RESULTS_DIR.
    Returns list of all result dicts.
    """
    os.makedirs(STUDY_RESULTS_DIR, exist_ok=True)
    np.random.seed(STUDY_SEED)

    # Noiseless simulator for VQE optimisation
    ideal_sim = make_ideal_simulator()

    all_results = []

    for N in STUDY_SYSTEM_SIZES:
        for h in STUDY_H_FIELDS:
            for L in STUDY_LAYERS:
                seed = STUDY_SEED + N * 100 + int(h * 10) + L
                logger.info("=" * 60)
                logger.info("N=%d  h=%.1f  L=%d", N, h, L)

                exact = get_exact_energy(N, CFG.J, h)
                ansatz, _ = build_ansatz(N, L)

                # ── Step 1: find parameters on noiseless simulator ────────────
                # Use reduced restarts/iterations for speed.
                # The noiseless VQE converges much faster than the noisy one.
                logger.info("  Running noiseless VQE...")
                t0 = time.perf_counter()
                best_params, _ = run_vqe(
                    ansatz, N, CFG.J, h, ideal_sim,
                    seed=seed,
                )
                logger.info("  Noiseless VQE done in %.1f s", time.perf_counter() - t0)

                # ── Step 2: evaluate ZNE scaling under each noise model ───────
                nm_results = {}
                for nm_key, nm_cfg in NOISE_MODELS.items():
                    logger.info("  Evaluating noise model: %s", nm_key)
                    nm = make_parametric_noise_model(
                        p_gate_1q=nm_cfg["p_gate_1q"],
                        p_gate_2q=nm_cfg["p_gate_2q"],
                        p_readout=nm_cfg["p_readout"],
                        n_qubits=N,
                    )
                    nm_results[nm_key] = run_zne_scaling_for_noise_model(
                        nm_key, nm, ansatz, best_params, N, CFG.J, h, seed,
                    )

                result = {
                    "N":     N,
                    "h":     h,
                    "L":     L,
                    "exact": float(exact),
                    **nm_results,
                }

                fname = os.path.join(
                    STUDY_RESULTS_DIR, f"zne_study_N{N}_h{h:.1f}_L{L}.json"
                )
                with open(fname, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info("  Saved: %s", fname)

                all_results.append(result)

    return all_results


def load_zne_study_results(results_dir: str = STUDY_RESULTS_DIR) -> list[dict]:
    """Load all ZNE study JSON files from results_dir."""
    results = []
    if not os.path.isdir(results_dir):
        logger.warning("ZNE study results directory '%s' not found.", results_dir)
        return results
    for fname in sorted(os.listdir(results_dir)):
        if fname.startswith("zne_study_") and fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                results.append(json.load(f))
    logger.info("Loaded %d ZNE study result files.", len(results))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: E(λ) vs λ — the key diagnostic figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_zne_lambda_scaling(
    study_results: list[dict],
    N: int = 6,
    h: float = 1.0,
    L: int = 2,
    fname: str = "plot_zne_lambda_scaling.pdf",
) -> None:
    """
    Plot E(lambda) vs lambda for all three noise models at a single (N, h, L).

    Each panel shows one noise model:
        - Points with ±1 SEM error bars at lambda in {1, 3, 5}
        - Richardson quadratic fit extrapolated to lambda=0
        - Horizontal dashed lines at exact energy and raw (lambda=1) energy
        - Extrapolated value marked with a star

    This is Figure X in the report demonstrating that ZNE mechanically works
    and that its effectiveness depends on the gate/readout noise ratio.

    Parameters
    ----------
    study_results : list[dict]
        Output of run_zne_noise_study() or load_zne_study_results().
    N, h, L : int, float, int
        Which (N, h, L) combination to plot. Should match STUDY_* parameters.
    fname : str
        Output filename. Saved to CFG.plots_dir.
    """
    # Find the correct result
    result = next(
        (r for r in study_results if r["N"] == N and r["h"] == h and r["L"] == L),
        None,
    )
    if result is None:
        logger.error(
            "No study result found for N=%d, h=%.1f, L=%d. "
            "Run run_zne_noise_study() first.", N, h, L
        )
        return

    exact = result["exact"]
    n_models = len(NOISE_MODELS)
    lambdas = np.array(STUDY_SCALE_FACTORS, dtype=float)

    fig, axes = plt.subplots(
        1, n_models,
        figsize=(5.0 * n_models, 5),
        sharey=False,
    )
    fig.suptitle(
        rf"ZNE Scaling: $E(\lambda)$ vs Noise Scale Factor  |  "
        rf"N={N},  $h$={h},  L={L}",
        fontsize=13, fontweight="bold",
    )

    # Richardson extrapolation curve: fit quadratic, evaluate at lambda in [0, 5.5]
    lambda_dense = np.linspace(0.0, max(lambdas) + 0.5, 200)

    for ax, (nm_key, nm_cfg) in zip(axes, NOISE_MODELS.items()):
        nm_result = result[nm_key]
        means = np.array(nm_result["scale_means"])
        sems  = np.array(nm_result["scale_sems"])
        extrap     = nm_result["extrapolated"]
        extrap_sem = nm_result["extrap_sem"]
        raw_energy = nm_result["raw_energy"]

        col = nm_cfg["color"]

        # ── Fit and plot Richardson polynomial ───────────────────────────────
        degree = len(lambdas) - 1
        coeffs = np.polyfit(lambdas, means, deg=degree)
        fit_curve = np.polyval(coeffs, lambda_dense)
        ax.plot(
            lambda_dense, fit_curve,
            color=col, linestyle="-", linewidth=1.8, alpha=0.7,
            label="Richardson fit",
            zorder=2,
        )

        # ── Measured E(lambda) points with error bars ─────────────────────────
        ax.errorbar(
            lambdas, means, yerr=sems,
            fmt="o", color=col, markersize=7, linewidth=1.5,
            capsize=4, capthick=1.5,
            label=r"$E(\lambda)$ ± 1 SEM",
            zorder=4,
        )

        # ── Extrapolated value at lambda=0 ────────────────────────────────────
        ax.errorbar(
            [0.0], [extrap], yerr=[extrap_sem],
            fmt="*", color=col, markersize=14, linewidth=1.5,
            capsize=4, capthick=1.5,
            label=rf"$E_\mathrm{{ZNE}}$ = {extrap:.4f}",
            zorder=5,
        )

        # ── Reference lines ───────────────────────────────────────────────────
        ax.axhline(
            exact, color="black", linestyle="--", linewidth=1.5,
            label=rf"Exact: {exact:.4f}", zorder=3,
        )

        # ── Relative errors as text annotations ───────────────────────────────
        rel_err_raw = abs(raw_energy - exact) / abs(exact)
        rel_err_zne = abs(extrap - exact) / abs(exact)
        ax.text(
            0.97, 0.97,
            rf"$\epsilon_\mathrm{{raw}}$ = {rel_err_raw:.3f}"
            "\n"
            rf"$\epsilon_\mathrm{{ZNE}}$ = {rel_err_zne:.3f}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
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
# Plot 2: ZNE relative error comparison across noise models
# ─────────────────────────────────────────────────────────────────────────────

def plot_zne_noise_comparison(
    study_results: list[dict],
    fname: str = "plot_zne_noise_comparison.pdf",
) -> None:
    """
    Grouped bar chart comparing raw vs ZNE relative error for each noise model,
    with one group of bars per (N, h, L) combination in the study.

    Error bars show ±1 SEM from get_energy_statistics (statistical shot noise),
    propagated through Richardson extrapolation for the ZNE bars.

    Each group of bars on the x-axis corresponds to one (N, h, L) combination.
    Within each group, three pairs of bars show raw vs ZNE for each noise model.

    This is the correct representation because:
        - The spread across (N, h, L) is physical signal, not measurement error.
          Collapsing it into a median ± percentile hides scientifically relevant
          trends (e.g. ZNE benefit growing with N).
        - The SEM from get_energy_statistics is the true statistical uncertainty
          on each individual energy estimate.

    Parameters
    ----------
    study_results : list[dict]
        Output of run_zne_noise_study() or load_zne_study_results().
    fname : str
        Output filename. Saved to CFG.plots_dir.
    """
    if not study_results:
        logger.error("No study results to plot.")
        return

    nm_keys = list(NOISE_MODELS.keys())
    n_models = len(nm_keys)

    # Sort results for consistent ordering on x-axis
    study_results_sorted = sorted(
        study_results, key=lambda r: (r["N"], r["h"], r["L"])
    )
    n_combos = len(study_results_sorted)

    # x-axis label for each (N, h, L) combo
    x_labels = [
        f"N={r['N']}, h={r['h']}, L={r['L']}"
        for r in study_results_sorted
    ]

    # Layout: one bar group per combo, within each group 2 bars (raw, ZNE)
    # per noise model. Total bars per group = 2 * n_models.
    # Bar width chosen so all bars in a group fit within unit spacing.
    n_bars_per_group = 2 * n_models
    bar_width = 0.8 / n_bars_per_group
    x = np.arange(n_combos)

    fig, ax = plt.subplots(figsize=(max(8, 3.5 * n_combos), 5.5))
    fig.suptitle(
        "ZNE Benefit vs Noise Model Composition\n"
        r"($\epsilon = |E - E_\mathrm{exact}| / |E_\mathrm{exact}|$, "
        "error bars = ±1 SEM from shot statistics)",
        fontsize=12, fontweight="bold",
    )

    # Hatch patterns distinguish raw vs ZNE within each noise model colour
    hatch_raw = ""       # solid fill = raw
    hatch_zne = "////"   # hatched = ZNE

    legend_handles = []

    for nm_idx, nm_key in enumerate(nm_keys):
        nm_cfg = NOISE_MODELS[nm_key]
        col = nm_cfg["color"]

        raw_vals, raw_errs = [], []
        zne_vals, zne_errs = [], []

        for r in study_results_sorted:
            exact = r["exact"]
            nm_r  = r[nm_key]

            raw_e   = nm_r["raw_energy"]
            raw_sem = nm_r["scale_sems"][0]   # SEM at lambda=1 from get_energy_statistics
            zne_e   = nm_r["extrapolated"]
            zne_sem = nm_r["extrap_sem"]       # Propagated SEM through Richardson

            raw_vals.append(abs(raw_e - exact) / abs(exact))
            raw_errs.append(raw_sem / abs(exact))  # SEM propagated to relative error
            zne_vals.append(abs(zne_e - exact) / abs(exact))
            zne_errs.append(zne_sem / abs(exact))

        # Offset positions: raw bars left of centre, ZNE bars right
        offset_raw = (2 * nm_idx - n_models + 0.5) * bar_width
        offset_zne = (2 * nm_idx - n_models + 1.5) * bar_width

        b_raw = ax.bar(
            x + offset_raw, raw_vals, bar_width,
            yerr=raw_errs,
            color=col, alpha=0.85, hatch=hatch_raw,
            capsize=3, error_kw={"linewidth": 1.2, "ecolor": "black"},
            label=f"{nm_cfg['label']} — raw",
        )
        b_zne = ax.bar(
            x + offset_zne, zne_vals, bar_width,
            yerr=zne_errs,
            color=col, alpha=0.55, hatch=hatch_zne,
            capsize=3, error_kw={"linewidth": 1.2, "ecolor": "black"},
            label=f"{nm_cfg['label']} — ZNE",
        )
        legend_handles.extend([b_raw, b_zne])

        # Annotate ZNE improvement percentage above each ZNE bar
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

    # Legend: group by noise model, show solid=raw hatched=ZNE
    ax.legend(
        handles=legend_handles,
        fontsize=8,
        framealpha=0.9,
        ncol=n_models,
        loc="upper right",
    )

    # Add a text note explaining hatch convention
    ax.text(
        0.01, 0.98,
        "Solid = Raw  |  Hatched = ZNE",
        transform=ax.transAxes,
        fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

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
    import os

    results_dir = STUDY_RESULTS_DIR
    existing = [
        f for f in os.listdir(results_dir)
        if f.startswith("zne_study_") and f.endswith(".json")
    ] if os.path.isdir(results_dir) else []

    if existing:
        ans = input(f"Found {len(existing)} saved results. Load? [y/n]: ").strip().lower()
        study_results = (
            load_zne_study_results() if ans == "y"
            else run_zne_noise_study()
        )
    else:
        study_results = run_zne_noise_study()

    # Generate both plots for the first (N, h, L) in the study
    # Adjust N, h, L to match STUDY_SYSTEM_SIZES/STUDY_H_FIELDS/STUDY_LAYERS
    plot_zne_lambda_scaling(
        study_results,
        N=STUDY_SYSTEM_SIZES[-1],   # largest N for clearest noise effect
        h=STUDY_H_FIELDS[0],
        L=STUDY_LAYERS[0],
    )
    plot_zne_noise_comparison(study_results)