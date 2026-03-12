"""
plotting.py
===========
All visualisation functions for the VQE mitigation benchmark.

Four plots:
    1. plot_relative_error   -- relative error vs N, all configs, all (h, L)
    2. plot_absolute_energy  -- absolute energy vs N with shaded SEM bands + zoomed inset panel
    3. plot_convergence      -- VQE optimisation curves for all (h, L, N), normalised as relative error
    4. plot_cost_vs_error    -- Pareto frontier plot

All plots saved as PDF (vector graphics: infinitely scalable, no pixelation).
PDFs are suitable for inclusion in a LaTeX report.

Import example:
    from plotting import plot_relative_error, plot_absolute_energy
    from plotting import plot_convergence, plot_cost_vs_error
    from plotting import plot_error_scaling, plot_parity_discard
"""

from __future__ import annotations
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from config import (CFG,
    MITIGATION_CONFIGS,
    STYLE,
    PHASE_LABELS,
    BenchmarkConfig,
    INDIVIDUAL_CONFIGS,
    COMBINED_CONFIGS,
)

os.makedirs(CFG.plots_dir, exist_ok=True)

logger = logging.getLogger(__name__)



# ─────────────────────────────────────────────────────────────────────────────
# 1. Relative error grid  (unchanged — this plot works)
# ─────────────────────────────────────────────────────────────────────────────

def plot_relative_error(
    all_results: list[dict],
    cfg: BenchmarkConfig = CFG,
    configs_to_show: list[str] | None = None,
    fname: str = "plot_relative_error.pdf",
) -> None:
    """
    Grid: relative error |E_mit - E_exact|/|E_exact| vs N.
    Rows = h (phase), Columns = L (depth). Log y-axis. Shaded ±1 SEM bands.

    Parameters
    ----------
    configs_to_show : list[str] or None
        Which configs to plot. None = all 8.
        Pass INDIVIDUAL_CONFIGS to show only the four individual methods.
    fname : str
        Output filename.
    """
    if configs_to_show is None:
        configs_to_show = list(STYLE.keys())

    fig, axes = plt.subplots(
        len(cfg.h_fields), len(cfg.layers),
        figsize=(6 * len(cfg.layers), 4 * len(cfg.h_fields)),
    )
    fig.suptitle(
        "Relative Error vs System Size  |  1D TFIM VQE Mitigation Benchmark",
        fontsize=15, fontweight="bold",
    )

    for h_idx, h_val in enumerate(cfg.h_fields):
        for l_idx, l_val in enumerate(cfg.layers):
            ax = axes[h_idx, l_idx]
            subset = sorted(
                [r for r in all_results if r["h"] == h_val and r["L"] == l_val],
                key=lambda r: r["N"],
            )
            ns = [r["N"] for r in subset]

            for config_name in configs_to_show:
                sty       = STYLE[config_name]
                rel_errs  = [r[config_name]["rel_err"] for r in subset]
                sems      = [r[config_name]["sem"] for r in subset]
                exact_abs = [abs(r["exact"]) for r in subset]
                rel_sems  = [s / e for s, e in zip(sems, exact_abs)]

                ax.semilogy(
                    ns, rel_errs,
                    color=sty["color"], marker=sty["marker"],
                    linestyle=sty["ls"], linewidth=sty["lw"],
                    label=config_name,
                )
                ax.fill_between(
                    ns,
                    [max(1e-6, r - s) for r, s in zip(rel_errs, rel_sems)],
                    [r + s for r, s in zip(rel_errs, rel_sems)],
                    color=sty["color"], alpha=0.15,
                )

            ax.set_ylim(1e-2, 1e0)

            ax.set_xticks(ns)
            ax.grid(True, which="both", alpha=0.3, linestyle=":")
            if h_idx == 0:
                ax.set_title(f"Depth  L = {l_val}", fontsize=12)
            if l_idx == 0:
                ax.set_ylabel(f"{PHASE_LABELS[h_val]}\nRelative Error", fontsize=10)
            if h_idx == len(cfg.h_fields) - 1:
                ax.set_xlabel("System Size  N  (qubits)", fontsize=10)

    handles = [
        plt.Line2D([0], [0], color=STYLE[c]["color"], marker=STYLE[c]["marker"],
                   linestyle=STYLE[c]["ls"], linewidth=1.5, label=c)
        for c in configs_to_show
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(configs_to_show)),
               bbox_to_anchor=(0.5, -0.02), fontsize=9, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(os.path.join(CFG.plots_dir, fname), bbox_inches="tight", dpi=150)
    plt.show()
    logger.info("Saved: %s", fname)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Absolute energy with zoomed inset
# ─────────────────────────────────────────────────────────────────────────────

def plot_absolute_energy(
    all_results: list[dict],
    cfg: BenchmarkConfig = CFG,
    zoom_N: int = 6,
    zoom_h: float = 1.0,
    zoom_L: int = 2,
) -> None:
    """
    Grid: absolute <H> vs N with shaded ±1 SEM bands.

    The SEM bands are genuinely small relative to the energy scale, so
    this also generates a separate zoomed figure for one representative
    (N, h, L) data point where the bands become visible.

    Parameters
    ----------
    zoom_N, zoom_h, zoom_L : int, float, int
        The (N, h, L) combination to zoom into for the inset figure.
        Default: N=6, h=1.0 (critical), L=2.
    """
    fig, axes = plt.subplots(
        len(cfg.h_fields), len(cfg.layers),
        figsize=(6 * len(cfg.layers), 4 * len(cfg.h_fields)),
    )
    fig.suptitle(
        "Absolute Ground-State Energy vs System Size  |  1D TFIM VQE",
        fontsize=15, fontweight="bold",
    )

    for h_idx, h_val in enumerate(cfg.h_fields):
        for l_idx, l_val in enumerate(cfg.layers):
            ax = axes[h_idx, l_idx]
            subset = sorted(
                [r for r in all_results if r["h"] == h_val and r["L"] == l_val],
                key=lambda r: r["N"],
            )
            ns         = [r["N"] for r in subset]
            exact_vals = [r["exact"] for r in subset]

            ax.plot(ns, exact_vals, "k--", lw=2, label="Exact (ED)", zorder=10)
            for config_name, sty in STYLE.items():
                means = np.array([r[config_name]["mean"] for r in subset])
                sems  = np.array([r[config_name]["sem"]  for r in subset])
                ax.plot(ns, means, color=sty["color"], marker=sty["marker"],
                        linestyle=sty["ls"], linewidth=sty["lw"], label=config_name)
                ax.fill_between(ns, means - sems, means + sems,
                                color=sty["color"], alpha=0.12)

            ax.set_xticks(ns)
            ax.grid(True, alpha=0.3, linestyle=":")
            if h_idx == 0:
                ax.set_title(f"Depth  L = {l_val}", fontsize=12)
            if l_idx == 0:
                ax.set_ylabel(f"{PHASE_LABELS[h_val]}\n<H>", fontsize=10)
            if h_idx == len(cfg.h_fields) - 1:
                ax.set_xlabel("System Size  N  (qubits)", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02), fontsize=9, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(os.path.join(CFG.plots_dir, "plot_absolute_energy.pdf"), bbox_inches="tight", dpi=150)
    plt.show()
    logger.info("Saved: plot_absolute_energy.pdf")

    # ── Zoomed figure: one (N, h, L) point, all configs, SEM bars visible ────
    zoom_match = [
        r for r in all_results
        if r["N"] == zoom_N and r["h"] == zoom_h and r["L"] == zoom_L
    ]
    if not zoom_match:
        logger.warning("No data for zoom point N=%d h=%.1f L=%d", zoom_N, zoom_h, zoom_L)
        return

    r = zoom_match[0]
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.set_title(
        f"Energy Estimates at N={zoom_N}, h={zoom_h}, L={zoom_L}  "
        f"(error bars = ±1 SEM)",
        fontsize=12,
    )

    x_positions = np.arange(len(STYLE))
    for i, (config_name, sty) in enumerate(STYLE.items()):
        mean = r[config_name]["mean"]
        sem  = r[config_name]["sem"]
        ax2.bar(i, mean, color=sty["color"], alpha=0.7, width=0.7,
                label=config_name)
        ax2.errorbar(i, mean, yerr=sem, fmt="none",
                     color="black", capsize=5, linewidth=1.5)

    ax2.axhline(r["exact"], color="black", linestyle="--", linewidth=2,
                label="Exact (ED)")
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(list(STYLE.keys()), rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("<H>", fontsize=11)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.plots_dir, "plot_absolute_energy_zoom.pdf"), bbox_inches="tight", dpi=150)
    plt.show()
    logger.info("Saved: plot_absolute_energy_zoom.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Convergence as normalised relative error
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(
    all_results: list[dict],
    cfg: BenchmarkConfig = CFG,
) -> None:
    """
    VQE convergence as normalised relative error.

    y-axis: (E_noisy(iter) - E_exact(N)) / |E_exact(N)|
    This is dimensionless and comparable across all N.
    Target is 0 (which is the x-axis itself). No reference line needed.

    Rows = L (depth), Columns = h (phase).
    All N values overlaid per panel, light-to-dark blue = small-to-large N.

    Note: brief dips below 0 are physical (shot noise can push the noisy
    energy below the variational bound momentarily). They are NOT errors.
    """
    n_rows = len(cfg.layers)
    n_cols = len(cfg.h_fields)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "VQE Convergence  |  Relative Error During Optimisation",
        fontsize=14, fontweight="bold",
    )

    cmap    = plt.cm.Blues
    colours = np.linspace(0.35, 0.92, len(cfg.system_sizes))

    for l_idx, l_val in enumerate(cfg.layers):
        for h_idx, h_val in enumerate(cfg.h_fields):
            ax = axes[l_idx, h_idx]

            for n_idx, N in enumerate(cfg.system_sizes):
                match = [r for r in all_results
                         if r["N"] == N and r["L"] == l_val and r["h"] == h_val]
                if not match:
                    continue

                # Use pre-computed normalised convergence from benchmark.py
                conv_rel = match[0].get("convergence_rel")
                if conv_rel is None:
                    # Fallback: compute on the fly from raw history
                    exact = match[0]["exact"]
                    conv_raw = match[0].get("convergence_raw", match[0].get("convergence", []))
                    conv_rel = [
                        abs(e - exact) / abs(exact) if exact != 0 else float("nan")
                        for e in conv_raw
                    ]

                ax.semilogy(
                    range(len(conv_rel)), conv_rel,
                    color=cmap(colours[n_idx]),
                    linewidth=1.2, label=f"N={N}",
                )

            if l_idx == 0:
                ax.set_title(f"{PHASE_LABELS[h_val]}", fontsize=11)
            if h_idx == 0:
                ax.set_ylabel(f"L={l_val}\n|E−E₀|/|E₀|", fontsize=10)
            if l_idx == n_rows - 1:
                ax.set_xlabel("Optimiser Iteration", fontsize=10)

            # Zero reference: dashed grey line at 0.0 on log scale is not
            # possible, so instead draw at a small epsilon as visual guide
            ax.axhline(1e-2, color="grey", linestyle=":", alpha=0.4, linewidth=0.8)
            ax.grid(True, which="both", alpha=0.25, linestyle=":")
            ax.legend(fontsize=7, ncol=2, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(CFG.plots_dir, "plot_convergence.pdf"), bbox_inches="tight", dpi=150)
    plt.show()
    logger.info("Saved: plot_convergence.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cost vs error — Pareto frontier, measured parity discard
# ─────────────────────────────────────────────────────────────────────────────

def _pareto_frontier(points: np.ndarray) -> np.ndarray:
    """
    Returns mask of Pareto-optimal points (lower cost AND lower error).
    A point is Pareto-optimal if no other point beats it on both axes.

    Parameters
    ----------
    points : np.ndarray, shape (M, 2)
        Column 0 = effective overhead (lower is better).
        Column 1 = median relative error (lower is better).

    Returns
    -------
    np.ndarray, shape (M,), dtype bool
        True where point is on the Pareto frontier.
    """
    M = len(points)
    on_frontier = np.ones(M, dtype=bool)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            # j dominates i if j is strictly better on at least one axis
            # and no worse on the other
            if points[j, 0] <= points[i, 0] and points[j, 1] <= points[i, 1]:
                if points[j, 0] < points[i, 0] or points[j, 1] < points[i, 1]:
                    on_frontier[i] = False
                    break
    return on_frontier


def plot_cost_vs_error(
    all_results: list[dict],
    cfg: BenchmarkConfig = CFG,
) -> None:
    """
    Pareto frontier cost-benefit plot, faceted by phase (h value).

    Three panels, one per h value. Each panel shows:
        - One point per mitigation config (x = effective overhead, y = median rel error)
        - Effective overhead uses MEASURED parity discard fraction (not assumed 50%)
        - Pareto frontier drawn as a step line
        - Methods on the frontier are labelled; dominated methods unlabelled

    Effective overhead = overhead_amortised * shot_multiplier
    where shot_multiplier = 1/(1-measured_discard_fraction).

    Why a Pareto plot rather than a scatter:
        A Pareto plot directly answers the question "given a computational
        budget, which method achieves the lowest error?" Methods on the
        frontier are the rational choices; methods inside it are dominated
        (another method achieves both lower error and lower cost).
    """
    fig, axes = plt.subplots(1, len(cfg.h_fields),
                             figsize=(5.5 * len(cfg.h_fields), 5),
                             sharey=False)
    fig.suptitle(
        "Mitigation Cost vs Error Reduction  |  Pareto Frontier by Phase\n"
        "(x-axis uses measured parity discard fraction, not assumed 50%)",
        fontsize=13, fontweight="bold",
    )

    non_raw_configs = [c for c in STYLE if c != "raw"]

    for h_idx, h_val in enumerate(cfg.h_fields):
        ax = axes[h_idx]
        h_results = [r for r in all_results if r["h"] == h_val]

        # For each config compute: median effective overhead and median rel error
        config_points: dict[str, tuple[float, float]] = {}

        for config_name in non_raw_configs:
            overheads = []
            errors    = []
            for r in h_results:
                cost_dict = r[config_name].get("cost", {})
                overhead  = cost_dict.get("overhead_effective")
                rel_err   = r[config_name]["rel_err"]
                if overhead is not None and not np.isnan(rel_err):
                    overheads.append(overhead)
                    errors.append(rel_err)

            if not overheads:
                continue

            config_points[config_name] = (
                float(np.median(overheads)),
                float(np.median(errors)),
            )

        if not config_points:
            ax.set_title(f"{PHASE_LABELS[h_val]}")
            continue

        # Also add raw as reference (overhead=1.0, use its median error)
        raw_errors = [r["raw"]["rel_err"] for r in h_results
                      if not np.isnan(r["raw"]["rel_err"])]
        if raw_errors:
            config_points["raw"] = (1.0, float(np.median(raw_errors)))

        # Compute Pareto frontier
        names  = list(config_points.keys())
        pts    = np.array([config_points[n] for n in names])
        front  = _pareto_frontier(pts)

        # Plot all points
        for i, name in enumerate(names):
            sty = STYLE[name]
            x, y = pts[i]
            marker_size = 180 if front[i] else 80
            edge_width  = 2.0 if front[i] else 0.5
            ax.scatter(x, y,
                       color=sty["color"], marker=sty["marker"],
                       s=marker_size, zorder=5,
                       edgecolors="black", linewidths=edge_width,
                       alpha=1.0 if front[i] else 0.45,
                       label=name)
            # Label only Pareto-optimal points to avoid clutter
            if front[i]:
                ax.annotate(
                    name,
                    (x, y),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=7.5,
                    color=sty["color"],
                    fontweight="bold",
                )

        # Draw Pareto frontier step line
        front_pts = pts[front]
        if len(front_pts) > 1:
            order = np.argsort(front_pts[:, 0])
            fp    = front_pts[order]
            ax.step(fp[:, 0], fp[:, 1], where="post",
                    color="black", linestyle="--", linewidth=1.2,
                    alpha=0.5, zorder=3, label="Pareto frontier")

        ax.set_xlabel(
            "Effective Overhead  (× raw)\n"
            "= circuit overhead × 1/(1−measured parity discard)",
            fontsize=9,
        )
        if h_idx == 0:
            ax.set_ylabel("Median Relative Error  |E−E₀|/|E₀|", fontsize=10)
        ax.set_title(f"{PHASE_LABELS[h_val]}", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25, linestyle=":")

        # Add note on parity discard
        parity_results = [r for r in h_results if "parity" in r]
        if parity_results:
            sample_discards = [
                r["parity"]["parity_discard"]
                for r in h_results
                if not np.isnan(r["parity"].get("parity_discard", float("nan")))
            ]
            if sample_discards:
                med_discard = np.median(sample_discards)
                ax.text(
                    0.97, 0.97,
                    f"Measured parity\ndiscard: {med_discard:.1%}",
                    transform=ax.transAxes,
                    fontsize=7.5, ha="right", va="top", color="grey",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

    # Shared legend at bottom
    handles = [
        plt.scatter([], [], color=STYLE[c]["color"], marker=STYLE[c]["marker"],
                    s=80, label=c)
        for c in list(STYLE.keys())
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), fontsize=8.5, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(os.path.join(CFG.plots_dir, "plot_cost_vs_error.pdf"), bbox_inches="tight", dpi=150)
    plt.show()
    logger.info("Saved: plot_cost_vs_error.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Error scaling with N
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_scaling(
    all_results: list[dict],
    cfg: BenchmarkConfig = CFG,
    fit_from_N: int = 4,
) -> None:
    """
    Fits and plots rel_error ∝ N^alpha for each (h, L, config).

    The slope alpha in a log-log regression of relative error vs N
    quantifies how rapidly each method's accuracy degrades with system size.
    Smaller alpha = more robust scaling.

    Uses only N >= fit_from_N to avoid the anomalously good N=2 results
    (the HEA exactly represents the N=2 ground state at any L>=1).

    Produces:
        - plot_error_scaling.pdf: log-log scatter + fit lines, faceted by h
        - Prints a table of alpha values per (h, L, config) to console
    """
    fig, axes = plt.subplots(
        len(cfg.h_fields), len(cfg.layers),
        figsize=(5 * len(cfg.layers), 4 * len(cfg.h_fields)),
    )
    fig.suptitle(
        r"Error Scaling  $\epsilon \propto N^\alpha$  |  1D TFIM VQE",
        fontsize=14, fontweight="bold",
    )

    print("\n" + "="*70)
    print(f"{'h':>4}  {'L':>2}  {'config':>16}  {'alpha':>6}  {'alpha_err':>9}")
    print("="*70)

    for h_idx, h_val in enumerate(cfg.h_fields):
        for l_idx, l_val in enumerate(cfg.layers):
            ax = axes[h_idx, l_idx]

            subset = sorted(
                [r for r in all_results
                 if r["h"] == h_val and r["L"] == l_val and r["N"] >= fit_from_N],
                key=lambda r: r["N"],
            )
            if len(subset) < 3:
                ax.set_title(f"L={l_val}, h={h_val} — insufficient data")
                continue

            ns = np.array([r["N"] for r in subset], dtype=float)
            log_ns = np.log(ns)

            for config_name, sty in STYLE.items():
                rel_errs = np.array([r[config_name]["rel_err"] for r in subset])
                valid    = ~np.isnan(rel_errs) & (rel_errs > 0)
                if valid.sum() < 2:
                    continue

                log_err = np.log(rel_errs[valid])
                log_n_v = log_ns[valid]

                # Linear fit in log-log space: log(err) = alpha*log(N) + beta
                coeffs, cov = np.polyfit(log_n_v, log_err, 1, cov=True)
                alpha     = coeffs[0]
                alpha_err = np.sqrt(cov[0, 0])

                # Plot scatter
                ax.scatter(ns[valid], rel_errs[valid],
                           color=sty["color"], marker=sty["marker"],
                           s=40, alpha=0.7, zorder=3)

                # Plot fit line over continuous N range
                n_fit   = np.linspace(fit_from_N, ns[-1], 50)
                err_fit = np.exp(coeffs[1]) * n_fit ** alpha
                ax.loglog(n_fit, err_fit,
                          color=sty["color"], linestyle="--",
                          linewidth=1.0, alpha=0.6,
                          label=f"{config_name} (α={alpha:.2f})")

                print(f"{h_val:>4.1f}  {l_val:>2}  {config_name:>16}  "
                      f"{alpha:>6.3f}  ±{alpha_err:.3f}")

            ax.set_xticks(ns.astype(int))
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.grid(True, which="both", alpha=0.3, linestyle=":")
            if h_idx == 0:
                ax.set_title(f"Depth  L = {l_val}", fontsize=11)
            if l_idx == 0:
                ax.set_ylabel(f"{PHASE_LABELS[h_val]}\nRelative Error", fontsize=10)
            if h_idx == len(cfg.h_fields) - 1:
                ax.set_xlabel("N  (qubits)", fontsize=10)
            ax.legend(fontsize=6, loc="upper left")

    print("="*70 + "\n")
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(os.path.join(CFG.plots_dir, "plot_error_scaling.pdf"), bbox_inches="tight", dpi=150)
    plt.show()
    logger.info("Saved: plot_error_scaling.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Parity discard fraction heatmap  (new — shows measured waste per regime)
# ─────────────────────────────────────────────────────────────────────────────

def plot_parity_discard(
    all_results: list[dict],
    cfg: BenchmarkConfig = CFG,
) -> None:
    """
    Heatmap of measured parity discard fraction vs (N, h) for each L.

    This replaces the assumed 50% with empirical data. Reveals:
        - Ordered phase (h=0.5): low discard (ground state has mostly even parity)
        - Disordered/high noise: higher discard
        - Larger N: more gates = more errors = higher discard
    """
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    n_layers = len(cfg.layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4))
    fig.suptitle(
        "Measured Parity Discard Fraction  |  parity config, both bases",
        fontsize=13, fontweight="bold",
    )

    vmin, vmax = 0.0, 0.5

    for l_idx, l_val in enumerate(cfg.layers):
        ax = axes[l_idx]
        matrix = np.full((len(cfg.system_sizes), len(cfg.h_fields)), np.nan)

        for n_idx, N in enumerate(cfg.system_sizes):
            for h_idx, h_val in enumerate(cfg.h_fields):
                match = [r for r in all_results
                         if r["N"] == N and r["h"] == h_val and r["L"] == l_val]
                if match:
                    d = match[0]["parity"].get("parity_discard", np.nan)
                    matrix[n_idx, h_idx] = d

        im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="YlOrRd",
                       aspect="auto", origin="lower")
        ax.set_xticks(range(len(cfg.h_fields)))
        ax.set_xticklabels([f"h={h}" for h in cfg.h_fields], fontsize=9)
        ax.set_yticks(range(len(cfg.system_sizes)))
        ax.set_yticklabels([f"N={N}" for N in cfg.system_sizes], fontsize=9)
        ax.set_title(f"L = {l_val}", fontsize=11)

        for n_idx in range(len(cfg.system_sizes)):
            for h_idx in range(len(cfg.h_fields)):
                val = matrix[n_idx, h_idx]
                if not np.isnan(val):
                    ax.text(h_idx, n_idx, f"{val:.2f}",
                            ha="center", va="center", fontsize=8,
                            color="white" if val > 0.3 else "black")

    fig.colorbar(im, ax=axes.tolist(), label="Discard Fraction", shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.plots_dir, "plot_parity_discard.pdf"), bbox_inches="tight", dpi=150)
    plt.show()
    logger.info("Saved: plot_parity_discard.pdf")