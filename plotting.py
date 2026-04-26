"""
plot_supplementary.py
=====================
Generates ALL dissertation figures from pre-saved JSON results.

Run this file from the project root directory:
    python plot_supplementary.py

Sections
--------
    A.  FakeBrisbane main benchmark  (loads from FakeBrisbane_results_updated_skip_rz/)
    B.  FakeFez main benchmark       (loads from FakeFez_results_updated_skip_rz/)
    C.  ZNE noise composition study  (loads from results_zne_study/)
    D.  Real hardware (ibm_fez)      (loads from results_hardware/)

For each benchmark device (A, B) the following plots are generated:
    1.  Relative error — individual mitigation methods
        (raw, RC, PPS, ZNE  — solid lines only)
    2.  Relative error — combined mitigation strategies
        (raw as reference + all four combinations — dashed lines)
    3.  VQE convergence history
    4.  Pareto frontier (cost vs error)
    5.  Parity discard heatmap
    6.  Parity discard delta heatmap  (parity − ro+parity)

One PDF is produced per (h, L) panel for plots 1–3,
one per h for plot 4, and one per L for plots 5–6.

File naming convention
----------------------
    {plot_type}_{device}_{h_or_L_descriptor}.pdf

    e.g.  rel_err_individual_brisbane_h1p0_L2.pdf
          convergence_fez_L2_h0p5.pdf
          pareto_brisbane_h1p0.pdf
          parity_discard_fez_L1.pdf

Design
------
    Typography  : Times New Roman / STIX (matches \\usepackage{newtxtext,newtxmath})
    Figure width: 6.30 in  (A4 − 2 × 2.5 cm margins)
    Height      : width / golden ratio ≈ 3.90 in  (adjusted per plot type)
    Colour      : ColorBrewer + Tableau palette, greyscale-safe via
                  simultaneous use of colour + marker + linestyle
    Solid lines : individual methods
    Dashed lines: combined methods
    No figure titles — captions written in LaTeX

    Panel plots (3×3 or 1×3 grids) are generated at reduced figsize so that
    text and markers remain legible when LaTeX displays each panel at
    ~0.31 textwidth (~2 inches).
"""

from __future__ import annotations
import glob
import json
import logging
import os
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

# ── Project imports ────────────────────────────────────────────────────────────
from config import CFG, MITIGATION_CONFIGS

os.makedirs(CFG.plots_dir, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# §0  Global style
# ═════════════════════════════════════════════════════════════════════════════

_TW = 6.30        # text width in inches (A4, 2.5 cm margins)
_GR = 1.618       # golden ratio
_FH = _TW / _GR   # default full-width figure height

# ── Panel sizes ───────────────────────────────────────────────────────────────
# Panels displayed in a 3×3 grid at 0.31\textwidth ≈ 2.0 in each.
# Generate at ~3.2 in wide so LaTeX shrinks by ~0.63×, keeping text legible.
_PANEL_W = 3.2     # panel figure width (inches)
_PANEL_H = 2.4     # panel figure height (inches)

# Panels displayed in a 1×3 grid at 0.32\textwidth ≈ 2.0 in each.
# Same target size as 3×3 panels.
_TRIPLET_W = 3.4
_TRIPLET_H = 2.6

matplotlib.rcParams.update({
    # Typography
    "font.family":          "serif",
    "font.serif":           ["Times New Roman", "Times",
                             "Nimbus Roman No9 L", "DejaVu Serif"],
    "font.size":            9,
    "axes.labelsize":       9,
    "axes.titlesize":       9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      7.5,
    "mathtext.fontset":     "stix",
    "mathtext.default":     "regular",
    # Lines
    "lines.linewidth":      1.2,
    "lines.markersize":     4.0,
    "patch.linewidth":      0.6,
    # Axes
    "axes.linewidth":       0.7,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.alpha":           0.25,
    "grid.linewidth":       0.5,
    "grid.linestyle":       ":",
    "axes.axisbelow":       True,
    # Ticks
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "xtick.major.size":     3.5,
    "ytick.major.size":     3.5,
    "xtick.minor.size":     2.0,
    "ytick.minor.size":     2.0,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    # Legend
    "legend.frameon":       True,
    "legend.framealpha":    0.90,
    "legend.edgecolor":     "0.75",
    "legend.borderpad":     0.4,
    "legend.handlelength":  2.4,
    "legend.handleheight":  0.9,
    "legend.labelspacing":  0.30,
    # Output
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.02,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
})

# ── Per-configuration visual style ────────────────────────────────────────────
_STYLE: dict[str, dict] = {
    "raw":           {"color": "#d62728", "marker": "o",  "ls": "-",  "lw": 1.2, "ms": 4.0},
    "readout":       {"color": "#1f77b4", "marker": "s",  "ls": "-",  "lw": 1.2, "ms": 4.0},
    "parity":        {"color": "#2ca02c", "marker": "^",  "ls": "-",  "lw": 1.2, "ms": 4.0},
    "zne":           {"color": "#ff7f0e", "marker": "D",  "ls": "-",  "lw": 1.2, "ms": 4.0},
    "ro+parity":     {"color": "#9467bd", "marker": "v",  "ls": "--", "lw": 1.2, "ms": 4.0},
    "ro+zne":        {"color": "#8c564b", "marker": "P",  "ls": "--", "lw": 1.2, "ms": 4.0},
    "parity+zne":    {"color": "#e377c2", "marker": "X",  "ls": "--", "lw": 1.2, "ms": 4.0},
    "ro+parity+zne": {"color": "#222222", "marker": "*",  "ls": "--", "lw": 1.8, "ms": 5.5},
}

# ── Panel-specific style overrides ────────────────────────────────────────────
# When generating panels for 3×3 or 1×3 grids, use thicker lines and larger
# markers/fonts so they survive LaTeX shrinking to ~0.31\textwidth.
_PANEL_STYLE: dict[str, dict] = {
    "raw":           {"color": "#d62728", "marker": "o",  "ls": "-",  "lw": 1.6, "ms": 5.0},
    "readout":       {"color": "#1f77b4", "marker": "s",  "ls": "-",  "lw": 1.6, "ms": 5.0},
    "parity":        {"color": "#2ca02c", "marker": "^",  "ls": "-",  "lw": 1.6, "ms": 5.0},
    "zne":           {"color": "#ff7f0e", "marker": "D",  "ls": "-",  "lw": 1.6, "ms": 5.0},
    "ro+parity":     {"color": "#9467bd", "marker": "v",  "ls": "--", "lw": 1.6, "ms": 5.0},
    "ro+zne":        {"color": "#8c564b", "marker": "P",  "ls": "--", "lw": 1.6, "ms": 5.0},
    "parity+zne":    {"color": "#e377c2", "marker": "X",  "ls": "--", "lw": 1.6, "ms": 5.0},
    "ro+parity+zne": {"color": "#222222", "marker": "*",  "ls": "--", "lw": 2.0, "ms": 6.5},
}

_LABELS: dict[str, str] = {
    "raw":           "Raw",
    "readout":       "R",
    "parity":        "P",
    "zne":           "Z",
    "ro+parity":     "R + P",
    "ro+zne":        "R + Z",
    "parity+zne":    "P + Z",
    "ro+parity+zne": "R + P + Z",
}

# Groups used for the split relative-error plots
_INDIVIDUAL = ["raw", "readout", "parity", "zne"]
_COMBINED   = ["raw", "ro+parity", "ro+zne", "parity+zne", "ro+parity+zne"]

# Hardware palette
_HW_COLOR  = "#d62728"
_SIM_COLOR = "#1f77b4"


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved: %s", path)


def _legend_handles(configs: list[str], panel: bool = False) -> list[plt.Line2D]:
    style_dict = _PANEL_STYLE if panel else _STYLE
    return [
        plt.Line2D(
            [0], [0],
            color=style_dict[c]["color"],
            marker=style_dict[c]["marker"],
            linestyle=style_dict[c]["ls"],
            linewidth=style_dict[c]["lw"],
            markersize=style_dict[c]["ms"],
            label=_LABELS[c],
        )
        for c in configs
    ]


def _hstr(h: float) -> str:
    """Safe filename fragment for h value: 1.0 → 'h1p0'."""
    return f"h{h:.1f}".replace(".", "p")


# ═════════════════════════════════════════════════════════════════════════════
# §1  Data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_benchmark_results(folder: str) -> list[dict]:
    results = []
    for path in glob.glob(os.path.join(folder, "N*.json")):
        try:
            with open(path) as f:
                results.append(json.load(f))
        except Exception as exc:
            logger.warning("Could not load %s: %s", path, exc)
    if not results:
        logger.warning("No results found in %s", folder)
    else:
        logger.info("Loaded %d result files from %s", len(results), folder)
    return results


def _infer_axes(results: list[dict]) -> tuple[list, list, list]:
    Ns = sorted(set(r["N"] for r in results))
    hs = sorted(set(r["h"] for r in results))
    Ls = sorted(set(r["L"] for r in results))
    return Ns, hs, Ls


# ═════════════════════════════════════════════════════════════════════════════
# §2  Main benchmark plots  (shared by Brisbane and Fez)
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 2.1  Relative error — one panel per (h, L)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_rel_err_panel(
    results:  list[dict],
    h_val:    float,
    l_val:    int,
    configs:  list[str],
    Ns:       list[int],
    path:     str,
) -> None:
    """Render one relative-error panel at panel size and save to path."""
    subset = sorted(
        [r for r in results if r["h"] == h_val and r["L"] == l_val],
        key=lambda r: r["N"],
    )
    if not subset:
        return

    ns = [r["N"] for r in subset]

    # ── Panel-optimised figure ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(_PANEL_W, _PANEL_H))

    # Override font sizes for panel legibility
    ax.tick_params(labelsize=10)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)

    for cfg_name in configs:
        sty      = _PANEL_STYLE[cfg_name]
        rel_errs = np.array([r[cfg_name]["rel_err"] for r in subset])
        sems     = np.array([r[cfg_name]["sem"]     for r in subset])
        exact_a  = np.array([abs(r["exact"])        for r in subset])
        rel_sems = sems / np.maximum(exact_a, 1e-12)

        ax.semilogy(
            ns, rel_errs,
            color=sty["color"], marker=sty["marker"],
            linestyle=sty["ls"], linewidth=sty["lw"], markersize=sty["ms"],
            label=_LABELS[cfg_name],
        )
        ax.fill_between(
            ns,
            np.maximum(rel_errs - rel_sems, 1e-6),
            rel_errs + rel_sems,
            color=sty["color"], alpha=0.12, linewidth=0,
        )

    ax.set_xlabel("System size $N$ (qubits)")
    ax.set_ylabel(r"Relative error $\epsilon$")
    ax.set_xticks(ns)
    ncol = 2 if len(configs) > 4 else 1
    ax.legend(handles=_legend_handles(configs, panel=True),
              ncol=ncol, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8,
              handlelength=1.5, handletextpad=0.4,
              borderpad=0.3, labelspacing=0.25)
    fig.tight_layout()
    _save(fig, path)


def plot_relative_error(
    results:    list[dict],
    device_tag: str,
) -> None:
    Ns, hs, Ls = _infer_axes(results)

    for h_val in hs:
        for l_val in Ls:
            for variant, configs in [("individual", _INDIVIDUAL),
                                      ("combined",   _COMBINED)]:
                fname = (f"rel_err_{variant}_{device_tag}"
                         f"_{_hstr(h_val)}_L{l_val}.pdf")
                _plot_rel_err_panel(
                    results, h_val, l_val, configs, Ns,
                    os.path.join(CFG.plots_dir, fname),
                )


# ─────────────────────────────────────────────────────────────────────────────
# 2.2  VQE convergence — one panel per (L, h)
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(
    results:    list[dict],
    device_tag: str,
) -> None:
    Ns, hs, Ls = _infer_axes(results)
    cmap   = plt.cm.Blues
    shades = np.linspace(0.30, 0.85, len(Ns))

    for l_val in Ls:
        for h_val in hs:
            # ── Panel-optimised figure ────────────────────────────────────
            fig, ax = plt.subplots(figsize=(_PANEL_W, _PANEL_H))
            ax.tick_params(labelsize=10)
            ax.xaxis.label.set_size(11)
            ax.yaxis.label.set_size(11)

            for n_idx, N in enumerate(Ns):
                match = [r for r in results
                         if r["N"] == N and r["L"] == l_val and r["h"] == h_val]
                if not match:
                    continue

                conv_rel = match[0].get("convergence_rel")
                if conv_rel is None:
                    exact    = match[0]["exact"]
                    conv_raw = match[0].get("convergence_raw",
                                            match[0].get("convergence", []))
                    conv_rel = [
                        abs(e - exact) / abs(exact) if exact != 0 else float("nan")
                        for e in conv_raw
                    ]
                if not conv_rel:
                    continue

                ax.semilogy(
                    range(len(conv_rel)), conv_rel,
                    color=cmap(shades[n_idx]),
                    linewidth=1.4,
                    label=f"$N={N}$",
                )

            ax.set_xlabel("Optimiser iteration")
            ax.set_ylabel(r"Relative error $\epsilon$")
            ax.legend(ncol=2, loc="upper right", fontsize=8,
                      handlelength=1.2, handletextpad=0.3,
                      borderpad=0.3, labelspacing=0.2,
                      columnspacing=0.6)
            fig.tight_layout()

            fname = f"convergence_{device_tag}_L{l_val}_{_hstr(h_val)}.pdf"
            _save(fig, os.path.join(CFG.plots_dir, fname))


# ─────────────────────────────────────────────────────────────────────────────
# 2.3  Pareto frontier — one panel per h value
# ─────────────────────────────────────────────────────────────────────────────

def _pareto_mask(points: np.ndarray) -> np.ndarray:
    M = len(points)
    mask = np.ones(M, dtype=bool)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            if (points[j, 0] <= points[i, 0] and
                    points[j, 1] <= points[i, 1] and
                    (points[j, 0] < points[i, 0] or points[j, 1] < points[i, 1])):
                mask[i] = False
                break
    return mask


def plot_pareto(
    results:    list[dict],
    device_tag: str,
) -> None:
    Ns, hs, Ls = _infer_axes(results)
    non_raw = [c for c in _STYLE if c != "raw"]

    for h_val in hs:
        h_results = [r for r in results if r["h"] == h_val]

        config_pts: dict[str, tuple[float, float]] = {}
        for cfg_name in non_raw:
            overheads, errors = [], []
            for r in h_results:
                oh  = r[cfg_name].get("cost", {}).get("overhead_effective")
                err = r[cfg_name]["rel_err"]
                if oh is not None and not np.isnan(err):
                    overheads.append(oh)
                    errors.append(err)
            if overheads:
                config_pts[cfg_name] = (
                    float(np.median(overheads)),
                    float(np.median(errors)),
                )

        raw_errs = [r["raw"]["rel_err"] for r in h_results
                    if not np.isnan(r["raw"]["rel_err"])]
        if raw_errs:
            config_pts["raw"] = (1.0, float(np.median(raw_errs)))
        if not config_pts:
            continue

        names  = list(config_pts.keys())
        pts    = np.array([config_pts[n] for n in names])
        front  = _pareto_mask(pts)

        # ── Panel-optimised figure (displayed at 0.32\textwidth) ──────────
        fig, ax = plt.subplots(figsize=(_TRIPLET_W, _TRIPLET_H))
        ax.tick_params(labelsize=10)
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)

        for i, name in enumerate(names):
            sty = _PANEL_STYLE[name]
            x, y = pts[i]
            is_f = front[i]
            ax.scatter(
                x, y,
                color=sty["color"], marker=sty["marker"],
                s=70 if is_f else 35,
                edgecolors="0.3" if is_f else "none",
                linewidths=0.9,
                alpha=1.0 if is_f else 0.40,
                zorder=4,
            )
            if is_f:
                ax.annotate(
                    _LABELS[name], (x, y),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=8, color=sty["color"], fontweight="bold",
                )

        front_pts = pts[front]
        if len(front_pts) > 1:
            order = np.argsort(front_pts[:, 0])
            fp    = front_pts[order]
            ax.step(fp[:, 0], fp[:, 1], where="post",
                    color="0.35", linestyle="--", linewidth=1.0, alpha=0.6, zorder=3)

        disc_vals = [
            r["parity"]["parity_discard"] for r in h_results
            if not np.isnan(r["parity"].get("parity_discard", float("nan")))
        ]
        if disc_vals:
            ax.text(0.97, 0.97,
                    f"Median parity discard: {np.median(disc_vals):.1%}",
                    transform=ax.transAxes, fontsize=8,
                    ha="right", va="top", color="0.45")

        ax.set_xlabel(
            r"Effective circuit overhead $\Omega_\mathrm{eff}$  $(\times\,\mathrm{raw})$"
        )
        ax.set_ylabel(r"Median relative error $\epsilon$")
        ax.set_yscale("log")

        scatter_handles = [
            plt.scatter([], [], color=_PANEL_STYLE[c]["color"],
                        marker=_PANEL_STYLE[c]["marker"], s=35, label=_LABELS[c])
            for c in _STYLE
        ]
        ax.legend(handles=scatter_handles, ncol=2, loc="upper right",
                  fontsize=7.5, handletextpad=0.3, columnspacing=0.6)

        fig.tight_layout()
        fname = f"pareto_{device_tag}_{_hstr(h_val)}.pdf"
        _save(fig, os.path.join(CFG.plots_dir, fname))


# ─────────────────────────────────────────────────────────────────────────────
# 2.4  Heatmap helpers
# ─────────────────────────────────────────────────────────────────────────────

def _heatmap(
    ax:         plt.Axes,
    data:       np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    vmin:       float,
    vmax:       float,
    cmap:       str,
    fmt:        str = ".2f",
) -> matplotlib.image.AxesImage:
    im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap,
                   aspect="auto", origin="lower")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(labelleft=True)
    thresh = vmin + 0.6 * (vmax - vmin)
    for ni in range(len(row_labels)):
        for hi in range(len(col_labels)):
            val = data[ni, hi]
            if not np.isnan(val):
                ax.text(hi, ni, format(val, fmt),
                        ha="center", va="center", fontsize=9,
                        color="white" if val > thresh else "black")
    return im


def _heatmap_fig(data, row_labels, col_labels, vmin, vmax, cmap,
                 xlabel, ylabel, cbar_label, fmt=".2f"):
    """Single-panel heatmap figure at triplet size."""
    fig, ax = plt.subplots(figsize=(_TRIPLET_W, _TRIPLET_H))
    ax.tick_params(labelsize=10)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
    im = _heatmap(ax, data, row_labels, col_labels, vmin, vmax, cmap, fmt)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.5  Parity discard heatmap — one panel per L
# ─────────────────────────────────────────────────────────────────────────────

def plot_parity_discard(
    results:    list[dict],
    device_tag: str,
) -> None:
    Ns, hs, Ls = _infer_axes(results)
    row_labels = [f"$N={N}$" for N in Ns]
    col_labels = [f"$h={h}$" for h in hs]

    for l_val in Ls:
        data = np.full((len(Ns), len(hs)), np.nan)
        for ni, N in enumerate(Ns):
            for hi, h in enumerate(hs):
                match = [r for r in results
                         if r["N"] == N and r["h"] == h and r["L"] == l_val]
                if match:
                    data[ni, hi] = match[0]["parity"].get("parity_discard", np.nan)

        fig = _heatmap_fig(
            data, row_labels, col_labels,
            vmin=0.0, vmax=0.5, cmap="YlOrRd",
            xlabel="Transverse field $h$",
            ylabel="System size $N$",
            cbar_label=r"Discard fraction $f_\mathrm{discard}$",
        )
        fname = f"parity_discard_{device_tag}_L{l_val}.pdf"
        _save(fig, os.path.join(CFG.plots_dir, fname))


# ─────────────────────────────────────────────────────────────────────────────
# 2.6  Parity discard delta heatmap — one panel per L
# ─────────────────────────────────────────────────────────────────────────────

def plot_parity_discard_delta(
    results:    list[dict],
    device_tag: str,
) -> None:
    Ns, hs, Ls = _infer_axes(results)
    row_labels = [f"$N={N}$" for N in Ns]
    col_labels = [f"$h={h}$" for h in hs]

    for l_val in Ls:
        data = np.full((len(Ns), len(hs)), np.nan)
        for ni, N in enumerate(Ns):
            for hi, h in enumerate(hs):
                match = [r for r in results
                         if r["N"] == N and r["h"] == h and r["L"] == l_val]
                if match:
                    r  = match[0]
                    p  = r.get("parity",    {}).get("parity_discard", np.nan)
                    rp = r.get("ro+parity", {}).get("parity_discard", np.nan)
                    data[ni, hi] = p - rp

        valid   = data[~np.isnan(data)]
        abs_max = max(abs(valid.min()), abs(valid.max()), 0.02) if len(valid) else 0.1

        fig = _heatmap_fig(
            data, row_labels, col_labels,
            vmin=-abs_max, vmax=abs_max, cmap="RdBu",
            xlabel="Transverse field $h$",
            ylabel="System size $N$",
            cbar_label=r"$\Delta f_\mathrm{discard}$  (parity $-$ ro+parity)",
            fmt="+.2f",
        )
        fname = f"parity_discard_delta_{device_tag}_L{l_val}.pdf"
        _save(fig, os.path.join(CFG.plots_dir, fname))


# ─────────────────────────────────────────────────────────────────────────────
# 2.7  Convenience wrapper: all six benchmark plots for one device
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_benchmark(results: list[dict], device_tag: str) -> None:
    if not results:
        logger.warning("No results for device '%s' — skipping.", device_tag)
        return
    logger.info("Generating benchmark plots for %s (%d result files) …",
                device_tag, len(results))
    plot_relative_error(results, device_tag)
    plot_convergence(results, device_tag)
    plot_pareto(results, device_tag)
    plot_parity_discard(results, device_tag)
    plot_parity_discard_delta(results, device_tag)
    logger.info("Benchmark plots done for %s.", device_tag)


# ═════════════════════════════════════════════════════════════════════════════
# §3  ZNE noise study plots
# ═════════════════════════════════════════════════════════════════════════════

def plot_zne_noise_comparison(
    noise_model_results: list[dict],
    fname: str = "plot_zne_noise_comparison.pdf",
) -> None:
    """
    Grouped bar chart: raw vs ZNE relative error for each noise model.
    Full-width figure — displayed at 0.8\\textwidth in LaTeX.
    """
    if not noise_model_results:
        logger.error("plot_zne_noise_comparison: empty input.")
        return

    _NM_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    n_models   = len(noise_model_results)
    w          = 0.35

    fig, ax = plt.subplots(figsize=(_TW, _FH))

    for nm_idx, nm in enumerate(noise_model_results):
        exact  = nm["exact"]
        colour = _NM_COLORS[nm_idx % len(_NM_COLORS)]
        label  = nm.get("label", f"Model {nm_idx + 1}")

        raw_err = abs(nm["raw_energy"] - exact) / abs(exact)
        raw_sem = nm.get("raw_sem", 0.0) / abs(exact)
        zne_err = abs(nm["zne_energy"] - exact) / abs(exact)
        zne_sem = nm.get("zne_sem", 0.0) / abs(exact)

        offset_raw = (2 * nm_idx - n_models + 0.5) * w
        offset_zne = (2 * nm_idx - n_models + 1.5) * w

        ax.bar(offset_raw, raw_err, w,
               yerr=raw_sem, color=colour, alpha=0.85,
               capsize=3, error_kw={"linewidth": 0.9, "ecolor": "0.3"},
               label=f"{label} — raw")

        ax.bar(offset_zne, zne_err, w,
               yerr=zne_sem, color=colour, alpha=0.45, hatch="////",
               capsize=3, error_kw={"linewidth": 0.9, "ecolor": "0.3"},
               label=f"{label} — ZNE")

        if raw_err > 0:
            imp = (raw_err - zne_err) / raw_err * 100
            ax.text(offset_zne, zne_err + zne_sem + 0.004,
                    f"${imp:+.0f}\\%$",
                    ha="center", va="bottom", fontsize=7,
                    color=colour, fontweight="bold")

    ax.set_xticks([])
    ax.set_ylabel(r"Relative error $\epsilon$")
    ax.set_ylim(bottom=0)
    ax.text(0.5, -0.04, r"$N=4,\ h=1.0,\ L=2$",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color="0.4")
    ax.text(0.02, 0.97, "Solid: raw\nHatched: ZNE",
            transform=ax.transAxes, fontsize=7, va="top", ha="left",
            color="0.45")
    ax.legend(ncol=min(n_models, 3), loc="upper right",
              handletextpad=0.4, columnspacing=0.8)

    fig.tight_layout()
    _save(fig, os.path.join(CFG.plots_dir, fname))


def plot_zne_improvement_vs_ratio(
    sweep_data: dict,
    fname: str = "plot_zne_improvement_vs_ratio.pdf",
) -> None:
    """
    ZNE improvement fraction vs gate-to-readout noise ratio r.
    Full-width figure — displayed at 0.8\\textwidth in LaTeX.
    """
    exact  = sweep_data["exact"]
    points = sorted(sweep_data.get("ratio_sweep", []), key=lambda p: p["r"])

    if not points:
        logger.error("plot_zne_improvement_vs_ratio: 'ratio_sweep' is empty.")
        return

    rs           = np.array([p["r"]          for p in points])
    raw_energies = np.array([p["raw_energy"] for p in points])
    zne_energies = np.array([p["zne_energy"] for p in points])
    raw_errs     = np.abs(raw_energies - exact) / abs(exact)
    zne_errs     = np.abs(zne_energies - exact) / abs(exact)
    improvement  = np.where(raw_errs > 1e-10,
                            (raw_errs - zne_errs) / raw_errs,
                            0.0)

    imp_sems = None
    if all("raw_sem" in p and "zne_sem" in p for p in points):
        raw_sems = np.array([p["raw_sem"] / abs(exact) for p in points])
        zne_sems = np.array([p["zne_sem"] / abs(exact) for p in points])
        with np.errstate(divide="ignore", invalid="ignore"):
            imp_sems = np.sqrt(
                (zne_sems / np.maximum(raw_errs, 1e-10)) ** 2 +
                (zne_errs * raw_sems / np.maximum(raw_errs, 1e-10) ** 2) ** 2
            )

    fig, ax = plt.subplots(figsize=(_TW * 0.72, _FH * 0.90))

    if imp_sems is not None:
        ax.fill_between(rs, improvement - imp_sems, improvement + imp_sems,
                        color=_HW_COLOR, alpha=0.15, linewidth=0)

    ax.plot(rs, improvement, color=_HW_COLOR, linewidth=1.4)

    for r_val, device_label in [(0.31, "Brisbane"), (0.51, "Fez")]:
        ax.axvline(r_val, color="0.40", linestyle="--", linewidth=0.9, zorder=4)
        yhi = ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 0.97
        align = "right" if r_val == 0.31 else "left"
        mult = 0.96 if r_val == 0.31 else 1.04
        ax.text(r_val * mult, yhi * 0.96,
                f"{device_label}\n$r={r_val}$",
                fontsize=7,
                color="0.35",
                va="top",
                ha=align)

    ax.set_xscale("log")
    ax.set_xlabel(
        r"Gate-to-readout error ratio  "
        r"$r = p_{\mathrm{gate,\,2q}}\,/\,p_{\mathrm{readout}}$"
    )
    ax.set_ylabel("ZNE improvement fraction")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )

    N = sweep_data.get("N", "?")
    h = sweep_data.get("h", "?")
    L = sweep_data.get("L", "?")
    ax.text(0.97, 0.05, rf"$N={N},\ h={h},\ L={L}$",
            transform=ax.transAxes, fontsize=7.5,
            ha="right", va="bottom", color="0.4")

    fig.tight_layout()
    _save(fig, os.path.join(CFG.plots_dir, fname))


# ═════════════════════════════════════════════════════════════════════════════
# §4  Hardware validation plots  (ibm_fez)
# ═════════════════════════════════════════════════════════════════════════════

def plot_zne_scaling(
    scale_results: dict,
    zne_energy:    float,
    zne_sem:       float,
    exact:         float,
    backend_name:  str = "ibm\\_fez",
    fname:         str = "hardware_zne_scaling.pdf",
) -> None:
    """
    E(λ) vs ZNE noise scale factor on real hardware.
    Full-width figure — displayed at 0.8\\textwidth in LaTeX.
    """
    scales = sorted(scale_results.keys())
    means  = [scale_results[s]["mean"] for s in scales]
    sems   = [scale_results[s]["sem"]  for s in scales]

    fig, ax = plt.subplots(figsize=(_TW * 0.62, _FH * 0.85))

    ax.errorbar(scales, means, yerr=sems,
                fmt="o-", color=_HW_COLOR, linewidth=1.2,
                markersize=5, capsize=3, capthick=0.8,
                label=f"Hardware ({backend_name})")

    ax.errorbar([0], [zne_energy], yerr=[zne_sem],
                fmt="*", color=_HW_COLOR, markersize=8,
                capsize=3, capthick=0.8, zorder=5,
                label=f"ZNE (Richardson),  $E_0 = {zne_energy:.4f}$")

    ax.plot([0, scales[0]], [zne_energy, means[0]],
            color=_HW_COLOR, linestyle="--", linewidth=0.8, alpha=0.50)

    ax.axhline(exact, color="0.20", linestyle=":", linewidth=1.1,
               label=f"Exact (ED),  $E_0 = {exact:.4f}$")

    ax.set_xlabel(r"Noise scale factor $\lambda$")
    ax.set_ylabel(r"Energy $\langle H \rangle$ (J)")
    ax.set_xticks([0] + scales)
    ax.legend(loc="lower right")

    fig.tight_layout()
    _save(fig, os.path.join(CFG.plots_dir, fname))


def plot_hw_comparison(
    hw_results:   dict,
    sim_results:  dict,
    exact:        float,
    backend_name: str = "ibm\\_fez",
    fname:        str = "hardware_benchmark_comparison.pdf",
) -> None:
    """
    Grouped bar chart: FakeFez simulation vs real hardware relative error.
    Full-width figure — displayed at 0.8\\textwidth in LaTeX.
    """
    configs = list(MITIGATION_CONFIGS.keys())
    x       = np.arange(len(configs))
    w       = 0.32

    fig, ax = plt.subplots(figsize=(_TW, _FH))

    for i, cfg_name in enumerate(configs):
        colour = _STYLE[cfg_name]["color"]

        sim_err = sim_results.get(cfg_name, {}).get("rel_err", np.nan)
        if not np.isnan(sim_err):
            ax.bar(x[i] - w / 2, sim_err, w,
                   color=colour, alpha=0.38,
                   edgecolor=colour, linewidth=0.6)

        hw_err = hw_results.get(cfg_name, {}).get("rel_err", np.nan)
        if not np.isnan(hw_err):
            ax.bar(x[i] + w / 2, hw_err, w,
                   color=colour, alpha=0.90,
                   edgecolor="0.25", linewidth=0.6)

    sim_proxy = mpatches.Patch(facecolor="0.60", alpha=0.38,
                               edgecolor="0.60", linewidth=0.6,
                               label="Simulator (FakeFez)")
    hw_proxy  = mpatches.Patch(facecolor="0.25", alpha=0.90,
                               edgecolor="0.25", linewidth=0.6,
                               label=f"Hardware ({backend_name})")
    ax.legend(handles=[sim_proxy, hw_proxy], loc="upper left")

    ax.set_xticks(x)
    ax.set_xticklabels([_LABELS[c] for c in configs], rotation=30, ha="right")
    ax.set_ylabel(r"Relative error $\epsilon$")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    _save(fig, os.path.join(CFG.plots_dir, fname))


# ═════════════════════════════════════════════════════════════════════════════
# §5  Master execution block
# ═════════════════════════════════════════════════════════════════════════════

def generate_all_plots() -> None:
    """
    Load all data from disk and generate every figure.
    """

    BRISBANE_DIR   = "FakeBrisbane_results_updated_skip_rz"
    FEZ_DIR        = "FakeFez_results_updated_skip_rz"
    ZNE_STUDY_DIR  = "results_zne_study"
    HW_DIR         = "results_hardware"

    # ── Section A: FakeBrisbane benchmark ────────────────────────────────────
    brisbane_results = load_benchmark_results(BRISBANE_DIR)
    plot_all_benchmark(brisbane_results, device_tag="brisbane")

    # ── Section B: FakeFez benchmark ─────────────────────────────────────────
    fez_results = load_benchmark_results(FEZ_DIR)
    plot_all_benchmark(fez_results, device_tag="fez")

    # ── Section C: ZNE noise study ────────────────────────────────────────────

    _MODEL_LABELS = {
        "readout_dom": "Readout-dominated",
        "balanced": "Balanced",
        "gate_dom": "Gate-dominated",
    }

    noise_comp_path = os.path.join(ZNE_STUDY_DIR, "zne_study_N2_h1.0_L2.json")
    if os.path.exists(noise_comp_path):
        with open(noise_comp_path) as f:
            raw = json.load(f)

        _META_KEYS = {"N", "h", "L", "exact"}
        exact_nc = raw.get("exact", 0.0)

        nm_data = []
        for key, val in raw.items():
            if key in _META_KEYS or not isinstance(val, dict):
                continue
            entry = {
                "label": _MODEL_LABELS.get(key, key.replace("_", " ").title()),
                "r": val.get("r", float("nan")),
                "raw_energy": val["raw_energy"],
                "raw_sem": val["scale_sems"][0] if val.get("scale_sems") else 0.0,
                "zne_energy": val["extrapolated"],
                "zne_sem": val.get("extrap_sem", 0.0),
                "exact": exact_nc,
            }
            nm_data.append(entry)

        if nm_data:
            plot_zne_noise_comparison(nm_data)
        else:
            logger.warning("No noise model entries found in %s.", noise_comp_path)
    else:
        logger.warning("Missing %s — skipping plot_zne_noise_comparison.", noise_comp_path)

    sweep_path = os.path.join(ZNE_STUDY_DIR, "zne_ratio_sweep_N4_h1.0_L2.json")
    if os.path.exists(sweep_path):
        with open(sweep_path) as f:
            raw = json.load(f)

        _META_KEYS = {"N", "h", "L", "exact"}
        exact_sw = raw.get("exact", 0.0)

        ratio_sweep = []
        for key, val in raw.items():
            if key in _META_KEYS or not isinstance(val, dict):
                continue
            ratio_sweep.append({
                "r": val["r"],
                "raw_energy": val["raw_energy"],
                "raw_sem": val["scale_sems"][0] if val.get("scale_sems") else 0.0,
                "zne_energy": val["extrapolated"],
                "zne_sem": val.get("extrap_sem", 0.0),
            })

        sweep_data = {
            "N": raw.get("N", 4),
            "h": raw.get("h", 1.0),
            "L": raw.get("L", 2),
            "exact": exact_sw,
            "ratio_sweep": ratio_sweep,
        }
        plot_zne_improvement_vs_ratio(sweep_data)
    else:
        logger.warning("Missing %s — skipping plot_zne_improvement_vs_ratio.",
                       sweep_path)

    # ── Section D: Real hardware ──────────────────────────────────────────────

    scaling_path = os.path.join(HW_DIR, "zne_scaling_N4_h1.0_L2.json")
    if os.path.exists(scaling_path):
        with open(scaling_path) as f:
            d = json.load(f)
        scale_results = {int(k): v for k, v in d["scale_results"].items()}
        plot_zne_scaling(
            scale_results = scale_results,
            zne_energy    = d["zne_energy"],
            zne_sem       = d["zne_sem"],
            exact         = d["exact"],
            backend_name  = d.get("backend", "ibm\\_fez"),
        )
    else:
        logger.warning("Missing %s — skipping plot_zne_scaling.", scaling_path)

    hw_bench_path = os.path.join(HW_DIR, "benchmark_N4_h1.0_L2.json")
    fez_sim_path  = os.path.join(FEZ_DIR, "N4_h1.0_L2.json")

    if os.path.exists(hw_bench_path) and os.path.exists(fez_sim_path):
        with open(hw_bench_path) as f:
            hw_data = json.load(f)
        with open(fez_sim_path) as f:
            sim_data = json.load(f)

        hw_results  = hw_data.get("results", hw_data)
        sim_results = {
            c: {"rel_err": sim_data[c]["rel_err"]}
            for c in MITIGATION_CONFIGS if c in sim_data
        }
        plot_hw_comparison(
            hw_results   = hw_results,
            sim_results  = sim_results,
            exact        = hw_data.get("exact", sim_data.get("exact", 0.0)),
            backend_name = hw_data.get("backend", "ibm\\_fez"),
        )
    else:
        for p in [hw_bench_path, fez_sim_path]:
            if not os.path.exists(p):
                logger.warning("Missing %s — skipping plot_hw_comparison.", p)

    logger.info("All plots complete.  Output: %s/", CFG.plots_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    generate_all_plots()