"""
config.py
=========
Single source of truth for all benchmark hyperparameters.

Import this in every other module:
    from config import CFG, MITIGATION_CONFIGS, STYLE, PHASE_LABELS
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Immutable configuration for the VQE error mitigation benchmark.

    frozen=True: any attempt to modify a field after creation raises
    FrozenInstanceError. This prevents accidental mid-run modification
    and makes the config safe to pass across processes.

    All parameters documented below.
    """

    # --- Physical system ---
    system_sizes: tuple[int, ...] = (2, 4, 6, 8, 10)
    # N values to sweep. Must be even: parity post-selection requires the
    # TFIM ground state to lie in the even-parity sector, which holds for
    # even N with open boundary conditions.

    h_fields: tuple[float, ...] = (0.5, 1.0, 2.0)
    # Transverse field strengths. 0.5=ordered, 1.0=critical, 2.0=disordered.

    J: float = 1.0
    # ZZ coupling. Fixed at 1.0 to set the energy scale.

    # --- Ansatz ---
    layers: tuple[int, ...] = (1, 2, 3)
    # Number of [RY block + CZ chain] layers. More layers = more expressible
    # but deeper circuits = more noise.

    # --- Shot counts ---
    shots_eval: int = 8192
    # Shots for final energy evaluation (mean + SEM reported from these).
    # Higher = lower shot noise. Standard choice in NISQ benchmarks.

    shots_optim: int = 1024
    # Shots during COBYLA optimisation. Reduced because the optimiser only
    # needs to distinguish better from worse parameters, not compute precise
    # energies. Final reported values always use shots_eval.

    shots_calibration: int = 2048
    # Shots for readout calibration circuits. 2048 gives ~0.3% precision on
    # readout error rates (typical ~1-3%), which is sufficient. Using
    # shots_eval=8192 here would be overcautious and waste ~6s per N.

    # --- Statistics ---
    n_reps: int = 10
    # Independent repetitions per energy evaluation for SEM estimation.
    # Each rep uses a different random seed -> statistically independent.
    # 10 reps gives SEM precision of ~30% (needs ~100 for 10% precision on SEM).

    # --- Optimisation ---
    n_restarts: int = 3
    # Random-start COBYLA runs per (N, h, L). Best result retained.
    # Guards against local minima and barren plateaus.

    maxiter: int = 300
    # Maximum COBYLA function evaluations per restart.

    rhobeg: float = float(np.pi / 4)
    # COBYLA initial simplex radius in parameter space (radians).
    # pi/4 ≈ 0.785 rad is a standard choice for angular parameters.

    # --- ZNE ---
    zne_scale_factors: tuple[int, ...] = (1, 3, 5)
    # Noise amplification factors. Must be odd integers >= 1.
    # 3 points enables quadratic Richardson extrapolation.

    # --- Infrastructure ---
    results_dir: str = "results"
    # Directory for incremental JSON result files.

    seed: int = 42
    # Global random seed for reproducibility (numpy + Aer simulator).

    max_parallel_threads_aer: int = 1
    # Aer's internal OpenMP thread count per process. Set to 1 when using
    # Python multiprocessing to avoid CPU core contention.


# Global config instance — import this everywhere
CFG = BenchmarkConfig()


# ---------------------------------------------------------------------------
# Mitigation configuration registry
# ---------------------------------------------------------------------------
# All 8 subsets of {readout, parity, zne}. Data-driven design: adding a new
# strategy means adding one entry here, not modifying any other code.

MITIGATION_CONFIGS: dict[str, dict[str, bool]] = {
    "raw":           {"use_readout": False, "use_parity": False, "use_zne": False},
    "readout":       {"use_readout": True,  "use_parity": False, "use_zne": False},
    "parity":        {"use_readout": False, "use_parity": True,  "use_zne": False},
    "zne":           {"use_readout": False, "use_parity": False, "use_zne": True},
    "ro+parity":     {"use_readout": True,  "use_parity": True,  "use_zne": False},
    "ro+zne":        {"use_readout": True,  "use_parity": False, "use_zne": True},
    "parity+zne":    {"use_readout": False, "use_parity": True,  "use_zne": True},
    "ro+parity+zne": {"use_readout": True,  "use_parity": True,  "use_zne": True},
}

# Configs to highlight as individual strategies (for single-method focus)
INDIVIDUAL_CONFIGS = ["raw", "readout", "parity", "zne"]
COMBINED_CONFIGS   = ["ro+parity", "ro+zne", "parity+zne", "ro+parity+zne"]


# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
# ColorBrewer qualitative palette — distinguishable for colourblind viewers
# and printable in greyscale. Consistent across all four plots.

STYLE: dict[str, dict] = {
    "raw":           {"color": "#e41a1c", "marker": "o", "ls": "-",  "lw": 1.5},
    "readout":       {"color": "#377eb8", "marker": "s", "ls": "-",  "lw": 1.5},
    "parity":        {"color": "#4daf4a", "marker": "^", "ls": "-",  "lw": 1.5},
    "zne":           {"color": "#ff7f00", "marker": "D", "ls": "-",  "lw": 1.5},
    "ro+parity":     {"color": "#984ea3", "marker": "v", "ls": "--", "lw": 1.5},
    "ro+zne":        {"color": "#a65628", "marker": "<", "ls": "--", "lw": 1.5},
    "parity+zne":    {"color": "#f781bf", "marker": ">", "ls": "--", "lw": 1.5},
    "ro+parity+zne": {"color": "#000000", "marker": "*", "ls": "-",  "lw": 2.5},
}

PHASE_LABELS: dict[float, str] = {
    0.5: "Ordered (h=0.5)",
    1.0: "Critical (h=1.0)",
    2.0: "Disordered (h=2.0)",
}