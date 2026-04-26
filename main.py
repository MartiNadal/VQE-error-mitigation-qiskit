"""
main.py
=======
Entry point for the VQE Error Mitigation Benchmark.

Run this file directly:
    python main.py

This script:
    1. Checks whether results already exist on disk (from a previous run).
       If they do, asks whether to reload or rerun.
    2. Runs the full benchmark sweep (or loads from disk).
    3. Generates all plots.
    4. Prints a summary table to the console.

Result management:
    Each (N, h, L) combination is saved to results/N{N}_h{h}_L{L}.json
    immediately after completion. If the script crashes mid-run, completed
    combinations are safe on disk. Re-running OVERWRITES existing files
    (since "w" mode is used in json.dump). To keep multiple runs, either:
        - Rename the results/ directory before re-running:
              mv results results_run1
        - Or change results_dir in config.py:
              results_dir: str = "results_run2"

    To reload results without re-running the benchmark:
        from benchmark import load_results
        all_results = load_results()
        (then call the plotting functions with all_results)
"""

from __future__ import annotations
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing
import os
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        RotatingFileHandler("vqe_benchmark.log",
                            maxBytes=5 * 1024 * 1024,
                            backupCount=5,
                            encoding="utf-8"), # Saves to a file
        logging.StreamHandler()   # Still prints to console
    ]
)

logging.getLogger("qiskit").setLevel(logging.WARNING)
logging.getLogger("qiskit_aer").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

from config import CFG, INDIVIDUAL_CONFIGS
from benchmark import run_benchmark, load_results
from plotting import generate_all_plots


def print_summary_table(all_results: list[dict]) -> None:
    configs = ["raw", "readout", "parity", "zne",
               "ro+parity", "ro+zne", "parity+zne", "ro+parity+zne"]
    print("\n" + "="*100)
    print(f"{'N':>3}  {'h':>4}  {'L':>2}  {'exact':>10}  " +
          "  ".join(f"{c[:10]:>10}" for c in configs))
    print("="*100)
    for r in sorted(all_results, key=lambda r: (r["N"], r["h"], r["L"])):
        row = "  ".join(
            f"{r[c]['rel_err']:>10.4f}" if not np.isnan(r[c]['rel_err'])
            else f"{'nan':>10}"
            for c in configs
        )
        print(f"{r['N']:>3}  {r['h']:>4.1f}  {r['L']:>2}  {r['exact']:>10.4f}  {row}")
    print("="*100 + "\n")


def main() -> None:
    logger.info("="*60)
    logger.info("VQE Error Mitigation Benchmark")
    logger.info("="*60)

    results_dir = CFG.results_dir
    existing = [f for f in os.listdir(results_dir)
                if f.endswith(".json")] if os.path.isdir(results_dir) else []

    if existing:
        ans = input(f"Found {len(existing)} saved results. Load? [y/n]: ").strip().lower()
        all_results = load_results(results_dir) if ans == "y" else run_benchmark(CFG)
    else:
        all_results = run_benchmark(CFG)

    print_summary_table(all_results)

    logger.info("Generating plots...")

    generate_all_plots()

    logger.info("Done.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()