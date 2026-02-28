from benchmark import load_results

from config import CFG
from benchmark import run_benchmark, load_results
from plotting import (
    plot_relative_error,
    plot_absolute_energy,
    plot_convergence,
    plot_cost_vs_error,
    plot_error_scaling,
    plot_parity_discard
)

all_results = load_results()

plot_relative_error(all_results, CFG)
plot_absolute_energy(all_results, CFG)
plot_convergence(all_results, CFG)
plot_cost_vs_error(all_results, CFG)
plot_error_scaling(all_results, CFG)
plot_parity_discard(all_results, CFG)