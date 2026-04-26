"""
hardware_benchmark.py
=====================
Full 8-configuration error mitigation benchmark on real IBM hardware.

Runs independently from main.py. Results saved to results_hardware/.

See hardware_zne_scaling.py for the full explanation of the circuit building
strategy, particularly the 156-qubit measurement fix using _get_physical_qubits()
and _measure_n_qubits().

CIRCUIT COUNT (N=4, n_reps=3)
------------------------------
    Readout calibration:      2×N = 8   at HW_CAL_SHOTS
    Non-ZNE configs (×4):     4 × 2 bases × 3 reps          = 24
    ZNE configs (×4):         4 × 3 scales × 2 bases × 3 reps = 72
    Total in one job:         104  (IBM limit: 300)

Usage
-----
    python hardware_EM_benchmark.py --backend ibm_fez
    python hardware_EM_benchmark.py --backend ibm_fez --dry-run
    python hardware_EM_benchmark.py --backend ibm_fez --no-readout-cal
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("qiskit").setLevel(logging.WARNING)
logging.getLogger("qiskit_aer").setLevel(logging.WARNING)
logger = logging.getLogger("hw_benchmark")

# ── Qiskit — hardware-specific ────────────────────────────────────────────────
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import XGate, SXGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# ── Project imports ───────────────────────────────────────────────────────────
from config import CFG, MITIGATION_CONFIGS, STYLE
from hamiltonian import get_exact_energy
from ansatz import build_ansatz
from energy import compute_zz_energy, compute_x_energy
from mitigation.zne import apply_zne_folding, zne_extrapolate, zne_error_propagation
from mitigation.readout import apply_readout_mitigation
from mitigation.parity import parity_post_selection
from benchmark import make_simulator
from optimizer import run_vqe

# ── IBM Quantum account ───────────────────────────────────────────────────────
# Note: Token is pulled from environment variables for security.
IBM_TOKEN    = os.environ.get("IBM_QUANTUM_TOKEN", "")
IBM_INSTANCE = "crn:v1:bluemix:public:quantum-computing:us-east:a/2c474368bfe24ffaaeb2d6e6453dcc8e:2ef5b970-de14-45e4-9afc-713f83e40df7::"
IBM_CHANNEL  = "ibm_cloud"

# ── Hardware study parameters ─────────────────────────────────────────────────
HW_N         = 4
HW_H         = 1.0
HW_L         = 2
HW_SHOTS     = 8192
HW_N_REPS    = 10
HW_CAL_SHOTS = 4096

RESULTS_DIR = "results_hardware"
SIM_RESULTS = os.path.join(CFG.results_dir, f"N{HW_N}_h{HW_H:.1f}_L{HW_L}.json")


# ─────────────────────────────────────────────────────────────────────────────
# Circuit helpers — shared logic with hardware_zne_scaling.py
# ─────────────────────────────────────────────────────────────────────────────
def _get_physical_qubits(qc_native: QuantumCircuit, n_logical: int) -> list[int]:
    """
    Return physical qubit indices [p0, p1, ..., p_{n-1}] in logical order.

    After transpilation to a 156-qubit device, the circuit uses n_logical
    physical qubits from the full device register. This function identifies
    them in logical order so measurements can be correctly associated with
    the TFIM spin indices used in compute_zz_energy / compute_x_energy.
    """
    if qc_native.layout is not None:
        try:
            phys = qc_native.layout.final_index_layout(filter_ancillas=True)
            return list(phys[:n_logical])
        except Exception as exc:
            logger.warning("layout.final_index_layout failed (%s), using fallback.", exc)

    # Fallback: qubits with gates, in order of first appearance
    seen: list[int] = []
    for inst in qc_native.data:
        if inst.operation.name not in ("barrier", "delay", "id", "measure", "reset"):
            for qubit in inst.qubits:
                idx = qc_native.find_bit(qubit).index
                if idx not in seen:
                    seen.append(idx)
                if len(seen) == n_logical:
                    return seen
    return seen[:n_logical]


def _decompose_sxdg(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Replace sxdg → [X, SX]. Equivalent up to global phase.
    sxdg is not in Heron's native gate set; IBM SamplerV2 rejects it.
    Proof: X·SX = SXdg (verified by matrix multiplication).
    """
    out = qc.copy()
    out.clear()
    for inst in qc.data:
        if inst.operation.name == "sxdg":
            out.append(XGate(), inst.qubits, [])
            out.append(SXGate(), inst.qubits, [])
        else:
            out.append(inst)
    return out


def _measure_n_qubits(qc: QuantumCircuit, phys_qubits: list[int]) -> QuantumCircuit:
    """
    Add ClassicalRegister(n, 'meas') and measure only the n ansatz physical
    qubits into it in logical order. Produces n-bit bitstrings.

    cr[i] ← physical qubit phys_qubits[i] = logical qubit i.
    After Qiskit little-endian reversal: bitstring[0] = logical qubit 0. ✓
    """
    n   = len(phys_qubits)
    cr  = ClassicalRegister(n, "meas")
    out = qc.copy()
    out.add_register(cr)
    for logical_i, phys_idx in enumerate(phys_qubits):
        out.measure(out.qubits[phys_idx], cr[logical_i])
    return out


def _apply_x_basis_rotation(qc: QuantumCircuit, phys_qubits: list[int]) -> QuantumCircuit:
    """
    Apply H = RZ(π/2)·SX·RZ(π/2) to each ansatz qubit only (not all 156).
    All three gates are native on Heron r2.
    """
    out = qc.copy()
    for phys_idx in phys_qubits:
        out.rz(np.pi / 2, out.qubits[phys_idx])
        out.sx(out.qubits[phys_idx])
        out.rz(np.pi / 2, out.qubits[phys_idx])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Parameter loading
# ─────────────────────────────────────────────────────────────────────────────
def load_best_params() -> np.ndarray:
    """Load optimal VQE parameters from simulator JSON; fallback to run_vqe."""
    if os.path.exists(SIM_RESULTS):
        with open(SIM_RESULTS) as f:
            data = json.load(f)
        if "best_params" in data:
            params = np.array(data["best_params"])
            logger.info("Loaded best_params from %s  shape=%s", SIM_RESULTS, params.shape)
            return params
        logger.warning("best_params not in JSON — running fallback VQE.")
    else:
        logger.warning("%s not found — running fallback VQE.", SIM_RESULTS)

    logger.info("Fallback VQE: N=%d  h=%.1f  L=%d", HW_N, HW_H, HW_L)
    sim       = make_simulator(threads=1)
    ansatz, _ = build_ansatz(HW_N, HW_L)
    params, _ = run_vqe(ansatz, HW_N, CFG.J, HW_H, sim, seed=CFG.seed)
    logger.info("Fallback VQE done | params shape: %s", params.shape)
    return params


# ─────────────────────────────────────────────────────────────────────────────
# Circuit building
# ─────────────────────────────────────────────────────────────────────────────
def build_calibration_circuits(
    N: int,
    backend,
    phys_qubits: list[int],
) -> tuple[list[QuantumCircuit], list[dict]]:
    """
    Build 2N readout calibration circuits pinned to the same physical qubits
    as the ansatz so calibration and evaluation are consistent.

    For qubit i (logical):
        - Circuit 2i:   all |0⟩, measure all N qubits → p(b|0) for qubit i
        - Circuit 2i+1: qubit i in |1⟩, measure all N qubits → p(b|1) for qubit i

    Uses initial_layout=phys_qubits so logical qubit i maps to phys_qubits[i].
    opt=0: basis translation only, no routing changes — calibration circuits
    are trivial (X + measure) so no routing is needed.
    """
    pm = generate_preset_pass_manager(
        optimization_level=0,
        backend=backend,
        initial_layout=phys_qubits,   # pin to same physical qubits as ansatz
    )
    circuits  = []
    index_map = []

    for qubit in range(N):
        # Prepare |0⟩ on all qubits (default state)
        qc0 = QuantumCircuit(N)
        qc0_compiled = pm.run(qc0)
        qc0_meas = _measure_n_qubits(qc0_compiled, phys_qubits)
        circuits.append(qc0_meas)
        index_map.append({"qubit": qubit, "state": 0})

        # Prepare |1⟩ on qubit `qubit`
        qc1 = QuantumCircuit(N)
        qc1.x(qubit)
        qc1_compiled = pm.run(qc1)
        qc1_meas = _measure_n_qubits(qc1_compiled, phys_qubits)
        circuits.append(qc1_meas)
        index_map.append({"qubit": qubit, "state": 1})

    logger.info("Built %d readout calibration circuits (pinned to qubits %s)",
                len(circuits), phys_qubits)
    return circuits, index_map


def build_evaluation_circuits(
    ansatz,
    params: np.ndarray,
    backend,
) -> tuple[list[QuantumCircuit], list[dict], list[int]]:
    """
    Build all 8-config evaluation circuits.

    Returns (circuits, index_map, phys_qubits).
    phys_qubits is shared with build_calibration_circuits so both use the
    same physical qubits, making readout correction valid.
    """
    pm        = generate_preset_pass_manager(optimization_level=1, backend=backend)
    qc_native = pm.run(ansatz.assign_parameters(params))

    phys_qubits = _get_physical_qubits(qc_native, HW_N)
    logger.info("Physical qubits (logical order): %s  (device has %d qubits)",
                phys_qubits, qc_native.num_qubits)

    qc_native = _decompose_sxdg(qc_native)

    # Pre-fold for all scale factors
    folded: dict[int, QuantumCircuit] = {}
    for scale in CFG.zne_scale_factors:
        if scale == 1:
            folded[scale] = qc_native.copy()
        else:
            folded[scale] = _decompose_sxdg(apply_zne_folding(qc_native, scale))

    circuits  = []
    index_map = []

    for config_name, cfg_flags in MITIGATION_CONFIGS.items():
        scales = CFG.zne_scale_factors if cfg_flags["use_zne"] else (1,)

        for scale in scales:
            qc_folded = folded[scale]

            for rep in range(HW_N_REPS):
                # Z-basis
                qc_z = _measure_n_qubits(qc_folded, phys_qubits)
                circuits.append(qc_z)
                index_map.append({
                    "config": config_name, "scale": scale,
                    "rep": rep, "basis": "Z",
                })

                # X-basis
                qc_x = _apply_x_basis_rotation(qc_folded, phys_qubits)
                qc_x = _measure_n_qubits(qc_x, phys_qubits)
                circuits.append(qc_x)
                index_map.append({
                    "config": config_name, "scale": scale,
                    "rep": rep, "basis": "X",
                })

    logger.info("Built %d evaluation circuits for %d configs",
                len(circuits), len(MITIGATION_CONFIGS))
    return circuits, index_map, phys_qubits


# ─────────────────────────────────────────────────────────────────────────────
# Hardware submission — single batched job
# ─────────────────────────────────────────────────────────────────────────────
def submit_all(
    cal_circuits: list[QuantumCircuit],
    eval_circuits: list[QuantumCircuit],
    backend,
    dry_run: bool = False,
):
    """
    Submit calibration + evaluation in ONE job — one queue wait.
    Per-PUB shot counts: calibration at HW_CAL_SHOTS, evaluation at HW_SHOTS.
    """
    n_cal   = len(cal_circuits)
    n_eval  = len(eval_circuits)
    n_total = n_cal + n_eval

    if n_total > 300:
        logger.error("Total circuits %d exceeds IBM limit 300. Reduce HW_N_REPS.", n_total)
        sys.exit(1)

    if dry_run:
        logger.info("DRY RUN: %d cal + %d eval = %d total circuits",
                    n_cal, n_eval, n_total)
        return None, None

    logger.info("Submitting %d circuits (%d cal + %d eval) to %s...",
                n_total, n_cal, n_eval, backend.name)
    pubs = (
            [(qc, [], HW_CAL_SHOTS) for qc in cal_circuits] +
            [(qc, [], HW_SHOTS) for qc in eval_circuits]
    )
    job    = Sampler(backend).run(pubs)
    logger.info("Job ID: %s | Waiting...", job.job_id())
    t0     = time.perf_counter()
    result = job.result()
    logger.info("Job complete in %.1f s", time.perf_counter() - t0)
    return result[:n_cal], result[n_cal:]


# ─────────────────────────────────────────────────────────────────────────────
# Result processing
# ─────────────────────────────────────────────────────────────────────────────
def _get_probs(result, idx: int, shots: int) -> dict[str, float]:
    counts = result[idx].data.meas.get_counts()
    return {b: c / shots for b, c in counts.items()}


def extract_readout_matrices(
    cal_result,
    cal_index_map: list[dict],
    N: int,
) -> list[np.ndarray]:
    """
    Build per-qubit readout correction matrices from calibration data.
    Returns inverted matrices in same format as get_readout_matrices() in
    mitigation/readout.py so apply_readout_mitigation() works directly.

    Bitstrings are N-bit (not 156-bit) because calibration circuits also
    used _measure_n_qubits(). Bit i = logical qubit i. Marginalising over
    all other qubits to get per-qubit readout error rates.
    """
    qubit_probs: dict[int, dict[int, dict]] = {q: {} for q in range(N)}
    for i, meta in enumerate(cal_index_map):
        qubit_probs[meta["qubit"]][meta["state"]] = _get_probs(
            cal_result, i, HW_CAL_SHOTS
        )

    inv_matrices = []
    for qubit in range(N):
        p0 = qubit_probs[qubit][0]
        p1 = qubit_probs[qubit][1]

        def marginal(probs_dict: dict, bit_val: int, q: int) -> float:
            return sum(
                prob for bs, prob in probs_dict.items()
                if int(bs[::-1][q]) == bit_val
            )

        p00 = marginal(p0, 0, qubit)
        p10 = marginal(p0, 1, qubit)
        p01 = marginal(p1, 0, qubit)
        p11 = marginal(p1, 1, qubit)

        logger.info("  Qubit %d | p(1|0)=%.4f  p(0|1)=%.4f", qubit, p10, p01)
        M = np.array([[p00, p01], [p10, p11]])
        try:
            inv_matrices.append(np.linalg.inv(M))
        except np.linalg.LinAlgError:
            logger.warning("  Qubit %d: singular matrix → using identity.", qubit)
            inv_matrices.append(np.eye(2))

    return inv_matrices


def process_all_configs(
    eval_result,
    eval_index_map: list[dict],
    readout_matrices: Optional[list[np.ndarray]],
    exact: float,
) -> dict[str, dict]:
    """
    Reconstruct per-config energy statistics.

    Mirrors run_single_combination in benchmark.py exactly:
    apply_readout_mitigation → parity_post_selection →
    compute_zz_energy + compute_x_energy → zne_extrapolate.

    All mitigation functions imported from project — identical code path
    to the simulator benchmark. Bitstrings are N-bit so energy functions
    receive exactly what they expect.
    """
    acc: dict = {
        cfg: {
            scale: {"Z": {}, "X": {}}
            for scale in (CFG.zne_scale_factors if flags["use_zne"] else (1,))
        }
        for cfg, flags in MITIGATION_CONFIGS.items()
    }
    for i, meta in enumerate(eval_index_map):
        acc[meta["config"]][meta["scale"]][meta["basis"]][meta["rep"]] = \
            _get_probs(eval_result, i, HW_SHOTS)

    results = {}
    for config_name, cfg_flags in MITIGATION_CONFIGS.items():
        use_readout = cfg_flags["use_readout"]
        use_parity  = cfg_flags["use_parity"]
        use_zne     = cfg_flags["use_zne"]
        scales      = CFG.zne_scale_factors if use_zne else (1,)

        scale_means, scale_sems_arr, scale_discards = [], [], []

        for scale in scales:
            rep_energies, rep_discards = [], []

            for rep in range(HW_N_REPS):
                probs_z = acc[config_name][scale]["Z"][rep]
                probs_x = acc[config_name][scale]["X"][rep]

                # apply_readout_mitigation — from mitigation/readout.py
                if use_readout and readout_matrices is not None:
                    counts_z = {b: int(p * HW_SHOTS) for b, p in probs_z.items()}
                    counts_x = {b: int(p * HW_SHOTS) for b, p in probs_x.items()}
                    probs_z  = apply_readout_mitigation(
                        counts_z, readout_matrices, HW_N, HW_SHOTS)
                    probs_x  = apply_readout_mitigation(
                        counts_x, readout_matrices, HW_N, HW_SHOTS)

                # parity_post_selection — from mitigation/parity.py
                discard_z, discard_x = 0.0, 0.0
                if use_parity:
                    probs_z, discard_z = parity_post_selection(probs_z)
                    probs_x, discard_x = parity_post_selection(probs_x)

                # compute_zz_energy / compute_x_energy — from energy.py
                energy = (compute_zz_energy(probs_z, HW_N, CFG.J)
                          + compute_x_energy(probs_x, HW_N, HW_H))
                rep_energies.append(energy)
                rep_discards.append(0.5 * (discard_z + discard_x))

            arr  = np.array(rep_energies)
            mean = float(np.mean(arr))
            sem  = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
            scale_means.append(mean)
            scale_sems_arr.append(sem)
            scale_discards.append(float(np.mean(rep_discards)))

        # zne_extrapolate / zne_error_propagation — from mitigation/zne.py
        if use_zne:
            final_mean = zne_extrapolate(scales, np.array(scale_means))
            final_sem  = zne_error_propagation(scales, np.array(scale_sems_arr))
        else:
            final_mean, final_sem = scale_means[0], scale_sems_arr[0]

        rel_err = abs(final_mean - exact) / abs(exact) if exact != 0 else float("nan")
        results[config_name] = {
            "mean": final_mean, "sem": final_sem, "rel_err": rel_err,
            "parity_discard": float(np.mean(scale_discards)),
        }
        logger.info("  [%s] E=%.4f ± %.4f  rel_err=%.4f",
                    config_name, final_mean, final_sem, rel_err)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────
def print_comparison_table(hw_results: dict, exact: float,
                           sim_results: Optional[dict], backend_name: str) -> None:
    print("\n" + "=" * 76)
    print(f"  HARDWARE BENCHMARK  |  N={HW_N}  h={HW_H}  L={HW_L}  |  {backend_name}")
    print("=" * 76)
    if sim_results:
        print(f"  {'Config':<16} {'Exact':>8} {'Sim err':>10} "
              f"{'HW err':>10}  {'Δ(HW-Sim)':>10}")
        print("  " + "-" * 58)
        for cfg in MITIGATION_CONFIGS:
            sim_err = sim_results.get(cfg, {}).get("rel_err", float("nan"))
            hw_err  = hw_results[cfg]["rel_err"]
            print(f"  {cfg:<16} {exact:>8.4f} {sim_err:>10.4f} "
                  f"{hw_err:>10.4f}  {hw_err - sim_err:>+10.4f}")
    else:
        print(f"  {'Config':<16} {'Exact':>8} {'HW mean':>12} {'HW err':>10}")
        print("  " + "-" * 48)
        for cfg in MITIGATION_CONFIGS:
            r = hw_results[cfg]
            print(f"  {cfg:<16} {exact:>8.4f} {r['mean']:>12.6f} "
                  f"{r['rel_err']:>10.4f}")
    print("=" * 76 + "\n")


def plot_comparison(hw_results: dict, exact: float,
                    sim_results: Optional[dict], backend_name: str) -> None:
    configs = list(MITIGATION_CONFIGS.keys())
    x       = np.arange(len(configs))
    width   = 0.35
    colours = [STYLE[c]["color"] for c in configs]
    hw_errs = [hw_results[c]["rel_err"] for c in configs]

    fig, ax = plt.subplots(figsize=(10, 4))
    offset  = width / 2 if sim_results else 0
    ax.bar(x + offset, hw_errs, width, color=colours, alpha=0.9,
           label=f"Hardware ({backend_name})")
    if sim_results:
        sim_errs = [sim_results.get(c, {}).get("rel_err", 0.0) for c in configs]
        ax.bar(x - width / 2, sim_errs, width, color=colours, alpha=0.35,
               label="Simulator (FakeFez)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Relative error  $|E - E_0| / |E_0|$", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")

    os.makedirs(CFG.plots_dir, exist_ok=True)
    path = os.path.join(CFG.plots_dir, "hardware_benchmark_comparison.pdf")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Plot saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(backend_name: str = "ibm_fez", dry_run: bool = False,
         no_readout_cal: bool = False) -> None:
    logger.info("=" * 64)
    logger.info("Hardware Benchmark  |  N=%d  h=%.1f  L=%d", HW_N, HW_H, HW_L)
    logger.info("=" * 64)

    if not dry_run:
        service = QiskitRuntimeService(
            channel=IBM_CHANNEL, token=IBM_TOKEN or None, instance=IBM_INSTANCE)
        backend = service.backend(backend_name)
        logger.info("Backend: %s  (%d qubits)", backend.name, backend.num_qubits)
    else:
        from qiskit_ibm_runtime.fake_provider import FakeFez
        backend = FakeFez()
        logger.info("DRY RUN — using FakeFez")

    exact       = get_exact_energy(HW_N, CFG.J, HW_H)
    best_params = load_best_params()
    ansatz, _   = build_ansatz(HW_N, HW_L)
    logger.info("Exact energy: %.6f", exact)

    # Build evaluation circuits first — needed to get phys_qubits
    eval_circuits, eval_index_map, phys_qubits = build_evaluation_circuits(
        ansatz, best_params, backend
    )

    # Build calibration circuits pinned to same physical qubits
    cal_circuits, cal_index_map = [], []
    if not no_readout_cal:
        cal_circuits, cal_index_map = build_calibration_circuits(
            HW_N, backend, phys_qubits
        )
    else:
        logger.warning("Readout calibration skipped.")

    # Single batched job
    cal_result, eval_result = submit_all(
        cal_circuits, eval_circuits, backend, dry_run=dry_run
    )
    if dry_run:
        logger.info("Dry run complete.")
        return

    # Extract readout matrices
    readout_matrices = None
    if cal_result is not None and cal_circuits:
        readout_matrices = extract_readout_matrices(
            cal_result, cal_index_map, HW_N
        )
        logger.info("Readout matrices: %d qubits calibrated.", len(readout_matrices))

    # Process evaluation results
    logger.info("Processing evaluation results...")
    hw_results = process_all_configs(
        eval_result, eval_index_map, readout_matrices, exact
    )

    # Load simulator results for comparison
    sim_config_results = None
    if os.path.exists(SIM_RESULTS):
        with open(SIM_RESULTS) as f:
            sim_data = json.load(f)
        sim_config_results = {
            cfg: {"rel_err": sim_data[cfg]["rel_err"], "mean": sim_data[cfg]["mean"]}
            for cfg in MITIGATION_CONFIGS if cfg in sim_data
        }

    print_comparison_table(hw_results, exact, sim_config_results, backend_name)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "N": HW_N, "h": HW_H, "L": HW_L, "backend": backend_name,
        "shots": HW_SHOTS, "n_reps": HW_N_REPS, "exact": exact,
        "physical_qubits": phys_qubits,
        "results": {cfg: dict(hw_results[cfg]) for cfg in hw_results},
        "simulator_results": sim_config_results,
    }
    path = os.path.join(RESULTS_DIR, f"benchmark_N{HW_N}_h{HW_H:.1f}_L{HW_L}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved: %s", path)

    plot_comparison(hw_results, exact, sim_config_results, backend_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full mitigation benchmark on IBM Quantum hardware")
    parser.add_argument("--backend", type=str, default="ibm_fez")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-readout-cal", action="store_true")
    args = parser.parse_args()
    main(backend_name=args.backend, dry_run=args.dry_run,
         no_readout_cal=args.no_readout_cal)