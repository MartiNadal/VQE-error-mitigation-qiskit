"""
hardware_zne_scaling.py
=======================
ZNE noise scaling verification on real IBM Quantum hardware (ibm_fez).

Runs independently from main.py. Results saved to results_hardware/.

Scientific purpose
------------------
Verifies that physical noise on real hardware scales monotonically with the
ZNE gate-folding factor. On real hardware this is guaranteed by physics.

N=4, h=1.0, L=1:
    N=4   — smallest non-trivial TFIM system
    h=1.0 — quantum critical point, maximum entanglement
    L=1   — minimal depth to limit decoherence on hardware

Circuit building strategy
-------------------------
1. bind_parameters + pm_opt1.run()
   Transpiles the 4-qubit ansatz to native Heron gates on 4 physical qubits
   within the 156-qubit device register. The circuit object has 156 qubits
   but only 4 are active.

2. _get_physical_qubits()
   Extracts the 4 physical qubit indices (in logical order) from the layout.
   Critical: we must measure exactly these 4 qubits to get 4-bit bitstrings
   that compute_zz_energy / compute_x_energy can interpret correctly.

3. _decompose_sxdg()
   Replaces sxdg (not native on Heron) with [X, SX]. Produced by
   apply_zne_folding when inverting sx gates.

4. apply_zne_folding()
   Folds the native circuit. Called after _decompose_sxdg so all gates
   are already ISA-compliant going in.

5. Measurement
   measure_n_qubits(): adds a 4-bit ClassicalRegister and measures only
   the 4 ansatz physical qubits in logical order. Do NOT use measure_all()
   — that adds 156 measurements and breaks the energy calculation.

6. X-basis
   H = RZ(π/2)·SX·RZ(π/2) applied only to the 4 ansatz physical qubits.

Usage
-----
    python hardware_zne_scaling.py --backend ibm_fez
    python hardware_zne_scaling.py --backend ibm_fez --dry-run
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
logger = logging.getLogger("hw_zne_scaling")

# ── Qiskit — hardware-specific ────────────────────────────────────────────────
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit.library import XGate, SXGate
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

# ── Project imports — reused directly from the main benchmark ─────────────────
from config import CFG, STYLE
from hamiltonian import get_exact_energy
from ansatz import build_ansatz
from energy import compute_zz_energy, compute_x_energy
from mitigation.zne import apply_zne_folding, zne_extrapolate, zne_error_propagation
from benchmark import make_simulator   # FakeFez AerSimulator
from optimizer import run_vqe          # COBYLA optimiser

# ── IBM Quantum account ───────────────────────────────────────────────────────
IBM_TOKEN    = os.environ.get("IBM_QUANTUM_TOKEN", "")
IBM_INSTANCE = "crn:v1:bluemix:public:quantum-computing:us-east:a/2c474368bfe24ffaaeb2d6e6453dcc8e:2ef5b970-de14-45e4-9afc-713f83e40df7::"
IBM_CHANNEL  = "ibm_cloud"

# ── Hardware study parameters ─────────────────────────────────────────────────
HW_N      = 4
HW_H      = 1.0
HW_L      = 2
HW_SHOTS  = 8192
HW_N_REPS = 10

RESULTS_DIR = "results_hardware"
SIM_RESULTS = os.path.join(CFG.results_dir, f"N{HW_N}_h{HW_H:.1f}_L{HW_L}.json")


# ─────────────────────────────────────────────────────────────────────────────
# Circuit helpers
# ─────────────────────────────────────────────────────────────────────────────
def _get_physical_qubits(qc_native: QuantumCircuit, n_logical: int) -> list[int]:
    """
    Return the physical qubit indices [p0, p1, ..., p_{n-1}] where p_i is the
    physical qubit in the device register corresponding to logical qubit i.

    After generate_preset_pass_manager().run(), the circuit occupies the full
    device register (156 qubits for ibm_fez). Only n_logical of these have
    gates. This function identifies them in logical order using the transpile
    layout, which is essential for correct energy calculation.

    Without this, measure_all() would add 156 measurements and the energy
    functions would process 156-bit bitstrings — giving completely wrong values.
    """
    if qc_native.layout is not None:
        try:
            # final_index_layout(filter_ancillas=True)[i] = physical qubit for
            # logical qubit i, after all layout and routing passes
            phys = qc_native.layout.final_index_layout(filter_ancillas=True)
            return list(phys[:n_logical])
        except Exception as exc:
            logger.warning("layout.final_index_layout failed (%s), using fallback.", exc)

    # Fallback: find qubits with gates, sorted by first appearance
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

    sxdg is produced by apply_zne_folding when inverting sx gates but is
    not in Heron's native gate set {cz, id, rz, sx, x}. IBM's SamplerV2
    rejects circuits containing it since March 2024.

    Proof: X·SX = SXdg  (verified by matrix multiplication).
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
    Add a ClassicalRegister(n, 'meas') and measure only the n physical qubits
    in logical order. Returns a new circuit with measurements appended.

    cr[i] ← physical qubit phys_qubits[i]  (= logical qubit i)

    After reversal in compute_zz_energy / compute_x_energy (Qiskit little-endian):
        bitstring[0] = cr[0] = logical qubit 0  ✓
        bitstring[1] = cr[1] = logical qubit 1  ✓
        ...

    DO NOT use measure_all() on a device-mapped circuit — it measures all
    156 qubits and the energy functions receive 156-bit bitstrings.
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
    Apply H = RZ(π/2)·SX·RZ(π/2) to each of the n ansatz qubits only.
    All three gates are native on Heron r2.

    Applied BEFORE _measure_n_qubits(). Never applied to all 156 qubits
    — only the n active ansatz qubits at their physical indices.
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
    """
    Load optimal VQE parameters from simulator results JSON.
    Falls back to run_vqe() on FakeFez (same function as main benchmark).
    """
    if os.path.exists(SIM_RESULTS):
        with open(SIM_RESULTS) as f:
            data = json.load(f)
        if "best_params" in data:
            params = np.array(data["best_params"])
            logger.info("Loaded best_params from %s  shape=%s", SIM_RESULTS, params.shape)
            return params
        logger.warning("best_params not in JSON — running fallback VQE on FakeFez.")
    else:
        logger.warning("%s not found — running fallback VQE on FakeFez.", SIM_RESULTS)

    logger.info("Fallback VQE: N=%d  h=%.1f  L=%d", HW_N, HW_H, HW_L)
    sim       = make_simulator(threads=1)
    ansatz, _ = build_ansatz(HW_N, HW_L)
    params, _ = run_vqe(ansatz, HW_N, CFG.J, HW_H, sim, seed=CFG.seed)
    logger.info("Fallback VQE done | params shape: %s", params.shape)
    return params

def _fold_noisy_gates_only(qc: QuantumCircuit, scale: int) -> QuantumCircuit:
    """
    Fold only physically noisy gates: cz and sx.
    Skips rz and x gates.

    On Heron r2:
        rz  — virtual Z rotation, software frame change, zero error
        x   — single-qubit, very low error (~1E-3), skip to keep circuits short
        sx  — single-qubit physical pulse, ~1E-3 error, fold
        cz  — two-qubit physical pulse, ~3E-3 error, fold (dominant noise)

    By folding only cz and sx, the gate count increase accurately reflects
    the noise amplification factor, giving a cleaner E(lambda) slope for
    Richardson extrapolation.
    """
    from qiskit.circuit.library import XGate, SXGate

    # Gates to skip — either virtual (rz) or too low-error to matter (x, id)
    _SKIP = {"rz", "id", "delay", "x"}

    if scale == 1:
        return qc.copy()

    n_folds = (scale - 1) // 2
    folded  = qc.copy()
    folded.clear()

    for inst in qc.data:
        if inst.operation.name in ("barrier", "measure", "reset"):
            folded.append(inst)
            continue

        folded.append(inst)

        # Only fold physical noisy gates
        if inst.operation.name in _SKIP:
            continue

        try:
            inv_op = inst.operation.inverse()
        except Exception:
            logger.warning("Gate '%s' has no inverse, skipping fold.",
                           inst.operation.name)
            continue

        # Replace sxdg with [X, SX] immediately
        if inv_op.name == "sxdg":
            for _ in range(n_folds):
                folded.append(XGate(), inst.qubits, [])
                folded.append(SXGate(), inst.qubits, [])
                folded.append(inst)
        else:
            for _ in range(n_folds):
                folded.append(inv_op, inst.qubits, inst.clbits)
                folded.append(inst)

    return folded

# ─────────────────────────────────────────────────────────────────────────────
# Circuit building
# ─────────────────────────────────────────────────────────────────────────────
def build_zne_circuits(
    ansatz,
    params: np.ndarray,
    backend,
    scale_factors: tuple[int, ...],
) -> tuple[list[QuantumCircuit], list[dict], list[int]]:
    """
    Build all ZNE measurement circuits.

    Returns (circuits, index_map, phys_qubits).
    phys_qubits is returned so result processing knows which qubits to read.
    """
    logger.info("Building ZNE circuits | scales=%s  reps=%d", scale_factors, HW_N_REPS)

    # Transpile ansatz ONCE to native Heron gates + physical layout
    pm        = generate_preset_pass_manager(optimization_level=1, backend=backend)
    qc_native = pm.run(ansatz.assign_parameters(params))

    # Extract the 4 physical qubit indices in logical order
    phys_qubits = _get_physical_qubits(qc_native, HW_N)
    logger.info("Physical qubits (logical order): %s  (device has %d qubits)",
                phys_qubits, qc_native.num_qubits)

    # Remove sxdg before folding
    qc_native = _decompose_sxdg(qc_native)

    circuits  = []
    index_map = []

    for scale in scale_factors:
        # Fold native circuit — re-run _decompose_sxdg because folding
        # re-introduces sxdg via .inverse() on sx gates
        if scale > 1:
            qc_folded = _fold_noisy_gates_only(qc_native, scale)
        else:
            qc_folded = qc_native.copy()

        gate_count = sum(
            1 for inst in qc_folded.data
            if inst.operation.name not in ("barrier", "delay")
        )
        logger.info("  scale=%d | active gates=%d", scale, gate_count)

        for rep in range(HW_N_REPS):
            # Z-basis: measure the 4 ansatz qubits only
            qc_z = _measure_n_qubits(qc_folded, phys_qubits)
            circuits.append(qc_z)
            index_map.append({"scale": scale, "rep": rep, "basis": "Z"})

            # X-basis: H on 4 ansatz qubits, then measure
            qc_x = _apply_x_basis_rotation(qc_folded, phys_qubits)
            qc_x = _measure_n_qubits(qc_x, phys_qubits)
            circuits.append(qc_x)
            index_map.append({"scale": scale, "rep": rep, "basis": "X"})

    logger.info("Built %d circuits total (%d scales × %d reps × 2 bases)",
                len(circuits), len(scale_factors), HW_N_REPS)

    # Log first few for verification
    for i, qc in enumerate(circuits[:4]):
        logger.info("  Circuit %d | depth=%d  gates=%d  qubits=%d  cbits=%d",
                    i, qc.depth(), len(qc.data), qc.num_qubits, qc.num_clbits)

    return circuits, index_map, phys_qubits


# ─────────────────────────────────────────────────────────────────────────────
# Hardware submission
# ─────────────────────────────────────────────────────────────────────────────
def submit_job(circuits: list[QuantumCircuit], backend, shots: int,
               dry_run: bool = False):
    """Submit all circuits in one SamplerV2 job — one queue wait."""
    if dry_run:
        logger.info("DRY RUN: %d circuits × %d shots (not submitted)",
                    len(circuits), shots)
        for i, qc in enumerate(circuits[:4]):
            logger.info("  Circuit %d | depth=%d  gates=%d  qubits=%d  cbits=%d",
                        i, qc.depth(), len(qc.data), qc.num_qubits, qc.num_clbits)
        return None

    logger.info("Submitting %d circuits to %s at %d shots each...",
                len(circuits), backend.name, shots)
    job    = Sampler(backend).run([(qc,) for qc in circuits], shots=shots)
    logger.info("Job ID: %s | Waiting...", job.job_id())
    t0     = time.perf_counter()
    result = job.result()
    logger.info("Job complete in %.1f s", time.perf_counter() - t0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Result processing
# ─────────────────────────────────────────────────────────────────────────────
def _get_probs(result, idx: int, shots: int) -> dict[str, float]:
    """Extract normalised probability dict from SamplerV2 result at index idx."""
    counts = result[idx].data.meas.get_counts()
    return {b: c / shots for b, c in counts.items()}


def process_results(result, index_map: list[dict], shots: int,
                    exact: float) -> dict[int, dict]:
    """
    Reconstruct per-scale energies using compute_zz_energy and compute_x_energy
    from energy.py — identical to the main benchmark.

    Bitstrings are HW_N=4 bits (not 156) because we used _measure_n_qubits().
    compute_zz_energy / compute_x_energy receive exactly what they expect.
    """
    acc: dict[int, dict[int, float]] = {}

    for i, meta in enumerate(index_map):
        scale = meta["scale"]
        rep   = meta["rep"]
        basis = meta["basis"]
        probs = _get_probs(result, i, shots)

        acc.setdefault(scale, {}).setdefault(rep, 0.0)
        if basis == "Z":
            acc[scale][rep] += compute_zz_energy(probs, HW_N, CFG.J)
        else:
            acc[scale][rep] += compute_x_energy(probs, HW_N, HW_H)

    scale_results = {}
    for scale, rep_dict in acc.items():
        energies = [rep_dict[r] for r in sorted(rep_dict)]
        arr      = np.array(energies)
        mean     = float(np.mean(arr))
        sem      = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
        rel_err  = abs(mean - exact) / abs(exact)
        scale_results[scale] = {"energies": energies, "mean": mean,
                                "sem": sem, "rel_err": rel_err}
        logger.info("  scale=%d | E=%.6f ± %.6f | rel_err=%.4f",
                    scale, mean, sem, rel_err)

    return scale_results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_zne_scaling(scale_results: dict, zne_energy: float, zne_sem: float,
                     exact: float, backend_name: str) -> None:
    scales = sorted(scale_results.keys())
    means  = [scale_results[s]["mean"] for s in scales]
    sems   = [scale_results[s]["sem"]  for s in scales]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(scales, means, yerr=sems, fmt="o-", color="#e41a1c",
                linewidth=1.8, markersize=7, capsize=4,
                label=f"Hardware ({backend_name})")
    ax.errorbar([0], [zne_energy], yerr=[zne_sem], fmt="*", color="#e41a1c",
                markersize=14, capsize=4,
                label=f"ZNE extrapolated  $E_0 = {zne_energy:.4f}$")
    ax.plot([0, scales[0]], [zne_energy, means[0]],
            color="#e41a1c", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axhline(exact, color="black", linestyle="-", linewidth=1.5,
               label=f"Exact  $E_0 = {exact:.4f}$")
    ax.set_xlabel("Noise scale factor  $\\lambda$", fontsize=11)
    ax.set_ylabel("Energy  $\\langle H \\rangle$  (J)", fontsize=11)
    ax.set_xticks([0] + scales)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")

    os.makedirs(CFG.plots_dir, exist_ok=True)
    path = os.path.join(CFG.plots_dir, "hardware_zne_scaling.pdf")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("Plot saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(backend_name: str = "ibm_fez", dry_run: bool = False) -> None:
    logger.info("=" * 64)
    logger.info("Hardware ZNE Scaling  |  N=%d  h=%.1f  L=%d", HW_N, HW_H, HW_L)
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
    logger.info("Exact: %.6f  |  params shape: %s", exact, best_params.shape)

    circuits, index_map, phys_qubits = build_zne_circuits(
        ansatz, best_params, backend, CFG.zne_scale_factors
    )

    result = submit_job(circuits, backend, HW_SHOTS, dry_run=dry_run)
    if dry_run:
        logger.info("Dry run complete.")
        return

    logger.info("Processing results...")
    scale_results = process_results(result, index_map, HW_SHOTS, exact)

    scales_sorted = sorted(scale_results.keys())
    means_arr     = np.array([scale_results[s]["mean"] for s in scales_sorted])
    sems_arr      = np.array([scale_results[s]["sem"]  for s in scales_sorted])
    zne_energy    = zne_extrapolate(CFG.zne_scale_factors, means_arr)
    zne_sem       = zne_error_propagation(CFG.zne_scale_factors, sems_arr)
    raw_rel_err   = scale_results[1]["rel_err"]
    zne_rel_err   = abs(zne_energy - exact) / abs(exact)
    errs          = [scale_results[s]["rel_err"] for s in scales_sorted]
    monotone      = all(errs[i] < errs[i + 1] for i in range(len(errs) - 1))

    print("\n" + "=" * 64)
    print(f"  ZNE SCALING  |  N={HW_N}  h={HW_H}  L={HW_L}  |  {backend_name}")
    print("=" * 64)
    for s in scales_sorted:
        r = scale_results[s]
        print(f"  scale={s} | E={r['mean']:>10.6f} ± {r['sem']:.6f} | "
              f"rel_err={r['rel_err']:.4f}")
    print("-" * 64)
    print(f"  ZNE extrap | E={zne_energy:>10.6f} ± {zne_sem:.6f} | "
          f"rel_err={zne_rel_err:.4f}")
    print(f"  Exact      | E={exact:>10.6f}")
    print(f"  Improvement: {100*(raw_rel_err-zne_rel_err)/raw_rel_err:+.1f}%"
          f"  ({'PASS' if zne_rel_err < raw_rel_err else 'FAIL'})")
    print(f"  Noise monotone: {'YES' if monotone else 'NO'}")
    print("=" * 64 + "\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output = {
        "N": HW_N, "h": HW_H, "L": HW_L, "backend": backend_name,
        "shots": HW_SHOTS, "n_reps": HW_N_REPS, "exact": exact,
        "physical_qubits": phys_qubits,
        "scale_factors": list(CFG.zne_scale_factors),
        "scale_results": {str(s): scale_results[s] for s in scale_results},
        "zne_energy": zne_energy, "zne_sem": zne_sem,
        "zne_rel_err": zne_rel_err, "raw_rel_err": raw_rel_err,
        "noise_monotone": monotone,
    }
    path = os.path.join(RESULTS_DIR, f"zne_scaling_N{HW_N}_h{HW_H:.1f}_L{HW_L}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved: %s", path)

    plot_zne_scaling(scale_results, zne_energy, zne_sem, exact, backend_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZNE noise scaling on IBM hardware")
    parser.add_argument("--backend", type=str, default="ibm_fez")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(backend_name=args.backend, dry_run=args.dry_run)