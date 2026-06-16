from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_r1.uav_isac_core_r1 import ScenarioConfig, UavIsacSimulator


LABELS: Dict[str, str] = {
    "proposed_full": "DCEE",
    "full_candidate_exact": "FC-Common",
    "anchored_mobility": "Anchored-Mobility",
    "regularized_execution": "Regularized-Execution",
    "static_uav": "Static-Hovering",
    "freshness_priority": "Freshness-Priority",
    "myopic_reoptimization": "Myopic Re-optimization",
    "yang_go_lyapunov": "GC-Lyapunov",
    "sun_receding_ao": "RHC-Alt.",
}


def read_config(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def emit(message: str, log_handle) -> None:
    print(message)
    log_handle.write(message + "\n")
    log_handle.flush()


def scenario_config(base: Dict[str, object], scenario: Dict[str, object]) -> ScenarioConfig:
    merged = dict(base)
    merged.update(scenario)
    return ScenarioConfig(
        num_vehicles=int(merged.get("num_vehicles", 6)),
        num_targets=int(merged.get("num_targets", 3)),
        benchmark_num_directions=int(merged.get("benchmark_num_directions", 16)),
        arrival_rate=float(merged.get("arrival_rate", 0.8)),
        V=float(merged.get("V", 12.0)),
        rho_r=float(merged.get("rho_r", 0.01)),
        eta_t=float(merged.get("eta_t", 2.0)),
        g0_sens=float(merged.get("g0_sens", 96.0)),
        mu_reduction_scale=float(merged.get("mu_reduction_scale", 12.0)),
        c_Y=float(merged.get("c_Y", 8.0)),
        c_Z=float(merged.get("c_Z", 8.0)),
        aoi_threshold=float(merged.get("aoi_threshold", 5.0)),
        uncertainty_threshold=float(merged.get("uncertainty_threshold", 3.5)),
        init_x=float(merged.get("init_x", 200.0)),
        init_y=float(merged.get("init_y", 0.0)),
        shortlist_num_local_dirs=int(merged.get("shortlist_num_local_dirs", 8)),
        shortlist_radius_scale=float(merged.get("shortlist_radius_scale", 0.45)),
        shortlist_size=int(merged.get("shortlist_size", 11)),
        sensing_success_gain=float(merged.get("sensing_success_gain", 1.2)),
        refresh_uncertainty_gain=float(merged.get("refresh_uncertainty_gain", 1.15)),
        nonrefresh_mu_factor=float(merged.get("nonrefresh_mu_factor", 0.15)),
        inner_solver_maxiter=int(merged.get("inner_solver_maxiter", 80)),
        inner_solver_ftol=float(merged.get("inner_solver_ftol", 1e-4)),
        sun_horizon_steps=int(merged.get("sun_horizon_steps", 2)),
        sun_outer_rounds=int(merged.get("sun_outer_rounds", 1)),
        sun_num_candidates=int(merged.get("sun_num_candidates", 8)),
    )


def summarize_slots(slot_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    keys = ["scenario", "scale_axis", "scale_value", "method"]
    for key_values, group in slot_df.groupby(keys, sort=False):
        scenario, axis, value, method = key_values
        runtime = group["runtime_ms"].astype(float)
        solver_runtime = group["solver_runtime_ms"].astype(float)
        candidate_count = group["candidate_family_size"].astype(float)
        rows.append({
            "scenario": scenario,
            "scale_axis": axis,
            "scale_value": value,
            "method": method,
            "method_label": LABELS.get(str(method), str(method)),
            "records": int(len(group)),
            "mean_ms_per_slot": float(runtime.mean()),
            "p95_ms_per_slot": float(runtime.quantile(0.95)),
            "mean_solver_ms_per_slot": float(solver_runtime.mean()),
            "p95_solver_ms_per_slot": float(solver_runtime.quantile(0.95)),
            "inner_solves_per_slot": float(candidate_count.mean()),
            "serial_solve_count_per_slot": float(candidate_count.mean()),
            "parallelizable_solve_count_per_slot": float(candidate_count.mean()),
            "mean_solver_iterations_per_slot": float(group["inner_iterations"].astype(float).mean()),
            "p95_solver_iterations_per_slot": float(group["inner_iterations"].astype(float).quantile(0.95)),
            "solver_success_rate": float(group["solver_success"].astype(bool).mean()),
            "mean_candidate_family_size": float(candidate_count.mean()),
        })
    return pd.DataFrame(rows)


def write_latex_table(summary_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep = summary_df.copy()
    keep = keep[[
        "scenario",
        "method_label",
        "mean_ms_per_slot",
        "p95_ms_per_slot",
        "inner_solves_per_slot",
        "mean_solver_iterations_per_slot",
        "solver_success_rate",
        "mean_candidate_family_size",
    ]]
    keep = keep.rename(columns={
        "scenario": "Scenario",
        "method_label": "Method",
        "mean_ms_per_slot": "Mean ms/slot",
        "p95_ms_per_slot": "p95 ms/slot",
        "inner_solves_per_slot": "Solves/slot",
        "mean_solver_iterations_per_slot": "Iter./slot",
        "solver_success_rate": "Success",
        "mean_candidate_family_size": "Family size",
    })
    for col in ["Mean ms/slot", "p95 ms/slot", "Solves/slot", "Iter./slot", "Family size"]:
        keep[col] = keep[col].astype(float).map(lambda value: f"{value:.2f}")
    keep["Success"] = keep["Success"].astype(float).map(lambda value: f"{100.0 * value:.1f}\\%")
    note = (
        "% Runtime and scalability diagnostics. Values are computed from lightweight R1 simulator runs.\n"
        "% Solves/slot and parallelizable solve count both report candidate-level post-motion resource solves.\n"
    )
    out_path.write_text(note + keep.to_latex(index=False, escape=False), encoding="utf-8")


def plot_scaling(summary_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    dcee = summary_df[summary_df["method"] == "proposed_full"].copy()
    dcee = dcee[dcee["scale_axis"].isin(["K", "S", "shortlist"])]
    if dcee.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    for ax, axis_name, xlabel in zip(axes, ["K", "S", "shortlist"], ["Vehicles K", "Targets S", "Shortlist size"]):
        part = dcee[dcee["scale_axis"] == axis_name].copy()
        if part.empty:
            ax.set_visible(False)
            continue
        part["scale_value_float"] = part["scale_value"].astype(float)
        part = part.sort_values("scale_value_float")
        ax.plot(part["scale_value_float"], part["mean_ms_per_slot"], marker="o", label="mean")
        ax.plot(part["scale_value_float"], part["p95_ms_per_slot"], marker="s", linestyle="--", label="p95")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ms/slot")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle("DCEE Runtime Scaling")
    fig.savefig(outdir / "runtime_scaling.pdf")
    fig.savefig(outdir / "runtime_scaling.png", dpi=200)
    plt.close(fig)


def run(config: Dict[str, object], outdir: Path, tables_dir: Path, figures_dir: Path, log_file: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    base = dict(config.get("base", {}))
    scenarios = list(config.get("scenarios", []))
    seeds = [int(seed) for seed in config.get("seed_values", [0, 1])]
    slots = int(config.get("slots", 20))

    all_slots: List[pd.DataFrame] = []
    started = time.perf_counter()
    with log_file.open("w", encoding="utf-8") as log_handle:
        emit(f"Patch 4 runtime start: scenarios={len(scenarios)}, seeds={seeds}, slots={slots}", log_handle)
        for scenario in scenarios:
            scenario_name = str(scenario["name"])
            methods = [str(method) for method in scenario.get("methods", config.get("methods", ["proposed_full"]))]
            scale_axis = str(scenario.get("scale_axis", "method"))
            scale_value = str(scenario.get("scale_value", scenario_name))
            cfg = scenario_config(base, scenario)
            sim = UavIsacSimulator(cfg)
            for method in methods:
                for seed in seeds:
                    job_start = time.perf_counter()
                    _, slot_df = sim.simulate_r1(method, horizon=slots, seed=seed)
                    slot_df["scenario"] = scenario_name
                    slot_df["scale_axis"] = scale_axis
                    slot_df["scale_value"] = scale_value
                    slot_df["method"] = method
                    slot_df["seed"] = seed
                    slot_df["num_vehicles"] = cfg.num_vehicles
                    slot_df["num_targets"] = cfg.num_targets
                    slot_df["shortlist_size_cfg"] = cfg.shortlist_size
                    all_slots.append(slot_df)
                    emit(
                        f"scenario={scenario_name} method={method} seed={seed} "
                        f"time={time.perf_counter() - job_start:.2f}s elapsed={time.perf_counter() - started:.2f}s",
                        log_handle,
                    )

    slot_all = pd.concat(all_slots, ignore_index=True)
    summary = summarize_slots(slot_all)
    slot_all.to_csv(outdir / "runtime_slot_metrics.csv", index=False)
    summary.to_csv(outdir / "runtime_summary.csv", index=False)
    write_latex_table(summary, tables_dir / "table_runtime_scalability.tex")
    plot_scaling(summary, figures_dir)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight R1 runtime and scalability diagnostics.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs_r1" / "runtime_scalability_light.json")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results_r1" / "runtime_scalability")
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "tables_r1")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures_r1")
    parser.add_argument("--log-file", type=Path, default=ROOT / "logs_r1" / "patch4_runtime_scalability.log")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    run(read_config(args.config), args.outdir, args.tables_dir, args.figures_dir, args.log_file)
    print(f"Wrote {args.outdir / 'runtime_summary.csv'}")
    print(f"Wrote {args.tables_dir / 'table_runtime_scalability.tex'}")
    print(f"Wrote {args.figures_dir / 'runtime_scaling.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
