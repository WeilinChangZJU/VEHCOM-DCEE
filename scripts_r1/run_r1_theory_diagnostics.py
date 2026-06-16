from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_r1.uav_isac_core_r1 import ScenarioConfig, SimState, UavIsacSimulator


LABELS: Dict[str, str] = {
    "proposed_full": "DCEE",
    "full_candidate_exact": "FC-Common",
    "anchored_mobility": "Anchored-Mobility",
    "static_uav": "Static-Hovering",
    "freshness_priority": "Freshness-Priority",
    "regularized_execution": "Regularized-Execution",
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


def build_config(base: Dict[str, object], overrides: Dict[str, object] | None = None) -> ScenarioConfig:
    merged = dict(base)
    if overrides:
        merged.update(overrides)
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
        arrival_burst_prob=float(merged.get("arrival_burst_prob", 0.15)),
        arrival_burst_scale=float(merged.get("arrival_burst_scale", 2.0)),
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


def pre_action_service(sim: UavIsacSimulator, state: SimState, exo: Dict[str, np.ndarray], action_result) -> Dict[str, float]:
    p, tau, s, u = sim.unpack_action(action_result.resource_action)
    rates = sim.rate_vector(action_result.post_motion_point, p, tau, state)
    service_bits = rates * sim.cfg.delta_t
    q_success = sim.q_success_vector(action_result.post_motion_point, s, u, state)
    mu = sim.mu_reduction_vector(action_result.post_motion_point, s, u, state, exo)
    data_slack_vec = service_bits - exo["arrivals"]
    return {
        "sum_arrivals": float(np.sum(exo["arrivals"])),
        "sum_service_bits": float(np.sum(service_bits)),
        "data_service_slack": float(np.sum(data_slack_vec)),
        "data_service_slack_min": float(np.min(data_slack_vec)),
        "data_service_violation": float(np.sum(data_slack_vec) < 0.0),
        "mean_pre_q_success": float(np.mean(q_success)),
        "mean_pre_mu_reduction": float(np.mean(mu)),
    }


def run_method_trajectory(
    sim: UavIsacSimulator,
    method: str,
    seed: int,
    horizon: int,
    profile: str,
    v_value: float,
) -> pd.DataFrame:
    state = sim.reset(seed)
    rows: List[Dict[str, object]] = []
    cumulative_penalty = 0.0
    cumulative_data_slack = 0.0
    for slot in range(horizon):
        exo = sim.build_exogenous(state)
        action_result = sim.choose_action_r1(state, exo, method=method)
        service = pre_action_service(sim, state, exo, action_result)
        metrics = sim.apply_action_r1(state, exo, action_result)
        cumulative_penalty += float(metrics["total_penalty"])
        cumulative_data_slack += service["data_service_slack"]
        aoi_slack = sim.cfg.aoi_threshold - state.A
        uncertainty_slack = sim.cfg.uncertainty_threshold - state.U
        rows.append({
            "profile": profile,
            "seed": seed,
            "slot": slot,
            "method": method,
            "method_label": LABELS.get(method, method),
            "V": float(v_value),
            "sum_Q": float(np.sum(state.Q)),
            "mean_Q": float(np.mean(state.Q)),
            "sum_Y": float(np.sum(state.Y)),
            "sum_Z": float(np.sum(state.Z)),
            "sum_virtual_queue": float(np.sum(state.Y) + np.sum(state.Z)),
            "mean_virtual_aoi_queue": float(np.mean(state.Y)),
            "mean_virtual_unc_queue": float(np.mean(state.Z)),
            "mean_AoI": float(np.mean(state.A)),
            "mean_uncertainty": float(np.mean(state.U)),
            "aoi_excess": float(np.mean(np.maximum(state.A - sim.cfg.aoi_threshold, 0.0))),
            "uncertainty_excess": float(np.mean(np.maximum(state.U - sim.cfg.uncertainty_threshold, 0.0))),
            "data_service_slack": service["data_service_slack"],
            "data_service_slack_min": service["data_service_slack_min"],
            "data_service_violation": service["data_service_violation"],
            "aoi_slack": float(np.mean(aoi_slack)),
            "aoi_slack_min": float(np.min(aoi_slack)),
            "aoi_violation": float(np.mean(aoi_slack) < 0.0),
            "uncertainty_slack": float(np.mean(uncertainty_slack)),
            "uncertainty_slack_min": float(np.min(uncertainty_slack)),
            "uncertainty_violation": float(np.mean(uncertainty_slack) < 0.0),
            "running_data_service_slack": float(cumulative_data_slack / (slot + 1)),
            "total_penalty": float(metrics["total_penalty"]),
            "running_penalty": float(cumulative_penalty / (slot + 1)),
            "travel_distance": float(metrics["travel_distance"]),
            "runtime_ms": float(action_result.runtime_ms),
            "solver_success": bool(action_result.solver_diag.success),
            "solver_nit": int(action_result.solver_diag.nit),
            "inner_iterations": int(action_result.metadata.get("inner_iterations", action_result.solver_diag.nit)),
            "candidate_family_size": int(len(action_result.candidate_family.points)),
            **service,
        })
    return pd.DataFrame(rows)


def summarize_queue(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (profile, method), group in df.groupby(["profile", "method"], sort=False):
        final_rows = group.sort_values("slot").groupby("seed", sort=False).tail(1)
        rows.append({
            "profile": profile,
            "method": method,
            "method_label": LABELS.get(str(method), str(method)),
            "num_seeds": int(group["seed"].nunique()),
            "slots_per_seed": int(group["slot"].nunique()),
            "mean_real_backlog": float(group["sum_Q"].mean()),
            "final_real_backlog_mean": float(final_rows["sum_Q"].mean()),
            "mean_virtual_backlog": float(group["sum_virtual_queue"].mean()),
            "final_virtual_backlog_mean": float(final_rows["sum_virtual_queue"].mean()),
            "mean_penalty": float(group["total_penalty"].mean()),
            "mean_running_penalty_final": float(final_rows["running_penalty"].mean()),
            "mean_aoi_excess": float(group["aoi_excess"].mean()),
            "mean_uncertainty_excess": float(group["uncertainty_excess"].mean()),
            "mean_travel": float(group["travel_distance"].mean()),
            "solver_success_rate": float(group["solver_success"].astype(bool).mean()),
            "mean_runtime_ms": float(group["runtime_ms"].mean()),
        })
    return pd.DataFrame(rows)


def summarize_slater(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    components = [
        ("data_service", "data_service_slack", "data_service_violation"),
        ("aoi", "aoi_slack", "aoi_violation"),
        ("uncertainty", "uncertainty_slack", "uncertainty_violation"),
    ]
    for (profile, method), group in df.groupby(["profile", "method"], sort=False):
        for component, slack_col, violation_col in components:
            values = group[slack_col].astype(float)
            rows.append({
                "profile": profile,
                "method": method,
                "method_label": LABELS.get(str(method), str(method)),
                "component": component,
                "mean_slack": float(values.mean()),
                "p5_slack": float(values.quantile(0.05)),
                "violation_ratio": float(group[violation_col].astype(float).mean()),
                "records": int(len(group)),
            })
    return pd.DataFrame(rows)


def summarize_v_tradeoff(v_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (profile, v_value, method), group in v_df.groupby(["profile", "V", "method"], sort=False):
        rows.append({
            "profile": profile,
            "V": float(v_value),
            "method": method,
            "method_label": LABELS.get(str(method), str(method)),
            "num_seeds": int(group["seed"].nunique()),
            "mean_penalty": float(group["total_penalty"].mean()),
            "mean_real_backlog": float(group["sum_Q"].mean()),
            "mean_virtual_backlog": float(group["sum_virtual_queue"].mean()),
            "mean_aoi_excess": float(group["aoi_excess"].mean()),
            "mean_uncertainty_excess": float(group["uncertainty_excess"].mean()),
            "mean_travel": float(group["travel_distance"].mean()),
            "mean_runtime_ms": float(group["runtime_ms"].mean()),
            "solver_success_rate": float(group["solver_success"].astype(bool).mean()),
        })
    return pd.DataFrame(rows).sort_values(["V", "method"]).reset_index(drop=True)


def finite_gap_running(source: Path) -> pd.DataFrame:
    if not source.exists():
        return pd.DataFrame(columns=["profile", "seed", "slot", "method", "method_label", "finite_gap", "running_mean_finite_gap"])
    usecols = ["profile", "seed", "slot", "method", "finite_gap"]
    df = pd.read_csv(source, usecols=usecols)
    df = df.sort_values(["profile", "seed", "method", "slot"]).reset_index(drop=True)
    df["method_label"] = df["method"].map(lambda value: LABELS.get(str(value), str(value)))
    df["running_mean_finite_gap"] = (
        df.groupby(["profile", "seed", "method"], sort=False)["finite_gap"]
        .expanding()
        .mean()
        .reset_index(level=[0, 1, 2], drop=True)
    )
    return df


def write_table(df: pd.DataFrame, columns: List[str], out_path: Path, note: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = df[columns].copy()
    for col in table.columns:
        if col in {"method", "method_label", "component", "profile"}:
            continue
        if pd.api.types.is_numeric_dtype(table[col]):
            if "ratio" in col or "rate" in col:
                table[col] = table[col].astype(float).map(lambda value: f"{100.0 * value:.1f}\\%")
            else:
                table[col] = table[col].astype(float).map(lambda value: f"{value:.3f}")
    out_path.write_text(note + "\n" + table.to_latex(index=False, escape=False), encoding="utf-8")


def plot_queue(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    plot_methods = ["proposed_full", "full_candidate_exact", "anchored_mobility", "static_uav", "freshness_priority"]
    part = df[df["method"].isin(plot_methods)].copy()
    mean_by_slot = (
        part.groupby(["slot", "method", "method_label"], as_index=False)[["sum_Q", "sum_virtual_queue", "running_penalty"]]
        .mean()
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), constrained_layout=True)
    for method, group in mean_by_slot.groupby("method", sort=False):
        label = str(group["method_label"].iloc[0])
        axes[0].plot(group["slot"], group["sum_Q"], label=label)
        axes[1].plot(group["slot"], group["sum_virtual_queue"], label=label)
        axes[2].plot(group["slot"], group["running_penalty"], label=label)
    axes[0].set_ylabel("Real queue")
    axes[1].set_ylabel("Virtual queue")
    axes[2].set_ylabel("Running penalty")
    for ax in axes:
        ax.set_xlabel("Slot")
        ax.grid(True, alpha=0.3)
    axes[2].legend(frameon=False, fontsize=8)
    fig.savefig(outdir / "queue_virtual_trajectories.pdf")
    fig.savefig(outdir / "queue_virtual_trajectories.png", dpi=200)
    plt.close(fig)


def plot_v_tradeoff(summary: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = summary.sort_values("V")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), constrained_layout=True)
    axes[0].plot(df["V"], df["mean_penalty"], marker="o")
    axes[1].plot(df["V"], df["mean_real_backlog"], marker="o")
    axes[2].plot(df["V"], df["mean_virtual_backlog"], marker="o")
    axes[0].set_ylabel("Penalty")
    axes[1].set_ylabel("Real backlog")
    axes[2].set_ylabel("Virtual backlog")
    for ax in axes:
        ax.set_xlabel("V")
        ax.grid(True, alpha=0.3)
    fig.savefig(outdir / "v_tradeoff.pdf")
    fig.savefig(outdir / "v_tradeoff.png", dpi=200)
    plt.close(fig)


def plot_slater_boxplot(df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    dcee = df[df["method"] == "proposed_full"].copy()
    data = [
        dcee["data_service_slack"].astype(float).values,
        dcee["aoi_slack"].astype(float).values,
        dcee["uncertainty_slack"].astype(float).values,
    ]
    fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
    ax.boxplot(data, tick_labels=["Data", "AoI", "Uncertainty"], showfliers=False)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Empirical slack")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(outdir / "slater_slack_boxplot.pdf")
    fig.savefig(outdir / "slater_slack_boxplot.png", dpi=200)
    plt.close(fig)


def run(config: Dict[str, object], outdir: Path, tables_dir: Path, figures_dir: Path, log_file: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    profile = str(config.get("profile", "balanced"))
    base = dict(config.get("base", {}))
    seeds = [int(seed) for seed in config.get("seed_values", list(range(10)))]
    slots = int(config.get("slots", 200))
    methods = [str(method) for method in config.get("methods", ["proposed_full"])]
    v_values = [float(value) for value in config.get("v_values", [4, 8, 12, 18, 24, 32, 48])]
    v_method = str(config.get("v_method", "proposed_full"))

    queue_frames: List[pd.DataFrame] = []
    v_frames: List[pd.DataFrame] = []
    started = time.perf_counter()
    with log_file.open("w", encoding="utf-8") as log_handle:
        emit(f"Patch 5 theory diagnostics start: methods={methods}, seeds={seeds}, slots={slots}", log_handle)
        cfg = build_config(base)
        for method in methods:
            sim = UavIsacSimulator(cfg)
            for seed in seeds:
                job_start = time.perf_counter()
                frame = run_method_trajectory(sim, method, seed, slots, profile, cfg.V)
                queue_frames.append(frame)
                emit(
                    f"queue method={method} seed={seed} time={time.perf_counter() - job_start:.2f}s "
                    f"elapsed={time.perf_counter() - started:.2f}s",
                    log_handle,
                )

        for v_value in v_values:
            cfg_v = build_config(base, {"V": v_value})
            sim_v = UavIsacSimulator(cfg_v)
            for seed in seeds:
                job_start = time.perf_counter()
                frame = run_method_trajectory(sim_v, v_method, seed, slots, profile, v_value)
                v_frames.append(frame)
                emit(
                    f"vtrade V={v_value:g} method={v_method} seed={seed} "
                    f"time={time.perf_counter() - job_start:.2f}s elapsed={time.perf_counter() - started:.2f}s",
                    log_handle,
                )

    queue_df = pd.concat(queue_frames, ignore_index=True)
    v_df = pd.concat(v_frames, ignore_index=True)
    queue_summary = summarize_queue(queue_df)
    slater_summary = summarize_slater(queue_df)
    v_summary = summarize_v_tradeoff(v_df)

    finite_source = ROOT / str(config.get("finite_gap_source", ""))
    gap_df = finite_gap_running(finite_source)

    queue_df.to_csv(outdir / "queue_trajectories.csv", index=False)
    queue_summary.to_csv(outdir / "queue_summary.csv", index=False)
    slater_summary.to_csv(outdir / "slater_slack_summary.csv", index=False)
    v_summary.to_csv(outdir / "v_tradeoff_summary.csv", index=False)
    gap_df.to_csv(outdir / "finite_gap_running.csv", index=False)

    write_table(
        slater_summary,
        ["method_label", "component", "mean_slack", "p5_slack", "violation_ratio"],
        tables_dir / "table_slater_slack.tex",
        "% Empirical slack diagnostics. These are tested-regime diagnostics, not a proof of general feasibility.",
    )
    write_table(
        v_summary,
        ["V", "mean_penalty", "mean_real_backlog", "mean_virtual_backlog", "mean_aoi_excess", "mean_uncertainty_excess", "mean_travel", "solver_success_rate"],
        tables_dir / "table_v_tradeoff.tex",
        "% DCEE V-tradeoff diagnostics over matched seeds.",
    )

    plot_queue(queue_df, figures_dir)
    plot_v_tradeoff(v_summary, figures_dir)
    plot_slater_boxplot(queue_df, figures_dir)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R1 queue, Slater, finite-gap, and V diagnostics.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs_r1" / "theory_diagnostics_light.json")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results_r1" / "theory_diagnostics")
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "tables_r1")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures_r1")
    parser.add_argument("--log-file", type=Path, default=ROOT / "logs_r1" / "patch5_theory_diagnostics.log")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    run(read_config(args.config), args.outdir, args.tables_dir, args.figures_dir, args.log_file)
    print(f"Wrote {args.outdir / 'queue_trajectories.csv'}")
    print(f"Wrote {args.outdir / 'slater_slack_summary.csv'}")
    print(f"Wrote {args.outdir / 'v_tradeoff_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
