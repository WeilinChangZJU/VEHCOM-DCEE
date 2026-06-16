from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

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
}


def read_config(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def emit(message: str, log_handle) -> None:
    print(message)
    log_handle.write(message + "\n")
    log_handle.flush()


def build_config(base: Dict[str, object]) -> ScenarioConfig:
    return ScenarioConfig(
        num_vehicles=int(base.get("num_vehicles", 6)),
        num_targets=int(base.get("num_targets", 3)),
        benchmark_num_directions=int(base.get("benchmark_num_directions", 16)),
        arrival_rate=float(base.get("arrival_rate", 0.8)),
        V=float(base.get("V", 12.0)),
        rho_r=float(base.get("rho_r", 0.01)),
        eta_t=float(base.get("eta_t", 2.0)),
        g0_sens=float(base.get("g0_sens", 96.0)),
        mu_reduction_scale=float(base.get("mu_reduction_scale", 12.0)),
        c_Y=float(base.get("c_Y", 8.0)),
        c_Z=float(base.get("c_Z", 8.0)),
        aoi_threshold=float(base.get("aoi_threshold", 5.0)),
        uncertainty_threshold=float(base.get("uncertainty_threshold", 3.5)),
        init_x=float(base.get("init_x", 200.0)),
        init_y=float(base.get("init_y", 0.0)),
        shortlist_num_local_dirs=int(base.get("shortlist_num_local_dirs", 8)),
        shortlist_radius_scale=float(base.get("shortlist_radius_scale", 0.45)),
        shortlist_size=int(base.get("shortlist_size", 11)),
        sensing_success_gain=float(base.get("sensing_success_gain", 1.2)),
        refresh_uncertainty_gain=float(base.get("refresh_uncertainty_gain", 1.15)),
        nonrefresh_mu_factor=float(base.get("nonrefresh_mu_factor", 0.15)),
        inner_solver_maxiter=int(base.get("inner_solver_maxiter", 80)),
        inner_solver_ftol=float(base.get("inner_solver_ftol", 1e-4)),
        sun_horizon_steps=int(base.get("sun_horizon_steps", 2)),
        sun_outer_rounds=int(base.get("sun_outer_rounds", 1)),
        sun_num_candidates=int(base.get("sun_num_candidates", 8)),
    )


class RobustnessSimulator(UavIsacSimulator):
    def __init__(self, cfg: ScenarioConfig, channel_mode: str, road_variant: str, primitive_noise: float):
        super().__init__(cfg)
        self.channel_mode = channel_mode
        self.road_variant = road_variant
        self.primitive_noise = float(primitive_noise)
        self.primitive_context = "true"

    def set_context(self, context: str) -> None:
        self.primitive_context = context

    def _noise_active(self) -> bool:
        return self.primitive_context == "estimated" and self.primitive_noise > 0.0

    def _field_noise(self, y: np.ndarray, ref: np.ndarray, salt: float) -> float:
        y = np.asarray(y, dtype=float)
        ref = np.asarray(ref, dtype=float)
        phase = 0.011 * y[0] + 0.017 * y[1] + 0.013 * ref[0] - 0.019 * ref[1] + salt
        return math.sin(phase) + 0.5 * math.cos(0.7 * phase + salt)

    def _mode_multiplier(self, y: np.ndarray, ref: np.ndarray, sensing: bool = False) -> float:
        diff = np.asarray(y, dtype=float) - np.asarray(ref, dtype=float)
        d2d = float(np.linalg.norm(diff))
        if self.channel_mode == "log_distance":
            return 1.0
        if self.channel_mode == "los_nlos":
            theta = math.atan2(self.cfg.altitude, max(d2d, 1.0))
            p_los = 1.0 / (1.0 + 9.61 * math.exp(-0.16 * (math.degrees(theta) - 9.61)))
            nlos_loss = 0.18 if not sensing else 0.25
            return p_los + (1.0 - p_los) * nlos_loss
        if self.channel_mode == "shadowed":
            sigma = 0.22 if not sensing else 0.18
            return float(math.exp(sigma * self._field_noise(y, ref, 17.0 if not sensing else 23.0)))
        raise ValueError(f"Unknown channel mode: {self.channel_mode}")

    def channel_gain(self, y: np.ndarray, r: np.ndarray) -> float:
        base = super().channel_gain(y, r) * self._mode_multiplier(y, r, sensing=False)
        if self._noise_active():
            base *= math.exp(self.primitive_noise * self._field_noise(y, r, 101.0))
        return float(max(base, 1e-12))

    def sensing_gain(self, y: np.ndarray, z: np.ndarray) -> float:
        base = super().sensing_gain(y, z) * self._mode_multiplier(y, z, sensing=True)
        if self._noise_active():
            base *= math.exp(self.primitive_noise * self._field_noise(y, z, 211.0))
        return float(max(base, 1e-12))

    def road_cost(self, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float)
        if self.road_variant == "gaussian":
            base = super().road_cost(y)
        elif self.road_variant == "lane_side":
            center = super().road_cost(y)
            lane_hazard = 0.65 * (1.0 / (1.0 + math.exp(-(abs(float(y[1])) - 22.0) / 3.0)))
            base = 0.55 * center + lane_hazard
        elif self.road_variant == "hotspot":
            x0, y0 = 760.0, -12.0
            hotspot = math.exp(-0.5 * (((float(y[0]) - x0) / 95.0) ** 2 + ((float(y[1]) - y0) / 10.0) ** 2))
            base = 0.45 * super().road_cost(y) + 0.85 * hotspot
        else:
            raise ValueError(f"Unknown road risk variant: {self.road_variant}")
        if self._noise_active():
            base += self.primitive_noise * 0.4 * self._field_noise(y, np.array([600.0, 0.0]), 307.0)
        return float(np.clip(base, 0.0, 2.0))

    def estimated_exogenous(self, exo_true: Dict[str, np.ndarray], state: SimState) -> Dict[str, np.ndarray]:
        exo = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in exo_true.items()}
        if self.primitive_noise > 0.0:
            factors = []
            for j in range(self.cfg.num_targets):
                noise = self._field_noise(state.x, state.target_pos[j], 401.0 + j)
                factors.append(math.exp(self.primitive_noise * noise))
            exo["U_base"] = np.clip(exo["U_base"] * np.asarray(factors), 0.0, self.cfg.uncertainty_max)
        return exo


def physical_parameter_table(cfg: ScenarioConfig) -> pd.DataFrame:
    rows = [
        ("corridor length", f"{cfg.corridor_x_max - cfg.corridor_x_min:.0f}", "m", "2D road corridor length"),
        ("corridor width", f"{cfg.corridor_y_max - cfg.corridor_y_min:.0f}", "m", "lateral operating width"),
        ("UAV altitude", f"{cfg.altitude:.0f}", "m", "fixed-altitude baseline"),
        ("slot duration", f"{cfg.delta_t:.1f}", "s", "control slot duration"),
        ("UAV max speed", f"{cfg.v_max:.1f}", "m/s", "mobility constraint"),
        ("bandwidth", f"{cfg.total_bandwidth:.1f}", "MHz-equivalent", "communication bandwidth scale"),
        ("total power", f"{cfg.total_power:.1f}", "W-equivalent", "joint communication/sensing budget"),
        ("vehicle speed range", f"{cfg.vehicle_speed_min:.1f}-{cfg.vehicle_speed_max:.1f}", "m/s", "moving vehicle process"),
        ("target speed range", f"{cfg.target_speed_min:.1f}-{cfg.target_speed_max:.1f}", "m/s", "moving target process"),
        ("road-risk field", "Gaussian/hotspot/lane-side", "dimensionless", "HD-map or traffic-risk proxy"),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value", "unit", "interpretation"])


def simulate_case(
    cfg: ScenarioConfig,
    case_type: str,
    case_name: str,
    channel_mode: str,
    road_variant: str,
    primitive_noise: float,
    method: str,
    seed: int,
    slots: int,
) -> pd.DataFrame:
    sim = RobustnessSimulator(cfg, channel_mode=channel_mode, road_variant=road_variant, primitive_noise=primitive_noise)
    state = sim.reset(seed)
    rows: List[Dict[str, object]] = []
    for slot in range(slots):
        exo_true = sim.build_exogenous(state)
        exo_est = sim.estimated_exogenous(exo_true, state)
        sim.set_context("estimated")
        action_result = sim.choose_action_r1(state, exo_est, method=method)
        sim.set_context("true")
        metrics = sim.apply_action_r1(state, exo_true, action_result)
        rows.append({
            "case_type": case_type,
            "case_name": case_name,
            "channel_mode": channel_mode,
            "road_variant": road_variant,
            "primitive_noise": primitive_noise,
            "seed": seed,
            "slot": slot,
            "method": method,
            "method_label": LABELS.get(method, method),
            "total_penalty": float(metrics["total_penalty"]),
            "queue_backlog": float(metrics["queue_backlog"]),
            "virtual_backlog": float(metrics["mean_virtual_aoi_queue"] + metrics["mean_virtual_unc_queue"]),
            "aoi_excess": float(metrics["aoi_violation"]),
            "uncertainty_excess": float(metrics["uncertainty_violation"]),
            "travel_distance": float(metrics["travel_distance"]),
            "runtime_ms": float(action_result.runtime_ms),
            "solver_success": bool(action_result.solver_diag.success),
            "inner_iterations": int(action_result.metadata.get("inner_iterations", action_result.solver_diag.nit)),
            "candidate_family_size": int(len(action_result.candidate_family.points)),
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    keys = ["case_type", "case_name", "channel_mode", "road_variant", "primitive_noise", "method", "method_label"]
    for key_values, group in df.groupby(keys, sort=False):
        row = {key: value for key, value in zip(keys, key_values)}
        row.update({
            "num_seeds": int(group["seed"].nunique()),
            "slots_per_seed": int(group["slot"].nunique()),
            "mean_penalty": float(group["total_penalty"].mean()),
            "mean_backlog": float(group["queue_backlog"].mean()),
            "mean_virtual_backlog": float(group["virtual_backlog"].mean()),
            "mean_aoi_excess": float(group["aoi_excess"].mean()),
            "mean_uncertainty_excess": float(group["uncertainty_excess"].mean()),
            "mean_travel": float(group["travel_distance"].mean()),
            "mean_runtime_ms": float(group["runtime_ms"].mean()),
            "solver_success_rate": float(group["solver_success"].astype(bool).mean()),
            "mean_inner_iterations": float(group["inner_iterations"].mean()),
            "mean_candidate_family_size": float(group["candidate_family_size"].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def write_latex(df: pd.DataFrame, path: Path, columns: List[str], note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = df[columns].copy()
    for col in table.columns:
        if col in {"case_name", "channel_mode", "road_variant", "method_label"}:
            continue
        if pd.api.types.is_numeric_dtype(table[col]):
            if "rate" in col:
                table[col] = table[col].astype(float).map(lambda value: f"{100.0 * value:.1f}\\%")
            else:
                table[col] = table[col].astype(float).map(lambda value: f"{value:.3f}")
    path.write_text(note + "\n" + table.to_latex(index=False, escape=False), encoding="utf-8")


def plot_channel(summary: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    dcee = summary[(summary["case_type"] == "channel") & (summary["method"] == "proposed_full")].copy()
    if dcee.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    ax.bar(dcee["channel_mode"], dcee["mean_penalty"])
    ax.set_ylabel("Mean penalty")
    ax.set_xlabel("Channel mode")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(outdir / "channel_sensitivity.pdf")
    fig.savefig(outdir / "channel_sensitivity.png", dpi=200)
    plt.close(fig)


def plot_noise(summary: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    dcee = summary[(summary["case_type"] == "primitive_noise") & (summary["method"] == "proposed_full")].copy()
    if dcee.empty:
        return
    dcee = dcee.sort_values("primitive_noise")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    axes[0].plot(100.0 * dcee["primitive_noise"], dcee["mean_penalty"], marker="o")
    axes[1].plot(100.0 * dcee["primitive_noise"], dcee["mean_backlog"], marker="s")
    axes[0].set_ylabel("Mean penalty")
    axes[1].set_ylabel("Mean backlog")
    for ax in axes:
        ax.set_xlabel("Primitive noise (%)")
        ax.grid(True, alpha=0.3)
    fig.savefig(outdir / "primitive_noise_robustness.pdf")
    fig.savefig(outdir / "primitive_noise_robustness.png", dpi=200)
    plt.close(fig)


def run(config: Dict[str, object], outdir: Path, tables_dir: Path, figures_dir: Path, log_file: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cfg = build_config(dict(config["base"]))
    seeds = [int(seed) for seed in config.get("seed_values", [0, 1, 2, 3, 4])]
    slots = int(config.get("slots", 100))
    methods = [str(method) for method in config.get("methods", ["proposed_full"])]
    channel_modes = [str(mode) for mode in config.get("channel_modes", ["log_distance", "los_nlos", "shadowed"])]
    noise_levels = [float(level) for level in config.get("primitive_noise_levels", [0.0, 0.05, 0.10, 0.20])]
    road_variants = [str(name) for name in config.get("road_risk_variants", ["gaussian", "lane_side", "hotspot"])]

    physical_parameter_table(cfg).to_csv(outdir / "physical_parameter_table.csv", index=False)
    write_latex(
        physical_parameter_table(cfg),
        tables_dir / "table_physical_settings.tex",
        ["parameter", "value", "unit", "interpretation"],
        "% SI-interpretable scenario settings. Some quantities remain simulator-equivalent scales.",
    )

    frames: List[pd.DataFrame] = []
    started = time.perf_counter()
    with log_file.open("w", encoding="utf-8") as log_handle:
        emit(f"Patch 6 physical robustness start: seeds={seeds}, slots={slots}, methods={methods}", log_handle)
        cases: List[Dict[str, object]] = []
        for mode in channel_modes:
            cases.append({"case_type": "channel", "case_name": mode, "channel_mode": mode, "road_variant": "gaussian", "primitive_noise": 0.0})
        for level in noise_levels:
            cases.append({"case_type": "primitive_noise", "case_name": f"noise_{int(round(100 * level))}", "channel_mode": "log_distance", "road_variant": "gaussian", "primitive_noise": level})
        for variant in road_variants:
            cases.append({"case_type": "road_risk", "case_name": variant, "channel_mode": "log_distance", "road_variant": variant, "primitive_noise": 0.0})

        seen = set()
        unique_cases: List[Dict[str, object]] = []
        for case in cases:
            key = tuple(case.items())
            if key not in seen:
                seen.add(key)
                unique_cases.append(case)

        for case in unique_cases:
            for method in methods:
                for seed in seeds:
                    job_start = time.perf_counter()
                    frame = simulate_case(cfg, method=method, seed=seed, slots=slots, **case)
                    frames.append(frame)
                    emit(
                        f"case={case['case_type']}:{case['case_name']} method={method} seed={seed} "
                        f"time={time.perf_counter() - job_start:.2f}s elapsed={time.perf_counter() - started:.2f}s",
                        log_handle,
                    )

    raw = pd.concat(frames, ignore_index=True)
    summary = summarize(raw)
    raw.to_csv(outdir / "physical_robustness_slot_metrics.csv", index=False)
    summary.to_csv(outdir / "physical_robustness_summary.csv", index=False)

    channel = summary[summary["case_type"] == "channel"].copy()
    noise = summary[summary["case_type"] == "primitive_noise"].copy()
    road = summary[summary["case_type"] == "road_risk"].copy()
    channel.to_csv(outdir / "channel_sensitivity_summary.csv", index=False)
    noise.to_csv(outdir / "primitive_noise_summary.csv", index=False)
    road.to_csv(outdir / "road_risk_variant_summary.csv", index=False)

    write_latex(
        channel,
        tables_dir / "table_channel_sensitivity.tex",
        ["channel_mode", "method_label", "mean_penalty", "mean_backlog", "mean_aoi_excess", "mean_uncertainty_excess", "solver_success_rate"],
        "% Channel sensitivity diagnostics. These are model-sensitivity runs, not measured-channel validation.",
    )
    write_latex(
        noise,
        tables_dir / "table_primitive_noise.tex",
        ["primitive_noise", "method_label", "mean_penalty", "mean_backlog", "mean_aoi_excess", "mean_uncertainty_excess", "solver_success_rate"],
        "% Primitive-noise robustness diagnostics. Noise affects estimated primitives used for action selection.",
    )
    write_latex(
        road,
        tables_dir / "table_road_risk_variants.tex",
        ["road_variant", "method_label", "mean_penalty", "mean_backlog", "mean_aoi_excess", "mean_uncertainty_excess", "solver_success_rate"],
        "% Road-risk variant diagnostics. Road cost is interpreted as an HD-map or traffic-risk proxy.",
    )
    plot_channel(summary, figures_dir)
    plot_noise(summary, figures_dir)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R1 physical/channel/primitive robustness diagnostics.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs_r1" / "physical_robustness_light.json")
    parser.add_argument("--outdir", type=Path, default=ROOT / "results_r1" / "physical_robustness")
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "tables_r1")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures_r1")
    parser.add_argument("--log-file", type=Path, default=ROOT / "logs_r1" / "patch6_physical_robustness.log")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    run(read_config(args.config), args.outdir, args.tables_dir, args.figures_dir, args.log_file)
    print(f"Wrote {args.outdir / 'channel_sensitivity_summary.csv'}")
    print(f"Wrote {args.outdir / 'primitive_noise_summary.csv'}")
    print(f"Wrote {args.outdir / 'road_risk_variant_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
