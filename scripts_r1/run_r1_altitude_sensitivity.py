from __future__ import annotations

import argparse
import json
import math
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

from src_r1.uav_isac_core_r1 import ActionResult, ScenarioConfig, SimState, UavIsacSimulator


LABELS: Dict[str, str] = {
    "DCEE-2D": "DCEE-2D",
    "DCEE-3D": "DCEE-3D",
    "Fixed-h80": "Fixed h=80 m",
    "Fixed-h100": "Fixed h=100 m",
    "Fixed-h120": "Fixed h=120 m",
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
        altitude=float(base.get("altitude", 100.0)),
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


class AltitudeSimulator(UavIsacSimulator):
    """UAV-ISAC simulator with a lightweight altitude context.

    The horizontal state remains the same as the 2D simulator.  Altitude is a
    scalar context that affects channel/sensing gains and flight cost.
    """

    def __init__(self, cfg: ScenarioConfig, initial_altitude: float):
        super().__init__(cfg)
        self.current_altitude = float(initial_altitude)
        self.eval_altitude = float(initial_altitude)
        self.flight_from_altitude = float(initial_altitude)

    def set_altitude_context(self, target_altitude: float, from_altitude: float | None = None) -> None:
        self.eval_altitude = float(target_altitude)
        self.flight_from_altitude = self.current_altitude if from_altitude is None else float(from_altitude)

    def channel_gain(self, y: np.ndarray, r: np.ndarray) -> float:
        d2 = float(np.sum((y - r) ** 2) + self.eval_altitude ** 2)
        return self.cfg.g0_comm / (d2 ** (0.5 * self.cfg.pathloss_exp_comm))

    def channel_gain_grad(self, y: np.ndarray, r: np.ndarray) -> np.ndarray:
        diff = y - r
        d2 = float(np.sum(diff ** 2) + self.eval_altitude ** 2)
        g = self.channel_gain(y, r)
        return -self.cfg.pathloss_exp_comm * g * diff / d2

    def sensing_gain(self, y: np.ndarray, z: np.ndarray) -> float:
        d2 = float(np.sum((y - z) ** 2) + self.eval_altitude ** 2)
        return self.cfg.g0_sens / (d2 ** (0.5 * self.cfg.pathloss_exp_sens))

    def sensing_gain_grad(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        diff = y - z
        d2 = float(np.sum(diff ** 2) + self.eval_altitude ** 2)
        beta = self.sensing_gain(y, z)
        return -self.cfg.pathloss_exp_sens * beta * diff / d2

    def flight_cost(self, y: np.ndarray, x: np.ndarray) -> float:
        horizontal = float(np.linalg.norm(y - x))
        vertical = float(abs(self.eval_altitude - self.flight_from_altitude))
        return float(math.sqrt(horizontal * horizontal + vertical * vertical) / max(self.cfg.delta, 1e-9))

    def flight_cost_grad(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        diff = y - x
        horizontal_sq = float(np.dot(diff, diff))
        vertical = float(self.eval_altitude - self.flight_from_altitude)
        norm3d = math.sqrt(horizontal_sq + vertical * vertical)
        if norm3d <= 1e-12:
            return np.zeros(2, dtype=float)
        return diff / (max(self.cfg.delta, 1e-9) * norm3d)

    def apply_altitude_action(
        self,
        state: SimState,
        exo: Dict[str, np.ndarray],
        action_result: ActionResult,
        target_altitude: float,
    ) -> Dict[str, float]:
        previous_altitude = float(self.current_altitude)
        self.set_altitude_context(target_altitude, from_altitude=previous_altitude)
        pre_x = state.x.copy()
        metrics = self.apply_action_r1(state, exo, action_result)
        horizontal = float(np.linalg.norm(action_result.post_motion_point - pre_x))
        vertical = float(abs(float(target_altitude) - previous_altitude))
        metrics.update({
            "altitude_m": float(target_altitude),
            "previous_altitude_m": previous_altitude,
            "vertical_travel_m": vertical,
            "horizontal_travel_m": horizontal,
            "travel_3d_m": float(math.sqrt(horizontal * horizontal + vertical * vertical)),
        })
        self.current_altitude = float(target_altitude)
        self.set_altitude_context(self.current_altitude, from_altitude=self.current_altitude)
        return metrics


def altitude_candidates_reachable(candidates: Iterable[float], current: float, vertical_step: float) -> List[float]:
    return [float(h) for h in candidates if abs(float(h) - float(current)) <= float(vertical_step) + 1e-9]


def choose_action_with_altitude(
    sim: AltitudeSimulator,
    state: SimState,
    exo: Dict[str, np.ndarray],
    method: str,
    altitude_candidates: List[float],
    vertical_step: float,
) -> Tuple[ActionResult, float, Dict[str, object]]:
    if method == "DCEE-2D":
        target_altitude = float(sim.current_altitude)
        sim.set_altitude_context(target_altitude, from_altitude=sim.current_altitude)
        action = sim.choose_action_r1(state, exo, method="proposed_full")
        action.metadata["altitude_candidate_count"] = 1
        action.metadata["altitude_scores"] = json.dumps({f"{target_altitude:.1f}": float(action.value)})
        return action, target_altitude, {"altitude_mode": "fixed_2d"}

    if method.startswith("Fixed-h"):
        target_altitude = float(method.replace("Fixed-h", ""))
        sim.current_altitude = target_altitude
        sim.set_altitude_context(target_altitude, from_altitude=target_altitude)
        action = sim.choose_action_r1(state, exo, method="proposed_full")
        action.metadata["altitude_candidate_count"] = 1
        action.metadata["altitude_scores"] = json.dumps({f"{target_altitude:.1f}": float(action.value)})
        return action, target_altitude, {"altitude_mode": "fixed_height"}

    if method != "DCEE-3D":
        raise ValueError(f"Unknown altitude method: {method}")

    reachable = altitude_candidates_reachable(altitude_candidates, sim.current_altitude, vertical_step)
    if not reachable:
        reachable = [float(sim.current_altitude)]
    action_results: List[Tuple[float, ActionResult]] = []
    total_runtime = 0.0
    scores: Dict[str, float] = {}
    from_alt = float(sim.current_altitude)
    for h in reachable:
        sim.set_altitude_context(h, from_altitude=from_alt)
        result = sim.choose_action_r1(state, exo, method="proposed_full")
        result.metadata["target_altitude_m"] = float(h)
        result.metadata["previous_altitude_m"] = from_alt
        action_results.append((h, result))
        total_runtime += float(result.runtime_ms)
        scores[f"{h:.1f}"] = float(result.value)
    target_altitude, selected = max(action_results, key=lambda item: float(item[1].value))
    selected.runtime_ms = total_runtime
    selected.metadata["altitude_candidate_count"] = len(reachable)
    selected.metadata["altitude_scores"] = json.dumps(scores, sort_keys=True)
    selected.metadata["target_altitude_m"] = float(target_altitude)
    selected.metadata["previous_altitude_m"] = from_alt
    return selected, float(target_altitude), {"altitude_mode": "candidate_search", "reachable_altitudes": reachable}


def simulate_method(
    cfg: ScenarioConfig,
    method: str,
    seed: int,
    slots: int,
    altitude_candidates: List[float],
    initial_altitude: float,
    vertical_step: float,
) -> pd.DataFrame:
    fixed_initial = initial_altitude
    if method.startswith("Fixed-h"):
        fixed_initial = float(method.replace("Fixed-h", ""))
    sim = AltitudeSimulator(cfg, initial_altitude=fixed_initial)
    sim.set_altitude_context(fixed_initial, from_altitude=fixed_initial)
    state = sim.reset(seed)
    rows: List[Dict[str, object]] = []
    for slot in range(slots):
        exo = sim.build_exogenous(state)
        t0 = time.perf_counter()
        action_result, target_altitude, meta = choose_action_with_altitude(
            sim,
            state,
            exo,
            method=method,
            altitude_candidates=altitude_candidates,
            vertical_step=vertical_step,
        )
        metrics = sim.apply_altitude_action(state, exo, action_result, target_altitude=target_altitude)
        total_runtime_ms = 1000.0 * (time.perf_counter() - t0)
        diag = action_result.solver_diag
        rows.append({
            "seed": int(seed),
            "slot": int(slot),
            "method": method,
            "method_label": LABELS.get(method, method),
            "altitude_mode": str(meta.get("altitude_mode", "")),
            "altitude_m": float(target_altitude),
            "altitude_candidate_count": int(action_result.metadata.get("altitude_candidate_count", 1)),
            "altitude_scores": str(action_result.metadata.get("altitude_scores", "{}")),
            "runtime_ms": float(total_runtime_ms),
            "controller_runtime_ms": float(action_result.runtime_ms),
            "solver_success": bool(diag.success),
            "solver_nit": int(diag.nit),
            "solver_runtime_ms": float(diag.runtime_ms),
            "solver_residual": float(diag.residual),
            **metrics,
        })
    return pd.DataFrame(rows)


def seed_level_summary(slot_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "total_penalty",
        "queue_backlog",
        "mean_virtual_aoi_queue",
        "mean_virtual_unc_queue",
        "aoi_violation",
        "uncertainty_violation",
        "horizontal_travel_m",
        "vertical_travel_m",
        "travel_3d_m",
        "runtime_ms",
        "solver_success",
        "solver_nit",
        "altitude_candidate_count",
    ]
    seed_rows = []
    for (method, seed), group in slot_df.groupby(["method", "seed"], sort=False):
        row: Dict[str, object] = {
            "method": method,
            "method_label": LABELS.get(str(method), str(method)),
            "seed": int(seed),
            "slots": int(len(group)),
        }
        for metric in metrics:
            row[metric] = float(pd.to_numeric(group[metric], errors="coerce").mean())
        seed_rows.append(row)
    seed_df = pd.DataFrame(seed_rows)
    summary_rows = []
    for method, group in seed_df.groupby("method", sort=False):
        row = {
            "method": method,
            "method_label": LABELS.get(str(method), str(method)),
            "seeds": int(group["seed"].nunique()),
            "slots_per_seed": int(group["slots"].iloc[0]),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def altitude_counts(slot_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in slot_df.groupby("method", sort=False):
        total = max(len(group), 1)
        for altitude, sub in group.groupby("altitude_m", sort=True):
            rows.append({
                "method": method,
                "method_label": LABELS.get(str(method), str(method)),
                "altitude_m": float(altitude),
                "count": int(len(sub)),
                "fraction": float(len(sub) / total),
            })
    return pd.DataFrame(rows)


def fmt(row: pd.Series, metric: str, digits: int = 3) -> str:
    mean = float(row.get(f"{metric}_mean", np.nan))
    std = float(row.get(f"{metric}_std", 0.0))
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_latex_table(summary_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Penalty & Backlog & AoI exc. & Unc. exc. & Vert. m & Success \\\\",
        "\\midrule",
    ]
    for _, row in summary_df.iterrows():
        success = 100.0 * float(row["solver_success_mean"])
        lines.append(
            f"{row['method_label']} & {fmt(row, 'total_penalty')} & {fmt(row, 'queue_backlog')} & "
            f"{fmt(row, 'aoi_violation')} & {fmt(row, 'uncertainty_violation')} & "
            f"{fmt(row, 'vertical_travel_m')} & {success:.1f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(summary_df: pd.DataFrame, counts_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    x = np.arange(len(summary_df))
    axes[0].bar(x, summary_df["total_penalty_mean"], yerr=summary_df["total_penalty_std"], capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary_df["method_label"], rotation=25, ha="right")
    axes[0].set_ylabel("Mean penalty")
    axes[0].grid(True, axis="y", alpha=0.3)

    dcee3d = counts_df[counts_df["method"] == "DCEE-3D"]
    if not dcee3d.empty:
        axes[1].bar(dcee3d["altitude_m"].astype(str), dcee3d["fraction"])
        axes[1].set_ylabel("DCEE-3D selection fraction")
        axes[1].set_xlabel("Altitude (m)")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].grid(True, axis="y", alpha=0.3)
    else:
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R1 lightweight altitude sensitivity diagnostics.")
    parser.add_argument("--config", type=Path, default=Path("configs_r1/altitude_sensitivity_light.json"))
    parser.add_argument("--outdir", type=Path, default=Path("results_r1/altitude_sensitivity"))
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()

    config = read_config(args.config)
    outdir = args.outdir
    tables_dir = ROOT / "tables_r1"
    figures_dir = ROOT / "figures_r1"
    logs_dir = ROOT / "logs_r1"
    outdir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    with (logs_dir / "patch8_altitude_sensitivity.log").open("w", encoding="utf-8") as log:
        emit("Patch 8 altitude sensitivity", log)
        emit(f"Config: {args.config}", log)
        cfg = build_config(config.get("base", {}))
        slots = int(args.slots if args.slots is not None else config.get("slots", 100))
        seeds = [int(x) for x in (args.seeds if args.seeds is not None else config.get("seed_values", [0, 1, 2, 3, 4]))]
        methods = [str(x) for x in (args.methods if args.methods is not None else config.get("methods", ["DCEE-2D", "DCEE-3D"]))]
        altitude_candidates = [float(x) for x in config.get("altitude_candidates_m", [80.0, 100.0, 120.0])]
        initial_altitude = float(config.get("initial_altitude_m", cfg.altitude))
        vertical_step = float(config.get("vertical_step_m", 20.0))
        emit(f"Seeds={seeds}, slots={slots}, methods={methods}", log)
        emit(f"Altitude candidates={altitude_candidates}, vertical_step={vertical_step}", log)

        frames = []
        t0 = time.perf_counter()
        for method in methods:
            for seed in seeds:
                emit(f"Running {method} seed={seed}", log)
                frames.append(
                    simulate_method(
                        cfg,
                        method=method,
                        seed=seed,
                        slots=slots,
                        altitude_candidates=altitude_candidates,
                        initial_altitude=initial_altitude,
                        vertical_step=vertical_step,
                    )
                )
        slot_df = pd.concat(frames, ignore_index=True)
        summary_df = seed_level_summary(slot_df)
        counts_df = altitude_counts(slot_df)

        slot_df.to_csv(outdir / "altitude_slot_metrics.csv", index=False)
        summary_df.to_csv(outdir / "altitude_summary.csv", index=False)
        counts_df.to_csv(outdir / "altitude_selection_counts.csv", index=False)
        write_latex_table(summary_df, tables_dir / "table_altitude_sensitivity.tex")
        write_figure(summary_df, counts_df, figures_dir / "altitude_sensitivity.pdf")

        manifest = {
            "config": str(args.config),
            "seeds": seeds,
            "slots": slots,
            "methods": methods,
            "altitude_candidates_m": altitude_candidates,
            "initial_altitude_m": initial_altitude,
            "vertical_step_m": vertical_step,
            "wall_clock_sec": float(time.perf_counter() - t0),
        }
        (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        emit(f"Wrote outputs to {outdir}", log)
        emit(summary_df[["method_label", "total_penalty_mean", "queue_backlog_mean", "aoi_violation_mean", "uncertainty_violation_mean", "vertical_travel_m_mean", "solver_success_mean"]].to_string(index=False), log)


if __name__ == "__main__":
    main()
