from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_r1.uav_isac_core_r1 import ScenarioConfig, UavIsacSimulator, aggregate_runs


def parse_controllers(text: Optional[str]) -> List[str]:
    if text is None or text.strip() == "":
        return ["proposed_full", "static_uav"]
    return [item.strip() for item in text.split(",") if item.strip()]


def emit(message: str, log_handle) -> None:
    print(message)
    log_handle.write(message + "\n")
    log_handle.flush()


def build_config(args: argparse.Namespace) -> ScenarioConfig:
    return ScenarioConfig(
        num_vehicles=args.vehicles,
        num_targets=args.targets,
        benchmark_num_directions=args.directions,
        arrival_rate=args.arrival_rate,
        V=args.V,
        rho_r=args.rho,
        eta_t=args.eta,
        g0_sens=args.g0_sens,
        mu_reduction_scale=args.mu_scale,
        c_Y=args.cY,
        c_Z=args.cZ,
        aoi_threshold=args.aoi_thr,
        uncertainty_threshold=args.unc_thr,
        init_x=args.init_x,
        init_y=args.init_y,
        shortlist_num_local_dirs=args.shortlist_dirs,
        shortlist_radius_scale=args.shortlist_radius_scale,
        shortlist_size=args.shortlist_size,
        sensing_success_gain=args.q_gain,
        refresh_uncertainty_gain=args.refresh_mu_gain,
        nonrefresh_mu_factor=args.nonrefresh_mu_factor,
        inner_solver_maxiter=args.inner_maxiter,
        inner_solver_ftol=args.inner_ftol,
        sun_horizon_steps=args.sun_horizon_steps,
        sun_outer_rounds=args.sun_outer_rounds,
        sun_num_candidates=args.sun_num_candidates,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="R1 smoke runner for side-effect-free controller interface.")
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--controllers", type=str, default="proposed_full,static_uav")
    parser.add_argument("--vehicles", type=int, default=6)
    parser.add_argument("--targets", type=int, default=3)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--arrival-rate", type=float, default=0.8)
    parser.add_argument("--V", type=float, default=12.0)
    parser.add_argument("--rho", type=float, default=0.01)
    parser.add_argument("--eta", type=float, default=2.0)
    parser.add_argument("--g0-sens", type=float, default=96.0)
    parser.add_argument("--mu-scale", type=float, default=12.0)
    parser.add_argument("--shortlist-dirs", type=int, default=8)
    parser.add_argument("--shortlist-radius-scale", type=float, default=0.45)
    parser.add_argument("--shortlist-size", type=int, default=7)
    parser.add_argument("--q-gain", type=float, default=1.20)
    parser.add_argument("--refresh-mu-gain", type=float, default=1.15)
    parser.add_argument("--nonrefresh-mu-factor", type=float, default=0.15)
    parser.add_argument("--cY", type=float, default=8.0)
    parser.add_argument("--cZ", type=float, default=8.0)
    parser.add_argument("--aoi-thr", type=float, default=5.0)
    parser.add_argument("--unc-thr", type=float, default=3.5)
    parser.add_argument("--init-x", type=float, default=200.0)
    parser.add_argument("--init-y", type=float, default=0.0)
    parser.add_argument("--inner-maxiter", type=int, default=30)
    parser.add_argument("--inner-ftol", type=float, default=1e-4)
    parser.add_argument("--sun-horizon-steps", type=int, default=2)
    parser.add_argument("--sun-outer-rounds", type=int, default=1)
    parser.add_argument("--sun-num-candidates", type=int, default=6)
    parser.add_argument("--outdir", type=str, default=str(ROOT / "results_r1" / "r1_smoke"))
    parser.add_argument("--log-file", type=str, default=str(ROOT / "logs_r1" / "r1_smoke.log"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    log_file = Path(args.log_file)
    outdir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    controllers = parse_controllers(args.controllers)
    cfg = build_config(args)
    sim = UavIsacSimulator(cfg)
    summaries = []
    slots = []
    total_jobs = max(1, args.seeds * len(controllers))
    job_id = 0
    started = time.perf_counter()

    with log_file.open("w", encoding="utf-8") as log_handle:
        emit(f"R1 smoke start: controllers={controllers}, slots={args.slots}, seeds={args.seeds}", log_handle)
        for seed_idx in range(args.seeds):
            seed_value = 1000 + seed_idx
            for controller in controllers:
                job_id += 1
                job_start = time.perf_counter()
                summary_df, slot_df = sim.simulate_r1(controller, horizon=args.slots, seed=seed_value)
                summary_df["seed"] = seed_value
                slot_df["seed"] = seed_value
                summaries.append(summary_df)
                slots.append(slot_df)
                emit(
                    f"[{job_id}/{total_jobs}] controller={controller} seed={seed_value} "
                    f"job_time={time.perf_counter() - job_start:.3f}s elapsed={time.perf_counter() - started:.3f}s",
                    log_handle,
                )

        summary_all = pd.concat(summaries, ignore_index=True)
        slot_all = pd.concat(slots, ignore_index=True)
        agg = aggregate_runs([df.drop(columns=["seed"], errors="ignore") for df in summaries])
        summary_all.to_csv(outdir / "r1_smoke_summary_raw.csv", index=False)
        slot_all.to_csv(outdir / "r1_smoke_slot_metrics.csv", index=False)
        agg.to_csv(outdir / "r1_smoke_summary_agg.csv", index=False)
        show_cols = [c for c in ["runtime_ms", "inner_iterations", "solver_residual", "candidate_family_size", "total_penalty"] if c in summary_all.columns]
        emit(str(summary_all.groupby("controller")[show_cols].mean()), log_handle)
        emit(f"R1 smoke outputs written to: {outdir}", log_handle)


if __name__ == "__main__":
    main()
