from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_r1.benchmark_r1 import (
    RESIDUAL_FAIL_TOL,
    RESIDUAL_WARN_TOL,
    build_common_candidate_family,
    build_dense_candidate_family,
    build_union_candidate_family,
    candidate_rows_for_csv,
    collect_method_candidate_family,
    compute_union_gap,
    coverage_rows_for_families,
    evaluate_finite_family_with_exo,
    repeated_exact_solve_at_executed_point,
    solver_summary_from_eval,
    summarize_benchmark_slots,
    summarize_consistency_slots,
)
from src_r1.uav_isac_core_r1 import ScenarioConfig, UavIsacSimulator


DEFAULT_METHODS = [
    "proposed_full",
    "regularized_execution",
    "anchored_mobility",
    "static_uav",
    "full_candidate_exact",
    "yang_go_lyapunov",
]


def load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def methods_from_config(config: Dict[str, object], override: str | None) -> List[str]:
    if override:
        return [item.strip() for item in override.split(",") if item.strip()]
    configured = config.get("methods") or config.get("controllers") or DEFAULT_METHODS
    return [str(item) for item in configured]


def build_config(config: Dict[str, object]) -> ScenarioConfig:
    return ScenarioConfig(
        num_vehicles=int(config.get("num_vehicles", 4)),
        num_targets=int(config.get("num_targets", 2)),
        benchmark_num_directions=int(config.get("benchmark_num_directions", 6)),
        arrival_rate=float(config.get("arrival_rate", 0.8)),
        V=float(config.get("V", 12.0)),
        rho_r=float(config.get("rho_r", 0.01)),
        eta_t=float(config.get("eta_t", config.get("eta", 2.0))),
        g0_sens=float(config.get("g0_sens", 96.0)),
        mu_reduction_scale=float(config.get("mu_reduction_scale", config.get("mu_scale", 12.0))),
        c_Y=float(config.get("c_Y", config.get("cY", 8.0))),
        c_Z=float(config.get("c_Z", config.get("cZ", 8.0))),
        aoi_threshold=float(config.get("aoi_threshold", config.get("aoi_thr", 5.0))),
        uncertainty_threshold=float(config.get("uncertainty_threshold", config.get("unc_thr", 3.5))),
        init_x=float(config.get("init_x", 200.0)),
        init_y=float(config.get("init_y", 0.0)),
        shortlist_num_local_dirs=int(config.get("shortlist_num_local_dirs", config.get("shortlist_dirs", 4))),
        shortlist_radius_scale=float(config.get("shortlist_radius_scale", 0.45)),
        shortlist_size=int(config.get("shortlist_size", 5)),
        sensing_success_gain=float(config.get("sensing_success_gain", config.get("q_gain", 1.20))),
        refresh_uncertainty_gain=float(config.get("refresh_uncertainty_gain", config.get("refresh_mu_gain", 1.15))),
        nonrefresh_mu_factor=float(config.get("nonrefresh_mu_factor", 0.15)),
        inner_solver_maxiter=int(config.get("inner_solver_maxiter", config.get("inner_maxiter", 25))),
        inner_solver_ftol=float(config.get("inner_solver_ftol", config.get("inner_ftol", 1e-4))),
        sun_horizon_steps=int(config.get("sun_horizon_steps", 2)),
        sun_outer_rounds=int(config.get("sun_outer_rounds", 1)),
        sun_num_candidates=int(config.get("sun_num_candidates", 5)),
    )


def write_latex_table(summary_df: pd.DataFrame, outfile: Path) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    if summary_df.empty:
        outfile.write_text("% No benchmark rows were generated.\n", encoding="utf-8")
        return
    cols = [
        "method",
        "mean_finite_gap",
        "p95_finite_gap",
        "mean_signed_residual",
        "max_positive_signed_violation",
        "solver_success_rate",
        "mean_nit",
        "mean_union_candidate_count",
        "mean_eval_runtime_ms",
    ]
    table_df = summary_df[[col for col in cols if col in summary_df.columns]].copy()
    table_df = table_df.rename(columns={
        "method": "Method",
        "mean_finite_gap": "Mean finite gap",
        "p95_finite_gap": "P95 finite gap",
        "mean_signed_residual": "Mean signed residual",
        "max_positive_signed_violation": "Max positive signed residual",
        "solver_success_rate": "Solver success",
        "mean_nit": "Mean nit",
        "mean_union_candidate_count": "Union size",
        "mean_eval_runtime_ms": "Eval ms",
    })
    latex = table_df.to_latex(index=False, float_format=lambda value: f"{value:.4g}")
    prefix = (
        "% Draft smoke table for Finite-Candidate Reference (FC-Union).\n"
        "% The reference is finite-family specific and does not imply continuous global optimality.\n"
    )
    outfile.write_text(prefix + latex, encoding="utf-8")


def build_solver_diagnostics_summary(candidate_df: pd.DataFrame, slot_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame()
    rows = []
    for (profile, method), group in candidate_df.groupby(["profile", "method"], sort=False):
        nit = group["nit"].astype(float)
        runtime_ms = group["runtime_ms"].astype(float)
        residual = group["residual"].astype(float)
        success = group["success"].astype(bool)
        primary_success = group["primary_success"].astype(bool) if "primary_success" in group.columns else success
        rescue_attempts = group["rescue_attempts"].astype(float) if "rescue_attempts" in group.columns else pd.Series(np.zeros(len(group)), index=group.index)
        slot_group = slot_df[(slot_df["profile"] == profile) & (slot_df["method"] == method)]
        rows.append({
            "profile": profile,
            "method": method,
            "candidate_evaluations": int(len(group)),
            "solver_success_rate": float(success.mean()),
            "solver_fail_count": int((~success).sum()),
            "primary_solver_success_rate": float(primary_success.mean()),
            "primary_solver_fail_count": int((~primary_success).sum()),
            "rescue_rate": float((rescue_attempts > 0).mean()),
            "rescue_success_count": int(group["rescue_success"].astype(bool).sum()) if "rescue_success" in group.columns else 0,
            "mean_nit": float(nit.mean()),
            "p95_nit": float(nit.quantile(0.95)),
            "mean_candidate_runtime_ms": float(runtime_ms.mean()),
            "p95_candidate_runtime_ms": float(runtime_ms.quantile(0.95)),
            "max_solver_residual": float(residual.max()),
            "mean_union_candidate_count": float(slot_group["union_candidate_count"].astype(float).mean()) if not slot_group.empty else np.nan,
            "mean_eval_runtime_ms": float(slot_group["eval_runtime_ms"].astype(float).mean()) if not slot_group.empty else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="R1 FC-Union benchmark repair smoke runner.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs_r1" / "benchmark_repair_smoke.json"))
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--seed-offset", type=int, default=1000)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--tables-dir", type=str, default=None)
    parser.add_argument("--include-dense", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    slots = int(args.slots if args.slots is not None else config.get("slots", 2))
    seeds = int(args.seeds if args.seeds is not None else config.get("seeds", 1))
    methods = methods_from_config(config, args.methods)
    profile = str(args.profile or config.get("profile", "balanced_smoke"))
    outdir = Path(args.outdir or config.get("outdir", ROOT / "results_r1" / "benchmark_repair"))
    tables_dir = Path(args.tables_dir or config.get("tables_dir", ROOT / "tables_r1"))
    include_dense = bool(args.include_dense or config.get("include_dense", False))
    dense_n_radial = int(config.get("dense_n_radial", 1))
    dense_n_angle = int(config.get("dense_n_angle", 8))
    warm_start_policy = str(config.get("warm_start_policy", "common"))

    outdir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_config(config)

    slot_records: List[Dict[str, object]] = []
    candidate_records: List[Dict[str, object]] = []
    coverage_records: List[Dict[str, object]] = []
    warning_records: List[Dict[str, object]] = []
    controller_cache_warning_records: List[Dict[str, object]] = []

    started = time.perf_counter()
    print(f"FC-Union smoke start: profile={profile}, methods={methods}, slots={slots}, seeds={seeds}")

    for seed_idx in range(seeds):
        seed_value = int(args.seed_offset) + seed_idx
        sims = {method: UavIsacSimulator(cfg) for method in methods}
        states = {method: sims[method].reset(seed_value) for method in methods}

        for slot in range(slots):
            for target_method in methods:
                sim = sims[target_method]
                state = states[target_method]
                exo = sim.build_exogenous(state)

                action_results = {}
                method_families = {}
                for source_method in methods:
                    action_result, family = collect_method_candidate_family(sim, state, source_method, exo=exo)
                    action_results[source_method] = action_result
                    method_families[source_method] = family

                common_family = build_common_candidate_family(sim, state)
                dense_family = build_dense_candidate_family(sim, state, dense_n_radial, dense_n_angle) if include_dense else None
                union_family = build_union_candidate_family(common_family, method_families, dense_family=dense_family)
                family_eval = evaluate_finite_family_with_exo(
                    sim,
                    state,
                    exo,
                    union_family,
                    warm_start_policy=warm_start_policy,
                )
                gap = compute_union_gap(action_results[target_method], family_eval)
                repeated = repeated_exact_solve_at_executed_point(
                    sim,
                    state,
                    exo,
                    action_results[target_method],
                    executed_value_cache=float(gap["executed_value_cache"]),
                    warm_start_policy=warm_start_policy,
                )
                solver_summary = solver_summary_from_eval(family_eval)

                slot_row = {
                    "slot": int(slot),
                    "seed": int(seed_value),
                    "profile": profile,
                    "method": target_method,
                    "executed_x": float(action_results[target_method].post_motion_point[0]),
                    "executed_y": float(action_results[target_method].post_motion_point[1]),
                    "union_best_x": float(family_eval.best_point[0]),
                    "union_best_y": float(family_eval.best_point[1]),
                    **gap,
                    **repeated,
                    "union_candidate_count": int(len(union_family.points)),
                    **solver_summary,
                    "eval_runtime_ms": float(family_eval.eval_runtime_ms),
                }
                slot_records.append(slot_row)
                candidate_records.extend(candidate_rows_for_csv(family_eval, slot=slot, seed=seed_value, profile=profile, method=target_method))
                coverage_records.extend(
                    coverage_rows_for_families(
                        method_families,
                        action_results,
                        slot=slot,
                        seed=seed_value,
                        profile=profile,
                        target_method=target_method,
                    )
                )
                warn_positive_signed = bool(slot_row["positive_signed_violation"] > RESIDUAL_WARN_TOL)
                fail_positive_signed = bool(slot_row["positive_signed_violation"] > RESIDUAL_FAIL_TOL)
                warn_controller_cache = bool(slot_row["controller_cache_residual_status"] in {"warn", "fail"})
                fail_controller_cache = bool(slot_row["controller_cache_residual_status"] == "fail")
                warn_repeated = bool(abs(float(slot_row["executed_repeated_residual"])) > RESIDUAL_WARN_TOL)
                fail_repeated = bool(abs(float(slot_row["executed_repeated_residual"])) > RESIDUAL_FAIL_TOL)
                warn_executed_missing = not bool(slot_row["executed_point_found_in_union"])
                warn_union_best_ineligible = not bool(slot_row.get("union_best_eligible_for_reference", True))
                if warn_controller_cache:
                    controller_cache_warning_records.append({
                        **slot_row,
                        "warn_controller_cache": warn_controller_cache,
                        "fail_controller_cache": fail_controller_cache,
                    })
                if warn_positive_signed or warn_repeated or warn_executed_missing or warn_union_best_ineligible:
                    warning_records.append({
                        **slot_row,
                        "warn_positive_signed": warn_positive_signed,
                        "fail_positive_signed": fail_positive_signed,
                        "warn_repeated_exact": warn_repeated,
                        "fail_repeated_exact": fail_repeated,
                        "warn_executed_missing": warn_executed_missing,
                        "warn_union_best_ineligible": warn_union_best_ineligible,
                    })

                metrics = sim.apply_action_r1(state, exo, action_results[target_method])
                del metrics

            print(f"seed={seed_value} slot={slot} elapsed={time.perf_counter() - started:.3f}s")

    candidate_df = pd.DataFrame(candidate_records)
    slot_df = pd.DataFrame(slot_records)
    summary_df = summarize_benchmark_slots(slot_records)
    consistency_df = summarize_consistency_slots(slot_records)
    coverage_df = pd.DataFrame(coverage_records)
    warnings_df = pd.DataFrame(warning_records)
    controller_cache_warnings_df = pd.DataFrame(controller_cache_warning_records)
    if warnings_df.empty:
        warnings_df = pd.DataFrame(columns=list(slot_df.columns) + [
            "warn_positive_signed",
            "fail_positive_signed",
            "warn_repeated_exact",
            "fail_repeated_exact",
            "warn_executed_missing",
            "warn_union_best_ineligible",
        ])
    if controller_cache_warnings_df.empty:
        controller_cache_warnings_df = pd.DataFrame(columns=list(slot_df.columns) + [
            "warn_controller_cache",
            "fail_controller_cache",
        ])

    candidate_df.to_csv(outdir / "candidate_values.csv", index=False)
    slot_df.to_csv(outdir / "slot_benchmark.csv", index=False)
    summary_df.to_csv(outdir / "benchmark_summary.csv", index=False)
    consistency_df.to_csv(outdir / "consistency_summary.csv", index=False)
    build_solver_diagnostics_summary(candidate_df, slot_df).to_csv(outdir / "solver_diagnostics_summary.csv", index=False)
    coverage_df.to_csv(outdir / "candidate_family_coverage.csv", index=False)
    warnings_df.to_csv(outdir / "benchmark_warnings.csv", index=False)
    controller_cache_warnings_df.to_csv(outdir / "controller_cache_warnings.csv", index=False)
    write_latex_table(summary_df, tables_dir / "table5_fc_union_benchmark_smoke.tex")

    max_positive = float(slot_df["positive_signed_violation"].max()) if not slot_df.empty else float("nan")
    success_rate = float(slot_df["solver_success_rate"].mean()) if not slot_df.empty else float("nan")
    max_repeated = float(slot_df["executed_repeated_residual"].abs().max()) if not slot_df.empty else float("nan")
    print(f"FC-Union smoke finished: outdir={outdir}")
    print(f"max_positive_signed_violation={max_positive:.6g}")
    print(f"max_abs_executed_repeated_residual={max_repeated:.6g}")
    print(f"warning_rows={len(warnings_df)}")
    print(f"mean_solver_success_rate={success_rate:.6g}")
    if max_positive > RESIDUAL_FAIL_TOL:
        print(f"WARNING: positive signed residual exceeds fail tolerance {RESIDUAL_FAIL_TOL}")


if __name__ == "__main__":
    main()
