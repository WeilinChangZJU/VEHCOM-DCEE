from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_r1.uav_isac_core_r1 import ScenarioConfig, UavIsacSimulator


METRIC_COLS = [
    "total_penalty",
    "queue_backlog",
    "aoi_violation",
    "uncertainty_violation",
    "mean_virtual_aoi_queue",
    "mean_virtual_unc_queue",
    "travel_distance",
    "solver_success",
    "solver_nit",
    "runtime_ms",
    "candidate_family_size",
]


def read_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def emit(message: str, log_handle) -> None:
    print(message)
    log_handle.write(message + "\n")
    log_handle.flush()


def build_config(base: Dict[str, Any], overrides: Dict[str, Any] | None = None) -> ScenarioConfig:
    data = dict(base)
    if overrides:
        data.update(overrides)
    return ScenarioConfig(
        num_vehicles=int(data.get("num_vehicles", 6)),
        num_targets=int(data.get("num_targets", 3)),
        benchmark_num_directions=int(data.get("benchmark_num_directions", 16)),
        arrival_rate=float(data.get("arrival_rate", 0.8)),
        V=float(data.get("V", 12.0)),
        rho_r=float(data.get("rho_r", 0.01)),
        eta_t=float(data.get("eta_t", 2.0)),
        g0_sens=float(data.get("g0_sens", 96.0)),
        mu_reduction_scale=float(data.get("mu_reduction_scale", 12.0)),
        c_Y=float(data.get("c_Y", 8.0)),
        c_Z=float(data.get("c_Z", 8.0)),
        aoi_threshold=float(data.get("aoi_threshold", 5.0)),
        uncertainty_threshold=float(data.get("uncertainty_threshold", 3.5)),
        init_x=float(data.get("init_x", 200.0)),
        init_y=float(data.get("init_y", 0.0)),
        shortlist_num_local_dirs=int(data.get("shortlist_num_local_dirs", 8)),
        shortlist_radius_scale=float(data.get("shortlist_radius_scale", 0.45)),
        shortlist_size=int(data.get("shortlist_size", 11)),
        sensing_success_gain=float(data.get("sensing_success_gain", 1.2)),
        refresh_uncertainty_gain=float(data.get("refresh_uncertainty_gain", 1.15)),
        nonrefresh_mu_factor=float(data.get("nonrefresh_mu_factor", 0.15)),
        inner_solver_maxiter=int(data.get("inner_solver_maxiter", 80)),
        inner_solver_ftol=float(data.get("inner_solver_ftol", 1e-4)),
        sun_horizon_steps=int(data.get("sun_horizon_steps", 2)),
        sun_outer_rounds=int(data.get("sun_outer_rounds", 1)),
        sun_num_candidates=int(data.get("sun_num_candidates", 8)),
    )


def scenario_config_to_dict(cfg: ScenarioConfig) -> Dict[str, Any]:
    keys = [
        "num_vehicles",
        "num_targets",
        "benchmark_num_directions",
        "arrival_rate",
        "V",
        "rho_r",
        "eta_t",
        "g0_sens",
        "mu_reduction_scale",
        "c_Y",
        "c_Z",
        "aoi_threshold",
        "uncertainty_threshold",
        "init_x",
        "init_y",
        "shortlist_num_local_dirs",
        "shortlist_radius_scale",
        "shortlist_size",
        "sensing_success_gain",
        "refresh_uncertainty_gain",
        "nonrefresh_mu_factor",
        "inner_solver_maxiter",
        "inner_solver_ftol",
        "sun_horizon_steps",
        "sun_outer_rounds",
        "sun_num_candidates",
    ]
    return {key: getattr(cfg, key) for key in keys}


def evaluate_method(
    cfg: ScenarioConfig,
    method: str,
    label: str,
    seeds: Iterable[int],
    slots: int,
    variant: str,
    log_handle,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    summaries: List[pd.DataFrame] = []
    slots_rows: List[pd.DataFrame] = []
    for seed in seeds:
        sim = UavIsacSimulator(cfg)
        t0 = time.perf_counter()
        summary_df, slot_df = sim.simulate_r1(method, horizon=int(slots), seed=int(seed))
        elapsed = time.perf_counter() - t0
        for df in (summary_df, slot_df):
            df["seed"] = int(seed)
            df["method"] = method
            df["method_label"] = label
            df["variant"] = variant
        summaries.append(summary_df)
        slots_rows.append(slot_df)
        emit(f"variant={variant} method={method} seed={seed} slots={slots} time={elapsed:.2f}s", log_handle)
    return summaries, slots_rows


def summarize_seed_rows(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, Any] = {col: value for col, value in zip(group_cols, keys)}
        row["num_seeds"] = int(group["seed"].nunique()) if "seed" in group.columns else int(len(group))
        for col in METRIC_COLS:
            if col in group.columns:
                values = pd.to_numeric(group[col], errors="coerce").dropna()
                if len(values):
                    row[f"{col}_mean"] = float(values.mean())
                    row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_mean_std(row: pd.Series, col: str, digits: int = 3) -> str:
    mean = row.get(f"{col}_mean")
    std = row.get(f"{col}_std")
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        std = 0.0
    return f"{float(mean):.{digits}f} $\\pm$ {float(std):.{digits}f}"


def write_latex_table(summary: pd.DataFrame, path: Path, caption_note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Penalty & Backlog & AoI exc. & Unc. exc. & Travel & Solver \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        solver = row.get("solver_success_mean", float("nan"))
        solver_txt = f"{100.0 * float(solver):.1f}\\%" if pd.notna(solver) else "--"
        lines.append(
            f"{row['method_label']} & {fmt_mean_std(row, 'total_penalty')} & "
            f"{fmt_mean_std(row, 'queue_backlog')} & {fmt_mean_std(row, 'aoi_violation')} & "
            f"{fmt_mean_std(row, 'uncertainty_violation')} & {fmt_mean_std(row, 'travel_distance')} & "
            f"{solver_txt} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"% {caption_note}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Limited Patch 9B DCEE tuning and held-out evaluation.")
    parser.add_argument("--config", type=Path, default=Path("configs_r1/patch9b_performance_strengthening_light.json"))
    parser.add_argument("--outdir", type=Path, default=Path("results_r1/patch9b_performance_strengthening"))
    parser.add_argument("--tables-dir", type=Path, default=Path("tables_r1"))
    parser.add_argument("--validation-slots", type=int, default=None)
    parser.add_argument("--test-slots", type=int, default=None)
    args = parser.parse_args()

    config = read_config(args.config)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = ROOT / "logs_r1"
    logs_dir.mkdir(parents=True, exist_ok=True)

    validation_slots = int(args.validation_slots or config.get("validation_slots", 100))
    test_slots = int(args.test_slots or config.get("test_slots", 100))
    validation_seeds = [int(x) for x in config.get("validation_seeds", [200, 201, 202])]
    test_seeds = [int(x) for x in config.get("test_seeds", [100, 101, 102, 103, 104])]
    base = config.get("base", {})
    variants = config.get("dcee_variants", [{"variant": "default", "overrides": {}}])

    with (logs_dir / "patch9b_performance_strengthening.log").open("w", encoding="utf-8") as log:
        emit("Patch 9B limited DCEE validation tuning", log)
        emit(f"Config: {args.config}", log)
        emit(f"Validation seeds={validation_seeds} slots={validation_slots}", log)
        emit(f"Test seeds={test_seeds} slots={test_slots}", log)

        validation_summaries: List[pd.DataFrame] = []
        validation_slots_rows: List[pd.DataFrame] = []
        for variant in variants:
            variant_name = str(variant.get("variant", "unnamed"))
            overrides = dict(variant.get("overrides", {}))
            cfg = build_config(base, overrides)
            summaries, slots_rows = evaluate_method(
                cfg,
                method="proposed_full",
                label=f"DCEE-{variant_name}",
                seeds=validation_seeds,
                slots=validation_slots,
                variant=variant_name,
                log_handle=log,
            )
            validation_summaries.extend(summaries)
            validation_slots_rows.extend(slots_rows)

        validation_raw = pd.concat(validation_summaries, ignore_index=True)
        validation_raw.to_csv(outdir / "dcee_tuning_validation_raw.csv", index=False)
        pd.concat(validation_slots_rows, ignore_index=True).to_csv(outdir / "dcee_tuning_validation_slot_metrics.csv", index=False)
        validation_summary = summarize_seed_rows(validation_raw, ["variant", "method", "method_label"])
        validation_summary = validation_summary.sort_values(["total_penalty_mean", "queue_backlog_mean"], ascending=[True, True])
        validation_summary.to_csv(outdir / "dcee_tuning_validation_summary.csv", index=False)

        best_variant_name = str(validation_summary.iloc[0]["variant"])
        best_variant = next(item for item in variants if str(item.get("variant")) == best_variant_name)
        best_cfg = build_config(base, dict(best_variant.get("overrides", {})))
        best_payload = {
            "selection_rule": "min validation total_penalty_mean; tie-break lower queue_backlog_mean",
            "validation_seeds": validation_seeds,
            "validation_slots": validation_slots,
            "test_seeds": test_seeds,
            "test_slots": test_slots,
            "best_variant": best_variant_name,
            "best_overrides": dict(best_variant.get("overrides", {})),
            "scenario_config": scenario_config_to_dict(best_cfg),
        }
        (outdir / "best_dcee_config.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
        emit(f"Selected DCEE variant: {best_variant_name}", log)

        test_summaries: List[pd.DataFrame] = []
        test_slots_rows: List[pd.DataFrame] = []
        default_cfg = build_config(base, {})
        for label, cfg, variant_name in [
            ("DCEE-default", default_cfg, "default"),
            ("DCEE-tuned", best_cfg, best_variant_name),
        ]:
            summaries, slots_rows = evaluate_method(
                cfg,
                method="proposed_full",
                label=label,
                seeds=test_seeds,
                slots=test_slots,
                variant=variant_name,
                log_handle=log,
            )
            test_summaries.extend(summaries)
            test_slots_rows.extend(slots_rows)

        for method_rec in config.get("test_methods", []):
            method = str(method_rec["method"])
            label = str(method_rec.get("label", method))
            summaries, slots_rows = evaluate_method(
                default_cfg,
                method=method,
                label=label,
                seeds=test_seeds,
                slots=test_slots,
                variant="default_backend",
                log_handle=log,
            )
            test_summaries.extend(summaries)
            test_slots_rows.extend(slots_rows)

        test_raw = pd.concat(test_summaries, ignore_index=True)
        test_raw.to_csv(outdir / "patch9b_test_raw.csv", index=False)
        pd.concat(test_slots_rows, ignore_index=True).to_csv(outdir / "patch9b_test_slot_metrics.csv", index=False)
        test_summary = summarize_seed_rows(test_raw, ["method_label", "method", "variant"])
        test_summary.to_csv(outdir / "patch9b_test_summary.csv", index=False)
        write_latex_table(
            test_summary,
            args.tables_dir / "table_patch9b_dcee_tuning.tex",
            "Limited validation tuning; held-out test seeds; not a new SOTA claim.",
        )
        manifest = {
            "config": str(args.config),
            "validation_seeds": validation_seeds,
            "test_seeds": test_seeds,
            "validation_slots": validation_slots,
            "test_slots": test_slots,
            "best_variant": best_variant_name,
            "outputs": [
                str(outdir / "dcee_tuning_validation_summary.csv"),
                str(outdir / "best_dcee_config.json"),
                str(outdir / "patch9b_test_summary.csv"),
                str(args.tables_dir / "table_patch9b_dcee_tuning.tex"),
            ],
        }
        (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        emit(test_summary[["method_label", "total_penalty_mean", "queue_backlog_mean", "solver_success_mean"]].to_string(index=False), log)
        emit(f"Wrote Patch 9B performance outputs to {outdir}", log)


if __name__ == "__main__":
    main()
