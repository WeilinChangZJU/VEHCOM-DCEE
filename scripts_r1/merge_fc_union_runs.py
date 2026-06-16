from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts_r1.make_table5_fc_union import build_seed_level_table, write_latex


def seed_dirs(root: Path) -> List[Path]:
    return sorted(path for path in root.glob("seed_*") if path.is_dir())


def read_completed_seed_dirs(root: Path) -> List[Path]:
    out: List[Path] = []
    for path in seed_dirs(root):
        manifest = path / "run_manifest.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("status") == "completed":
            out.append(path)
    return out


def read_csvs(dirs: List[Path], filename: str, *, required: bool = True) -> pd.DataFrame:
    frames = []
    for path in dirs:
        csv_path = path / filename
        if csv_path.exists():
            try:
                frame = pd.read_csv(csv_path)
            except Exception as exc:
                if required:
                    raise RuntimeError(f"Failed to read required CSV {csv_path}: {exc}") from exc
                print(f"Warning: skipping optional CSV {csv_path}: {exc}")
                continue
            if frame.empty and len(frame.columns) <= 1 and not required:
                print(f"Warning: skipping empty optional CSV {csv_path}")
                continue
            frame["seed_dir"] = path.name
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def critical_warnings_from_slot_df(slot_df: pd.DataFrame, warn_tol: float = 1e-4, fail_tol: float = 1e-3) -> pd.DataFrame:
    if slot_df.empty:
        return pd.DataFrame()
    union_ineligible = (
        ~slot_df["union_best_eligible_for_reference"].astype(bool)
        if "union_best_eligible_for_reference" in slot_df.columns
        else pd.Series(False, index=slot_df.index)
    )
    mask = (
        (slot_df["positive_signed_violation"].astype(float) > warn_tol)
        | (slot_df["executed_repeated_residual"].astype(float).abs() > warn_tol)
        | (~slot_df["executed_point_found_in_union"].astype(bool))
        | union_ineligible
    )
    out = slot_df[mask].copy()
    if out.empty:
        return pd.DataFrame(columns=list(slot_df.columns) + [
            "warn_positive_signed",
            "fail_positive_signed",
            "warn_repeated_exact",
            "fail_repeated_exact",
            "warn_executed_missing",
        ])
    out["warn_positive_signed"] = out["positive_signed_violation"].astype(float) > warn_tol
    out["fail_positive_signed"] = out["positive_signed_violation"].astype(float) > fail_tol
    out["warn_repeated_exact"] = out["executed_repeated_residual"].astype(float).abs() > warn_tol
    out["fail_repeated_exact"] = out["executed_repeated_residual"].astype(float).abs() > fail_tol
    out["warn_executed_missing"] = ~out["executed_point_found_in_union"].astype(bool)
    out["warn_union_best_ineligible"] = (
        ~out["union_best_eligible_for_reference"].astype(bool)
        if "union_best_eligible_for_reference" in out.columns else False
    )
    return out


def controller_cache_warnings_from_slot_df(slot_df: pd.DataFrame) -> pd.DataFrame:
    if slot_df.empty or "controller_cache_residual_status" not in slot_df.columns:
        return pd.DataFrame()
    out = slot_df[slot_df["controller_cache_residual_status"].isin(["warn", "fail"])].copy()
    if out.empty:
        return pd.DataFrame(columns=list(slot_df.columns) + [
            "warn_controller_cache",
            "fail_controller_cache",
        ])
    out["warn_controller_cache"] = out["controller_cache_residual_status"].isin(["warn", "fail"])
    out["fail_controller_cache"] = out["controller_cache_residual_status"] == "fail"
    return out


def seed_level_summary(slot_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (profile, seed, method), group in slot_df.groupby(["profile", "seed", "method"], sort=False):
        rows.append({
            "profile": profile,
            "seed": int(seed),
            "method": method,
            "mean_finite_gap": float(group["finite_gap"].astype(float).mean()),
            "p95_finite_gap": float(group["finite_gap"].astype(float).quantile(0.95)),
            "mean_signed_residual": float(group["signed_residual"].astype(float).mean()),
            "solver_success_rate": float(group["solver_success_rate"].astype(float).mean()),
            "final_solver_success_rate": float(group["final_solver_success_rate"].astype(float).mean()) if "final_solver_success_rate" in group.columns else float(group["solver_success_rate"].astype(float).mean()),
            "primary_solver_success_rate": float(group["primary_solver_success_rate"].astype(float).mean()) if "primary_solver_success_rate" in group.columns else float(group["solver_success_rate"].astype(float).mean()),
            "rescue_rate": float(group["rescue_rate"].astype(float).mean()) if "rescue_rate" in group.columns else 0.0,
            "mean_nit": float(group["mean_nit"].astype(float).mean()),
            "mean_union_candidate_count": float(group["union_candidate_count"].astype(float).mean()),
            "max_positive_signed_violation": float(group["positive_signed_violation"].astype(float).max()),
            "max_abs_repeated_residual": float(group["executed_repeated_residual"].astype(float).abs().max()),
            "executed_point_found_rate": float(group["executed_point_found_in_union"].astype(bool).mean()),
        })
    return pd.DataFrame(rows)


def table5_seed_level_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    methods = list(seed_df["method"].drop_duplicates())
    reference = seed_df.copy()
    rows.append({
        "profile": str(reference["profile"].iloc[0]) if not reference.empty else "balanced",
        "method": "FC-Union",
        "finite_gap_mean": 0.0,
        "finite_gap_std": 0.0,
        "p95_gap_mean": 0.0,
        "signed_residual_mean": 0.0,
        "signed_residual_std": 0.0,
        "solver_success_rate_mean": float(reference["solver_success_rate"].astype(float).mean()) if not reference.empty else np.nan,
        "rescue_rate_mean": float(reference["rescue_rate"].astype(float).mean()) if (not reference.empty and "rescue_rate" in reference.columns) else 0.0,
        "mean_nit": float(reference["mean_nit"].astype(float).mean()) if not reference.empty else np.nan,
        "union_size_mean": float(reference["mean_union_candidate_count"].astype(float).mean()) if not reference.empty else np.nan,
        "num_seeds": int(reference["seed"].nunique()) if not reference.empty else 0,
    })
    for method in methods:
        group = seed_df[seed_df["method"] == method].copy()
        rows.append({
            "profile": str(group["profile"].iloc[0]),
            "method": method,
            "finite_gap_mean": float(group["mean_finite_gap"].mean()),
            "finite_gap_std": float(group["mean_finite_gap"].std(ddof=0)),
            "p95_gap_mean": float(group["p95_finite_gap"].mean()),
            "signed_residual_mean": float(group["mean_signed_residual"].mean()),
            "signed_residual_std": float(group["mean_signed_residual"].std(ddof=0)),
            "solver_success_rate_mean": float(group["solver_success_rate"].mean()),
            "rescue_rate_mean": float(group["rescue_rate"].mean()) if "rescue_rate" in group.columns else 0.0,
            "mean_nit": float(group["mean_nit"].mean()),
            "union_size_mean": float(group["mean_union_candidate_count"].mean()),
            "num_seeds": int(group["seed"].nunique()),
        })
    return pd.DataFrame(rows)


def solver_diagnostics_summary(candidate_df: pd.DataFrame, slot_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (profile, method), group in candidate_df.groupby(["profile", "method"], sort=False):
        slot_group = slot_df[(slot_df["profile"] == profile) & (slot_df["method"] == method)]
        success = group["success"].astype(bool)
        primary_success = group["primary_success"].astype(bool) if "primary_success" in group.columns else success
        rescue_attempts = group["rescue_attempts"].astype(float) if "rescue_attempts" in group.columns else pd.Series(np.zeros(len(group)), index=group.index)
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
            "mean_nit": float(group["nit"].astype(float).mean()),
            "p95_nit": float(group["nit"].astype(float).quantile(0.95)),
            "max_solver_residual": float(group["residual"].astype(float).max()),
            "mean_eval_runtime_ms": float(slot_group["eval_runtime_ms"].astype(float).mean()) if not slot_group.empty else np.nan,
        })
    return pd.DataFrame(rows)


def rescue_diagnostics_summary(candidate_df: pd.DataFrame, slot_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if candidate_df.empty:
        return pd.DataFrame()
    for (profile, method), group in candidate_df.groupby(["profile", "method"], sort=False):
        primary_success = group["primary_success"].astype(bool) if "primary_success" in group.columns else group["success"].astype(bool)
        final_success = group["final_success"].astype(bool) if "final_success" in group.columns else group["success"].astype(bool)
        eligible = group["eligible_for_reference"].astype(bool) if "eligible_for_reference" in group.columns else final_success
        rescue_attempts = group["rescue_attempts"].astype(float) if "rescue_attempts" in group.columns else pd.Series(np.zeros(len(group)), index=group.index)
        changed = group["value_changed_by_rescue"].astype(bool) if "value_changed_by_rescue" in group.columns else pd.Series(np.zeros(len(group), dtype=bool), index=group.index)
        slot_group = slot_df[(slot_df["profile"] == profile) & (slot_df["method"] == method)]
        best_final_success = (
            slot_group["union_best_final_success"].astype(bool).mean()
            if "union_best_final_success" in slot_group.columns and not slot_group.empty else np.nan
        )
        best_eligible = (
            slot_group["union_best_eligible_for_reference"].astype(bool).mean()
            if "union_best_eligible_for_reference" in slot_group.columns and not slot_group.empty else np.nan
        )
        rows.append({
            "profile": profile,
            "method": method,
            "candidate_evaluations": int(len(group)),
            "primary_solver_success_rate": float(primary_success.mean()),
            "final_solver_success_rate": float(final_success.mean()),
            "primary_solver_fail_count": int((~primary_success).sum()),
            "final_solver_fail_count": int((~final_success).sum()),
            "rescue_rate": float((rescue_attempts > 0).mean()),
            "mean_rescue_attempts": float(rescue_attempts.mean()),
            "rescue_success_count": int(group["rescue_success"].astype(bool).sum()) if "rescue_success" in group.columns else 0,
            "value_changed_by_rescue_rate": float(changed.mean()),
            "eligible_for_reference_rate": float(eligible.mean()),
            "union_best_final_success_rate": float(best_final_success),
            "union_best_eligible_for_reference_rate": float(best_eligible),
        })
    return pd.DataFrame(rows)


def coverage_summary(coverage_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (profile, method), group in coverage_df.groupby(["profile", "method"], sort=False):
        rows.append({
            "profile": profile,
            "method": method,
            "rows": int(len(group)),
            "min_candidate_count": int(group["candidate_count"].astype(int).min()),
            "max_candidate_count": int(group["candidate_count"].astype(int).max()),
            "executed_point_found_rate": float(group["has_executed_point"].astype(bool).mean()),
            "has_full_family_rate": float(group["has_full_family"].astype(bool).mean()),
            "has_shortlist_rate": float(group["has_shortlist"].astype(bool).mean()),
            "notes": "; ".join(sorted(set(str(x) for x in group["notes"].dropna()))),
        })
    return pd.DataFrame(rows)


def union_best_source_counts(slot_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["profile", "method", "union_best_source_method", "union_best_source_label", "union_best_is_executed_point"]
    if not all(col in slot_df.columns for col in cols):
        return pd.DataFrame()
    return (
        slot_df.groupby(cols, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["profile", "method", "count"], ascending=[True, True, False])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge seed-level FC-Union benchmark outputs.")
    parser.add_argument("--root", type=str, default=str(ROOT / "results_r1" / "benchmark_full_balanced"))
    parser.add_argument("--merged-dir", type=str, default=None)
    parser.add_argument("--table-output", type=str, default=str(ROOT / "tables_r1" / "table5_fc_union_balanced.tex"))
    parser.add_argument("--table-preview", type=str, default=str(ROOT / "tables_r1" / "table5_fc_union_balanced_preview.csv"))
    args = parser.parse_args()

    root = Path(args.root)
    merged_dir = Path(args.merged_dir) if args.merged_dir else root / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    dirs = read_completed_seed_dirs(root)
    if not dirs:
        raise RuntimeError(f"No completed seed directories found under {root}")

    slot_df = read_csvs(dirs, "slot_benchmark.csv")
    candidate_df = read_csvs(dirs, "candidate_values.csv")
    coverage_df = read_csvs(dirs, "candidate_family_coverage.csv")
    consistency_df = read_csvs(dirs, "consistency_summary.csv", required=False)
    warnings_df = critical_warnings_from_slot_df(slot_df)
    controller_cache_warnings_df = controller_cache_warnings_from_slot_df(slot_df)

    slot_df.to_csv(merged_dir / "slot_benchmark.csv", index=False)
    candidate_df.to_csv(merged_dir / "candidate_values.csv", index=False)
    coverage_df.to_csv(merged_dir / "candidate_family_coverage.csv", index=False)
    warnings_df.to_csv(merged_dir / "benchmark_warnings_all.csv", index=False)
    controller_cache_warnings_df.to_csv(merged_dir / "controller_cache_warnings_all.csv", index=False)
    consistency_df.to_csv(merged_dir / "consistency_summary_all.csv", index=False)

    seed_df = seed_level_summary(slot_df)
    table5_df = table5_seed_level_summary(seed_df)
    solver_df = solver_diagnostics_summary(candidate_df, slot_df)
    rescue_df = rescue_diagnostics_summary(candidate_df, slot_df)
    coverage_sum = coverage_summary(coverage_df)
    source_counts = union_best_source_counts(slot_df)

    seed_df.to_csv(merged_dir / "seed_summary.csv", index=False)
    table5_df.to_csv(merged_dir / "table5_seed_level_summary.csv", index=False)
    solver_df.to_csv(merged_dir / "solver_diagnostics_summary.csv", index=False)
    rescue_df.to_csv(merged_dir / "rescue_diagnostics_summary.csv", index=False)
    coverage_sum.to_csv(merged_dir / "candidate_family_coverage_summary.csv", index=False)
    source_counts.to_csv(merged_dir / "union_best_source_counts.csv", index=False)

    table = build_seed_level_table(seed_df)
    write_latex(table, Path(args.table_output))
    Path(args.table_preview).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.table_preview, index=False)
    print(f"Merged {len(dirs)} seed directories into {merged_dir}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
