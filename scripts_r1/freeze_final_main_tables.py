from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


LABELS: Dict[str, str] = {
    "proposed_full": "DCEE",
    "full_candidate_exact": "FC-Common",
    "anchored_mobility": "Anchored-Mobility",
    "regularized_execution": "Regularized-Execution",
    "static_uav": "Static-Hovering",
    "freshness_priority": "Freshness-Priority",
    "yang_go_lyapunov": "GC-Lyapunov",
    "sun_receding_ao": "RHC-Alt.",
    "myopic_reoptimization": "Myopic Re-optimization",
}

ORDER: List[str] = [
    "proposed_full",
    "full_candidate_exact",
    "anchored_mobility",
    "regularized_execution",
    "static_uav",
    "freshness_priority",
    "yang_go_lyapunov",
    "sun_receding_ao",
    "myopic_reoptimization",
]

MECHANISM_METHODS = [
    "proposed_full",
    "full_candidate_exact",
    "anchored_mobility",
    "regularized_execution",
]

MECHANISM_NOTES = {
    "proposed_full": "recourse scoring + exact execution",
    "full_candidate_exact": "common finite family",
    "anchored_mobility": "anchored mobility score",
    "regularized_execution": "regularized direct execution",
}


def summarize_seed_level(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, rec in raw.iterrows():
        method = str(rec["controller"])
        rows.append({
            "method": method,
            "method_label": LABELS.get(method, method),
            "seed": int(rec["seed"]),
            "horizon": int(rec["horizon"]),
            "total_penalty": float(rec.get("total_penalty", np.nan)),
            "queue_backlog": float(rec.get("queue_backlog", np.nan)),
            "aoi_violation": float(rec.get("aoi_violation", np.nan)),
            "uncertainty_violation": float(rec.get("uncertainty_violation", np.nan)),
            "mean_virtual_aoi_queue": float(rec.get("mean_virtual_aoi_queue", np.nan)),
            "mean_virtual_unc_queue": float(rec.get("mean_virtual_unc_queue", np.nan)),
            "travel_distance": float(rec.get("travel_distance", np.nan)),
            "solver_success": float(rec.get("solver_success", np.nan)),
            "solver_nit": float(rec.get("solver_nit", np.nan)),
            "runtime_ms": float(rec.get("runtime_ms", np.nan)),
            "candidate_family_size": float(rec.get("candidate_family_size", np.nan)),
        })
    return pd.DataFrame(rows)


def aggregate(seed_df: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    metrics = [
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
    rows = []
    for method in methods:
        group = seed_df[seed_df["method"] == method]
        if group.empty:
            continue
        row = {
            "method": method,
            "method_label": LABELS.get(method, method),
            "num_seeds": int(group["seed"].nunique()),
            "slots_per_seed": int(group["horizon"].iloc[0]),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(row: pd.Series, metric: str, digits: int = 3) -> str:
    mean = float(row.get(f"{metric}_mean", np.nan))
    std = float(row.get(f"{metric}_std", 0.0))
    if not np.isfinite(mean):
        return "--"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_main_table(summary: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Penalty & Backlog & AoI exc. & Unc. exc. & Travel & Solver \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        solver = 100.0 * float(row.get("solver_success_mean", np.nan))
        lines.append(
            f"{row['method_label']} & {fmt(row, 'total_penalty')} & {fmt(row, 'queue_backlog')} & "
            f"{fmt(row, 'aoi_violation')} & {fmt(row, 'uncertainty_violation')} & "
            f"{fmt(row, 'travel_distance')} & {solver:.1f}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mechanism_table(summary: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Method & Mechanism & Penalty & Backlog & AoI exc. & Unc. exc. \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        note = MECHANISM_NOTES.get(str(row["method"]), "")
        lines.append(
            f"{row['method_label']} & {note} & {fmt(row, 'total_penalty')} & "
            f"{fmt(row, 'queue_backlog')} & {fmt(row, 'aoi_violation')} & "
            f"{fmt(row, 'uncertainty_violation')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze final R1 main-performance and ablation tables.")
    parser.add_argument("--input", type=Path, default=Path("results_r1/final_main/r1_smoke_summary_raw.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("results_r1/final_main"))
    parser.add_argument("--tables-dir", type=Path, default=Path("tables_r1"))
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    seed_df = summarize_seed_level(raw)
    main_summary = aggregate(seed_df, ORDER)
    mechanism_summary = aggregate(seed_df, MECHANISM_METHODS)
    mechanism_summary["mechanism_note"] = mechanism_summary["method"].map(MECHANISM_NOTES)

    args.outdir.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(args.outdir / "main_performance_seed_summary.csv", index=False)
    main_summary.to_csv(args.outdir / "main_performance_summary.csv", index=False)
    mechanism_summary.to_csv(args.outdir / "mechanism_ablation_summary.csv", index=False)
    write_main_table(main_summary, args.tables_dir / "table_main_performance.tex")
    write_mechanism_table(mechanism_summary, args.tables_dir / "table_mechanism_ablation.tex")
    print(f"Wrote {args.outdir / 'main_performance_summary.csv'}")
    print(f"Wrote {args.outdir / 'mechanism_ablation_summary.csv'}")
    print(f"Wrote {args.tables_dir / 'table_main_performance.tex'}")
    print(f"Wrote {args.tables_dir / 'table_mechanism_ablation.tex'}")


if __name__ == "__main__":
    main()
