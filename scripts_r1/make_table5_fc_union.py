from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LABELS: Dict[str, str] = {
    "proposed_full": "DCEE",
    "regularized_execution": "Regularized-Execution",
    "anchored_mobility": "Anchored-Mobility",
    "static_uav": "Static-Hovering",
    "full_candidate_exact": "FC-Common",
    "yang_go_lyapunov": "GC-Lyapunov",
    "sun_receding_ao": "RHC-Alt.",
    "freshness_priority": "Freshness-Priority",
    "myopic_reoptimization": "Myopic Re-optimization",
}

ORDER: List[str] = [
    "FC-Union",
    "proposed_full",
    "full_candidate_exact",
    "regularized_execution",
    "anchored_mobility",
    "static_uav",
    "yang_go_lyapunov",
    "sun_receding_ao",
    "freshness_priority",
    "myopic_reoptimization",
]


def mean_pm_std(series: pd.Series, precision: int = 3) -> str:
    values = series.astype(float)
    return f"{values.mean():.{precision}f} $\\pm$ {values.std(ddof=0):.{precision}f}"


def pct(series: pd.Series, precision: int = 1) -> str:
    return f"{100.0 * series.astype(float).mean():.{precision}f}\\%"


def build_table(slot_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    reference_success = slot_df["solver_success_rate"].astype(float)
    reference_rescue = slot_df["rescue_rate"].astype(float) if "rescue_rate" in slot_df.columns else pd.Series([0.0] * len(slot_df))
    reference_nit = slot_df["mean_nit"].astype(float)
    reference_size = slot_df["union_candidate_count"].astype(float)
    rows.append({
        "method_code": "FC-Union",
        "Method": "FC-Union",
        "Finite gap": "0.000 $\\pm$ 0.000",
        "p95 gap": "0.000",
        "Signed residual": "0.000 $\\pm$ 0.000",
        "Success": pct(reference_success),
        "Rescue": pct(reference_rescue),
        "Iter.": f"{reference_nit.mean():.2f}",
        "Union size": f"{reference_size.mean():.1f}",
    })

    for method, group in slot_df.groupby("method", sort=False):
        rows.append({
            "method_code": method,
            "Method": LABELS.get(method, method),
            "Finite gap": mean_pm_std(group["finite_gap"]),
            "p95 gap": f"{group['finite_gap'].astype(float).quantile(0.95):.3f}",
            "Signed residual": mean_pm_std(group["signed_residual"]),
            "Success": pct(group["solver_success_rate"]),
            "Rescue": pct(group["rescue_rate"]) if "rescue_rate" in group.columns else "0.0\\%",
            "Iter.": f"{group['mean_nit'].astype(float).mean():.2f}",
            "Union size": f"{group['union_candidate_count'].astype(float).mean():.1f}",
        })
    table = pd.DataFrame(rows)
    order_map = {name: idx for idx, name in enumerate(ORDER)}
    table["_order"] = table["method_code"].map(lambda value: order_map.get(value, len(order_map)))
    return table.sort_values(["_order", "method_code"]).drop(columns=["_order", "method_code"]).reset_index(drop=True)


def build_seed_level_table(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    reference = seed_df.copy()
    reference_rescue = reference["rescue_rate"] if "rescue_rate" in reference.columns else pd.Series([0.0] * len(reference))
    rows.append({
        "method_code": "FC-Union",
        "Method": "FC-Union",
        "Finite gap": "0.000 $\\pm$ 0.000",
        "p95 gap": "0.000",
        "Signed residual": "0.000 $\\pm$ 0.000",
        "Success": pct(reference["solver_success_rate"]) if not reference.empty else "nan\\%",
        "Rescue": pct(reference_rescue) if not reference.empty else "nan\\%",
        "Iter.": f"{reference['mean_nit'].astype(float).mean():.2f}" if not reference.empty else "nan",
        "Union size": f"{reference['mean_union_candidate_count'].astype(float).mean():.1f}" if not reference.empty else "nan",
    })
    for method, group in seed_df.groupby("method", sort=False):
        rows.append({
            "method_code": method,
            "Method": LABELS.get(method, method),
            "Finite gap": mean_pm_std(group["mean_finite_gap"]),
            "p95 gap": f"{group['p95_finite_gap'].astype(float).mean():.3f}",
            "Signed residual": mean_pm_std(group["mean_signed_residual"]),
            "Success": pct(group["solver_success_rate"]),
            "Rescue": pct(group["rescue_rate"]) if "rescue_rate" in group.columns else "0.0\\%",
            "Iter.": f"{group['mean_nit'].astype(float).mean():.2f}",
            "Union size": f"{group['mean_union_candidate_count'].astype(float).mean():.1f}",
        })
    table = pd.DataFrame(rows)
    order_map = {name: idx for idx, name in enumerate(ORDER)}
    table["_order"] = table["method_code"].map(lambda value: order_map.get(value, len(order_map)))
    return table.sort_values(["_order", "method_code"]).drop(columns=["_order", "method_code"]).reset_index(drop=True)


def write_latex(table: pd.DataFrame, outfile: Path, num_seeds: int | None = None) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    latex = table.to_latex(index=False, escape=False)
    seed_note = (
        f"% Entries report mean +/- standard deviation over {num_seeds} matched random seeds; each seed is first averaged over 200 slots.\n"
        if num_seeds is not None
        else "% Entries report seed-level mean +/- standard deviation when a seed summary is provided.\n"
    )
    caption = (
        "% FC-Union finite-family benchmark and solver diagnostics in the balanced scenario.\n"
        "% The finite gap is nonnegative by construction and is computed relative to the union finite candidate family.\n"
        "% The signed residual is reported only as a numerical diagnostic.\n"
        "% Success and rescue rate are reported after applying the same deterministic rescue procedure to all candidate-level inner solves.\n"
        + seed_note
        + "% The p95 gap reports the mean of seed-level 95th percentiles when the seed-level builder is used.\n"
        + "% Candidates that remain unsuccessful after rescue are retained in diagnostics but are not eligible to define FC-Union.\n"
    )
    outfile.write_text(caption + latex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Table 5 draft from FC-Union slot benchmark CSV.")
    parser.add_argument("--input", type=str, default=str(ROOT / "results_r1" / "benchmark_full_balanced" / "slot_benchmark.csv"))
    parser.add_argument("--output", type=str, default=str(ROOT / "tables_r1" / "table5_fc_union_balanced.tex"))
    parser.add_argument("--preview-csv", type=str, default=str(ROOT / "tables_r1" / "table5_fc_union_balanced_preview.csv"))
    parser.add_argument("--seed-summary", type=str, default=None)
    parser.add_argument("--num-seeds", type=int, default=None)
    args = parser.parse_args()

    if args.seed_summary:
        table = build_seed_level_table(pd.read_csv(args.seed_summary))
    else:
        slot_df = pd.read_csv(args.input)
        table = build_table(slot_df)
    write_latex(table, Path(args.output), num_seeds=args.num_seeds)
    Path(args.preview_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.preview_csv, index=False)
    print(table.to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
