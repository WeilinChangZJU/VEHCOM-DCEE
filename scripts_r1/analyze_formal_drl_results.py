from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 12345) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def summarize(eval_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "total_penalty",
        "queue_backlog",
        "mean_virtual_aoi_queue",
        "mean_virtual_unc_queue",
        "aoi_violation",
        "uncertainty_violation",
        "travel_distance",
        "solver_success",
        "solver_nit",
        "solver_runtime_ms",
        "inference_runtime_ms",
        "reward",
    ]
    rows: List[Dict[str, Any]] = []
    for method, group in eval_df.groupby("method", sort=False):
        row: Dict[str, Any] = {"method": method, "num_test_seeds": int(group["test_seed"].nunique())}
        for metric in metrics:
            if metric in group.columns:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                if len(values):
                    row[f"{metric}_mean"] = float(values.mean())
                    row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise(eval_df: pd.DataFrame, baseline: str = "DCEE-tuned") -> pd.DataFrame:
    metrics = ["total_penalty", "queue_backlog", "aoi_violation", "uncertainty_violation", "travel_distance"]
    base = eval_df[eval_df["method"] == baseline].copy()
    rows: List[Dict[str, Any]] = []
    for method in eval_df["method"].drop_duplicates():
        if method == baseline:
            continue
        other = eval_df[eval_df["method"] == method].copy()
        merged = base.merge(other, on="test_seed", suffixes=("_dcee", "_other"))
        if merged.empty:
            continue
        for metric in metrics:
            d_col = f"{metric}_dcee"
            o_col = f"{metric}_other"
            if d_col not in merged.columns or o_col not in merged.columns:
                continue
            diff = pd.to_numeric(merged[d_col], errors="coerce") - pd.to_numeric(merged[o_col], errors="coerce")
            diff = diff.dropna().to_numpy(dtype=float)
            if diff.size == 0:
                continue
            ci_low, ci_high = bootstrap_ci(diff)
            other_mean = float(pd.to_numeric(merged[o_col], errors="coerce").mean())
            dcee_mean = float(pd.to_numeric(merged[d_col], errors="coerce").mean())
            reduction = (other_mean - dcee_mean) / other_mean if abs(other_mean) > 1e-12 else float("nan")
            rows.append({
                "baseline": baseline,
                "comparison": method,
                "metric": metric,
                "paired_diff_mean_dcee_minus_other": float(diff.mean()),
                "paired_diff_std": float(diff.std(ddof=1)) if diff.size > 1 else 0.0,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "dcee_mean": dcee_mean,
                "other_mean": other_mean,
                "relative_reduction_when_positive": float(reduction),
                "num_paired_seeds": int(diff.size),
            })
    return pd.DataFrame(rows)


def fmt(row: pd.Series, metric: str, digits: int = 3) -> str:
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        std = 0.0
    return f"{float(mean):.{digits}f} $\\pm$ {float(std):.{digits}f}"


def write_tables(summary: pd.DataFrame, training: pd.DataFrame, dcee_stage2: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Penalty & Backlog & AoI exc. & Unc. exc. & Solver & Policy ms \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        solver = row.get("solver_success_mean")
        solver_txt = f"{100.0 * float(solver):.1f}\\%" if pd.notna(solver) else "--"
        infer = row.get("inference_runtime_ms_mean")
        infer_txt = f"{float(infer):.2f}" if pd.notna(infer) else "--"
        lines.append(
            f"{row['method']} & {fmt(row, 'total_penalty')} & {fmt(row, 'queue_backlog')} & "
            f"{fmt(row, 'aoi_violation')} & {fmt(row, 'uncertainty_violation')} & {solver_txt} & {infer_txt} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / "table_formal_drl_comparison.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    proto = [
        "\\begin{tabular}{lrrrl}",
        "\\toprule",
        "Algorithm & Train seeds & Steps & Wall time (s) & Status \\\\",
        "\\midrule",
    ]
    if not training.empty:
        for alg, group in training.groupby("algorithm", sort=False):
            seeds = ",".join(str(int(x)) for x in sorted(group["train_seed"].unique()))
            steps = int(pd.to_numeric(group["completed_steps"], errors="coerce").max())
            wall = float(pd.to_numeric(group["wall_clock_sec"], errors="coerce").sum())
            status = ",".join(sorted(str(x) for x in group["status"].unique()))
            proto.append(f"{alg} & {seeds} & {steps} & {wall:.1f} & {status} \\\\")
    proto.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / "table_formal_drl_training_protocol.tex").write_text("\n".join(proto) + "\n", encoding="utf-8")

    tune = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "DCEE config & Penalty & Backlog & Solver \\\\",
        "\\midrule",
    ]
    if not dcee_stage2.empty:
        for _, row in dcee_stage2.iterrows():
            solver = 100.0 * float(row.get("solver_success_mean", 0.0))
            tune.append(
                f"{row['variant']} & {float(row['total_penalty_mean']):.3f} $\\pm$ {float(row['total_penalty_std']):.3f} & "
                f"{float(row['queue_backlog_mean']):.3f} $\\pm$ {float(row['queue_backlog_std']):.3f} & {solver:.1f}\\% \\\\"
            )
    tune.extend(["\\bottomrule", "\\end{tabular}"])
    (tables_dir / "table_formal_dcee_tuning.tex").write_text("\n".join(tune) + "\n", encoding="utf-8")


def write_figures(summary: pd.DataFrame, pairwise_df: pd.DataFrame, trace: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not trace.empty and {"algorithm", "train_seed", "step", "mean_reward_window"}.issubset(trace.columns):
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for (alg, seed), group in trace.groupby(["algorithm", "train_seed"], sort=False):
            ax.plot(group["step"], group["mean_reward_window"], alpha=0.8, label=f"{alg}-{seed}")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Mean reward window")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(figures_dir / "formal_drl_learning_curves.pdf")
        plt.close(fig)

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        ax.scatter(summary["total_penalty_mean"], summary["queue_backlog_mean"], s=55)
        for _, row in summary.iterrows():
            ax.annotate(str(row["method"]), (row["total_penalty_mean"], row["queue_backlog_mean"]), fontsize=8)
        ax.set_xlabel("Penalty")
        ax.set_ylabel("Backlog")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "formal_drl_penalty_backlog_tradeoff.pdf")
        plt.close(fig)

    if not pairwise_df.empty:
        plot_df = pairwise_df[pairwise_df["metric"].isin(["total_penalty", "queue_backlog"])].copy()
        if not plot_df.empty:
            labels = [f"{r.comparison}\n{r.metric}" for r in plot_df.itertuples()]
            means = plot_df["paired_diff_mean_dcee_minus_other"].to_numpy(dtype=float)
            low = plot_df["bootstrap_ci_low"].to_numpy(dtype=float)
            high = plot_df["bootstrap_ci_high"].to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(7.0, 4.0))
            y = np.arange(len(labels))
            ax.errorbar(means, y, xerr=[means - low, high - means], fmt="o", capsize=3)
            ax.axvline(0.0, color="black", linewidth=1.0)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("DCEE-tuned minus comparator")
            ax.grid(True, axis="x", alpha=0.3)
            fig.tight_layout()
            fig.savefig(figures_dir / "formal_drl_pairwise_ci.pdf")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze formal DRL and DCEE tuning results.")
    parser.add_argument("--root-dir", type=Path, default=Path("results_r1/formal_drl_tuning"))
    parser.add_argument("--tables-dir", type=Path, default=Path("tables_r1"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures_r1"))
    args = parser.parse_args()

    eval_path = args.root_dir / "drl_test_by_episode.csv"
    if not eval_path.exists():
        raise FileNotFoundError(eval_path)
    eval_df = pd.read_csv(eval_path)
    summary = summarize(eval_df)
    pairwise_df = pairwise(eval_df)
    training = pd.read_csv(args.root_dir / "drl_training_summary.csv") if (args.root_dir / "drl_training_summary.csv").exists() else pd.DataFrame()
    trace = pd.read_csv(args.root_dir / "drl_training_trace.csv") if (args.root_dir / "drl_training_trace.csv").exists() else pd.DataFrame()
    dcee_stage2 = pd.read_csv(args.root_dir / "dcee_tuning_stage2.csv") if (args.root_dir / "dcee_tuning_stage2.csv").exists() else pd.DataFrame()

    summary.to_csv(args.root_dir / "formal_comparison_summary.csv", index=False)
    pairwise_df.to_csv(args.root_dir / "formal_pairwise_statistics.csv", index=False)
    if not eval_df.empty:
        runtime_cols = [c for c in ["method", "test_seed", "inference_runtime_ms", "solver_runtime_ms"] if c in eval_df.columns]
        if runtime_cols:
            runtime_summary = eval_df.groupby("method", sort=False)[runtime_cols[2:]].agg(["mean", "std"]).reset_index()
            runtime_summary.to_csv(args.root_dir / "inference_runtime_summary.csv", index=False)
    write_tables(summary, training, dcee_stage2, args.tables_dir)
    write_figures(summary, pairwise_df, trace, args.figures_dir)
    manifest = {
        "inputs": [str(eval_path)],
        "outputs": [
            str(args.root_dir / "formal_comparison_summary.csv"),
            str(args.root_dir / "formal_pairwise_statistics.csv"),
            str(args.tables_dir / "table_formal_drl_comparison.tex"),
            str(args.tables_dir / "table_formal_drl_training_protocol.tex"),
            str(args.tables_dir / "table_formal_dcee_tuning.tex"),
            str(args.figures_dir / "formal_drl_learning_curves.pdf"),
            str(args.figures_dir / "formal_drl_penalty_backlog_tradeoff.pdf"),
            str(args.figures_dir / "formal_drl_pairwise_ci.pdf"),
        ],
    }
    (args.root_dir / "formal_analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
