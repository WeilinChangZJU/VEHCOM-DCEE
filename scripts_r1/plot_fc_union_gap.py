from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LABELS = {
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


def plot_running_mean(slot_df: pd.DataFrame, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    for method, group in slot_df.groupby("method", sort=False):
        ordered = group.sort_values(["seed", "slot"]).copy()
        ordered["run_index"] = range(1, len(ordered) + 1)
        ordered["running_gap"] = ordered["finite_gap"].astype(float).expanding().mean()
        ax.plot(ordered["run_index"], ordered["running_gap"], linewidth=1.6, label=LABELS.get(method, method))
    ax.set_xlabel("Evaluated slot index")
    ax.set_ylabel("Running mean finite gap")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile)
    fig.savefig(outfile.with_suffix(".png"))
    plt.close(fig)


def plot_histogram(slot_df: pd.DataFrame, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=180)
    for method, group in slot_df.groupby("method", sort=False):
        values = group["finite_gap"].astype(float)
        ax.hist(values, bins=24, alpha=0.35, label=LABELS.get(method, method))
    ax.set_xlabel("Finite gap")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile)
    fig.savefig(outfile.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot FC-Union finite-gap diagnostics.")
    parser.add_argument("--input", type=str, default=str(ROOT / "results_r1" / "benchmark_full_balanced" / "slot_benchmark.csv"))
    parser.add_argument("--outdir", type=str, default=str(ROOT / "figures_r1"))
    parser.add_argument("--tag", type=str, default="balanced")
    args = parser.parse_args()

    slot_df = pd.read_csv(args.input)
    outdir = Path(args.outdir)
    plot_running_mean(slot_df, outdir / f"finite_gap_running_mean_{args.tag}.pdf")
    plot_histogram(slot_df, outdir / f"finite_gap_histogram_{args.tag}.pdf")
    print(f"Wrote figures to {outdir}")


if __name__ == "__main__":
    main()
