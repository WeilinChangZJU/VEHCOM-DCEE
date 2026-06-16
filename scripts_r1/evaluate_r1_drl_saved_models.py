from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts_r1.run_r1_drl_baselines import (
    build_config,
    build_reward_config,
    evaluate_model,
    seed_level_summary,
)
from src_r1.uav_isac_core_r1 import ScenarioConfig, UavIsacSimulator


def read_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_dcee_label(cfg: ScenarioConfig, label: str, test_seed: int, slots: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sim = UavIsacSimulator(cfg)
    summary_df, slot_df = sim.simulate_r1("proposed_full", horizon=slots, seed=test_seed)
    rows: List[Dict[str, Any]] = []
    for _, rec in slot_df.iterrows():
        row = rec.to_dict()
        row.update({
            "method": label,
            "test_seed": int(test_seed),
            "train_seed": -1,
            "reward": float(-row.get("total_penalty", 0.0)),
            "inference_runtime_ms": float(row.get("runtime_ms", 0.0)),
        })
        rows.append(row)
    summary = summary_df.iloc[0].to_dict()
    summary.update({
        "method": label,
        "test_seed": int(test_seed),
        "train_seed": -1,
        "inference_runtime_ms": float(slot_df["runtime_ms"].mean()),
    })
    return rows, summary


def cfg_from_best_payload(path: Path) -> ScenarioConfig:
    payload = read_config(path)
    return build_config(payload["scenario_config"])


def fmt_mean_std(row: pd.Series, col: str, digits: int = 3) -> str:
    mean = row.get(f"{col}_mean")
    std = row.get(f"{col}_std")
    if pd.isna(mean):
        return "--"
    if pd.isna(std):
        std = 0.0
    return f"{float(mean):.{digits}f} $\\pm$ {float(std):.{digits}f}"


def write_table(summary: pd.DataFrame, table_path: Path, slots: int, test_seeds: List[int]) -> None:
    table_path.parent.mkdir(parents=True, exist_ok=True)
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
            f"{row['method']} & {fmt_mean_std(row, 'total_penalty')} & "
            f"{fmt_mean_std(row, 'queue_backlog')} & {fmt_mean_std(row, 'aoi_violation')} & "
            f"{fmt_mean_std(row, 'uncertainty_violation')} & {solver_txt} & {infer_txt} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        f"% Saved PPO/SAC models; no retraining; test seeds={test_seeds}; slots={slots}.",
    ])
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved PPO/SAC R1 models on extended matched seeds.")
    parser.add_argument("--config", type=Path, default=Path("configs_r1/drl_baselines_extended_eval.json"))
    parser.add_argument("--outdir", type=Path, default=Path("results_r1/drl_baselines_extended_eval"))
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--tuned-config", type=Path, default=Path("results_r1/patch9b_performance_strengthening/best_dcee_config.json"))
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--test-seeds", type=str, default=None, help="Comma-separated test seeds.")
    args = parser.parse_args()

    config = read_config(args.config)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir or Path(config.get("model_dir", "results_r1/drl_baselines/models"))
    slots = int(args.slots or config.get("slots", 100))
    if args.test_seeds:
        test_seeds = [int(x) for x in args.test_seeds.split(",") if x.strip()]
    else:
        test_seeds = [int(x) for x in config.get("test_seeds", [100, 101, 102, 103, 104])]

    cfg_default = build_config(config.get("base", {}))
    reward_cfg = build_reward_config(config.get("reward", {}))
    cfg_tuned = cfg_from_best_payload(args.tuned_config) if args.tuned_config.exists() else None

    eval_rows: List[Dict[str, Any]] = []
    eval_summary_rows: List[Dict[str, Any]] = []

    for seed in test_seeds:
        rows, summary = evaluate_dcee_label(cfg_default, "DCEE-default", seed, slots)
        eval_rows.extend(rows)
        eval_summary_rows.append(summary)
        if cfg_tuned is not None:
            rows, summary = evaluate_dcee_label(cfg_tuned, "DCEE-tuned", seed, slots)
            eval_rows.extend(rows)
            eval_summary_rows.append(summary)

    try:
        from stable_baselines3 import PPO, SAC

        model_specs = [
            ("PPO-Mobility-Exact", PPO, model_dir / "ppo_seed_0.zip"),
            ("SAC-Mobility-Exact", SAC, model_dir / "sac_seed_0.zip"),
        ]
        for label, cls, path in model_specs:
            if not path.exists():
                continue
            model = cls.load(path)
            for seed in test_seeds:
                rows, summary = evaluate_model(label, model, cfg_default, reward_cfg, seed, slots, train_seed=0)
                eval_rows.extend(rows)
                eval_summary_rows.append(summary)
    except Exception as exc:
        (outdir / "dependency_error.txt").write_text(str(exc), encoding="utf-8")

    eval_df = pd.DataFrame(eval_summary_rows)
    slot_df = pd.DataFrame(eval_rows)
    summary_df = seed_level_summary(eval_df) if not eval_df.empty else pd.DataFrame()
    eval_df.to_csv(outdir / "rl_eval_by_episode.csv", index=False)
    slot_df.to_csv(outdir / "rl_test_slot_metrics.csv", index=False)
    summary_df.to_csv(outdir / "rl_eval_summary.csv", index=False)
    runtime_cols = [c for c in ["method", "train_seed", "test_seed", "slot", "inference_runtime_ms", "solver_runtime_ms"] if c in slot_df.columns]
    slot_df[runtime_cols].to_csv(outdir / "rl_inference_runtime.csv", index=False)
    write_table(summary_df, ROOT / "tables_r1" / "table_drl_comparison_extended.tex", slots=slots, test_seeds=test_seeds)
    manifest = {
        "config": str(args.config),
        "model_dir": str(model_dir),
        "tuned_config": str(args.tuned_config) if args.tuned_config.exists() else "",
        "slots": slots,
        "test_seeds": test_seeds,
        "note": "Saved-model evaluation only; no retraining.",
        "outputs": [
            str(outdir / "rl_eval_summary.csv"),
            str(outdir / "rl_test_slot_metrics.csv"),
            str(ROOT / "tables_r1" / "table_drl_comparison_extended.tex"),
        ],
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
