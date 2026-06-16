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

from scripts_r1.evaluate_r1_drl_saved_models import evaluate_dcee_label
from scripts_r1.run_r1_drl_baselines import build_config, build_reward_config, evaluate_model, seed_level_summary


def read_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(algorithm: str, path: Path):
    from stable_baselines3 import DDPG, PPO, SAC

    cls_map = {"PPO": PPO, "SAC": SAC, "DDPG": DDPG}
    cls = cls_map.get(algorithm.upper())
    if cls is None:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return cls.load(path)


def summarize_dcee(df: pd.DataFrame) -> pd.DataFrame:
    return seed_level_summary(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal held-out evaluation for DCEE-tuned and DRL checkpoints.")
    parser.add_argument("--config", type=Path, default=Path("configs_r1/formal_drl_training.json"))
    parser.add_argument("--outdir", type=Path, default=Path("results_r1/formal_drl_tuning"))
    parser.add_argument("--dcee-config", type=Path, default=Path("results_r1/formal_drl_tuning/best_dcee_config_formal.json"))
    parser.add_argument("--selected-checkpoints", type=Path, default=Path("results_r1/formal_drl_tuning/selected_drl_checkpoints.json"))
    parser.add_argument("--slots", type=int, default=None)
    args = parser.parse_args()

    cfg_json = read_config(args.config)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    test_seeds = [int(x) for x in cfg_json.get("test_seeds", list(range(100, 110)))]
    test_slots = int(args.slots or cfg_json.get("test_slots", 200))
    cfg_default = build_config(cfg_json.get("base", {}))
    reward_cfg = build_reward_config(cfg_json.get("reward", {}))
    tuned_payload = read_config(args.dcee_config)
    cfg_tuned = build_config(tuned_payload["scenario_config"])
    selected = read_config(args.selected_checkpoints) if args.selected_checkpoints.exists() else {}

    eval_rows: List[Dict[str, Any]] = []
    eval_summary_rows: List[Dict[str, Any]] = []

    for seed in test_seeds:
        rows, summary = evaluate_dcee_label(cfg_default, "DCEE-default", seed, test_slots)
        eval_rows.extend(rows)
        eval_summary_rows.append(summary)
        rows, summary = evaluate_dcee_label(cfg_tuned, "DCEE-tuned", seed, test_slots)
        eval_rows.extend(rows)
        eval_summary_rows.append(summary)

    for algorithm, rec in selected.items():
        model_path = Path(rec["model_path"])
        if not model_path.exists():
            continue
        model = load_model(algorithm, model_path)
        label = str(rec.get("method", f"{algorithm}-Mobility-Exact"))
        for seed in test_seeds:
            rows, summary = evaluate_model(label, model, cfg_default, reward_cfg, seed, test_slots, train_seed=int(rec["train_seed"]))
            for row in rows:
                row.update({
                    "algorithm": algorithm,
                    "checkpoint_step": int(rec["checkpoint_step"]),
                    "model_path": str(model_path),
                })
            summary.update({
                "algorithm": algorithm,
                "checkpoint_step": int(rec["checkpoint_step"]),
                "model_path": str(model_path),
            })
            eval_rows.extend(rows)
            eval_summary_rows.append(summary)

    eval_by_episode = pd.DataFrame(eval_summary_rows)
    slot_df = pd.DataFrame(eval_rows)
    summary_df = seed_level_summary(eval_by_episode) if not eval_by_episode.empty else pd.DataFrame()
    eval_by_episode.to_csv(outdir / "drl_test_by_episode.csv", index=False)
    slot_df.to_csv(outdir / "drl_test_slot_metrics.csv", index=False)
    summary_df.to_csv(outdir / "drl_test_summary.csv", index=False)
    dcee_summary = summary_df[summary_df["method"].isin(["DCEE-default", "DCEE-tuned"])].copy() if not summary_df.empty else pd.DataFrame()
    dcee_summary.to_csv(outdir / "dcee_test_summary.csv", index=False)
    runtime_cols = [c for c in ["method", "train_seed", "test_seed", "slot", "inference_runtime_ms", "solver_runtime_ms"] if c in slot_df.columns]
    slot_df[runtime_cols].to_csv(outdir / "formal_inference_runtime.csv", index=False)
    manifest = {
        "config": str(args.config),
        "dcee_config": str(args.dcee_config),
        "selected_checkpoints": str(args.selected_checkpoints),
        "test_seeds": test_seeds,
        "test_slots": test_slots,
        "outputs": [
            str(outdir / "drl_test_summary.csv"),
            str(outdir / "drl_test_by_episode.csv"),
            str(outdir / "drl_test_slot_metrics.csv"),
        ],
    }
    (outdir / "formal_eval_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
