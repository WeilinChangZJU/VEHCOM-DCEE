from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts_r1.run_r1_drl_baselines import (
    RewardTraceCallback,
    build_config,
    build_reward_config,
    evaluate_model,
    make_env,
    package_version,
)


LABELS = {
    "PPO": "PPO-Mobility-Exact",
    "SAC": "SAC-Mobility-Exact",
    "DDPG": "DDPG-Mobility-Exact",
}


def read_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def emit(message: str, log_handle) -> None:
    print(message)
    log_handle.write(message + "\n")
    log_handle.flush()


def algorithm_class_and_kwargs(algorithm: str, kwargs: Dict[str, Any]):
    import torch
    from stable_baselines3 import DDPG, PPO, SAC
    from stable_baselines3.common.noise import NormalActionNoise

    torch.set_num_threads(1)
    alg = algorithm.upper()
    clean_kwargs = dict(kwargs)
    if alg == "PPO":
        return PPO, clean_kwargs
    if alg == "SAC":
        return SAC, clean_kwargs
    if alg == "DDPG":
        clean_kwargs.setdefault("action_noise", NormalActionNoise(mean=np.zeros(2), sigma=0.10 * np.ones(2)))
        return DDPG, clean_kwargs
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def summarize_validation(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    summary_rows: List[Dict[str, Any]] = []
    for keys, group in rows.groupby(["algorithm", "method", "train_seed", "checkpoint_step", "model_path"], sort=False):
        algorithm, method, train_seed, checkpoint_step, model_path = keys
        row: Dict[str, Any] = {
            "algorithm": algorithm,
            "method": method,
            "train_seed": int(train_seed),
            "checkpoint_step": int(checkpoint_step),
            "model_path": model_path,
            "validation_episodes": int(len(group)),
        }
        for col in [
            "total_penalty",
            "queue_backlog",
            "aoi_violation",
            "uncertainty_violation",
            "travel_distance",
            "solver_success",
            "inference_runtime_ms",
        ]:
            if col in group.columns:
                values = pd.to_numeric(group[col], errors="coerce").dropna()
                if len(values):
                    row[f"{col}_mean"] = float(values.mean())
                    row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["algorithm", "total_penalty_mean", "queue_backlog_mean", "checkpoint_step"],
            ascending=[True, True, True, True],
        )
    return summary


def select_checkpoints(validation_summary: pd.DataFrame) -> Dict[str, Any]:
    selected: Dict[str, Any] = {}
    for algorithm, group in validation_summary.groupby("algorithm", sort=False):
        best = group.sort_values(["total_penalty_mean", "queue_backlog_mean", "checkpoint_step"]).iloc[0]
        selected[str(algorithm)] = {
            "method": str(best["method"]),
            "train_seed": int(best["train_seed"]),
            "checkpoint_step": int(best["checkpoint_step"]),
            "model_path": str(best["model_path"]),
            "validation_total_penalty_mean": float(best["total_penalty_mean"]),
            "validation_queue_backlog_mean": float(best["queue_backlog_mean"]),
            "selection_rule": "minimum validation mean penalty; tie-break lower backlog and earlier checkpoint",
        }
    return selected


def train_algorithm_seed(
    algorithm: str,
    train_seed: int,
    cfg,
    reward_cfg,
    training_horizon: int,
    train_steps: int,
    checkpoints: List[int],
    algo_kwargs: Dict[str, Any],
    validation_seeds: List[int],
    validation_slots: int,
    models_dir: Path,
    log_handle,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    training_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    validation_rows: List[Dict[str, Any]] = []
    method = LABELS.get(algorithm.upper(), algorithm.upper())
    t0 = time.perf_counter()
    status = "completed"
    message = ""
    completed_steps = 0
    saved_models: List[str] = []

    try:
        cls, kwargs = algorithm_class_and_kwargs(algorithm, algo_kwargs)
        env = make_env(cfg, horizon=training_horizon, seed=train_seed, reward_cfg=reward_cfg)
        callback = RewardTraceCallback(algorithm=algorithm.upper(), train_seed=train_seed, log_interval=250)
        model = cls("MlpPolicy", env, seed=int(train_seed), **kwargs)
        previous = 0
        for checkpoint in checkpoints:
            if checkpoint > train_steps:
                continue
            delta = int(checkpoint - previous)
            if delta <= 0:
                continue
            emit(f"Training {algorithm} seed={train_seed}: {previous}->{checkpoint}", log_handle)
            model.learn(
                total_timesteps=delta,
                reset_num_timesteps=(previous == 0),
                callback=callback.callback,
                progress_bar=False,
            )
            completed_steps = int(checkpoint)
            previous = int(checkpoint)
            model_path = models_dir / f"{algorithm.lower()}_seed_{train_seed}_step_{checkpoint}.zip"
            model.save(model_path)
            saved_models.append(str(model_path))
            for val_seed in validation_seeds:
                _, summary = evaluate_model(method, model, cfg, reward_cfg, val_seed, validation_slots, train_seed=train_seed)
                summary.update({
                    "algorithm": algorithm.upper(),
                    "checkpoint_step": int(checkpoint),
                    "model_path": str(model_path),
                })
                validation_rows.append(summary)
        trace_rows.extend(callback.rows)
    except Exception as exc:  # pragma: no cover - training stability is environment-dependent.
        status = "failed"
        message = str(exc)
        emit(f"{algorithm} seed={train_seed} failed: {message}", log_handle)

    wall = time.perf_counter() - t0
    training_rows.append({
        "algorithm": algorithm.upper(),
        "method": method,
        "train_seed": int(train_seed),
        "target_train_steps": int(train_steps),
        "completed_steps": int(completed_steps),
        "status": status,
        "message": message,
        "wall_clock_sec": float(wall),
        "saved_models": json.dumps(saved_models),
        "python_version": sys.version.split()[0],
        "stable_baselines3_version": package_version("stable-baselines3"),
        "gymnasium_version": package_version("gymnasium"),
        "torch_version": package_version("torch"),
        "numpy_version": package_version("numpy"),
    })
    return training_rows, trace_rows, validation_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal PPO/SAC/DDPG mobility-only training for Patch 10.")
    parser.add_argument("--config", type=Path, default=Path("configs_r1/formal_drl_training.json"))
    parser.add_argument("--outdir", type=Path, default=Path("results_r1/formal_drl_tuning"))
    parser.add_argument("--algorithms", nargs="*", default=None)
    parser.add_argument("--train-steps", type=int, default=None, help="Override train steps for all algorithms.")
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()

    cfg_json = read_config(args.config)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    models_dir = outdir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = ROOT / "logs_r1"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "patch10_formal_drl_training.log"

    cfg = build_config(cfg_json.get("base", {}))
    reward_cfg = build_reward_config(cfg_json.get("reward", {}))
    train_seeds = [int(x) for x in cfg_json.get("train_seeds", [0, 1, 2])]
    validation_seeds = [int(x) for x in cfg_json.get("validation_seeds", [20, 21, 22, 23, 24])]
    training_horizon = int(cfg_json.get("training_horizon", 200))
    validation_slots = int(cfg_json.get("validation_slots", 100))
    algorithms = [str(x).upper() for x in (args.algorithms or cfg_json.get("algorithms", ["PPO", "SAC"]))]
    train_steps_cfg = {str(k).upper(): int(v) for k, v in cfg_json.get("train_steps", {}).items()}
    checkpoints = sorted(int(x) for x in cfg_json.get("checkpoints", [10000, 25000, 50000]))
    if args.train_steps is not None:
        train_steps_cfg = {alg: int(args.train_steps) for alg in algorithms}
        checkpoints = [x for x in checkpoints if x <= int(args.train_steps)]
        if int(args.train_steps) not in checkpoints:
            checkpoints.append(int(args.train_steps))
            checkpoints = sorted(set(checkpoints))

    all_training: List[Dict[str, Any]] = []
    all_trace: List[Dict[str, Any]] = []
    all_validation: List[Dict[str, Any]] = []

    with log_path.open("w", encoding="utf-8") as log:
        emit("Patch 10 formal DRL training", log)
        emit(f"Config: {args.config}", log)
        emit(f"Algorithms={algorithms}, train_seeds={train_seeds}, validation_seeds={validation_seeds}", log)
        t0 = time.perf_counter()

        for algorithm in algorithms:
            kwargs = cfg_json.get(f"{algorithm.lower()}_kwargs", {})
            steps = int(train_steps_cfg.get(algorithm, 50000))
            for train_seed in train_seeds:
                expected_final = models_dir / f"{algorithm.lower()}_seed_{train_seed}_step_{steps}.zip"
                if args.skip_completed and expected_final.exists():
                    emit(f"Skipping existing {expected_final}", log)
                    continue
                training, trace, validation = train_algorithm_seed(
                    algorithm,
                    train_seed,
                    cfg,
                    reward_cfg,
                    training_horizon,
                    steps,
                    checkpoints,
                    kwargs,
                    validation_seeds,
                    validation_slots,
                    models_dir,
                    log,
                )
                all_training.extend(training)
                all_trace.extend(trace)
                all_validation.extend(validation)

        training_df = pd.DataFrame(all_training)
        trace_df = pd.DataFrame(all_trace)
        validation_df = pd.DataFrame(all_validation)
        validation_summary = summarize_validation(validation_df)
        selected = select_checkpoints(validation_summary) if not validation_summary.empty else {}

        training_df.to_csv(outdir / "drl_training_summary.csv", index=False)
        trace_df.to_csv(outdir / "drl_training_trace.csv", index=False)
        validation_df.to_csv(outdir / "drl_checkpoint_validation.csv", index=False)
        validation_summary.to_csv(outdir / "drl_checkpoint_validation_summary.csv", index=False)
        (outdir / "selected_drl_checkpoints.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

        manifest = {
            "config": str(args.config),
            "algorithms": algorithms,
            "train_seeds": train_seeds,
            "validation_seeds": validation_seeds,
            "training_horizon": training_horizon,
            "validation_slots": validation_slots,
            "checkpoints": checkpoints,
            "elapsed_sec": time.perf_counter() - t0,
            "selected": selected,
            "outputs": [
                str(outdir / "drl_training_summary.csv"),
                str(outdir / "drl_checkpoint_validation.csv"),
                str(outdir / "selected_drl_checkpoints.json"),
            ],
        }
        (outdir / "drl_training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        emit(json.dumps(selected, indent=2), log)
        emit(f"DRL training elapsed={manifest['elapsed_sec']:.1f}s", log)


if __name__ == "__main__":
    main()
