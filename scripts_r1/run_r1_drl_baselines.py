from __future__ import annotations

import argparse
import json
import os
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Windows Anaconda can load Intel OpenMP through both SciPy and PyTorch.
# This runner is diagnostic-only; limiting threads and allowing duplicate
# OpenMP avoids aborting before SB3 training starts.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_r1.rl_env_r1 import RLMobilityEnv, RLRewardConfig, evaluate_policy_on_env
from src_r1.uav_isac_core_r1 import ScenarioConfig, UavIsacSimulator


LABELS = {
    "DCEE": "DCEE",
    "Random-Mobility-Exact": "Random-Mobility-Exact",
    "PPO": "PPO-Mobility-Exact",
    "SAC": "SAC-Mobility-Exact",
}


def read_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def emit(message: str, log_handle) -> None:
    print(message)
    log_handle.write(message + "\n")
    log_handle.flush()


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"


def build_config(base: Dict[str, Any]) -> ScenarioConfig:
    return ScenarioConfig(
        num_vehicles=int(base.get("num_vehicles", 6)),
        num_targets=int(base.get("num_targets", 3)),
        benchmark_num_directions=int(base.get("benchmark_num_directions", 16)),
        arrival_rate=float(base.get("arrival_rate", 0.8)),
        V=float(base.get("V", 12.0)),
        rho_r=float(base.get("rho_r", 0.01)),
        eta_t=float(base.get("eta_t", 2.0)),
        g0_sens=float(base.get("g0_sens", 96.0)),
        mu_reduction_scale=float(base.get("mu_reduction_scale", 12.0)),
        c_Y=float(base.get("c_Y", 8.0)),
        c_Z=float(base.get("c_Z", 8.0)),
        aoi_threshold=float(base.get("aoi_threshold", 5.0)),
        uncertainty_threshold=float(base.get("uncertainty_threshold", 3.5)),
        init_x=float(base.get("init_x", 200.0)),
        init_y=float(base.get("init_y", 0.0)),
        shortlist_num_local_dirs=int(base.get("shortlist_num_local_dirs", 8)),
        shortlist_radius_scale=float(base.get("shortlist_radius_scale", 0.45)),
        shortlist_size=int(base.get("shortlist_size", 11)),
        sensing_success_gain=float(base.get("sensing_success_gain", 1.2)),
        refresh_uncertainty_gain=float(base.get("refresh_uncertainty_gain", 1.15)),
        nonrefresh_mu_factor=float(base.get("nonrefresh_mu_factor", 0.15)),
        inner_solver_maxiter=int(base.get("inner_solver_maxiter", 80)),
        inner_solver_ftol=float(base.get("inner_solver_ftol", 1e-4)),
        sun_horizon_steps=int(base.get("sun_horizon_steps", 2)),
        sun_outer_rounds=int(base.get("sun_outer_rounds", 1)),
        sun_num_candidates=int(base.get("sun_num_candidates", 8)),
    )


def build_reward_config(data: Dict[str, Any]) -> RLRewardConfig:
    return RLRewardConfig(
        penalty_weight=float(data.get("penalty_weight", 1.0)),
        queue_weight=float(data.get("queue_weight", 0.004)),
        virtual_weight=float(data.get("virtual_weight", 0.015)),
        aoi_violation_weight=float(data.get("aoi_violation_weight", 0.25)),
        uncertainty_violation_weight=float(data.get("uncertainty_violation_weight", 0.35)),
        reward_scale=float(data.get("reward_scale", 1.0)),
    )


def make_env(cfg: ScenarioConfig, horizon: int, seed: int, reward_cfg: RLRewardConfig) -> RLMobilityEnv:
    return RLMobilityEnv(cfg=cfg, horizon=horizon, seed=seed, reward_cfg=reward_cfg, use_rescue_solver=True)


class RewardTraceCallback:
    def __init__(self, algorithm: str, train_seed: int, log_interval: int = 50):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Callback(BaseCallback):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer

            def _on_step(self) -> bool:
                rewards = self.locals.get("rewards")
                if rewards is not None:
                    self.outer.recent_rewards.extend(float(x) for x in np.asarray(rewards).reshape(-1))
                if self.n_calls % self.outer.log_interval == 0 and self.outer.recent_rewards:
                    window = self.outer.recent_rewards[-self.outer.log_interval:]
                    self.outer.rows.append({
                        "algorithm": self.outer.algorithm,
                        "train_seed": self.outer.train_seed,
                        "step": int(self.num_timesteps),
                        "mean_reward_window": float(np.mean(window)),
                    })
                return True

        self.algorithm = algorithm
        self.train_seed = int(train_seed)
        self.log_interval = int(log_interval)
        self.recent_rewards: List[float] = []
        self.rows: List[Dict[str, Any]] = []
        self.callback = _Callback(self)


def train_model(
    algorithm: str,
    cfg: ScenarioConfig,
    reward_cfg: RLRewardConfig,
    horizon: int,
    train_seed: int,
    train_steps: int,
    algo_kwargs: Dict[str, Any],
    model_dir: Path,
) -> Tuple[Any, Dict[str, Any], List[Dict[str, Any]]]:
    try:
        import torch
        from stable_baselines3 import PPO, SAC

        torch.set_num_threads(1)
    except Exception as exc:  # pragma: no cover - environment dependent.
        return None, {
            "algorithm": algorithm,
            "train_seed": int(train_seed),
            "train_steps": int(train_steps),
            "status": "dependency_failed",
            "message": str(exc),
            "wall_clock_sec": 0.0,
            "model_path": "",
        }, []

    env = make_env(cfg, horizon=horizon, seed=train_seed, reward_cfg=reward_cfg)
    cls = {"PPO": PPO, "SAC": SAC}.get(algorithm.upper())
    if cls is None:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    model_path = model_dir / f"{algorithm.lower()}_seed_{train_seed}.zip"
    callback = RewardTraceCallback(algorithm=algorithm, train_seed=train_seed)
    status = "completed"
    message = ""
    t0 = time.perf_counter()
    model = None
    try:
        model = cls("MlpPolicy", env, seed=int(train_seed), **algo_kwargs)
        model.learn(total_timesteps=int(train_steps), callback=callback.callback, progress_bar=False)
        model.save(model_path)
    except Exception as exc:
        status = "failed"
        message = str(exc)
    wall = time.perf_counter() - t0
    row = {
        "algorithm": algorithm,
        "method_label": LABELS.get(algorithm, algorithm),
        "train_seed": int(train_seed),
        "train_steps": int(train_steps),
        "status": status,
        "message": message,
        "wall_clock_sec": float(wall),
        "model_path": str(model_path if status == "completed" else ""),
        "python_version": sys.version.split()[0],
        "stable_baselines3_version": package_version("stable-baselines3"),
        "gymnasium_version": package_version("gymnasium"),
        "torch_version": package_version("torch"),
        "numpy_version": package_version("numpy"),
    }
    return model, row, callback.rows


def evaluate_model(
    method_label: str,
    model: Any,
    cfg: ScenarioConfig,
    reward_cfg: RLRewardConfig,
    test_seed: int,
    slots: int,
    train_seed: int | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    env = make_env(cfg, horizon=slots, seed=test_seed, reward_cfg=reward_cfg)
    rows, summary = evaluate_policy_on_env(env, model, seed=test_seed)
    for row in rows:
        row.update({
            "method": method_label,
            "test_seed": int(test_seed),
            "train_seed": -1 if train_seed is None else int(train_seed),
        })
    summary.update({
        "method": method_label,
        "test_seed": int(test_seed),
        "train_seed": -1 if train_seed is None else int(train_seed),
    })
    return rows, summary


def evaluate_dcee(cfg: ScenarioConfig, test_seed: int, slots: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sim = UavIsacSimulator(cfg)
    summary_df, slot_df = sim.simulate_r1("proposed_full", horizon=slots, seed=test_seed)
    rows: List[Dict[str, Any]] = []
    for _, rec in slot_df.iterrows():
        row = rec.to_dict()
        row.update({
            "method": "DCEE",
            "test_seed": int(test_seed),
            "train_seed": -1,
            "reward": float(-row.get("total_penalty", 0.0)),
            "inference_runtime_ms": float(row.get("runtime_ms", np.nan)),
        })
        rows.append(row)
    summary = summary_df.iloc[0].to_dict()
    summary.update({
        "method": "DCEE",
        "test_seed": int(test_seed),
        "train_seed": -1,
        "inference_runtime_ms": float(slot_df["runtime_ms"].mean()),
    })
    return rows, summary


def seed_level_summary(eval_summary: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
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
    rows = []
    for method, group in eval_summary.groupby("method", sort=False):
        row: Dict[str, Any] = {"method": method, "episodes": int(len(group))}
        for col in numeric_cols:
            if col in group.columns:
                values = pd.to_numeric(group[col], errors="coerce").dropna()
                if len(values):
                    row[f"{col}_mean"] = float(values.mean())
                    row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_mean_std(row: pd.Series, col: str, digits: int = 3) -> str:
    mean = row.get(f"{col}_mean", np.nan)
    std = row.get(f"{col}_std", np.nan)
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        std = 0.0
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def write_tables(
    summary: pd.DataFrame,
    train_df: pd.DataFrame,
    tables_dir: Path,
    test_seeds: List[int],
    test_slots: int,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        "DCEE": "DCEE",
        "Random-Mobility-Exact": "Random-Mobility-Exact",
        "PPO-Mobility-Exact": "PPO-Mobility-Exact",
        "SAC-Mobility-Exact": "SAC-Mobility-Exact",
    }
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Method & Penalty & Backlog & AoI exc. & Unc. exc. & Solver & Policy ms \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        method = labels.get(str(row["method"]), str(row["method"]))
        solver = row.get("solver_success_mean", np.nan)
        solver_txt = f"{100.0 * solver:.1f}\\%" if np.isfinite(solver) else "--"
        infer = row.get("inference_runtime_ms_mean", np.nan)
        infer_txt = f"{infer:.2f}" if np.isfinite(infer) else "--"
        lines.append(
            f"{method} & {fmt_mean_std(row, 'total_penalty')} & "
            f"{fmt_mean_std(row, 'queue_backlog')} & "
            f"{fmt_mean_std(row, 'aoi_violation')} & "
            f"{fmt_mean_std(row, 'uncertainty_violation')} & "
            f"{solver_txt} & {infer_txt} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
    ])
    (tables_dir / "table_drl_comparison.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    proto_lines = [
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        "Algorithm & Train seeds & Steps & Test seeds & Test slots & Status \\\\",
        "\\midrule",
    ]
    if train_df.empty:
        proto_lines.append("No learning run & 0 & 0 & 0 & 0 & missing \\\\")
    else:
        test_seed_txt = ",".join(str(int(x)) for x in test_seeds)
        for alg, group in train_df.groupby("method_label", sort=False):
            steps = ",".join(str(int(x)) for x in sorted(group["train_steps"].unique()))
            seeds = ",".join(str(int(x)) for x in sorted(group["train_seed"].unique()))
            statuses = ",".join(sorted(str(x) for x in group["status"].unique()))
            proto_lines.append(f"{alg} & {seeds} & {steps} & {test_seed_txt} & {int(test_slots)} & {statuses} \\\\")
    proto_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
    ])
    (tables_dir / "table_drl_training_protocol.tex").write_text("\n".join(proto_lines) + "\n", encoding="utf-8")


def write_figures(trace_df: pd.DataFrame, summary: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not trace_df.empty:
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        for (alg, seed), group in trace_df.groupby(["algorithm", "train_seed"], sort=False):
            ax.plot(group["step"], group["mean_reward_window"], label=f"{alg} seed {seed}")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Mean reward window")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "drl_learning_curve.pdf")
        plt.close(fig)
    else:
        (figures_dir / "drl_learning_curve.pdf").write_text("No learning curve available.\n", encoding="utf-8")

    if not summary.empty and "total_penalty_mean" in summary.columns:
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        methods = summary["method"].astype(str).tolist()
        means = summary["total_penalty_mean"].astype(float).to_numpy()
        errs = summary.get("total_penalty_std", pd.Series(np.zeros(len(summary)))).astype(float).to_numpy()
        ax.bar(methods, means, yerr=errs, capsize=3)
        ax.set_ylabel("Mean penalty")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_dir / "drl_performance_comparison.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R1 DRL mobility-only baselines.")
    parser.add_argument("--config", type=Path, default=Path("configs_r1/drl_baselines_light.json"))
    parser.add_argument("--outdir", type=Path, default=Path("results_r1/drl_baselines"))
    parser.add_argument("--slots", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None, help="Override all algorithm training steps.")
    parser.add_argument("--algorithms", nargs="*", default=None)
    parser.add_argument("--smoke-only", action="store_true", help="Run random environment smoke without SB3 training.")
    args = parser.parse_args()

    config = read_config(args.config)
    outdir = args.outdir
    logs_dir = ROOT / "logs_r1"
    tables_dir = ROOT / "tables_r1"
    figures_dir = ROOT / "figures_r1"
    models_dir = outdir / "models"
    outdir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    with (logs_dir / "patch7_drl_baselines.log").open("w", encoding="utf-8") as log:
        emit("Patch 7 DRL mobility-only baselines", log)
        emit(f"Config: {args.config}", log)
        emit(
            "Versions: "
            f"python={sys.version.split()[0]}, "
            f"stable-baselines3={package_version('stable-baselines3')}, "
            f"gymnasium={package_version('gymnasium')}, "
            f"torch={package_version('torch')}",
            log,
        )

        cfg = build_config(config.get("base", {}))
        reward_cfg = build_reward_config(config.get("reward", {}))
        slots = int(args.slots if args.slots is not None else config.get("slots", 60))
        train_seeds = [int(x) for x in config.get("train_seeds", [0])]
        test_seeds = [int(x) for x in config.get("test_seeds", [100, 101, 102])]
        algorithms = [str(x).upper() for x in (args.algorithms if args.algorithms is not None else config.get("algorithms", ["PPO", "SAC"]))]
        train_steps_cfg = {str(k).upper(): int(v) for k, v in config.get("train_steps", {}).items()}
        if args.train_steps is not None:
            train_steps_cfg = {alg: int(args.train_steps) for alg in algorithms}

        training_rows: List[Dict[str, Any]] = []
        trace_rows: List[Dict[str, Any]] = []
        eval_rows: List[Dict[str, Any]] = []
        eval_summary_rows: List[Dict[str, Any]] = []
        trained_models: List[Tuple[str, int, Any]] = []

        if config.get("include_dcee_reference", True):
            emit("Evaluating DCEE reference on matched test seeds.", log)
            for seed in test_seeds:
                rows, summary = evaluate_dcee(cfg, seed, slots)
                eval_rows.extend(rows)
                eval_summary_rows.append(summary)

        if config.get("include_random_reference", True):
            emit("Evaluating Random-Mobility-Exact reference.", log)
            for seed in test_seeds:
                rows, summary = evaluate_model("Random-Mobility-Exact", "random", cfg, reward_cfg, seed, slots)
                eval_rows.extend(rows)
                eval_summary_rows.append(summary)

        if args.smoke_only:
            emit("Smoke-only mode: skipping SB3 training.", log)
        else:
            for alg in algorithms:
                kwargs = config.get(f"{alg.lower()}_kwargs", {})
                for train_seed in train_seeds:
                    steps = int(train_steps_cfg.get(alg, 3000))
                    emit(f"Training {alg} seed={train_seed} steps={steps}.", log)
                    model, train_row, trace = train_model(
                        alg,
                        cfg,
                        reward_cfg,
                        horizon=slots,
                        train_seed=train_seed,
                        train_steps=steps,
                        algo_kwargs=kwargs,
                        model_dir=models_dir,
                    )
                    training_rows.append(train_row)
                    trace_rows.extend(trace)
                    if model is not None and train_row["status"] == "completed":
                        trained_models.append((alg, train_seed, model))
                    else:
                        emit(f"{alg} seed={train_seed} did not complete: {train_row.get('message', '')}", log)

            for alg, train_seed, model in trained_models:
                label = LABELS.get(alg, alg)
                emit(f"Evaluating {label} trained_seed={train_seed}.", log)
                for seed in test_seeds:
                    rows, summary = evaluate_model(label, model, cfg, reward_cfg, seed, slots, train_seed=train_seed)
                    eval_rows.extend(rows)
                    eval_summary_rows.append(summary)

        training_df = pd.DataFrame(training_rows)
        trace_df = pd.DataFrame(trace_rows)
        eval_df = pd.DataFrame(eval_summary_rows)
        slot_df = pd.DataFrame(eval_rows)
        summary_df = seed_level_summary(eval_df) if not eval_df.empty else pd.DataFrame()

        training_df.to_csv(outdir / "rl_training_summary.csv", index=False)
        trace_df.to_csv(outdir / "rl_training_trace.csv", index=False)
        eval_df.to_csv(outdir / "rl_eval_by_episode.csv", index=False)
        summary_df.to_csv(outdir / "rl_eval_summary.csv", index=False)
        slot_df.to_csv(outdir / "rl_test_slot_metrics.csv", index=False)
        runtime_cols = [c for c in ["method", "train_seed", "test_seed", "slot", "inference_runtime_ms", "solver_runtime_ms"] if c in slot_df.columns]
        slot_df[runtime_cols].to_csv(outdir / "rl_inference_runtime.csv", index=False)

        write_tables(summary_df, training_df, tables_dir, test_seeds=test_seeds, test_slots=slots)
        write_figures(trace_df, summary_df, figures_dir)

        manifest = {
            "config": str(args.config),
            "slots": slots,
            "train_seeds": train_seeds,
            "test_seeds": test_seeds,
            "algorithms": algorithms,
            "smoke_only": bool(args.smoke_only),
            "outputs": [
                str(outdir / "rl_training_summary.csv"),
                str(outdir / "rl_eval_summary.csv"),
                str(outdir / "rl_test_slot_metrics.csv"),
                str(outdir / "rl_inference_runtime.csv"),
                str(tables_dir / "table_drl_comparison.tex"),
                str(tables_dir / "table_drl_training_protocol.tex"),
                str(figures_dir / "drl_learning_curve.pdf"),
                str(figures_dir / "drl_performance_comparison.pdf"),
            ],
        }
        (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        emit(f"Wrote outputs to {outdir}.", log)
        if not summary_df.empty:
            emit(summary_df[["method", "total_penalty_mean", "queue_backlog_mean", "solver_success_mean"]].to_string(index=False), log)


if __name__ == "__main__":
    main()
