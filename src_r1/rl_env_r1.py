from __future__ import annotations

"""Mobility-only RL wrapper for the R1 UAV-ISAC simulator.

The wrapper keeps resource allocation out of the learning action.  A policy
selects only a 2D mobility displacement; the post-motion communication and
sensing action is then computed by the same exact resource solver used by the
R1 controllers.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - only used when Gymnasium is unavailable.
    gym = None
    spaces = None

from .uav_isac_core_r1 import (
    ActionResult,
    CandidateFamily,
    ScenarioConfig,
    SolverDiag,
    UavIsacSimulator,
)


@dataclass
class RLRewardConfig:
    penalty_weight: float = 1.0
    queue_weight: float = 0.004
    virtual_weight: float = 0.015
    aoi_violation_weight: float = 0.25
    uncertainty_violation_weight: float = 0.35
    reward_scale: float = 1.0


class RLMobilityEnv(gym.Env if gym is not None else object):
    """Gymnasium-compatible mobility-only environment.

    Action: continuous 2D vector in [-1, 1]^2.  The vector is projected to the
    current reachable disk and corridor bounds.  Resource variables are not
    learned; they are solved by `exact_value_with_rescue_diag` after motion.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: ScenarioConfig,
        horizon: int = 200,
        seed: int = 0,
        reward_cfg: Optional[RLRewardConfig] = None,
        use_rescue_solver: bool = True,
        solver_feas_tol: float = 1e-6,
    ):
        if gym is None or spaces is None:
            raise RuntimeError("Gymnasium is required for RLMobilityEnv.")
        super().__init__()
        self.cfg = cfg
        self.horizon = int(horizon)
        self.base_seed = int(seed)
        self.reward_cfg = reward_cfg or RLRewardConfig()
        self.use_rescue_solver = bool(use_rescue_solver)
        self.solver_feas_tol = float(solver_feas_tol)
        self.sim = UavIsacSimulator(cfg)
        self.state = None
        self.slot = 0
        self.last_info: Dict[str, Any] = {}

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(self.observation_dim(),), dtype=np.float32)

    def observation_dim(self) -> int:
        cfg = self.cfg
        return (
            2
            + cfg.num_vehicles
            + 4 * cfg.num_targets
            + 2 * cfg.num_vehicles
            + 2 * cfg.num_targets
            + 8
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        del options
        if seed is None:
            seed = self.base_seed
        self.base_seed = int(seed)
        self.state = self.sim.reset(int(seed))
        self.slot = 0
        self.last_info = {}
        return self._observation(), {}

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Environment must be reset before step().")
        action_arr = np.asarray(action, dtype=float).reshape(2)
        action_arr = np.clip(action_arr, -1.0, 1.0)
        norm = float(np.linalg.norm(action_arr))
        if norm > 1.0:
            action_arr = action_arr / max(norm, 1e-12)

        exo = self.sim.build_exogenous(self.state)
        pre_x = self.state.x.copy()
        target = pre_x + self.cfg.delta * action_arr
        y = self.sim.project_to_reachable(pre_x, target)

        solve_start = self.sim.exact_value_with_rescue_diag if self.use_rescue_solver else self.sim.exact_value_with_diag
        result = solve_start(
            y,
            self.state,
            exo,
            warm_start=self.state.last_exact_action,
            use_true_weights=True,
            **({"feas_tol": self.solver_feas_tol} if self.use_rescue_solver else {}),
        )
        diag: SolverDiag = result["solver_diag"]
        family = CandidateFamily(
            method="rl_mobility_exact",
            points=np.asarray([y], dtype=float),
            executed_point=y.copy(),
            labels=["rl/executed"],
        )
        action_result = ActionResult(
            method="rl_mobility_exact",
            velocity=(y - pre_x) / max(self.cfg.delta_t, 1e-12),
            post_motion_point=y.copy(),
            resource_action=np.asarray(result["action"], dtype=float).copy(),
            value=float(result["value"]),
            candidate_family=family,
            solver_diag=diag,
            runtime_ms=float(diag.runtime_ms),
            metadata={
                "controller_value": float(result["value"]),
                "controller_value_type": "unregularized_exact",
                "shortlist_size": 1,
                "next_last_exact_action": np.asarray(result["action"], dtype=float).copy(),
                "next_last_reg_action": None if self.state.last_reg_action is None else self.state.last_reg_action.copy(),
                "final_success": bool(result.get("final_success", diag.success)),
                "rescue_attempts": int(result.get("rescue_attempts", 0)),
                "rescue_success": bool(result.get("rescue_success", False)),
                "eligible_for_reference": bool(result.get("eligible_for_reference", diag.success)),
            },
        )
        metrics = self.sim.apply_action_r1(self.state, exo, action_result)
        self.slot += 1
        reward = self._reward(metrics)
        truncated = self.slot >= self.horizon
        terminated = False
        info = {
            **metrics,
            "slot": self.slot - 1,
            "executed_x": float(y[0]),
            "executed_y": float(y[1]),
            "solver_success": bool(diag.success),
            "solver_nit": int(diag.nit),
            "solver_runtime_ms": float(diag.runtime_ms),
            "solver_residual": float(diag.residual),
            "exact_value": float(action_result.value),
            "rescue_attempts": int(action_result.metadata["rescue_attempts"]),
            "rescue_success": bool(action_result.metadata["rescue_success"]),
            "eligible_for_reference": bool(action_result.metadata["eligible_for_reference"]),
            "reward": float(reward),
        }
        self.last_info = info
        return self._observation(), float(reward), terminated, truncated, info

    def _reward(self, metrics: Dict[str, float]) -> float:
        rcfg = self.reward_cfg
        cost = (
            rcfg.penalty_weight * float(metrics.get("total_penalty", 0.0))
            + rcfg.queue_weight * float(metrics.get("queue_backlog", 0.0))
            + rcfg.virtual_weight
            * (
                float(metrics.get("mean_virtual_aoi_queue", 0.0))
                + float(metrics.get("mean_virtual_unc_queue", 0.0))
            )
            + rcfg.aoi_violation_weight * float(metrics.get("aoi_violation", 0.0))
            + rcfg.uncertainty_violation_weight * float(metrics.get("uncertainty_violation", 0.0))
        )
        return -rcfg.reward_scale * cost

    def _observation(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Environment has no state.")
        s = self.state
        cfg = self.cfg
        x_norm = np.array([
            (s.x[0] - cfg.corridor_x_min) / max(cfg.width, 1e-12),
            (s.x[1] - cfg.corridor_y_min) / max(cfg.height, 1e-12),
        ])
        q_norm = s.Q / max(10.0, 5.0 * cfg.arrival_rate)
        a_norm = s.A / max(float(cfg.aoi_max), 1.0)
        u_norm = s.U / max(float(cfg.uncertainty_max), 1.0)
        y_norm = s.Y / 25.0
        z_norm = s.Z / 25.0
        veh_rel = (s.vehicle_pos - s.x) / np.array([max(cfg.width, 1e-12), max(cfg.height, 1e-12)])
        tgt_rel = (s.target_pos - s.x) / np.array([max(cfg.width, 1e-12), max(cfg.height, 1e-12)])
        veh_dist = np.linalg.norm(s.vehicle_pos - s.x, axis=1)
        tgt_dist = np.linalg.norm(s.target_pos - s.x, axis=1)
        summary = np.array([
            self.slot / max(self.horizon, 1),
            np.mean(s.Q) / max(10.0, 5.0 * cfg.arrival_rate),
            np.mean(s.A) / max(float(cfg.aoi_max), 1.0),
            np.mean(s.U) / max(float(cfg.uncertainty_max), 1.0),
            np.mean(s.Y) / 25.0,
            np.mean(s.Z) / 25.0,
            np.min(veh_dist) / max(cfg.width, 1e-12),
            np.min(tgt_dist) / max(cfg.width, 1e-12),
        ])
        obs = np.concatenate([
            x_norm,
            q_norm,
            a_norm,
            u_norm,
            y_norm,
            z_norm,
            veh_rel.reshape(-1),
            tgt_rel.reshape(-1),
            summary,
        ]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)


def evaluate_policy_on_env(
    env: RLMobilityEnv,
    policy: Any,
    seed: int,
    deterministic: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    obs, _ = env.reset(seed=seed)
    rows: List[Dict[str, Any]] = []
    done = False
    while not done:
        t0 = time.perf_counter()
        if policy == "random":
            action = env.action_space.sample()
        elif callable(policy):
            action = policy(obs)
        else:
            action, _ = policy.predict(obs, deterministic=deterministic)
        inference_ms = 1000.0 * (time.perf_counter() - t0)
        next_obs, reward, terminated, truncated, info = env.step(action)
        rows.append({"reward": float(reward), "inference_runtime_ms": float(inference_ms), **info})
        obs = next_obs
        done = bool(terminated or truncated)
    summary = {}
    if rows:
        keys = rows[0].keys()
        for key in keys:
            values = [row[key] for row in rows if isinstance(row.get(key), (int, float, np.floating, np.integer, bool))]
            if values:
                summary[key] = float(np.mean(values))
    summary["seed"] = int(seed)
    summary["horizon"] = int(env.horizon)
    return rows, summary
