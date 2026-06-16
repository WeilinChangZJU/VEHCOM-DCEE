from __future__ import annotations

import numpy as np

from src_r1.rl_env_r1 import RLMobilityEnv
from src_r1.uav_isac_core_r1 import ScenarioConfig


def test_rl_env_mobility_exact_step_smoke():
    cfg = ScenarioConfig(
        num_vehicles=3,
        num_targets=2,
        benchmark_num_directions=8,
        inner_solver_maxiter=40,
        sun_horizon_steps=1,
        sun_outer_rounds=1,
        sun_num_candidates=4,
    )
    env = RLMobilityEnv(cfg=cfg, horizon=2, seed=7)
    obs, _ = env.reset(seed=7)
    assert obs.shape == env.observation_space.shape

    next_obs, reward, terminated, truncated, info = env.step(np.array([0.25, -0.5], dtype=np.float32))
    assert next_obs.shape == env.observation_space.shape
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    assert info["solver_success"]
    assert info["eligible_for_reference"]
    assert info["solver_residual"] <= 1e-6
    assert info["travel_distance"] <= cfg.delta + 1e-8
