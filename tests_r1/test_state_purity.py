from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_r1.diagnostics_r1 import assert_same_signature, state_signature
from src_r1.uav_isac_core_r1 import ActionResult, ScenarioConfig, UavIsacSimulator


METHODS = [
    "proposed_full",
    "regularized_execution",
    "anchored_mobility",
    "static_uav",
    "full_candidate_exact",
    "yang_go_lyapunov",
    "yang_ge_equal",
    "sun_receding_ao",
    "aoi_only",
    "greedy_myopic",
    "freshness_priority",
    "myopic_reoptimization",
]


def make_test_simulator() -> UavIsacSimulator:
    cfg = ScenarioConfig(
        num_vehicles=3,
        num_targets=2,
        benchmark_num_directions=6,
        shortlist_size=5,
        shortlist_num_local_dirs=4,
        inner_solver_maxiter=25,
        inner_solver_ftol=1e-4,
        sun_horizon_steps=2,
        sun_outer_rounds=1,
        sun_num_candidates=5,
    )
    return UavIsacSimulator(cfg)


def assert_candidate_family_contains_execution(result: ActionResult) -> None:
    points = result.candidate_family.points
    executed = result.candidate_family.executed_point
    if points.ndim != 2 or points.shape[1] != 2:
        raise AssertionError(f"Invalid candidate-family shape for {result.method}: {points.shape}")
    if not any(np.allclose(point, executed, rtol=0.0, atol=1e-8) for point in points):
        raise AssertionError(f"Executed point missing from candidate family for {result.method}")


def test_choose_action_r1_has_no_state_side_effects() -> None:
    for method in METHODS:
        sim = make_test_simulator()
        state = sim.reset(seed=2026)
        state.last_exact_action = sim.initial_action_guess()
        state.last_reg_action = sim.initial_action_guess()
        exo = sim.build_exogenous(state)
        before = state_signature(state)
        result = sim.choose_action_r1(state, exo, method=method)
        after = state_signature(state)
        assert_same_signature(before, after)
        if not isinstance(result, ActionResult):
            raise AssertionError(f"{method} did not return ActionResult")
        if result.method != method:
            raise AssertionError(f"Unexpected method label: {result.method} != {method}")
        assert_candidate_family_contains_execution(result)


if __name__ == "__main__":
    test_choose_action_r1_has_no_state_side_effects()
    print(f"state purity passed for {len(METHODS)} methods")
