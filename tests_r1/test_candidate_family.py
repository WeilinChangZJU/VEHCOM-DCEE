from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_r1.benchmark_r1 import collect_method_candidate_family
from src_r1.uav_isac_core_r1 import ScenarioConfig, UavIsacSimulator


METHODS = [
    "proposed_full",
    "regularized_execution",
    "anchored_mobility",
    "static_uav",
    "full_candidate_exact",
    "yang_go_lyapunov",
]


def make_sim() -> UavIsacSimulator:
    return UavIsacSimulator(
        ScenarioConfig(
            num_vehicles=3,
            num_targets=2,
            benchmark_num_directions=6,
            shortlist_size=5,
            shortlist_num_local_dirs=4,
            inner_solver_maxiter=25,
            sun_horizon_steps=2,
            sun_outer_rounds=1,
            sun_num_candidates=5,
        )
    )


def contains(points: np.ndarray, point: np.ndarray) -> bool:
    return any(np.allclose(candidate, point, rtol=0.0, atol=1e-7) for candidate in points)


def test_candidate_families_include_executed_points() -> None:
    sim = make_sim()
    state = sim.reset(seed=2026)
    exo = sim.build_exogenous(state)
    for method in METHODS:
        action_result, family = collect_method_candidate_family(sim, state, method, exo=exo)
        if not contains(family.points, action_result.post_motion_point):
            raise AssertionError(f"{method} family does not include executed point")
        if len(family.labels) != len(family.points):
            raise AssertionError(f"{method} labels do not match candidate points")


def test_dcee_and_common_family_labels_are_exposed() -> None:
    sim = make_sim()
    state = sim.reset(seed=2027)
    exo = sim.build_exogenous(state)
    _, dcee_family = collect_method_candidate_family(sim, state, "proposed_full", exo=exo)
    if not any("proposed_full/grad" in label for label in dcee_family.labels):
        raise AssertionError("DCEE/proposed_full family does not expose gradient shortlist labels")

    _, common_family = collect_method_candidate_family(sim, state, "full_candidate_exact", exo=exo)
    if not any(label.startswith("common/") for label in common_family.labels):
        raise AssertionError("full_candidate_exact family does not expose common finite candidates")


if __name__ == "__main__":
    test_candidate_families_include_executed_points()
    test_dcee_and_common_family_labels_are_exposed()
    print("candidate family tests passed")
