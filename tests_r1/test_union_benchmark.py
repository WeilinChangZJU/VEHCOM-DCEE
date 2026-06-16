from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_r1.benchmark_r1 import (
    VALUE_TOL,
    build_common_candidate_family,
    build_union_candidate_family,
    collect_method_candidate_family,
    compute_union_gap,
    evaluate_finite_family_with_exo,
    repeated_exact_solve_at_executed_point,
)
from src_r1.uav_isac_core_r1 import ScenarioConfig, UavIsacSimulator


METHODS = ["proposed_full", "regularized_execution", "anchored_mobility", "static_uav", "full_candidate_exact", "yang_go_lyapunov"]


def make_sim() -> UavIsacSimulator:
    return UavIsacSimulator(
        ScenarioConfig(
            num_vehicles=3,
            num_targets=2,
            benchmark_num_directions=6,
            shortlist_size=5,
            shortlist_num_local_dirs=4,
            inner_solver_maxiter=25,
        )
    )


def contains(points: np.ndarray, point: np.ndarray) -> bool:
    return any(np.allclose(candidate, point, rtol=0.0, atol=1e-7) for candidate in points)


def test_union_family_includes_common_and_executed_points() -> None:
    sim = make_sim()
    state = sim.reset(seed=2028)
    exo = sim.build_exogenous(state)
    common = build_common_candidate_family(sim, state)
    action_results = {}
    families = {}
    for method in METHODS:
        action_result, family = collect_method_candidate_family(sim, state, method, exo=exo)
        action_results[method] = action_result
        families[method] = family
    union = build_union_candidate_family(common, families)

    for point in common.points:
        if not contains(union.points, point):
            raise AssertionError("Union family is missing a common candidate")
    for method, action_result in action_results.items():
        if not contains(union.points, action_result.post_motion_point):
            raise AssertionError(f"Union family is missing executed point for {method}")


def test_union_gap_is_nonnegative_by_construction() -> None:
    sim = make_sim()
    state = sim.reset(seed=2029)
    exo = sim.build_exogenous(state)
    common = build_common_candidate_family(sim, state)
    action_results = {}
    families = {}
    for method in METHODS:
        action_result, family = collect_method_candidate_family(sim, state, method, exo=exo)
        action_results[method] = action_result
        families[method] = family
    union = build_union_candidate_family(common, families)
    eval_result = evaluate_finite_family_with_exo(sim, state, exo, union)
    for method in METHODS:
        gap = compute_union_gap(action_results[method], eval_result)
        if gap["finite_gap"] < -VALUE_TOL:
            raise AssertionError(f"Negative finite gap for {method}: {gap['finite_gap']}")
        if gap["positive_signed_violation"] > VALUE_TOL:
            raise AssertionError(f"Positive signed residual for {method}: {gap['positive_signed_violation']}")
        if not gap["executed_point_found_in_union"]:
            raise AssertionError(f"Executed point not found for {method}")
        if gap["executed_point_min_distance_to_union"] > 1e-7:
            raise AssertionError(f"Executed point distance too large for {method}")


def test_non_comparable_controller_values_do_not_warn() -> None:
    sim = make_sim()
    state = sim.reset(seed=2031)
    exo = sim.build_exogenous(state)
    common = build_common_candidate_family(sim, state)
    action_results = {}
    families = {}
    for method in ["regularized_execution", "anchored_mobility", "static_uav"]:
        action_result, family = collect_method_candidate_family(sim, state, method, exo=exo)
        action_results[method] = action_result
        families[method] = family
    union = build_union_candidate_family(common, families)
    eval_result = evaluate_finite_family_with_exo(sim, state, exo, union)

    reg_gap = compute_union_gap(action_results["regularized_execution"], eval_result)
    if reg_gap["controller_value_type"] != "regularized_design":
        raise AssertionError("regularized_execution value type was not classified as regularized_design")
    if reg_gap["exec_value_comparable"]:
        raise AssertionError("regularized_execution should not be controller-cache comparable")
    if reg_gap["controller_cache_residual_status"] != "not_comparable":
        raise AssertionError("regularized_execution should not trigger a controller-cache warning")

    anchored_gap = compute_union_gap(action_results["anchored_mobility"], eval_result)
    if anchored_gap["controller_value_type"] != "anchored_score":
        raise AssertionError("anchored_mobility value type was not classified as anchored_score")
    if anchored_gap["exec_value_comparable"]:
        raise AssertionError("anchored_mobility native score should not be controller-cache comparable")
    if anchored_gap["controller_cache_residual_status"] != "not_comparable":
        raise AssertionError("anchored_mobility native score should not trigger a controller-cache warning")

    static_gap = compute_union_gap(action_results["static_uav"], eval_result)
    if static_gap["controller_value_type"] != "unregularized_exact":
        raise AssertionError("static_uav should be classified as unregularized_exact")
    if not static_gap["exec_value_comparable"]:
        raise AssertionError("static_uav should be controller-cache comparable")


def test_repeated_exact_solve_fields_are_populated() -> None:
    sim = make_sim()
    state = sim.reset(seed=2032)
    exo = sim.build_exogenous(state)
    action_result, family = collect_method_candidate_family(sim, state, "static_uav", exo=exo)
    union = build_union_candidate_family(build_common_candidate_family(sim, state), [family])
    eval_result = evaluate_finite_family_with_exo(sim, state, exo, union)
    gap = compute_union_gap(action_result, eval_result)
    repeated = repeated_exact_solve_at_executed_point(
        sim,
        state,
        exo,
        action_result,
        executed_value_cache=float(gap["executed_value_cache"]),
    )
    for key in ["executed_repeated_value", "executed_repeated_residual", "executed_repeated_success"]:
        if key not in repeated:
            raise AssertionError(f"Missing repeated exact solve field: {key}")
    if not repeated["executed_repeated_success"]:
        raise AssertionError("Repeated exact solve did not succeed")


if __name__ == "__main__":
    test_union_family_includes_common_and_executed_points()
    test_union_gap_is_nonnegative_by_construction()
    test_non_comparable_controller_values_do_not_warn()
    test_repeated_exact_solve_fields_are_populated()
    print("union benchmark tests passed")
