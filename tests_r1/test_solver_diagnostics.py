from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src_r1.benchmark_r1 import (
    build_common_candidate_family,
    collect_method_candidate_family,
    compute_union_gap,
    evaluate_finite_family_with_exo,
    repeated_exact_solve_at_executed_point,
    solver_summary_from_eval,
)
from src_r1.uav_isac_core_r1 import ScenarioConfig, UavIsacSimulator


def make_sim() -> UavIsacSimulator:
    return UavIsacSimulator(
        ScenarioConfig(
            num_vehicles=3,
            num_targets=2,
            benchmark_num_directions=4,
            inner_solver_maxiter=25,
        )
    )


def test_solver_diagnostics_are_populated() -> None:
    sim = make_sim()
    state = sim.reset(seed=2030)
    exo = sim.build_exogenous(state)
    family = build_common_candidate_family(sim, state)
    eval_result = evaluate_finite_family_with_exo(sim, state, exo, family)
    if not eval_result.rows:
        raise AssertionError("No candidate evaluation rows were produced")
    for row in eval_result.rows:
        for key in ["success", "nit", "objective", "exact_objective", "residual", "runtime_ms", "warm_start_type", "message"]:
            if key not in row:
                raise AssertionError(f"Missing solver diagnostic field: {key}")
        if not isinstance(row["success"], (bool, np.bool_)):
            raise AssertionError("success field is not boolean")
        if int(row["nit"]) < 0:
            raise AssertionError("nit field is negative")
        if float(row["runtime_ms"]) < 0.0:
            raise AssertionError("runtime_ms field is negative")
        if not np.isfinite(float(row["residual"])):
            raise AssertionError("residual field is not finite")

    summary = solver_summary_from_eval(eval_result)
    for key in ["solver_success_rate", "solver_fail_count", "mean_nit", "p95_nit"]:
        if key not in summary:
            raise AssertionError(f"Missing solver summary field: {key}")


def test_repeated_exact_solve_diagnostics_are_populated() -> None:
    sim = make_sim()
    state = sim.reset(seed=2033)
    exo = sim.build_exogenous(state)
    action_result, family = collect_method_candidate_family(sim, state, "static_uav", exo=exo)
    eval_result = evaluate_finite_family_with_exo(sim, state, exo, family)
    gap = compute_union_gap(action_result, eval_result)
    repeated = repeated_exact_solve_at_executed_point(
        sim,
        state,
        exo,
        action_result,
        executed_value_cache=float(gap["executed_value_cache"]),
    )
    required = [
        "executed_repeated_value",
        "executed_repeated_residual",
        "executed_repeated_success",
        "executed_repeated_nit",
        "executed_repeated_runtime_ms",
        "executed_repeated_message",
    ]
    for key in required:
        if key not in repeated:
            raise AssertionError(f"Missing repeated exact solve field: {key}")
    if not isinstance(repeated["executed_repeated_success"], (bool, np.bool_)):
        raise AssertionError("executed_repeated_success is not boolean")
    if not np.isfinite(float(repeated["executed_repeated_residual"])):
        raise AssertionError("executed_repeated_residual is not finite")


if __name__ == "__main__":
    test_solver_diagnostics_are_populated()
    test_repeated_exact_solve_diagnostics_are_populated()
    print("solver diagnostics tests passed")
