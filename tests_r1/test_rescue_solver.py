from __future__ import annotations

import numpy as np

from src_r1.benchmark_r1 import (
    build_common_candidate_family,
    build_union_candidate_family,
    evaluate_finite_family_with_exo,
)
from src_r1.uav_isac_core_r1 import ScenarioConfig, UavIsacSimulator


def test_rescue_fields_and_union_best_eligibility() -> None:
    cfg = ScenarioConfig(num_vehicles=3, num_targets=2, benchmark_num_directions=4, inner_solver_maxiter=20)
    sim = UavIsacSimulator(cfg)
    state = sim.reset(seed=17)
    exo = sim.build_exogenous(state)
    common = build_common_candidate_family(sim, state)
    union = build_union_candidate_family(common, [common], dense_family=None)
    result = evaluate_finite_family_with_exo(sim, state, exo, union)

    assert result.best_index >= 0
    best_row = result.rows[result.best_index]
    assert bool(best_row["eligible_for_reference"])
    assert bool(best_row["final_success"])
    assert np.isfinite(float(best_row["final_value"]))

    required = [
        "primary_success",
        "final_success",
        "rescue_attempts",
        "rescue_success",
        "selected_start_type",
        "feasibility_residual",
        "eligible_for_reference",
        "primary_value",
        "final_value",
        "value_changed_by_rescue",
        "primary_message",
        "final_message",
    ]
    for idx, row in enumerate(result.rows):
        for key in required:
            assert key in row
        if not bool(row["eligible_for_reference"]):
            assert idx != result.best_index
