from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts_r1.make_table5_fc_union import build_seed_level_table
from scripts_r1.merge_fc_union_runs import seed_level_summary, table5_seed_level_summary


def test_seed_level_summary_and_table_reference_row() -> None:
    slot_df = pd.DataFrame([
        {"profile": "balanced", "seed": 1000, "method": "proposed_full", "finite_gap": 0.0, "signed_residual": 0.0, "solver_success_rate": 1.0, "mean_nit": 4.0, "union_candidate_count": 10, "positive_signed_violation": 0.0, "executed_repeated_residual": 0.0, "executed_point_found_in_union": True},
        {"profile": "balanced", "seed": 1000, "method": "proposed_full", "finite_gap": 2.0, "signed_residual": -2.0, "solver_success_rate": 1.0, "mean_nit": 6.0, "union_candidate_count": 10, "positive_signed_violation": 0.0, "executed_repeated_residual": 0.0, "executed_point_found_in_union": True},
        {"profile": "balanced", "seed": 1001, "method": "proposed_full", "finite_gap": 4.0, "signed_residual": -4.0, "solver_success_rate": 0.9, "mean_nit": 8.0, "union_candidate_count": 12, "positive_signed_violation": 0.0, "executed_repeated_residual": 0.0, "executed_point_found_in_union": True},
        {"profile": "balanced", "seed": 1001, "method": "proposed_full", "finite_gap": 6.0, "signed_residual": -6.0, "solver_success_rate": 0.9, "mean_nit": 10.0, "union_candidate_count": 12, "positive_signed_violation": 0.0, "executed_repeated_residual": 0.0, "executed_point_found_in_union": True},
    ])
    seed_df = seed_level_summary(slot_df)
    if len(seed_df) != 2:
        raise AssertionError("Expected one row per seed/method")
    table_summary = table5_seed_level_summary(seed_df)
    proposed = table_summary[table_summary["method"] == "proposed_full"].iloc[0]
    if abs(float(proposed["finite_gap_mean"]) - 3.0) > 1e-12:
        raise AssertionError("Seed-level mean finite gap is incorrect")
    if abs(float(proposed["finite_gap_std"]) - 2.0) > 1e-12:
        raise AssertionError("Seed-level std should be computed over seed means")
    table = build_seed_level_table(seed_df)
    if table.iloc[0]["Method"] != "FC-Union":
        raise AssertionError("Table should start with FC-Union reference row")


if __name__ == "__main__":
    test_seed_level_summary_and_table_reference_row()
    print("seed aggregation tests passed")
