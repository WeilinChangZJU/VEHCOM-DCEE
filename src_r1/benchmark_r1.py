from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .diagnostics_r1 import assert_same_signature, state_signature
from .uav_isac_core_r1 import ActionResult, CandidateFamily, SimState, SolverDiag, UavIsacSimulator


VALUE_TOL = 1e-6
POINT_TOL = 1e-7
FEAS_TOL = 1e-6
RESIDUAL_WARN_TOL = 1e-4
RESIDUAL_FAIL_TOL = 1e-3


@dataclass
class FamilyEvalResult:
    family: CandidateFamily
    rows: List[Dict[str, object]]
    best_index: int
    best_point: np.ndarray
    best_value: float
    best_action: np.ndarray
    eval_runtime_ms: float


def point_key(point: np.ndarray, tol: float = POINT_TOL) -> Tuple[int, ...]:
    arr = np.asarray(point, dtype=float)
    return tuple(np.round(arr / tol).astype(np.int64).tolist())


def _prefix_label(prefix: str, label: str) -> str:
    label = str(label)
    if label.startswith(prefix + "/"):
        return label
    return f"{prefix}/{label}"


def _family_contains_point(family: CandidateFamily, point: np.ndarray, tol: float = POINT_TOL) -> bool:
    target = np.asarray(point, dtype=float)
    return any(np.allclose(candidate, target, rtol=0.0, atol=tol) for candidate in family.points)


def dedupe_candidates(
    points: Iterable[Sequence[float]],
    labels: Optional[Iterable[str]] = None,
    *,
    sim: Optional[UavIsacSimulator] = None,
    state: Optional[SimState] = None,
    tol: float = POINT_TOL,
) -> Tuple[np.ndarray, List[str]]:
    point_list = [np.asarray(point, dtype=float).reshape(-1) for point in points]
    label_list = list(labels) if labels is not None else [f"candidate_{i}" for i in range(len(point_list))]
    if len(label_list) != len(point_list):
        raise ValueError("labels length must match points length")

    unique: List[np.ndarray] = []
    unique_labels: List[str] = []
    key_to_index: Dict[Tuple[int, ...], int] = {}

    for point, label in zip(point_list, label_list):
        if point.size != 2 or not np.all(np.isfinite(point)):
            continue
        candidate = point.astype(float)
        if sim is not None and state is not None:
            candidate = sim.project_to_reachable(state.x, candidate)
        key = point_key(candidate, tol=tol)
        if key in key_to_index:
            idx = key_to_index[key]
            existing = set(part for part in unique_labels[idx].split("|") if part)
            if str(label) not in existing:
                unique_labels[idx] = unique_labels[idx] + "|" + str(label)
            continue
        key_to_index[key] = len(unique)
        unique.append(candidate.copy())
        unique_labels.append(str(label))

    if not unique:
        return np.zeros((0, 2), dtype=float), []
    return np.vstack(unique).astype(float), unique_labels


def build_common_candidate_family(sim: UavIsacSimulator, state: SimState) -> CandidateFamily:
    raw_points = sim.reachable_candidates(state.x)
    labels = ["common/stay" if i == 0 else f"common/reachable_{i}" for i in range(len(raw_points))]
    points, labels = dedupe_candidates(raw_points, labels, sim=sim, state=state)
    return CandidateFamily(
        method="FC-Common diagnostic",
        points=points,
        executed_point=state.x.copy(),
        labels=labels,
    )


def build_dense_candidate_family(
    sim: UavIsacSimulator,
    state: SimState,
    n_radial: int = 3,
    n_angle: int = 32,
) -> CandidateFamily:
    points: List[np.ndarray] = [state.x.copy()]
    labels: List[str] = ["dense/stay"]
    n_radial = max(1, int(n_radial))
    n_angle = max(4, int(n_angle))
    for ridx, radius_scale in enumerate(np.linspace(1.0 / n_radial, 1.0, n_radial), start=1):
        radius = sim.cfg.delta * float(radius_scale)
        for aidx, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, n_angle, endpoint=False)):
            point = sim.project_to_reachable(
                state.x,
                state.x + radius * np.array([np.cos(angle), np.sin(angle)], dtype=float),
            )
            points.append(point)
            labels.append(f"dense/r{ridx}_a{aidx}")
    points_arr, labels = dedupe_candidates(points, labels, sim=sim, state=state)
    return CandidateFamily(
        method="Dense diagnostic",
        points=points_arr,
        executed_point=state.x.copy(),
        labels=labels,
    )


def _ensure_prefixed_family(method: str, family: CandidateFamily) -> CandidateFamily:
    labels = [_prefix_label(method, label) for label in family.labels]
    return CandidateFamily(
        method=method,
        points=np.asarray(family.points, dtype=float).copy(),
        executed_point=np.asarray(family.executed_point, dtype=float).copy(),
        labels=labels,
    )


def collect_method_candidate_family(
    sim: UavIsacSimulator,
    state: SimState,
    method: str,
    exo: Optional[Dict[str, np.ndarray]] = None,
    rng: object = None,
) -> Tuple[ActionResult, CandidateFamily]:
    del rng
    if exo is None:
        raise ValueError("collect_method_candidate_family requires the slot exogenous primitives")
    before = state_signature(state)
    action_result = sim.choose_action_r1(state, exo, method=method)
    after = state_signature(state)
    assert_same_signature(before, after)

    if method in {"full_candidate_exact", "FC-Union", "fc_union", "finite_candidate_reference"}:
        common = build_common_candidate_family(sim, state)
        points = list(common.points) + [action_result.post_motion_point.copy()]
        labels = list(common.labels) + [f"{method}/executed"]
        points_arr, labels = dedupe_candidates(points, labels, sim=sim, state=state)
        family = CandidateFamily(method=method, points=points_arr, executed_point=action_result.post_motion_point.copy(), labels=labels)
    else:
        family = _ensure_prefixed_family(method, action_result.candidate_family)
        if not _family_contains_point(family, action_result.post_motion_point):
            points = list(family.points) + [action_result.post_motion_point.copy()]
            labels = list(family.labels) + [f"{method}/executed"]
            points_arr, labels = dedupe_candidates(points, labels, sim=sim, state=state)
            family = CandidateFamily(method=method, points=points_arr, executed_point=action_result.post_motion_point.copy(), labels=labels)

    if not _family_contains_point(family, action_result.post_motion_point):
        raise AssertionError(f"Executed point missing from candidate family for {method}")
    return action_result, family


def build_union_candidate_family(
    common_family: CandidateFamily,
    method_families: Dict[str, CandidateFamily] | Sequence[CandidateFamily],
    dense_family: Optional[CandidateFamily] = None,
) -> CandidateFamily:
    if isinstance(method_families, dict):
        families = list(method_families.values())
    else:
        families = list(method_families)

    points: List[np.ndarray] = []
    labels: List[str] = []
    for point, label in zip(common_family.points, common_family.labels):
        points.append(np.asarray(point, dtype=float).copy())
        labels.append(_prefix_label("common", label.replace("common/", "")))
    for family in families:
        for point, label in zip(family.points, family.labels):
            points.append(np.asarray(point, dtype=float).copy())
            labels.append(str(label))
        points.append(np.asarray(family.executed_point, dtype=float).copy())
        labels.append(f"{family.method}/executed")
    if dense_family is not None:
        for point, label in zip(dense_family.points, dense_family.labels):
            points.append(np.asarray(point, dtype=float).copy())
            labels.append(_prefix_label("dense", label.replace("dense/", "")))

    points_arr, labels = dedupe_candidates(points, labels)
    executed = points_arr[0].copy() if len(points_arr) else np.zeros(2, dtype=float)
    return CandidateFamily(
        method="Finite-Candidate Reference (FC-Union)",
        points=points_arr,
        executed_point=executed,
        labels=labels,
    )


def evaluate_finite_family(
    sim: UavIsacSimulator,
    state: SimState,
    candidate_family: CandidateFamily,
    warm_start_policy: str = "common",
    *,
    exo: Optional[Dict[str, np.ndarray]] = None,
) -> FamilyEvalResult:
    if exo is None:
        raise ValueError("evaluate_finite_family requires exo=slot_exogenous_primitives")
    before = state_signature(state)
    started = time.perf_counter()
    rows: List[Dict[str, object]] = []
    best_index = -1
    best_value = -np.inf
    best_point = np.zeros(2, dtype=float)
    best_action = sim.initial_action_guess()
    cache: Dict[Tuple[int, ...], Dict[str, object]] = {}

    if warm_start_policy not in {"common", "initial", "none", "state"}:
        raise ValueError(f"Unknown warm_start_policy: {warm_start_policy}")
    base_warm_start = None if warm_start_policy in {"initial", "none"} else state.last_exact_action

    for idx, (point, label) in enumerate(zip(candidate_family.points, candidate_family.labels)):
        key = point_key(point)
        if key not in cache:
            eval_result = sim.exact_value_with_rescue_diag(
                np.asarray(point, dtype=float),
                state,
                exo=exo,
                warm_start=base_warm_start,
                use_true_weights=True,
                feas_tol=FEAS_TOL,
            )
            cache[key] = eval_result
        else:
            eval_result = cache[key]

        value = float(eval_result["value"])
        action = np.asarray(eval_result["action"], dtype=float)
        diag: SolverDiag = eval_result["solver_diag"]
        row = {
            "candidate_id": idx,
            "source_label": str(label),
            "x": float(point[0]),
            "y": float(point[1]),
            "value": value,
            "success": bool(diag.success),
            "nit": int(diag.nit),
            "objective": float(diag.objective),
            "exact_objective": float(diag.exact_objective),
            "residual": float(diag.residual),
            "runtime_ms": float(diag.runtime_ms),
            "warm_start_type": str(diag.warm_start_type),
            "message": str(diag.message),
            "primary_success": bool(eval_result.get("primary_success", diag.success)),
            "final_success": bool(eval_result.get("final_success", diag.success)),
            "rescue_attempts": int(eval_result.get("rescue_attempts", 0)),
            "rescue_success": bool(eval_result.get("rescue_success", False)),
            "selected_start_type": str(eval_result.get("selected_start_type", diag.warm_start_type)),
            "feasibility_residual": float(eval_result.get("feasibility_residual", diag.residual)),
            "eligible_for_reference": bool(eval_result.get("eligible_for_reference", diag.success)),
            "primary_value": float(eval_result.get("primary_value", value)),
            "final_value": float(eval_result.get("final_value", value)),
            "value_changed_by_rescue": bool(eval_result.get("value_changed_by_rescue", False)),
            "primary_message": str(eval_result.get("primary_message", diag.message)),
            "final_message": str(eval_result.get("final_message", diag.message)),
            "_action": action,
        }
        rows.append(row)
        if bool(row["eligible_for_reference"]) and value > best_value:
            best_index = idx
            best_value = value
            best_point = np.asarray(point, dtype=float).copy()
            best_action = action.copy()

    after = state_signature(state)
    assert_same_signature(before, after)
    if best_index < 0:
        raise RuntimeError("No eligible FC-Union candidate after deterministic rescue evaluation")
    eval_runtime_ms = 1000.0 * (time.perf_counter() - started)
    return FamilyEvalResult(
        family=candidate_family,
        rows=rows,
        best_index=best_index,
        best_point=best_point,
        best_value=float(best_value),
        best_action=best_action,
        eval_runtime_ms=float(eval_runtime_ms),
    )


def evaluate_finite_family_with_exo(
    sim: UavIsacSimulator,
    state: SimState,
    exo: Dict[str, np.ndarray],
    candidate_family: CandidateFamily,
    warm_start_policy: str = "common",
) -> FamilyEvalResult:
    exo_copy = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in exo.items()}
    return evaluate_finite_family(sim, state, candidate_family, warm_start_policy=warm_start_policy, exo=exo_copy)


def find_point_in_family_eval(
    family_eval: FamilyEvalResult,
    point: np.ndarray,
    tol: float = POINT_TOL,
) -> Dict[str, object]:
    target = np.asarray(point, dtype=float)
    best_distance = np.inf
    best_row: Optional[Dict[str, object]] = None
    for row in family_eval.rows:
        candidate = np.array([float(row["x"]), float(row["y"])], dtype=float)
        distance = float(np.linalg.norm(candidate - target))
        if distance < best_distance:
            best_distance = distance
            best_row = row
        if np.allclose(candidate, target, rtol=0.0, atol=tol):
            return {
                "found": True,
                "min_distance": distance,
                "source_label": str(row["source_label"]),
                "row": row,
            }
    return {
        "found": False,
        "min_distance": float(best_distance),
        "source_label": "" if best_row is None else str(best_row["source_label"]),
        "row": best_row,
    }


def _controller_cache_residual_status(value_type: str, residual: float) -> Tuple[bool, str]:
    comparable = value_type == "unregularized_exact"
    if not comparable:
        return False, "not_comparable"
    abs_residual = abs(float(residual))
    if abs_residual > RESIDUAL_FAIL_TOL:
        return True, "fail"
    if abs_residual > RESIDUAL_WARN_TOL:
        return True, "warn"
    return True, "ok"


def repeated_exact_solve_at_executed_point(
    sim: UavIsacSimulator,
    state: SimState,
    exo: Dict[str, np.ndarray],
    action_result: ActionResult,
    executed_value_cache: float,
    warm_start_policy: str = "common",
) -> Dict[str, object]:
    before = state_signature(state)
    warm_start = None if warm_start_policy in {"initial", "none"} else state.last_exact_action
    repeated = sim.exact_value_with_rescue_diag(
        action_result.post_motion_point,
        state,
        exo,
        warm_start=warm_start,
        use_true_weights=True,
        feas_tol=FEAS_TOL,
    )
    after = state_signature(state)
    assert_same_signature(before, after)
    value = float(repeated["value"])
    diag: SolverDiag = repeated["solver_diag"]
    residual = value - float(executed_value_cache)
    return {
        "executed_repeated_value": value,
        "executed_repeated_residual": float(residual),
        "executed_repeated_success": bool(diag.success),
        "executed_repeated_nit": int(diag.nit),
        "executed_repeated_runtime_ms": float(diag.runtime_ms),
        "executed_repeated_message": str(diag.message),
        "executed_repeated_final_success": bool(repeated.get("final_success", diag.success)),
        "executed_repeated_rescue_attempts": int(repeated.get("rescue_attempts", 0)),
        "executed_repeated_selected_start_type": str(repeated.get("selected_start_type", diag.warm_start_type)),
    }


def compute_union_gap(action_result: ActionResult, family_eval: FamilyEvalResult) -> Dict[str, object]:
    executed_lookup = find_point_in_family_eval(family_eval, action_result.post_motion_point)
    executed_row = executed_lookup["row"]
    if executed_row is None or not executed_lookup["found"]:
        raise AssertionError(f"Executed point for {action_result.method} is missing from the evaluated union family")
    executed_value_cache = float(executed_row["value"])
    union_best_value = float(family_eval.best_value)
    signed_residual = executed_value_cache - union_best_value
    finite_gap = max(0.0, union_best_value - executed_value_cache)
    positive_signed_violation = max(0.0, signed_residual)
    controller_value = float(action_result.metadata.get("controller_value", action_result.value))
    controller_value_type = str(action_result.metadata.get("controller_value_type", "unknown"))
    controller_cache_residual = controller_value - executed_value_cache
    exec_value_comparable, controller_cache_status = _controller_cache_residual_status(
        controller_value_type,
        controller_cache_residual,
    )
    best_row = family_eval.rows[family_eval.best_index]
    union_best_source_label = str(best_row["source_label"])
    union_best_source_parts = [
        part.split("/", 1)[0]
        for part in union_best_source_label.split("|")
        if "/" in part
    ]
    union_best_source_method = "|".join(sorted(set(union_best_source_parts))) if union_best_source_parts else "unknown"
    union_best_is_executed_point = "executed" in union_best_source_label
    union_best_distance_to_executed = float(np.linalg.norm(family_eval.best_point - action_result.post_motion_point))
    union_best_final_success = bool(best_row.get("final_success", best_row.get("success", False)))
    union_best_eligible_for_reference = bool(best_row.get("eligible_for_reference", union_best_final_success))
    return {
        "executed_value_cache": executed_value_cache,
        "controller_value": controller_value,
        "controller_value_type": controller_value_type,
        "union_best_value": union_best_value,
        "union_best_source_label": union_best_source_label,
        "union_best_source_method": union_best_source_method,
        "union_best_is_executed_point": bool(union_best_is_executed_point),
        "union_best_distance_to_executed": union_best_distance_to_executed,
        "union_best_final_success": union_best_final_success,
        "union_best_eligible_for_reference": union_best_eligible_for_reference,
        "union_best_rescue_attempts": int(best_row.get("rescue_attempts", 0)),
        "union_best_selected_start_type": str(best_row.get("selected_start_type", best_row.get("warm_start_type", "unknown"))),
        "failed_candidate_selected_as_union_best": bool(not union_best_eligible_for_reference),
        "finite_gap": finite_gap,
        "signed_residual": signed_residual,
        "positive_signed_violation": positive_signed_violation,
        "controller_cache_residual": float(controller_cache_residual),
        "controller_cache_residual_status": controller_cache_status,
        "exec_value_comparable": bool(exec_value_comparable),
        "executed_point_found_in_union": bool(executed_lookup["found"]),
        "executed_point_min_distance_to_union": float(executed_lookup["min_distance"]),
        "executed_point_source_label": str(executed_lookup["source_label"]),
    }


def solver_summary_from_eval(family_eval: FamilyEvalResult) -> Dict[str, float]:
    rows = family_eval.rows
    success = np.array([bool(row["success"]) for row in rows], dtype=bool)
    final_success = np.array([bool(row.get("final_success", row["success"])) for row in rows], dtype=bool)
    rescue_attempts = np.array([float(row.get("rescue_attempts", 0)) for row in rows], dtype=float)
    nit = np.array([float(row["nit"]) for row in rows], dtype=float)
    return {
        "solver_success_rate": float(np.mean(final_success.astype(float))) if rows else np.nan,
        "solver_fail_count": int(np.sum(~final_success)) if rows else 0,
        "final_solver_success_rate": float(np.mean(final_success.astype(float))) if rows else np.nan,
        "primary_solver_success_rate": float(np.mean(success.astype(float))) if rows else np.nan,
        "rescue_rate": float(np.mean((rescue_attempts > 0).astype(float))) if rows else np.nan,
        "mean_nit": float(np.mean(nit)) if nit.size else np.nan,
        "p95_nit": float(np.percentile(nit, 95)) if nit.size else np.nan,
    }


def summarize_benchmark_slots(slot_records: List[Dict[str, object]]) -> pd.DataFrame:
    if not slot_records:
        return pd.DataFrame()
    df = pd.DataFrame(slot_records)
    rows: List[Dict[str, object]] = []
    for (profile, method), group in df.groupby(["profile", "method"], sort=False):
        rows.append({
            "profile": profile,
            "method": method,
            "mean_finite_gap": float(group["finite_gap"].mean()),
            "std_finite_gap": float(group["finite_gap"].std(ddof=0)),
            "p95_finite_gap": float(group["finite_gap"].quantile(0.95)),
            "mean_signed_residual": float(group["signed_residual"].mean()),
            "max_positive_signed_violation": float(group["positive_signed_violation"].max()),
            "mean_controller_cache_residual": float(group["controller_cache_residual"].mean()),
            "comparable_controller_cache_warn_count": int(group["controller_cache_residual_status"].isin(["warn", "fail"]).sum()),
            "mean_executed_repeated_residual": float(group["executed_repeated_residual"].mean()),
            "max_abs_executed_repeated_residual": float(group["executed_repeated_residual"].abs().max()),
            "executed_point_found_rate": float(group["executed_point_found_in_union"].astype(float).mean()),
            "solver_success_rate": float(group["solver_success_rate"].mean()),
            "final_solver_success_rate": float(group["final_solver_success_rate"].mean()) if "final_solver_success_rate" in group else float(group["solver_success_rate"].mean()),
            "primary_solver_success_rate": float(group["primary_solver_success_rate"].mean()) if "primary_solver_success_rate" in group else float(group["solver_success_rate"].mean()),
            "rescue_rate": float(group["rescue_rate"].mean()) if "rescue_rate" in group else 0.0,
            "mean_nit": float(group["mean_nit"].mean()),
            "p95_nit": float(group["p95_nit"].mean()),
            "mean_union_candidate_count": float(group["union_candidate_count"].mean()),
            "mean_eval_runtime_ms": float(group["eval_runtime_ms"].mean()),
        })
    return pd.DataFrame(rows)


def summarize_consistency_slots(slot_records: List[Dict[str, object]]) -> pd.DataFrame:
    if not slot_records:
        return pd.DataFrame()
    df = pd.DataFrame(slot_records)
    rows: List[Dict[str, object]] = []
    for (profile, method), group in df.groupby(["profile", "method"], sort=False):
        comparable = group[group["exec_value_comparable"].astype(bool)].copy()
        rows.append({
            "profile": profile,
            "method": method,
            "rows": int(len(group)),
            "controller_value_type": "|".join(sorted(str(x) for x in group["controller_value_type"].dropna().unique())),
            "controller_cache_status_ok": int((group["controller_cache_residual_status"] == "ok").sum()),
            "controller_cache_status_not_comparable": int((group["controller_cache_residual_status"] == "not_comparable").sum()),
            "controller_cache_status_warn": int((group["controller_cache_residual_status"] == "warn").sum()),
            "controller_cache_status_fail": int((group["controller_cache_residual_status"] == "fail").sum()),
            "mean_abs_controller_cache_residual_comparable": (
                float(comparable["controller_cache_residual"].abs().mean()) if not comparable.empty else np.nan
            ),
            "max_abs_controller_cache_residual_comparable": (
                float(comparable["controller_cache_residual"].abs().max()) if not comparable.empty else np.nan
            ),
            "mean_executed_repeated_residual": float(group["executed_repeated_residual"].mean()),
            "max_abs_executed_repeated_residual": float(group["executed_repeated_residual"].abs().max()),
            "executed_repeated_success_rate": float(group["executed_repeated_success"].astype(float).mean()),
            "executed_repeated_final_success_rate": (
                float(group["executed_repeated_final_success"].astype(float).mean())
                if "executed_repeated_final_success" in group else float(group["executed_repeated_success"].astype(float).mean())
            ),
            "executed_point_found_rate": float(group["executed_point_found_in_union"].astype(float).mean()),
            "max_executed_point_distance": float(group["executed_point_min_distance_to_union"].max()),
        })
    return pd.DataFrame(rows)


def candidate_rows_for_csv(
    family_eval: FamilyEvalResult,
    *,
    slot: int,
    seed: int,
    profile: str,
    method: str,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for row in family_eval.rows:
        clean = {key: value for key, value in row.items() if key != "_action"}
        out.append({
            "slot": int(slot),
            "seed": int(seed),
            "profile": profile,
            "method": method,
            **clean,
        })
    return out


def coverage_rows_for_families(
    method_families: Dict[str, CandidateFamily],
    action_results: Dict[str, ActionResult],
    *,
    slot: int,
    seed: int,
    profile: str,
    target_method: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method, family in method_families.items():
        labels = list(family.labels)
        has_shortlist = any("grad" in label or "shortlist" in label or "reachable" in label for label in labels)
        has_full_family = method == "full_candidate_exact" or any("common/" in label for label in labels)
        has_executed = _family_contains_point(family, action_results[method].post_motion_point)
        notes = "full common finite family" if has_full_family else ("shortlist/search family" if has_shortlist else "executed point only")
        rows.append({
            "slot": int(slot),
            "seed": int(seed),
            "profile": profile,
            "target_method": target_method,
            "method": method,
            "candidate_count": int(len(family.points)),
            "has_executed_point": bool(has_executed),
            "has_full_family": bool(has_full_family),
            "has_shortlist": bool(has_shortlist),
            "notes": notes,
        })
    return rows
