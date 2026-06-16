from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _copy_array_or_none(value: Any) -> Any:
    if value is None:
        return None
    return np.asarray(value).copy()


def state_signature(state: Any) -> Dict[str, Any]:
    return {
        "x": np.asarray(state.x).copy(),
        "Q": np.asarray(state.Q).copy(),
        "A": np.asarray(state.A).copy(),
        "U": np.asarray(state.U).copy(),
        "Y": np.asarray(state.Y).copy(),
        "Z": np.asarray(state.Z).copy(),
        "vehicle_pos": np.asarray(state.vehicle_pos).copy(),
        "vehicle_vel": np.asarray(state.vehicle_vel).copy(),
        "target_pos": np.asarray(state.target_pos).copy(),
        "target_vel": np.asarray(state.target_vel).copy(),
        "t": int(state.t),
        "last_exact_action": _copy_array_or_none(state.last_exact_action),
        "last_reg_action": _copy_array_or_none(state.last_reg_action),
    }


def assert_same_signature(before: Dict[str, Any], after: Dict[str, Any]) -> None:
    if before.keys() != after.keys():
        raise AssertionError(f"Signature keys differ: {before.keys()} != {after.keys()}")
    for key in before:
        left = before[key]
        right = after[key]
        if left is None or right is None:
            if left is not None or right is not None:
                raise AssertionError(f"State field changed for {key}: {left} != {right}")
            continue
        if isinstance(left, np.ndarray):
            if not isinstance(right, np.ndarray) or left.shape != right.shape or not np.array_equal(left, right):
                raise AssertionError(f"State array changed for {key}")
        elif left != right:
            raise AssertionError(f"State field changed for {key}: {left} != {right}")
