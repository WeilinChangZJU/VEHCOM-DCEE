from __future__ import annotations

from typing import Dict

import numpy as np

from .uav_isac_core_r1 import ActionResult, SimState, UavIsacSimulator


def choose_action_r1(
    simulator: UavIsacSimulator,
    state: SimState,
    exo: Dict[str, np.ndarray],
    method: str,
) -> ActionResult:
    return simulator.choose_action_r1(state, exo, method=method)
