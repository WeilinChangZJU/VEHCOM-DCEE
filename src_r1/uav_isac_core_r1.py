from __future__ import annotations

"""Final theory-aligned UAV-ISAC simulator with richer baseline support.

This version extends the V7 simulator with:
1) stronger controller baselines inspired by Yang 2022 and Sun 2021,
2) configurable shortlist-size ablations,
3) stronger diagnostics for long-horizon constraint validation,
4) analytic gradient diagnostics for the regularized post-motion envelope.

The implementation is still intentionally lightweight and fully causal.
It is designed for comparative simulation studies rather than hardware control.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


Array = np.ndarray


@dataclass
class SolverDiag:
    success: bool
    nit: int
    objective: float
    exact_objective: float
    residual: float
    message: str
    runtime_ms: float
    warm_start_type: str


@dataclass
class CandidateFamily:
    method: str
    points: Array
    executed_point: Array
    labels: List[str]


@dataclass
class ActionResult:
    method: str
    velocity: Array
    post_motion_point: Array
    resource_action: Array
    value: float
    candidate_family: CandidateFamily
    solver_diag: SolverDiag
    runtime_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioConfig:
    corridor_x_min: float = 0.0
    corridor_x_max: float = 1200.0
    corridor_y_min: float = -40.0
    corridor_y_max: float = 40.0
    altitude: float = 100.0
    delta_t: float = 1.0
    v_max: float = 15.0
    num_vehicles: int = 6
    num_targets: int = 3
    total_bandwidth: float = 5.0
    noise_psd: float = 1e-3
    total_power: float = 8.0
    sensing_peak_power: float = 4.0
    sensing_time_cost: float = 0.35
    lambda_F: float = 0.3
    lambda_E: float = 0.25
    lambda_S: float = 0.15
    lambda_R: float = 0.3
    c_Y: float = 8.0
    c_Z: float = 8.0
    V: float = 12.0
    rho_r: float = 0.01
    eta_t: float = 2.0
    grad_eps: float = 3.0
    g0_comm: float = 8000.0
    g0_sens: float = 96.0
    pathloss_exp_comm: float = 2.2
    pathloss_exp_sens: float = 2.0
    mu_reduction_scale: float = 12.0
    aoi_max: int = 20
    uncertainty_max: float = 12.0
    aoi_threshold: float = 5.0
    uncertainty_threshold: float = 3.5
    arrival_rate: float = 1.0
    arrival_burst_prob: float = 0.15
    arrival_burst_scale: float = 2.0
    vehicle_speed_min: float = 8.0
    vehicle_speed_max: float = 18.0
    target_speed_min: float = 3.0
    target_speed_max: float = 8.0
    target_process_noise: float = 0.25
    uncertainty_base_alpha: float = 0.96
    uncertainty_base_noise: float = 0.35
    road_center_x: float = 600.0
    road_sigma_x: float = 180.0
    road_sigma_y: float = 18.0
    init_x: Optional[float] = 200.0
    init_y: float = 0.0
    benchmark_num_directions: int = 24
    inner_solver_maxiter: int = 60
    inner_solver_ftol: float = 1e-4
    warm_start_blend: float = 0.4
    shortlist_num_local_dirs: int = 8
    shortlist_radius_scale: float = 0.45
    shortlist_use_half_step: bool = True
    shortlist_size: int = 11
    sensing_success_gain: float = 1.20
    refresh_uncertainty_gain: float = 1.15
    nonrefresh_mu_factor: float = 0.15
    short_horizon_steps: int = 3
    short_horizon_discount: float = 0.75
    short_horizon_num_candidates: int = 12
    short_horizon_outer_rounds: int = 2
    sun_horizon_steps: int = 4
    sun_horizon_discount: float = 0.80
    sun_num_candidates: int = 12
    sun_outer_rounds: int = 3
    center_tracking_blend: float = 0.90
    equal_comm_power_fraction: float = 0.55
    equal_sensing_power_fraction: float = 0.45

    @property
    def delta(self) -> float:
        return self.v_max * self.delta_t

    @property
    def width(self) -> float:
        return self.corridor_x_max - self.corridor_x_min

    @property
    def height(self) -> float:
        return self.corridor_y_max - self.corridor_y_min

    @property
    def action_dim(self) -> int:
        return 2 * self.num_vehicles + 2 * self.num_targets

    @property
    def sensing_time_cost_vec(self) -> Array:
        return self.sensing_time_cost * np.ones(self.num_targets, dtype=float)

    @property
    def action_box_norm_sq(self) -> float:
        k = self.num_vehicles
        s = self.num_targets
        return (
            k * (self.total_power ** 2)
            + k * (1.0 ** 2)
            + s * (1.0 ** 2)
            + s * (self.sensing_peak_power ** 2)
        )


@dataclass
class SimState:
    x: Array
    Q: Array
    A: Array
    U: Array
    Y: Array
    Z: Array
    vehicle_pos: Array
    vehicle_vel: Array
    target_pos: Array
    target_vel: Array
    t: int = 0
    last_exact_action: Optional[Array] = None
    last_reg_action: Optional[Array] = None


class UavIsacSimulator:
    def __init__(self, cfg: ScenarioConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # Reset and exogenous generation
    # ------------------------------------------------------------------
    def reset(self, seed: int = 0) -> SimState:
        self.rng = np.random.default_rng(seed)
        x0 = np.array([
            self.cfg.init_x if self.cfg.init_x is not None else 0.5 * (self.cfg.corridor_x_min + self.cfg.corridor_x_max),
            self.cfg.init_y,
        ], dtype=float)
        vehicle_x = self.rng.uniform(self.cfg.corridor_x_min, self.cfg.corridor_x_max, size=self.cfg.num_vehicles)
        lane_choices = self.rng.choice([-1.0, 1.0], size=self.cfg.num_vehicles)
        vehicle_y = lane_choices * self.rng.uniform(6.0, 18.0, size=self.cfg.num_vehicles)
        vehicle_pos = np.column_stack([vehicle_x, vehicle_y])
        vehicle_vel = self.rng.uniform(self.cfg.vehicle_speed_min, self.cfg.vehicle_speed_max, size=self.cfg.num_vehicles)

        target_x = self.rng.uniform(
            self.cfg.corridor_x_min + 0.1 * self.cfg.width,
            self.cfg.corridor_x_max - 0.1 * self.cfg.width,
            size=self.cfg.num_targets,
        )
        target_y = self.rng.uniform(self.cfg.corridor_y_min * 0.5, self.cfg.corridor_y_max * 0.5, size=self.cfg.num_targets)
        target_pos = np.column_stack([target_x, target_y])
        target_angle = self.rng.uniform(-math.pi / 3.0, math.pi / 3.0, size=self.cfg.num_targets)
        target_speed = self.rng.uniform(self.cfg.target_speed_min, self.cfg.target_speed_max, size=self.cfg.num_targets)
        target_vel = np.column_stack([target_speed * np.cos(target_angle), target_speed * np.sin(target_angle)])

        Q0 = self.rng.uniform(0.0, 2.0, size=self.cfg.num_vehicles)
        A0 = self.rng.integers(1, min(4, self.cfg.aoi_max), size=self.cfg.num_targets).astype(float)
        U0 = self.rng.uniform(0.8, 2.0, size=self.cfg.num_targets)
        Y0 = np.zeros(self.cfg.num_targets, dtype=float)
        Z0 = np.zeros(self.cfg.num_targets, dtype=float)
        return SimState(
            x=x0,
            Q=Q0,
            A=A0,
            U=U0,
            Y=Y0,
            Z=Z0,
            vehicle_pos=vehicle_pos,
            vehicle_vel=vehicle_vel,
            target_pos=target_pos,
            target_vel=target_vel,
        )

    def clone_state(self, state: SimState) -> SimState:
        return SimState(
            x=state.x.copy(),
            Q=state.Q.copy(),
            A=state.A.copy(),
            U=state.U.copy(),
            Y=state.Y.copy(),
            Z=state.Z.copy(),
            vehicle_pos=state.vehicle_pos.copy(),
            vehicle_vel=state.vehicle_vel.copy(),
            target_pos=state.target_pos.copy(),
            target_vel=state.target_vel.copy(),
            t=state.t,
            last_exact_action=None if state.last_exact_action is None else state.last_exact_action.copy(),
            last_reg_action=None if state.last_reg_action is None else state.last_reg_action.copy(),
        )

    def build_exogenous(self, state: SimState) -> Dict[str, Array]:
        arrivals = self.rng.poisson(self.cfg.arrival_rate, size=self.cfg.num_vehicles).astype(float)
        burst_mask = self.rng.uniform(size=self.cfg.num_vehicles) < self.cfg.arrival_burst_prob
        if np.any(burst_mask):
            arrivals[burst_mask] += self.rng.poisson(self.cfg.arrival_burst_scale, size=int(np.sum(burst_mask)))
        u_base = np.clip(
            self.cfg.uncertainty_base_alpha * state.U
            + self.cfg.uncertainty_base_noise
            + self.rng.normal(0.0, 0.15, size=self.cfg.num_targets),
            0.0,
            self.cfg.uncertainty_max,
        )
        return {"arrivals": arrivals, "U_base": u_base}

    # ------------------------------------------------------------------
    # Primitive models
    # ------------------------------------------------------------------
    def clip_point(self, y: Array) -> Array:
        return np.array([
            np.clip(y[0], self.cfg.corridor_x_min, self.cfg.corridor_x_max),
            np.clip(y[1], self.cfg.corridor_y_min, self.cfg.corridor_y_max),
        ], dtype=float)

    def project_to_reachable(self, x: Array, y: Array) -> Array:
        box_proj = self.clip_point(y)
        diff = box_proj - x
        dist = float(np.linalg.norm(diff))
        if dist <= self.cfg.delta + 1e-12:
            return box_proj
        if dist <= 1e-12:
            return x.copy()
        return x + (self.cfg.delta / dist) * diff

    def is_reachable(self, x: Array, y: Array, tol: float = 1e-8) -> bool:
        in_box = (
            self.cfg.corridor_x_min - tol <= y[0] <= self.cfg.corridor_x_max + tol
            and self.cfg.corridor_y_min - tol <= y[1] <= self.cfg.corridor_y_max + tol
        )
        return in_box and float(np.linalg.norm(y - x)) <= self.cfg.delta + tol

    def assert_reachable(self, x: Array, y: Array, tol: float = 1e-8) -> None:
        if not self.is_reachable(x, y, tol=tol):
            raise RuntimeError(
                f"Infeasible motion candidate detected: x={x}, y={y}, dist={np.linalg.norm(y - x):.6f}, delta={self.cfg.delta:.6f}"
            )

    def channel_gain(self, y: Array, r: Array) -> float:
        d2 = float(np.sum((y - r) ** 2) + self.cfg.altitude ** 2)
        return self.cfg.g0_comm / (d2 ** (0.5 * self.cfg.pathloss_exp_comm))

    def channel_gain_grad(self, y: Array, r: Array) -> Array:
        diff = y - r
        d2 = float(np.sum(diff ** 2) + self.cfg.altitude ** 2)
        g = self.channel_gain(y, r)
        return -self.cfg.pathloss_exp_comm * g * diff / d2

    def sensing_gain(self, y: Array, z: Array) -> float:
        d2 = float(np.sum((y - z) ** 2) + self.cfg.altitude ** 2)
        return self.cfg.g0_sens / (d2 ** (0.5 * self.cfg.pathloss_exp_sens))

    def sensing_gain_grad(self, y: Array, z: Array) -> Array:
        diff = y - z
        d2 = float(np.sum(diff ** 2) + self.cfg.altitude ** 2)
        beta = self.sensing_gain(y, z)
        return -self.cfg.pathloss_exp_sens * beta * diff / d2

    def rate_vector(self, y: Array, p: Array, tau: Array, state: SimState) -> Array:
        rates = np.zeros(self.cfg.num_vehicles, dtype=float)
        for k in range(self.cfg.num_vehicles):
            if tau[k] <= 1e-9 or p[k] <= 1e-9:
                continue
            gk = self.channel_gain(y, state.vehicle_pos[k])
            snr = gk * p[k] / (self.cfg.noise_psd * self.cfg.total_bandwidth * max(tau[k], 1e-9))
            rates[k] = tau[k] * self.cfg.total_bandwidth * np.log2(1.0 + snr)
        return rates

    def rate_grad_vector(self, y: Array, p: Array, tau: Array, state: SimState) -> Array:
        grads = np.zeros((self.cfg.num_vehicles, 2), dtype=float)
        for k in range(self.cfg.num_vehicles):
            if tau[k] <= 1e-9 or p[k] <= 1e-9:
                continue
            gk = self.channel_gain(y, state.vehicle_pos[k])
            dg = self.channel_gain_grad(y, state.vehicle_pos[k])
            const = p[k] / (self.cfg.noise_psd * self.cfg.total_bandwidth * max(tau[k], 1e-9))
            snr = gk * const
            grads[k] = tau[k] * self.cfg.total_bandwidth / math.log(2.0) * (1.0 / (1.0 + snr)) * const * dg
        return grads

    def q_success_vector(self, y: Array, s: Array, u: Array, state: SimState) -> Array:
        q = np.zeros(self.cfg.num_targets, dtype=float)
        for j in range(self.cfg.num_targets):
            if s[j] <= 1e-9 or u[j] <= 1e-9:
                continue
            beta = self.sensing_gain(y, state.target_pos[j])
            beta_eff = self.cfg.sensing_success_gain * beta
            q[j] = s[j] * (1.0 - np.exp(-beta_eff * u[j] / max(s[j], 1e-9)))
            q[j] = float(np.clip(q[j], 0.0, 1.0))
        return q

    def q_success_grad_vector(self, y: Array, s: Array, u: Array, state: SimState) -> Array:
        grads = np.zeros((self.cfg.num_targets, 2), dtype=float)
        for j in range(self.cfg.num_targets):
            if s[j] <= 1e-9 or u[j] <= 1e-9:
                continue
            beta = self.sensing_gain(y, state.target_pos[j])
            dbeta = self.sensing_gain_grad(y, state.target_pos[j])
            beta_eff = self.cfg.sensing_success_gain * beta
            exponent = np.exp(-beta_eff * u[j] / max(s[j], 1e-9))
            grads[j] = exponent * self.cfg.sensing_success_gain * u[j] * dbeta
        return grads

    def mu_reduction_vector(self, y: Array, s: Array, u: Array, state: SimState, exo: Dict[str, Array]) -> Array:
        q = self.q_success_vector(y, s, u, state)
        mu = self.cfg.refresh_uncertainty_gain * self.cfg.mu_reduction_scale * q
        return np.minimum(mu, exo["U_base"])

    def mu_reduction_grad_vector(self, y: Array, s: Array, u: Array, state: SimState, exo: Dict[str, Array]) -> Array:
        dq = self.q_success_grad_vector(y, s, u, state)
        grads = self.cfg.refresh_uncertainty_gain * self.cfg.mu_reduction_scale * dq
        mu_uncapped = self.cfg.refresh_uncertainty_gain * self.cfg.mu_reduction_scale * self.q_success_vector(y, s, u, state)
        active = mu_uncapped < exo["U_base"] - 1e-9
        grads[~active] = 0.0
        return grads

    def flight_cost(self, y: Array, x: Array) -> float:
        return float(np.linalg.norm(y - x) / max(self.cfg.delta, 1e-9))

    def flight_cost_grad(self, y: Array, x: Array) -> Array:
        diff = y - x
        norm = float(np.linalg.norm(diff))
        if norm <= 1e-12:
            return np.zeros(2, dtype=float)
        return diff / (max(self.cfg.delta, 1e-9) * norm)

    def road_cost(self, y: Array) -> float:
        dx = (y[0] - self.cfg.road_center_x) / self.cfg.road_sigma_x
        dy = y[1] / self.cfg.road_sigma_y
        hotspot = np.exp(-0.5 * (dx * dx + dy * dy))
        lateral_pen = (abs(y[1]) / max(abs(self.cfg.corridor_y_max), 1.0)) ** 2
        return float(0.6 * hotspot + 0.4 * lateral_pen)

    def road_cost_grad(self, y: Array) -> Array:
        dx = (y[0] - self.cfg.road_center_x) / self.cfg.road_sigma_x
        dy = y[1] / self.cfg.road_sigma_y
        hotspot = np.exp(-0.5 * (dx * dx + dy * dy))
        grad_hotspot = np.array([
            0.6 * hotspot * (-dx / self.cfg.road_sigma_x),
            0.6 * hotspot * (-dy / self.cfg.road_sigma_y),
        ], dtype=float)
        y_max = max(abs(self.cfg.corridor_y_max), 1.0)
        grad_lateral = np.array([0.0, 0.8 * y[1] / (y_max ** 2)], dtype=float)
        return grad_hotspot + grad_lateral

    # ------------------------------------------------------------------
    # Objective and solver helpers
    # ------------------------------------------------------------------
    def unpack_action(self, action: Array) -> Tuple[Array, Array, Array, Array]:
        K = self.cfg.num_vehicles
        S = self.cfg.num_targets
        p = action[:K]
        tau = action[K:2 * K]
        s = action[2 * K:2 * K + S]
        u = action[2 * K + S:2 * K + 2 * S]
        return p, tau, s, u

    def initial_action_guess(self) -> Array:
        K = self.cfg.num_vehicles
        S = self.cfg.num_targets
        p = np.full(K, 0.25 * self.cfg.total_power / max(K, 1), dtype=float)
        tau = np.full(K, 1.0 / max(K, 1), dtype=float)
        s = np.full(S, min(1.0 / max(S, 1), 0.9), dtype=float)
        u = np.minimum(0.25 * self.cfg.total_power / max(S, 1), self.cfg.sensing_peak_power * s)
        return np.concatenate([p, tau, s, u])

    def gamma_value(self, y: Array, action: Array, state: SimState, exo: Dict[str, Array],
                    use_true_weights: bool = True, regularized: bool = False,
                    rho_r: Optional[float] = None) -> float:
        p, tau, s, u = self.unpack_action(action)
        rates = self.rate_vector(y, p, tau, state)
        q = self.q_success_vector(y, s, u, state)
        mu = self.mu_reduction_vector(y, s, u, state, exo)
        if use_true_weights:
            q_weight = state.Q
            y_weight = self.cfg.c_Y * state.Y * (np.minimum(state.A + 1.0, self.cfg.aoi_max) - 1.0)
            z_weight = self.cfg.c_Z * state.Z
        else:
            q_weight = np.ones_like(state.Q)
            y_weight = np.ones_like(state.A)
            z_weight = np.ones_like(state.U)
        energy = self.cfg.delta_t * (np.sum(p) + np.sum(u))
        val = (
            self.cfg.delta_t * np.dot(q_weight, rates)
            + np.dot(y_weight, q)
            + np.dot(z_weight, mu)
            - self.cfg.V * (self.cfg.lambda_E * energy + self.cfg.lambda_S * np.sum(1.0 - s))
        )
        if regularized:
            reg = self.cfg.rho_r if rho_r is None else rho_r
            val -= 0.5 * reg * float(np.dot(action, action))
        return float(val)

    def gamma_value_from_weights(self, y: Array, action: Array, state: SimState, exo: Dict[str, Array],
                                 q_weight: Array, y_weight: Array, z_weight: Array,
                                 regularized: bool = False, rho_r: Optional[float] = None) -> float:
        p, tau, s, u = self.unpack_action(action)
        rates = self.rate_vector(y, p, tau, state)
        q = self.q_success_vector(y, s, u, state)
        mu = self.mu_reduction_vector(y, s, u, state, exo)
        energy = self.cfg.delta_t * (np.sum(p) + np.sum(u))
        val = (
            self.cfg.delta_t * np.dot(q_weight, rates)
            + np.dot(y_weight, q)
            + np.dot(z_weight, mu)
            - self.cfg.V * (self.cfg.lambda_E * energy + self.cfg.lambda_S * np.sum(1.0 - s))
        )
        if regularized:
            reg = self.cfg.rho_r if rho_r is None else rho_r
            val -= 0.5 * reg * float(np.dot(action, action))
        return float(val)

    def gamma_grad_given_action(self, y: Array, action: Array, state: SimState, exo: Dict[str, Array],
                                use_true_weights: bool = True) -> Array:
        p, tau, s, u = self.unpack_action(action)
        rate_grads = self.rate_grad_vector(y, p, tau, state)
        q_grads = self.q_success_grad_vector(y, s, u, state)
        mu_grads = self.mu_reduction_grad_vector(y, s, u, state, exo)
        if use_true_weights:
            q_weight = state.Q
            y_weight = self.cfg.c_Y * state.Y * (np.minimum(state.A + 1.0, self.cfg.aoi_max) - 1.0)
            z_weight = self.cfg.c_Z * state.Z
        else:
            q_weight = np.ones_like(state.Q)
            y_weight = np.ones_like(state.A)
            z_weight = np.ones_like(state.U)
        grad = self.cfg.delta_t * np.sum(q_weight[:, None] * rate_grads, axis=0)
        grad += np.sum(y_weight[:, None] * q_grads, axis=0)
        grad += np.sum(z_weight[:, None] * mu_grads, axis=0)
        return grad

    def spatial_cost_value(self, y: Array, x: Array) -> float:
        return float(self.cfg.V * (self.cfg.lambda_F * self.flight_cost(y, x) + self.cfg.lambda_R * self.road_cost(y)))

    def spatial_cost_grad(self, y: Array, x: Array) -> Array:
        return self.cfg.V * (self.cfg.lambda_F * self.flight_cost_grad(y, x) + self.cfg.lambda_R * self.road_cost_grad(y))

    def total_penalty(self, y: Array, x: Array, action: Array) -> Dict[str, float]:
        p, tau, s, u = self.unpack_action(action)
        flight = self.flight_cost(y, x)
        energy = self.cfg.delta_t * float(np.sum(p) + np.sum(u))
        sensing_shortfall = float(np.sum(1.0 - s))
        road = self.road_cost(y)
        penalty = self.cfg.lambda_F * flight + self.cfg.lambda_E * energy + self.cfg.lambda_S * sensing_shortfall + self.cfg.lambda_R * road
        return {
            "flight_cost": flight,
            "resource_energy": energy,
            "sensing_shortfall": sensing_shortfall,
            "road_cost": road,
            "total_penalty": penalty,
        }

    def build_constraints(self) -> List[Dict[str, object]]:
        cfg = self.cfg
        K = cfg.num_vehicles
        S = cfg.num_targets
        constraints: List[Dict[str, object]] = [
            {"type": "ineq", "fun": lambda a: cfg.total_power - float(np.sum(a[:K]) + np.sum(a[2 * K + S:2 * K + 2 * S]))},
            {"type": "ineq", "fun": lambda a: 1.0 - float(np.sum(a[K:2 * K]))},
            {"type": "ineq", "fun": lambda a: 1.0 - float(np.dot(cfg.sensing_time_cost_vec, a[2 * K:2 * K + S]))},
        ]
        for j in range(S):
            constraints.append(
                {"type": "ineq", "fun": lambda a, jj=j: cfg.sensing_peak_power * a[2 * K + jj] - a[2 * K + S + jj]}
            )
        return constraints

    def build_bounds(self) -> List[Tuple[float, float]]:
        cfg = self.cfg
        K = cfg.num_vehicles
        S = cfg.num_targets
        bounds: List[Tuple[float, float]] = []
        bounds.extend([(0.0, cfg.total_power)] * K)
        bounds.extend([(0.0, 1.0)] * K)
        bounds.extend([(0.0, 1.0)] * S)
        bounds.extend([(0.0, cfg.sensing_peak_power)] * S)
        return bounds

    def action_constraint_residual(self, action: Array) -> float:
        cfg = self.cfg
        K = cfg.num_vehicles
        S = cfg.num_targets
        p, tau, s, u = self.unpack_action(action)
        bounds = self.build_bounds()
        lower = np.array([b[0] for b in bounds], dtype=float)
        upper = np.array([b[1] for b in bounds], dtype=float)
        residuals = [
            float(np.max(np.maximum(lower - action, 0.0))),
            float(np.max(np.maximum(action - upper, 0.0))),
            max(0.0, float(np.sum(p) + np.sum(u) - cfg.total_power)),
            max(0.0, float(np.sum(tau) - 1.0)),
            max(0.0, float(np.dot(cfg.sensing_time_cost_vec, s) - 1.0)),
        ]
        residuals.extend(max(0.0, float(u[j] - cfg.sensing_peak_power * s[j])) for j in range(S))
        return float(max(residuals))

    def project_action_feasible(self, action: Array) -> Array:
        bounds = self.build_bounds()
        lower = np.array([b[0] for b in bounds], dtype=float)
        upper = np.array([b[1] for b in bounds], dtype=float)
        a = np.clip(np.asarray(action, dtype=float).copy(), lower, upper)
        K = self.cfg.num_vehicles
        S = self.cfg.num_targets
        p, tau, s, u = self.unpack_action(a)
        tau_sum = float(np.sum(tau))
        if tau_sum > 1.0:
            tau *= 1.0 / max(tau_sum, 1e-12)
        s_cost = float(np.dot(self.cfg.sensing_time_cost_vec, s))
        if s_cost > 1.0:
            s *= 1.0 / max(s_cost, 1e-12)
        u[:] = np.minimum(u, self.cfg.sensing_peak_power * s)
        power_sum = float(np.sum(p) + np.sum(u))
        if power_sum > self.cfg.total_power:
            scale = self.cfg.total_power / max(power_sum, 1e-12)
            p *= scale
            u *= scale
        a[:K] = p
        a[K:2 * K] = tau
        a[2 * K:2 * K + S] = s
        a[2 * K + S:2 * K + 2 * S] = np.minimum(u, self.cfg.sensing_peak_power * s)
        return np.clip(a, lower, upper)

    def communication_heavy_action_guess(self) -> Array:
        K = self.cfg.num_vehicles
        S = self.cfg.num_targets
        p = np.full(K, 0.75 * self.cfg.total_power / max(K, 1), dtype=float)
        tau = np.full(K, 1.0 / max(K, 1), dtype=float)
        s = np.full(S, min(0.15, 1.0 / max(S, 1)), dtype=float)
        u = np.minimum(0.05 * self.cfg.total_power / max(S, 1), self.cfg.sensing_peak_power * s)
        return self.project_action_feasible(np.concatenate([p, tau, s, u]))

    def sensing_heavy_action_guess(self) -> Array:
        K = self.cfg.num_vehicles
        S = self.cfg.num_targets
        p = np.full(K, 0.20 * self.cfg.total_power / max(K, 1), dtype=float)
        tau = np.full(K, 1.0 / max(K, 1), dtype=float)
        s_level = min(1.0, 0.95 / max(float(np.sum(self.cfg.sensing_time_cost_vec)), 1e-12))
        s = np.full(S, s_level, dtype=float)
        u = np.minimum(0.70 * self.cfg.total_power / max(S, 1), self.cfg.sensing_peak_power * s)
        return self.project_action_feasible(np.concatenate([p, tau, s, u]))

    def low_power_action_guess(self) -> Array:
        K = self.cfg.num_vehicles
        S = self.cfg.num_targets
        p = np.full(K, 0.05 * self.cfg.total_power / max(K, 1), dtype=float)
        tau = np.full(K, 1.0 / max(K, 1), dtype=float)
        s = np.full(S, min(0.25, 1.0 / max(S, 1)), dtype=float)
        u = np.minimum(0.05 * self.cfg.total_power / max(S, 1), self.cfg.sensing_peak_power * s)
        return self.project_action_feasible(np.concatenate([p, tau, s, u]))

    def deterministic_rescue_starts(self, warm_start: Optional[Array] = None) -> List[Tuple[str, Array]]:
        starts: List[Tuple[str, Array]] = []
        if warm_start is not None:
            starts.append(("primary_warm_start", self.project_action_feasible(warm_start)))
        else:
            starts.append(("primary_default", self.project_action_feasible(self.initial_action_guess())))
        starts.extend([
            ("previous_exact", self.project_action_feasible(warm_start)) if warm_start is not None else ("previous_exact", self.project_action_feasible(self.initial_action_guess())),
            ("neutral_balanced", self.project_action_feasible(self.initial_action_guess())),
            ("communication_heavy", self.communication_heavy_action_guess()),
            ("sensing_heavy", self.sensing_heavy_action_guess()),
            ("low_power", self.low_power_action_guess()),
        ])
        unique: List[Tuple[str, Array]] = []
        for label, start in starts:
            if any(np.allclose(start, existing, rtol=0.0, atol=1e-10) for _, existing in unique):
                continue
            unique.append((label, start))
        return unique

    def _solve_inner_from_start(self, y: Array, state: SimState, exo: Dict[str, Array], *,
                                regularized: bool = False, use_true_weights: bool = True,
                                x0: Array, warm_start_type: str) -> Dict[str, object]:
        bounds = self.build_bounds()
        constraints = self.build_constraints()

        def objective(a: Array) -> float:
            return -self.gamma_value(y, a, state, exo, use_true_weights=use_true_weights, regularized=regularized)

        lower = np.array([b[0] for b in bounds], dtype=float)
        upper = np.array([b[1] for b in bounds], dtype=float)
        x0 = np.clip(np.asarray(x0, dtype=float).copy(), lower, upper)
        solve_start = time.perf_counter()
        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.cfg.inner_solver_maxiter, "ftol": self.cfg.inner_solver_ftol, "disp": False},
        )
        runtime_ms = 1000.0 * (time.perf_counter() - solve_start)
        action = np.clip(res.x, lower, upper)
        gamma_val = self.gamma_value(y, action, state, exo, use_true_weights=use_true_weights, regularized=regularized)
        exact_val = self.gamma_value(y, action, state, exo, use_true_weights=use_true_weights, regularized=False)
        return {
            "action": action,
            "success": bool(res.success),
            "nit": int(getattr(res, "nit", 0)),
            "objective": float(gamma_val),
            "exact_objective": float(exact_val),
            "residual": self.action_constraint_residual(action),
            "message": str(res.message),
            "runtime_ms": float(runtime_ms),
            "warm_start_type": warm_start_type,
        }

    def solve_inner(self, y: Array, state: SimState, exo: Dict[str, Array], *,
                    regularized: bool = False, use_true_weights: bool = True,
                    warm_start: Optional[Array] = None) -> Dict[str, object]:
        x0 = self.initial_action_guess() if warm_start is None else warm_start.copy()
        warm_start_type = "default" if warm_start is None else "provided"
        return self._solve_inner_from_start(
            y,
            state,
            exo,
            regularized=regularized,
            use_true_weights=use_true_weights,
            x0=x0,
            warm_start_type=warm_start_type,
        )

    def solution_eligible(self, sol: Dict[str, object], feas_tol: float = 1e-6) -> bool:
        return (
            bool(sol.get("success", False))
            and np.isfinite(float(sol.get("objective", np.nan)))
            and np.isfinite(float(sol.get("exact_objective", np.nan)))
            and float(sol.get("residual", np.inf)) <= feas_tol
        )

    def solve_inner_with_rescue(self, y: Array, state: SimState, exo: Dict[str, Array], *,
                                regularized: bool = False, use_true_weights: bool = True,
                                warm_start: Optional[Array] = None,
                                feas_tol: float = 1e-6) -> Dict[str, object]:
        starts = self.deterministic_rescue_starts(warm_start)
        attempts: List[Dict[str, object]] = []
        attempt_idx = 0
        max_attempts = 12
        while attempt_idx < len(starts) and attempt_idx < max_attempts:
            start_type, x0 = starts[attempt_idx]
            sol = self._solve_inner_from_start(
                y,
                state,
                exo,
                regularized=regularized,
                use_true_weights=use_true_weights,
                x0=x0,
                warm_start_type=start_type,
            )
            sol["start_type"] = start_type
            sol["attempt_index"] = int(attempt_idx)
            sol["eligible"] = bool(self.solution_eligible(sol, feas_tol=feas_tol))
            attempts.append(sol)
            if attempt_idx == 0 and sol["eligible"]:
                break
            if not sol["eligible"] and float(sol.get("residual", np.inf)) > feas_tol:
                projected = self.project_action_feasible(np.asarray(sol["action"], dtype=float))
                if self.action_constraint_residual(projected) <= feas_tol:
                    duplicate = any(np.allclose(projected, existing, rtol=0.0, atol=1e-10) for _, existing in starts)
                    if not duplicate:
                        starts.append((f"projected_returned_{attempt_idx}", projected))
            attempt_idx += 1

        primary = attempts[0]
        eligible = [sol for sol in attempts if bool(sol.get("eligible", False))]
        if eligible:
            final = max(eligible, key=lambda sol: float(sol["exact_objective"]))
            final_success = True
        else:
            finite_feasible = [
                sol for sol in attempts
                if np.isfinite(float(sol.get("exact_objective", np.nan)))
                and float(sol.get("residual", np.inf)) <= feas_tol
            ]
            finite = [
                sol for sol in attempts
                if np.isfinite(float(sol.get("exact_objective", np.nan)))
            ]
            fallback = finite_feasible if finite_feasible else finite
            final = max(fallback, key=lambda sol: float(sol["exact_objective"])) if fallback else primary
            final_success = False

        total_runtime_ms = float(sum(float(sol.get("runtime_ms", 0.0)) for sol in attempts))
        total_nit = int(sum(int(sol.get("nit", 0)) for sol in attempts))
        final = dict(final)
        final.update({
            "success": bool(final_success),
            "nit": total_nit,
            "runtime_ms": total_runtime_ms,
            "message": str(final.get("message", "")),
            "warm_start_type": str(final.get("start_type", final.get("warm_start_type", "unknown"))),
            "primary_success": bool(primary.get("success", False)),
            "primary_message": str(primary.get("message", "")),
            "primary_objective": float(primary.get("objective", np.nan)),
            "primary_exact_objective": float(primary.get("exact_objective", np.nan)),
            "primary_residual": float(primary.get("residual", np.nan)),
            "primary_nit": int(primary.get("nit", 0)),
            "primary_runtime_ms": float(primary.get("runtime_ms", np.nan)),
            "final_success": bool(final_success),
            "rescue_attempts": max(0, len(attempts) - 1),
            "rescue_success": bool(final_success and not bool(primary.get("eligible", False))),
            "selected_start_type": str(final.get("start_type", final.get("warm_start_type", "unknown"))),
            "feasibility_residual": float(final.get("residual", np.nan)),
            "eligible_for_reference": bool(final_success and float(final.get("residual", np.inf)) <= feas_tol),
            "attempts": attempts,
        })
        return final

    def regularized_value(self, y: Array, state: SimState, exo: Dict[str, Array], warm_start: Optional[Array] = None,
                          use_true_weights: bool = True) -> Tuple[float, Array, int]:
        sol = self.solve_inner(y, state, exo, regularized=True, use_true_weights=use_true_weights, warm_start=warm_start)
        j_rho = sol["objective"] - self.spatial_cost_value(y, state.x)
        return float(j_rho), sol["action"], int(sol["nit"])

    def exact_value(self, y: Array, state: SimState, exo: Dict[str, Array], warm_start: Optional[Array] = None,
                    use_true_weights: bool = True) -> Tuple[float, Array, int]:
        sol = self.solve_inner(y, state, exo, regularized=False, use_true_weights=use_true_weights, warm_start=warm_start)
        j = sol["objective"] - self.spatial_cost_value(y, state.x)
        return float(j), sol["action"], int(sol["nit"])

    def solver_diag_from_solution(self, sol: Dict[str, object]) -> SolverDiag:
        return SolverDiag(
            success=bool(sol.get("success", False)),
            nit=int(sol.get("nit", 0)),
            objective=float(sol.get("objective", np.nan)),
            exact_objective=float(sol.get("exact_objective", np.nan)),
            residual=float(sol.get("residual", np.nan)),
            message=str(sol.get("message", "")),
            runtime_ms=float(sol.get("runtime_ms", np.nan)),
            warm_start_type=str(sol.get("warm_start_type", "unknown")),
        )

    def exact_value_with_diag(self, y: Array, state: SimState, exo: Dict[str, Array],
                              warm_start: Optional[Array] = None,
                              use_true_weights: bool = True) -> Dict[str, object]:
        sol = self.solve_inner(y, state, exo, regularized=False, use_true_weights=use_true_weights, warm_start=warm_start)
        j = float(sol["objective"]) - self.spatial_cost_value(y, state.x)
        return {
            "value": float(j),
            "action": sol["action"],
            "solver_diag": self.solver_diag_from_solution(sol),
        }

    def exact_value_with_rescue_diag(self, y: Array, state: SimState, exo: Dict[str, Array],
                                     warm_start: Optional[Array] = None,
                                     use_true_weights: bool = True,
                                     feas_tol: float = 1e-6) -> Dict[str, object]:
        sol = self.solve_inner_with_rescue(
            y,
            state,
            exo,
            regularized=False,
            use_true_weights=use_true_weights,
            warm_start=warm_start,
            feas_tol=feas_tol,
        )
        j = float(sol["objective"]) - self.spatial_cost_value(y, state.x)
        primary_j = float(sol["primary_objective"]) - self.spatial_cost_value(y, state.x)
        return {
            "value": float(j),
            "action": sol["action"],
            "solver_diag": self.solver_diag_from_solution(sol),
            "primary_value": float(primary_j),
            "final_value": float(j),
            "primary_success": bool(sol["primary_success"]),
            "final_success": bool(sol["final_success"]),
            "rescue_attempts": int(sol["rescue_attempts"]),
            "rescue_success": bool(sol["rescue_success"]),
            "selected_start_type": str(sol["selected_start_type"]),
            "feasibility_residual": float(sol["feasibility_residual"]),
            "eligible_for_reference": bool(sol["eligible_for_reference"]),
            "primary_message": str(sol["primary_message"]),
            "final_message": str(sol["message"]),
            "primary_nit": int(sol["primary_nit"]),
            "primary_runtime_ms": float(sol["primary_runtime_ms"]),
            "value_changed_by_rescue": bool(abs(float(j) - float(primary_j)) > 1e-6),
        }

    def aoi_weight_vector(self, state: SimState) -> Array:
        return self.cfg.c_Y * state.Y * (np.minimum(state.A + 1.0, self.cfg.aoi_max) - 1.0)

    def aoi_only_value(self, y: Array, state: SimState, exo: Dict[str, Array], warm_start: Optional[Array] = None,
                       regularized: bool = False) -> Tuple[float, Array, int]:
        q_weight = np.zeros_like(state.Q)
        y_weight = self.aoi_weight_vector(state)
        z_weight = np.zeros_like(state.U)
        bounds = self.build_bounds()
        constraints = self.build_constraints()

        def objective(a: Array) -> float:
            return -self.gamma_value_from_weights(y, a, state, exo, q_weight, y_weight, z_weight, regularized=regularized)

        x0 = self.initial_action_guess() if warm_start is None else warm_start.copy()
        lower = np.array([b[0] for b in bounds], dtype=float)
        upper = np.array([b[1] for b in bounds], dtype=float)
        x0 = np.clip(x0, lower, upper)
        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.cfg.inner_solver_maxiter, "ftol": self.cfg.inner_solver_ftol, "disp": False},
        )
        action = np.clip(res.x, lower, upper)
        val = self.gamma_value_from_weights(y, action, state, exo, q_weight, y_weight, z_weight, regularized=regularized) - self.spatial_cost_value(y, state.x)
        return float(val), action, int(getattr(res, "nit", 0))

    # ------------------------------------------------------------------
    # Mean-state prediction and rollout
    # ------------------------------------------------------------------
    def predict_exogenous_mean(self, state: SimState) -> Dict[str, Array]:
        arrivals = np.full(self.cfg.num_vehicles, self.cfg.arrival_rate, dtype=float)
        u_base = np.clip(self.cfg.uncertainty_base_alpha * state.U + self.cfg.uncertainty_base_noise, 0.0, self.cfg.uncertainty_max)
        return {"arrivals": arrivals, "U_base": u_base}

    def advance_positions_deterministic(self, state: SimState) -> None:
        state.vehicle_pos[:, 0] += state.vehicle_vel * self.cfg.delta_t
        width = self.cfg.width
        state.vehicle_pos[:, 0] = self.cfg.corridor_x_min + np.mod(state.vehicle_pos[:, 0] - self.cfg.corridor_x_min, width)
        state.target_pos += state.target_vel * self.cfg.delta_t
        state.target_pos = np.column_stack([
            np.clip(state.target_pos[:, 0], self.cfg.corridor_x_min, self.cfg.corridor_x_max),
            np.clip(state.target_pos[:, 1], self.cfg.corridor_y_min, self.cfg.corridor_y_max),
        ])

    def predict_next_state_mean(self, state: SimState, exo: Dict[str, Array], y: Array, action: Array) -> SimState:
        p, tau, s, u = self.unpack_action(action)
        rates = self.rate_vector(y, p, tau, state)
        q = self.q_success_vector(y, s, u, state)
        mu = self.mu_reduction_vector(y, s, u, state, exo)
        service_bits = rates * self.cfg.delta_t
        Q_next = np.maximum(state.Q - service_bits, 0.0) + exo["arrivals"]
        A_plus = np.minimum(state.A + 1.0, self.cfg.aoi_max)
        A_next = A_plus - q * (A_plus - 1.0)
        U_next = np.clip(exo["U_base"] - mu, 0.0, self.cfg.uncertainty_max)
        Y_next = np.maximum(state.Y + A_next - self.cfg.aoi_threshold, 0.0)
        Z_next = np.maximum(state.Z + U_next - self.cfg.uncertainty_threshold, 0.0)
        next_state = SimState(
            x=y.copy(),
            Q=Q_next.copy(),
            A=A_next.copy(),
            U=U_next.copy(),
            Y=Y_next.copy(),
            Z=Z_next.copy(),
            vehicle_pos=state.vehicle_pos.copy(),
            vehicle_vel=state.vehicle_vel.copy(),
            target_pos=state.target_pos.copy(),
            target_vel=state.target_vel.copy(),
            t=state.t + 1,
            last_exact_action=action.copy(),
            last_reg_action=action.copy(),
        )
        self.advance_positions_deterministic(next_state)
        return next_state

    def short_horizon_ao_rollout_value(self, y0: Array, state: SimState, exo: Dict[str, Array],
                                       warm_start: Optional[Array] = None) -> Dict[str, object]:
        discount = self.cfg.short_horizon_discount
        horizon = self.cfg.short_horizon_steps
        score = 0.0
        total_nit = 0
        cur_state = self.clone_state(state)
        cur_exo = {"arrivals": exo["arrivals"].copy(), "U_base": exo["U_base"].copy()}
        chosen_action = None
        chosen_y = y0.copy()
        for h in range(horizon):
            if h == 0:
                y = y0.copy()
            else:
                best_val = -np.inf
                best_y = cur_state.x.copy()
                best_action_h = None
                nit_h = 0
                for cand in self.reachable_candidates(cur_state.x, num_directions=self.cfg.short_horizon_num_candidates):
                    val, act, nit = self.exact_value(cand, cur_state, cur_exo, warm_start=cur_state.last_exact_action, use_true_weights=True)
                    if val > best_val:
                        best_val = val
                        best_y = cand
                        best_action_h = act
                        nit_h = nit
                total_nit += nit_h
                y = best_y
                action_h = best_action_h
            if h == 0 or chosen_action is None:
                val, action_h, nit = self.exact_value(y, cur_state, cur_exo, warm_start=warm_start if h == 0 else cur_state.last_exact_action, use_true_weights=True)
                total_nit += nit
            else:
                val = self.gamma_value(y, action_h, cur_state, cur_exo, use_true_weights=True, regularized=False) - self.spatial_cost_value(y, cur_state.x)
            score += (discount ** h) * float(val)
            if h == 0:
                chosen_action = action_h.copy()
                chosen_y = y.copy()
            cur_state = self.predict_next_state_mean(cur_state, cur_exo, y, action_h)
            cur_exo = self.predict_exogenous_mean(cur_state)
        return {"score": float(score), "action": chosen_action, "y": chosen_y, "inner_iterations": total_nit}

    def numerical_gradient_regularized_value(self, state: SimState, exo: Dict[str, Array],
                                             warm_start: Optional[Array] = None, use_true_weights: bool = True) -> Tuple[Array, Array, int]:
        eps = self.cfg.grad_eps
        x = state.x
        grad = np.zeros(2, dtype=float)
        total_nit = 0
        current_val, current_action, nit0 = self.regularized_value(x, state, exo, warm_start=warm_start, use_true_weights=use_true_weights)
        total_nit += nit0
        for d in range(2):
            e = np.zeros(2, dtype=float)
            e[d] = 1.0
            y_plus = self.clip_point(x + eps * e)
            y_minus = self.clip_point(x - eps * e)
            val_plus, _, nit_p = self.regularized_value(y_plus, state, exo, warm_start=current_action, use_true_weights=use_true_weights)
            val_minus, _, nit_m = self.regularized_value(y_minus, state, exo, warm_start=current_action, use_true_weights=use_true_weights)
            total_nit += nit_p + nit_m
            denom = max(y_plus[d] - y_minus[d], 1e-9)
            grad[d] = (val_plus - val_minus) / denom
        return grad, current_action, total_nit

    def analytic_gradient_regularized_value(self, state: SimState, exo: Dict[str, Array],
                                            warm_start: Optional[Array] = None, use_true_weights: bool = True) -> Tuple[Array, Array, int]:
        _, action, nit = self.regularized_value(state.x, state, exo, warm_start=warm_start, use_true_weights=use_true_weights)
        grad = self.gamma_grad_given_action(state.x, action, state, exo, use_true_weights=use_true_weights) - self.spatial_cost_grad(state.x, state.x)
        return grad.astype(float), action, nit

    # ------------------------------------------------------------------
    # Mobility helpers and candidate sets
    # ------------------------------------------------------------------
    def reachable_candidates(self, x: Array, num_directions: Optional[int] = None) -> List[Array]:
        n = self.cfg.benchmark_num_directions if num_directions is None else num_directions
        candidates: List[Array] = [x.copy()]
        for ang in np.linspace(0.0, 2.0 * math.pi, n, endpoint=False):
            nu = np.array([math.cos(ang), math.sin(ang)], dtype=float)
            y = self.project_to_reachable(x, x + self.cfg.delta * nu)
            candidates.append(y)
        xs = np.linspace(self.cfg.corridor_x_min, self.cfg.corridor_x_max, 5)
        ys = np.linspace(self.cfg.corridor_y_min, self.cfg.corridor_y_max, 3)
        for xg in xs:
            for yg in ys:
                y = np.array([xg, yg], dtype=float)
                if np.linalg.norm(y - x) <= self.cfg.delta + 1e-9:
                    candidates.append(y)
        uniq: List[Array] = []
        seen = set()
        for y in candidates:
            key = (round(float(y[0]), 4), round(float(y[1]), 4))
            if key not in seen:
                seen.add(key)
                uniq.append(y)
        return uniq

    def local_candidates_around(self, x: Array, center: Array, num_candidates: int) -> List[Array]:
        if num_candidates <= 1:
            return [x.copy()]
        candidates = [x.copy(), self.project_to_reachable(x, center)]
        remaining = max(0, num_candidates - len(candidates))
        if remaining > 0:
            base = self.project_to_reachable(x, center)
            rings = max(1, int(math.ceil(remaining / max(1, self.cfg.shortlist_num_local_dirs))))
            generated = 0
            for ring in range(1, rings + 1):
                radius = self.cfg.shortlist_radius_scale * self.cfg.delta * ring / rings
                dirs_this = min(max(4, self.cfg.shortlist_num_local_dirs), remaining - generated)
                for ang in np.linspace(0.0, 2.0 * math.pi, dirs_this, endpoint=False):
                    cand = self.project_to_reachable(x, base + radius * np.array([math.cos(ang), math.sin(ang)], dtype=float))
                    candidates.append(cand)
                    generated += 1
                    if generated >= remaining:
                        break
                if generated >= remaining:
                    break
        uniq: List[Array] = []
        seen = set()
        for cand in candidates:
            key = (round(float(cand[0]), 4), round(float(cand[1]), 4))
            if key not in seen:
                seen.add(key)
                uniq.append(cand)
        return uniq[:num_candidates]

    def mobility_from_gradient(self, x: Array, grad: Array, step_scale: float = 1.0) -> Tuple[Array, Array]:
        g = self.cfg.delta * grad
        nu_raw = step_scale * self.cfg.eta_t * g
        norm_nu = float(np.linalg.norm(nu_raw))
        if norm_nu > 1.0:
            nu_raw = nu_raw / norm_nu
        y = self.project_to_reachable(x, x + self.cfg.delta * nu_raw)
        nu = (y - x) / max(self.cfg.delta, 1e-9)
        norm_nu2 = float(np.linalg.norm(nu))
        if norm_nu2 > 1.0:
            nu = nu / norm_nu2
            y = self.project_to_reachable(x, x + self.cfg.delta * nu)
        return nu, y

    def step_towards_point(self, x: Array, target: Array) -> Tuple[Array, Array]:
        target_clipped = self.clip_point(target)
        diff = target_clipped - x
        norm = float(np.linalg.norm(diff))
        if norm <= 1e-12:
            y = x.copy()
        else:
            y = x + min(self.cfg.delta, norm) * diff / norm
            y = self.project_to_reachable(x, y)
        nu = (y - x) / max(self.cfg.delta, 1e-9)
        return nu, y

    def geometric_center(self, state: SimState, use_weights: bool = False, include_targets: bool = True) -> Array:
        pts = []
        wts = []
        for k in range(self.cfg.num_vehicles):
            pts.append(state.vehicle_pos[k])
            if use_weights:
                wts.append(float(state.Q[k] + 0.25))
            else:
                wts.append(1.0)
        if include_targets:
            urgency = self.cfg.c_Y * state.Y + self.cfg.c_Z * state.Z + 0.25
            for j in range(self.cfg.num_targets):
                pts.append(state.target_pos[j])
                wts.append(float(urgency[j] if use_weights else 1.0))
        pts_arr = np.asarray(pts, dtype=float)
        w_arr = np.asarray(wts, dtype=float)
        center = np.average(pts_arr, axis=0, weights=w_arr)
        return self.clip_point(center)

    def build_gradient_shortlist(self, x: Array, grad: Array, target_size: Optional[int] = None) -> List[Tuple[str, Array]]:
        n_target = max(1, self.cfg.shortlist_size if target_size is None else int(target_size))
        candidates: List[Tuple[str, Array]] = [("stay", x.copy())]
        if n_target == 1:
            return candidates
        _, y_grad = self.mobility_from_gradient(x, grad, step_scale=1.0)
        candidates.append(("grad_full", y_grad))
        if n_target == 2:
            return candidates
        if self.cfg.shortlist_use_half_step:
            _, y_half = self.mobility_from_gradient(x, grad, step_scale=0.5)
            candidates.append(("grad_half", y_half))
        remaining = max(0, n_target - len(candidates))
        if remaining > 0:
            center = y_grad.copy()
            rings = max(1, int(math.ceil(remaining / max(1, self.cfg.shortlist_num_local_dirs))))
            generated = 0
            for ring in range(1, rings + 1):
                radius = self.cfg.shortlist_radius_scale * self.cfg.delta * ring / rings
                dirs_this = min(max(4, self.cfg.shortlist_num_local_dirs), remaining - generated)
                for ang in np.linspace(0.0, 2.0 * math.pi, dirs_this, endpoint=False):
                    cand = self.project_to_reachable(x, center + radius * np.array([math.cos(ang), math.sin(ang)], dtype=float))
                    candidates.append((f"local_r{ring}_{ang:.2f}", cand))
                    generated += 1
                    if generated >= remaining:
                        break
                if generated >= remaining:
                    break
        uniq: List[Tuple[str, Array]] = []
        seen = set()
        for label, cand in candidates:
            key = (round(float(cand[0]), 4), round(float(cand[1]), 4))
            if key not in seen:
                seen.add(key)
                uniq.append((label, cand))
        return uniq[:n_target]

    def select_best_exact_shortlist(self, shortlist: List[Tuple[str, Array]], state: SimState, exo: Dict[str, Array],
                                    warm_start: Optional[Array] = None, use_true_weights: bool = True) -> Dict[str, object]:
        best = None
        total_nit = 0
        best_warm = warm_start
        for label, cand in shortlist:
            val, act, nit = self.exact_value(cand, state, exo, warm_start=best_warm, use_true_weights=use_true_weights)
            total_nit += nit
            if best is None or val > best["exact_value"]:
                best = {"candidate_type": label, "y": cand, "action": act, "exact_value": float(val)}
                best_warm = act
        assert best is not None
        best["shortlist_size"] = len(shortlist)
        best["inner_iterations"] = total_nit
        return best

    def benchmark_oracle(self, state: SimState, exo: Dict[str, Array], use_true_weights: bool = True) -> Dict[str, object]:
        best_val = -np.inf
        best_y = state.x.copy()
        best_action = self.initial_action_guess()
        total_nit = 0
        warm = state.last_exact_action
        for y in self.reachable_candidates(state.x):
            val, act, nit = self.exact_value(y, state, exo, warm_start=warm, use_true_weights=use_true_weights)
            total_nit += nit
            if val > best_val:
                best_val = val
                best_y = y
                best_action = act
                warm = act
        return {"Psi": float(best_val), "best_y": best_y, "best_action": best_action, "nit": total_nit}

    def equal_allocation_action(self, y: Array, state: SimState, exo: Dict[str, Array]) -> Array:
        K = self.cfg.num_vehicles
        S = self.cfg.num_targets
        p = np.zeros(K, dtype=float)
        tau = np.zeros(K, dtype=float)
        s = np.zeros(S, dtype=float)
        u = np.zeros(S, dtype=float)

        active_k = np.where(state.Q > 1e-6)[0]
        if active_k.size == 0:
            active_k = np.arange(K)
        tau_share = 1.0 / max(int(active_k.size), 1)
        tau[active_k] = tau_share
        p_budget = self.cfg.equal_comm_power_fraction * self.cfg.total_power
        p[active_k] = p_budget / max(int(active_k.size), 1)

        urgency = self.cfg.c_Y * state.Y + self.cfg.c_Z * state.Z + 1e-3
        active_s = np.where(urgency > 1e-6)[0]
        if active_s.size == 0:
            active_s = np.arange(S)
        s_share = min(1.0, 1.0 / max(self.cfg.sensing_time_cost * int(active_s.size), 1e-9))
        s[active_s] = 0.95 * s_share
        u_budget = self.cfg.equal_sensing_power_fraction * self.cfg.total_power
        u_share = u_budget / max(int(active_s.size), 1)
        for j in active_s:
            u[j] = min(u_share, self.cfg.sensing_peak_power * s[j])

        action = np.concatenate([p, tau, s, u])
        return action

    def sun_receding_ao_controller(self, state: SimState, exo: Dict[str, Array]) -> Dict[str, object]:
        horizon = max(1, self.cfg.sun_horizon_steps)
        rounds = max(1, self.cfg.sun_outer_rounds)
        num_candidates = max(4, self.cfg.sun_num_candidates)

        path: List[Array] = []
        tmp_state = self.clone_state(state)
        for _ in range(horizon):
            ref = self.geometric_center(tmp_state, use_weights=True, include_targets=True)
            _, y = self.step_towards_point(tmp_state.x, ref)
            path.append(y)
            tmp_state.x = y.copy()
            self.advance_positions_deterministic(tmp_state)

        total_nit = 0
        actions: List[Array] = [self.initial_action_guess() for _ in range(horizon)]
        for _ in range(rounds):
            cur_state = self.clone_state(state)
            cur_exo = {"arrivals": exo["arrivals"].copy(), "U_base": exo["U_base"].copy()}
            new_actions: List[Array] = []
            warm = state.last_exact_action
            for h in range(horizon):
                _, act, nit = self.exact_value(path[h], cur_state, cur_exo, warm_start=warm, use_true_weights=True)
                total_nit += nit
                new_actions.append(act)
                warm = act
                cur_state = self.predict_next_state_mean(cur_state, cur_exo, path[h], act)
                cur_exo = self.predict_exogenous_mean(cur_state)
            actions = new_actions

            cur_state = self.clone_state(state)
            cur_exo = {"arrivals": exo["arrivals"].copy(), "U_base": exo["U_base"].copy()}
            new_path: List[Array] = []
            for h in range(horizon):
                ref = self.geometric_center(cur_state, use_weights=True, include_targets=True)
                candidates = self.local_candidates_around(cur_state.x, ref, num_candidates)
                best_score = -np.inf
                best_y = candidates[0]
                for cand in candidates:
                    score = self.gamma_value(cand, actions[h], cur_state, cur_exo, use_true_weights=True, regularized=False) - self.spatial_cost_value(cand, cur_state.x)
                    if score > best_score:
                        best_score = score
                        best_y = cand
                new_path.append(best_y)
                cur_state = self.predict_next_state_mean(cur_state, cur_exo, best_y, actions[h])
                cur_exo = self.predict_exogenous_mean(cur_state)
            path = new_path

        y = path[0].copy()
        val, action, nit = self.exact_value(y, state, exo, warm_start=actions[0], use_true_weights=True)
        total_nit += nit
        nu = (y - state.x) / max(self.cfg.delta, 1e-9)
        return {
            "nu": nu,
            "y": y,
            "action": action,
            "inner_iterations": total_nit,
            "candidate_type": "sun_receding_ao",
            "execution": "sun_ao_exact_exec",
            "shortlist_size": num_candidates,
            "shortlist_best_exact_value": float(val),
        }

    # ------------------------------------------------------------------
    # Controllers
    # ------------------------------------------------------------------
    def choose_action(self, controller: str, state: SimState, exo: Dict[str, Array]) -> Dict[str, object]:
        start = time.perf_counter()
        inner_nit = 0
        shortlist_size = 0
        candidate_type = "na"
        shortlist_best_exact_value = np.nan
        grad_norm = np.nan
        grad_error_rel = np.nan
        candidate_points: List[Array] = []
        candidate_labels: List[str] = []

        if controller == "proposed_full":
            grad, reg_action_x, nit = self.numerical_gradient_regularized_value(state, exo, warm_start=state.last_reg_action, use_true_weights=True)
            inner_nit += nit
            grad_norm = float(np.linalg.norm(grad))
            shortlist = self.build_gradient_shortlist(state.x, grad)
            candidate_points = [cand.copy() for _, cand in shortlist]
            candidate_labels = [str(label) for label, _ in shortlist]
            shortlist_size = len(shortlist)
            best = self.select_best_exact_shortlist(shortlist, state, exo, warm_start=state.last_exact_action, use_true_weights=True)
            candidate_type = str(best["candidate_type"])
            shortlist_best_exact_value = float(best["exact_value"])
            inner_nit += int(best["inner_iterations"])
            y = best["y"]
            nu = (y - state.x) / max(self.cfg.delta, 1e-9)
            action = best["action"]
            exec_label = "exact_shortlist"
            state.last_reg_action = reg_action_x
            state.last_exact_action = action

        elif controller == "regularized_execution":
            grad, reg_action_x, nit = self.numerical_gradient_regularized_value(state, exo, warm_start=state.last_reg_action, use_true_weights=True)
            inner_nit += nit
            grad_norm = float(np.linalg.norm(grad))
            nu, y = self.mobility_from_gradient(state.x, grad)
            candidate_points = [y.copy()]
            candidate_labels = ["gradient_step"]
            sol_exec = self.solve_inner(y, state, exo, regularized=True, use_true_weights=True, warm_start=state.last_reg_action)
            inner_nit += sol_exec["nit"]
            action = sol_exec["action"]
            exec_label = "regularized"
            state.last_reg_action = action
            state.last_exact_action = action

        elif controller == "anchored_mobility":
            sol_anchor = self.solve_inner(state.x, state, exo, regularized=False, use_true_weights=True, warm_start=state.last_exact_action)
            inner_nit += sol_anchor["nit"]
            anchor_action = sol_anchor["action"]
            best_score = -np.inf
            y = state.x.copy()
            shortlist = self.reachable_candidates(state.x)
            candidate_points = [cand.copy() for cand in shortlist]
            candidate_labels = [f"reachable_{i}" for i in range(len(shortlist))]
            for cand in shortlist:
                score = self.gamma_value(cand, anchor_action, state, exo, use_true_weights=True, regularized=False) - self.spatial_cost_value(cand, state.x)
                if score > best_score:
                    best_score = score
                    y = cand
            nu = (y - state.x) / max(self.cfg.delta, 1e-9)
            sol_exec = self.solve_inner(y, state, exo, regularized=False, use_true_weights=True, warm_start=anchor_action)
            inner_nit += sol_exec["nit"]
            action = sol_exec["action"]
            exec_label = "exact_after_anchored_eval"
            shortlist_size = len(self.reachable_candidates(state.x))
            shortlist_best_exact_value = float(best_score)
            candidate_type = "anchored_full_search"
            state.last_exact_action = action

        elif controller == "aoi_only":
            best_score = -np.inf
            best_y = state.x.copy()
            best_action = self.initial_action_guess()
            shortlist = self.reachable_candidates(state.x)
            candidate_points = [cand.copy() for cand in shortlist]
            candidate_labels = [f"reachable_{i}" for i in range(len(shortlist))]
            shortlist_size = len(shortlist)
            for cand in shortlist:
                val, act, nit = self.aoi_only_value(cand, state, exo, warm_start=state.last_exact_action, regularized=False)
                inner_nit += nit
                if val > best_score:
                    best_score = val
                    best_y = cand
                    best_action = act
            y = best_y
            nu = (y - state.x) / max(self.cfg.delta, 1e-9)
            action = best_action
            exec_label = "aoi_only_exact"
            shortlist_best_exact_value = float(best_score)
            candidate_type = "aoi_best"
            state.last_exact_action = action

        elif controller == "short_horizon_ao":
            best = None
            shortlist = [("cand", cand) for cand in self.reachable_candidates(state.x, num_directions=self.cfg.short_horizon_num_candidates)]
            candidate_points = [cand.copy() for _, cand in shortlist]
            candidate_labels = [f"rollout_{i}" for i in range(len(shortlist))]
            shortlist_size = len(shortlist)
            for _, cand in shortlist:
                rollout = self.short_horizon_ao_rollout_value(cand, state, exo, warm_start=state.last_exact_action)
                inner_nit += int(rollout["inner_iterations"])
                if best is None or rollout["score"] > best["score"]:
                    best = rollout
            assert best is not None
            y = best["y"]
            nu = (y - state.x) / max(self.cfg.delta, 1e-9)
            action = best["action"]
            exec_label = "short_horizon_ao"
            shortlist_best_exact_value = float(best["score"])
            candidate_type = "rollout_best"
            state.last_exact_action = action

        elif controller == "greedy_myopic":
            best_score = -np.inf
            best_y = state.x.copy()
            best_action = self.initial_action_guess()
            shortlist = self.reachable_candidates(state.x)
            candidate_points = [cand.copy() for cand in shortlist]
            candidate_labels = [f"reachable_{i}" for i in range(len(shortlist))]
            shortlist_size = len(shortlist)
            for cand in shortlist:
                val, act, nit = self.exact_value(cand, state, exo, warm_start=state.last_exact_action, use_true_weights=False)
                inner_nit += nit
                if val > best_score:
                    best_score = val
                    best_y = cand
                    best_action = act
            y = best_y
            nu = (y - state.x) / max(self.cfg.delta, 1e-9)
            action = best_action
            exec_label = "myopic"
            shortlist_best_exact_value = float(best_score)
            candidate_type = "myopic_best"
            state.last_exact_action = action

        elif controller == "static_uav":
            y = state.x.copy()
            nu = np.zeros(2, dtype=float)
            candidate_points = [y.copy()]
            candidate_labels = ["static"]
            sol_exec = self.solve_inner(y, state, exo, regularized=False, use_true_weights=True, warm_start=state.last_exact_action)
            inner_nit += sol_exec["nit"]
            action = sol_exec["action"]
            exec_label = "static_exact"
            state.last_exact_action = action

        elif controller == "full_candidate_exact":
            shortlist = self.reachable_candidates(state.x)
            candidate_points = [cand.copy() for cand in shortlist]
            candidate_labels = [f"reachable_{i}" for i in range(len(shortlist))]
            oracle = self.benchmark_oracle(state, exo, use_true_weights=True)
            inner_nit += int(oracle["nit"])
            y = oracle["best_y"]
            action = oracle["best_action"]
            nu = (y - state.x) / max(self.cfg.delta, 1e-9)
            exec_label = "full_candidate_exact"
            shortlist_size = len(self.reachable_candidates(state.x))
            shortlist_best_exact_value = float(oracle["Psi"])
            candidate_type = "full_candidate"
            state.last_exact_action = action

        elif controller == "yang_go_lyapunov":
            center = self.geometric_center(state, use_weights=False, include_targets=True)
            nu, y = self.step_towards_point(state.x, center)
            candidate_points = [y.copy()]
            candidate_labels = ["geometric_center"]
            sol_exec = self.solve_inner(y, state, exo, regularized=False, use_true_weights=True, warm_start=state.last_exact_action)
            inner_nit += sol_exec["nit"]
            action = sol_exec["action"]
            exec_label = "yang_go_lyap"
            candidate_type = "geometric_center"
            state.last_exact_action = action

        elif controller == "yang_ge_equal":
            center = self.geometric_center(state, use_weights=False, include_targets=True)
            nu, y = self.step_towards_point(state.x, center)
            candidate_points = [y.copy()]
            candidate_labels = ["geometric_center"]
            action = self.equal_allocation_action(y, state, exo)
            exec_label = "yang_ge_equal"
            candidate_type = "geometric_center_equal"
            state.last_exact_action = action

        elif controller == "sun_receding_ao":
            ctl = self.sun_receding_ao_controller(state, exo)
            inner_nit += int(ctl["inner_iterations"])
            y = ctl["y"]
            nu = ctl["nu"]
            action = ctl["action"]
            candidate_points = [y.copy()]
            candidate_labels = ["sun_receding_path0"]
            exec_label = str(ctl["execution"])
            shortlist_size = int(ctl["shortlist_size"])
            shortlist_best_exact_value = float(ctl["shortlist_best_exact_value"])
            candidate_type = str(ctl["candidate_type"])
            state.last_exact_action = action

        else:
            raise ValueError(f"Unknown controller: {controller}")

        runtime = time.perf_counter() - start
        return {
            "controller": controller,
            "nu": nu,
            "y": y,
            "action": action,
            "runtime": runtime,
            "inner_iterations": inner_nit,
            "execution": exec_label,
            "shortlist_size": int(shortlist_size),
            "candidate_type": candidate_type,
            "shortlist_best_exact_value": float(shortlist_best_exact_value) if np.isfinite(shortlist_best_exact_value) else np.nan,
            "grad_norm": float(grad_norm) if np.isfinite(grad_norm) else np.nan,
            "grad_error_rel": float(grad_error_rel) if np.isfinite(grad_error_rel) else np.nan,
            "candidate_points": [cand.copy() for cand in candidate_points],
            "candidate_labels": list(candidate_labels),
        }

    def _canonical_controller_name(self, method: str) -> str:
        aliases = {
            "FC-Union": "full_candidate_exact",
            "fc_union": "full_candidate_exact",
            "finite_candidate_reference": "full_candidate_exact",
            "freshness_priority": "aoi_only",
            "myopic_reoptimization": "greedy_myopic",
        }
        return aliases.get(method, method)

    def _candidate_family_from_choice(self, method: str, chosen: Dict[str, object]) -> CandidateFamily:
        y = np.asarray(chosen["y"], dtype=float).copy()
        raw_points = chosen.get("candidate_points", [])
        raw_labels = chosen.get("candidate_labels", [])
        points_list = [np.asarray(p, dtype=float).copy() for p in raw_points] if raw_points else [y.copy()]
        labels = [str(label) for label in raw_labels] if raw_labels else ["executed"]
        if len(labels) != len(points_list):
            labels = [f"candidate_{i}" for i in range(len(points_list))]
        has_executed = any(np.allclose(p, y, rtol=0.0, atol=1e-8) for p in points_list)
        if not has_executed:
            points_list.append(y.copy())
            labels.append("executed")
        points = np.vstack(points_list).astype(float)
        return CandidateFamily(method=method, points=points, executed_point=y, labels=labels)

    def _solver_diag_from_choice(self, chosen: Dict[str, object], value: float, residual: float) -> SolverDiag:
        return SolverDiag(
            success=True,
            nit=int(chosen.get("inner_iterations", 0)),
            objective=float(value),
            exact_objective=float(value),
            residual=float(residual),
            message="R1 controller action diagnostic; detailed per-solve metadata is retained by solve_inner and will be aggregated in Patch 2.",
            runtime_ms=float(chosen.get("runtime", 0.0)) * 1000.0,
            warm_start_type="controller_path",
        )

    def _controller_value_type(self, method: str, legacy_method: str) -> str:
        del method
        exact_methods = {
            "proposed_full",
            "static_uav",
            "full_candidate_exact",
            "yang_go_lyapunov",
        }
        if legacy_method in exact_methods:
            return "unregularized_exact"
        if legacy_method == "regularized_execution":
            return "regularized_design"
        if legacy_method == "anchored_mobility":
            return "anchored_score"
        if legacy_method in {"aoi_only", "yang_ge_equal"}:
            return "heuristic_score"
        if legacy_method in {"greedy_myopic", "short_horizon_ao", "sun_receding_ao"}:
            return "surrogate_score"
        return "unknown"

    def _controller_reported_value(self, legacy_method: str, chosen: Dict[str, object], computed_value: float) -> float:
        native_score_methods = {
            "proposed_full",
            "anchored_mobility",
            "aoi_only",
            "greedy_myopic",
            "full_candidate_exact",
            "sun_receding_ao",
            "short_horizon_ao",
        }
        if legacy_method in native_score_methods:
            native_value = chosen.get("shortlist_best_exact_value", np.nan)
            if np.isfinite(native_value):
                return float(native_value)
        return float(computed_value)

    def choose_action_r1(self, state: SimState, exo: Dict[str, Array], method: str = "proposed_full") -> ActionResult:
        legacy_method = self._canonical_controller_name(method)
        state_for_choice = self.clone_state(state)
        exo_for_choice = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in exo.items()
        }
        chosen = self.choose_action(legacy_method, state_for_choice, exo_for_choice)
        y = np.asarray(chosen["y"], dtype=float).copy()
        action = np.asarray(chosen["action"], dtype=float).copy()
        nu = np.asarray(chosen["nu"], dtype=float).copy()
        value = self.gamma_value(y, action, state, exo, use_true_weights=True, regularized=False) - self.spatial_cost_value(y, state.x)
        family = self._candidate_family_from_choice(method, chosen)
        solver_diag = self._solver_diag_from_choice(chosen, float(value), self.action_constraint_residual(action))
        controller_value_type = self._controller_value_type(method, legacy_method)
        controller_value = self._controller_reported_value(legacy_method, chosen, float(value))
        metadata: Dict[str, Any] = {
            "legacy_method": legacy_method,
            "controller_value": float(controller_value),
            "controller_value_type": controller_value_type,
            "execution": str(chosen.get("execution", "na")),
            "candidate_type": str(chosen.get("candidate_type", "na")),
            "shortlist_size": int(chosen.get("shortlist_size", 0)),
            "shortlist_best_exact_value": float(chosen.get("shortlist_best_exact_value", np.nan)) if np.isfinite(chosen.get("shortlist_best_exact_value", np.nan)) else np.nan,
            "grad_norm": float(chosen.get("grad_norm", np.nan)) if np.isfinite(chosen.get("grad_norm", np.nan)) else np.nan,
            "grad_error_rel": float(chosen.get("grad_error_rel", np.nan)) if np.isfinite(chosen.get("grad_error_rel", np.nan)) else np.nan,
            "inner_iterations": int(chosen.get("inner_iterations", 0)),
            "next_last_exact_action": None if state_for_choice.last_exact_action is None else state_for_choice.last_exact_action.copy(),
            "next_last_reg_action": None if state_for_choice.last_reg_action is None else state_for_choice.last_reg_action.copy(),
        }
        return ActionResult(
            method=method,
            velocity=nu,
            post_motion_point=y,
            resource_action=action,
            value=float(value),
            candidate_family=family,
            solver_diag=solver_diag,
            runtime_ms=float(chosen.get("runtime", 0.0)) * 1000.0,
            metadata=metadata,
        )

    def apply_action_r1(self, state: SimState, exo: Dict[str, Array], action_result: ActionResult) -> Dict[str, float]:
        chosen = {
            "y": action_result.post_motion_point,
            "action": action_result.resource_action,
            "shortlist_size": action_result.metadata.get("shortlist_size", len(action_result.candidate_family.points)),
            "shortlist_best_exact_value": action_result.metadata.get("shortlist_best_exact_value", np.nan),
            "grad_norm": action_result.metadata.get("grad_norm", np.nan),
            "grad_error_rel": action_result.metadata.get("grad_error_rel", np.nan),
        }
        metrics = self.apply_action(state, exo, chosen)
        next_exact = action_result.metadata.get("next_last_exact_action")
        next_reg = action_result.metadata.get("next_last_reg_action")
        state.last_exact_action = None if next_exact is None else np.asarray(next_exact, dtype=float).copy()
        state.last_reg_action = None if next_reg is None else np.asarray(next_reg, dtype=float).copy()
        return metrics

    def simulate_r1(self, method: str, horizon: int, seed: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        state = self.reset(seed)
        slot_rows: List[Dict[str, object]] = []
        for t in range(horizon):
            exo = self.build_exogenous(state)
            action_result = self.choose_action_r1(state, exo, method=method)
            metrics = self.apply_action_r1(state, exo, action_result)
            diag = action_result.solver_diag
            slot_rows.append({
                "slot": t,
                "controller": method,
                "x": float(state.x[0]),
                "y": float(state.x[1]),
                "runtime_ms": float(action_result.runtime_ms),
                "inner_iterations": int(action_result.metadata.get("inner_iterations", diag.nit)),
                "solver_success": bool(diag.success),
                "solver_nit": int(diag.nit),
                "solver_residual": float(diag.residual),
                "solver_runtime_ms": float(diag.runtime_ms),
                "candidate_family_size": int(len(action_result.candidate_family.points)),
                "execution": str(action_result.metadata.get("execution", "na")),
                "candidate_type": str(action_result.metadata.get("candidate_type", "na")),
                "executed_value": float(action_result.value),
                **metrics,
            })
        slot_df = pd.DataFrame(slot_rows)
        summary = slot_df.drop(columns=["slot", "x", "y"]).mean(numeric_only=True).to_dict()
        summary["controller"] = method
        summary["seed"] = seed
        summary["horizon"] = horizon
        return pd.DataFrame([summary]), slot_df

    # ------------------------------------------------------------------
    # State update and rollout
    # ------------------------------------------------------------------
    def update_vehicle_positions(self, state: SimState) -> None:
        state.vehicle_pos[:, 0] += state.vehicle_vel * self.cfg.delta_t
        width = self.cfg.width
        state.vehicle_pos[:, 0] = self.cfg.corridor_x_min + np.mod(state.vehicle_pos[:, 0] - self.cfg.corridor_x_min, width)

    def update_target_positions(self, state: SimState) -> None:
        noise = self.rng.normal(0.0, self.cfg.target_process_noise, size=state.target_vel.shape)
        state.target_vel = state.target_vel + noise
        speed = np.linalg.norm(state.target_vel, axis=1)
        speed = np.clip(speed, self.cfg.target_speed_min, self.cfg.target_speed_max)
        direction = np.arctan2(state.target_vel[:, 1], state.target_vel[:, 0])
        state.target_vel[:, 0] = speed * np.cos(direction)
        state.target_vel[:, 1] = speed * np.sin(direction)
        state.target_pos += state.target_vel * self.cfg.delta_t
        for j in range(self.cfg.num_targets):
            if state.target_pos[j, 0] < self.cfg.corridor_x_min or state.target_pos[j, 0] > self.cfg.corridor_x_max:
                state.target_vel[j, 0] *= -1.0
            if state.target_pos[j, 1] < self.cfg.corridor_y_min or state.target_pos[j, 1] > self.cfg.corridor_y_max:
                state.target_vel[j, 1] *= -1.0
        state.target_pos = np.column_stack([
            np.clip(state.target_pos[:, 0], self.cfg.corridor_x_min, self.cfg.corridor_x_max),
            np.clip(state.target_pos[:, 1], self.cfg.corridor_y_min, self.cfg.corridor_y_max),
        ])

    def apply_action(self, state: SimState, exo: Dict[str, Array], chosen: Dict[str, object]) -> Dict[str, float]:
        y = chosen["y"]
        action = chosen["action"]
        self.assert_reachable(state.x, y)
        p, tau, s, u = self.unpack_action(action)
        rates = self.rate_vector(y, p, tau, state)
        q = self.q_success_vector(y, s, u, state)
        mu = self.mu_reduction_vector(y, s, u, state, exo)
        refresh = self.rng.uniform(size=self.cfg.num_targets) < q
        service_bits = rates * self.cfg.delta_t

        prev_queue_plus_arrivals = float(np.sum(state.Q + exo["arrivals"]))
        prev_mean_aoi = float(np.mean(state.A))
        prev_mean_unc = float(np.mean(state.U))

        Q_next = np.maximum(state.Q - service_bits, 0.0) + exo["arrivals"]
        A_plus = np.minimum(state.A + 1.0, self.cfg.aoi_max)
        A_next = np.where(refresh, 1.0, A_plus)
        U_next = np.clip(exo["U_base"] - mu, 0.0, self.cfg.uncertainty_max)
        Y_next = np.maximum(state.Y + A_next - self.cfg.aoi_threshold, 0.0)
        Z_next = np.maximum(state.Z + U_next - self.cfg.uncertainty_threshold, 0.0)

        travel_distance = float(np.linalg.norm(y - state.x))
        penalty_parts = self.total_penalty(y, state.x, action)
        exact_value = self.gamma_value(y, action, state, exo, use_true_weights=True, regularized=False) - self.spatial_cost_value(y, state.x)
        reg_value = self.gamma_value(y, action, state, exo, use_true_weights=True, regularized=True) - self.spatial_cost_value(y, state.x)

        state.x = y.copy()
        state.Q = Q_next
        state.A = A_next
        state.U = U_next
        state.Y = Y_next
        state.Z = Z_next
        state.t += 1
        self.update_vehicle_positions(state)
        self.update_target_positions(state)

        out = {
            **penalty_parts,
            "mean_service_rate": float(np.mean(rates)),
            "mean_q_success": float(np.mean(q)),
            "mean_mu_reduction": float(np.mean(mu)),
            "queue_backlog": float(np.sum(Q_next)),
            "mean_aoi": float(np.mean(A_next)),
            "mean_uncertainty": float(np.mean(U_next)),
            "aoi_violation": float(np.mean(np.maximum(A_next - self.cfg.aoi_threshold, 0.0))),
            "uncertainty_violation": float(np.mean(np.maximum(U_next - self.cfg.uncertainty_threshold, 0.0))),
            "stable_queue_indicator": float(np.sum(Q_next) <= 1.05 * prev_queue_plus_arrivals),
            "refresh_rate": float(np.mean(refresh.astype(float))),
            "travel_distance": travel_distance,
            "motion_violation": 0.0,
            "exact_value": float(exact_value),
            "regularized_value_at_exec": float(reg_value),
            "shortlist_size": float(chosen.get("shortlist_size", 0.0)),
            "shortlist_best_exact_value": float(chosen.get("shortlist_best_exact_value", np.nan)) if np.isfinite(chosen.get("shortlist_best_exact_value", np.nan)) else np.nan,
            "grad_norm": float(chosen.get("grad_norm", np.nan)) if np.isfinite(chosen.get("grad_norm", np.nan)) else np.nan,
            "grad_error_rel": float(chosen.get("grad_error_rel", np.nan)) if np.isfinite(chosen.get("grad_error_rel", np.nan)) else np.nan,
            "mean_virtual_aoi_queue": float(np.mean(Y_next)),
            "mean_virtual_unc_queue": float(np.mean(Z_next)),
            "delta_mean_aoi": float(np.mean(A_next) - prev_mean_aoi),
            "delta_mean_uncertainty": float(np.mean(U_next) - prev_mean_unc),
            "tx_power_sum": float(np.sum(p)),
            "sens_power_sum": float(np.sum(u)),
            "bw_sum": float(np.sum(tau)),
            "sens_time_sum": float(np.sum(self.cfg.sensing_time_cost_vec * s)),
        }
        return out

    def simulate(self, controller: str, horizon: int, seed: int = 0, compute_benchmark_gap: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        state = self.reset(seed)
        slot_rows: List[Dict[str, float]] = []
        for t in range(horizon):
            exo = self.build_exogenous(state)
            chosen = self.choose_action(controller, state, exo)
            benchmark_gap = np.nan
            psi_val = np.nan
            if compute_benchmark_gap:
                benchmark = self.benchmark_oracle(state, exo, use_true_weights=True)
                psi_val = benchmark["Psi"]
                benchmark_gap = psi_val - (self.gamma_value(chosen["y"], chosen["action"], state, exo, use_true_weights=True, regularized=False) - self.spatial_cost_value(chosen["y"], state.x))
            metrics = self.apply_action(state, exo, chosen)
            slot_record = {
                "slot": t,
                "controller": controller,
                "x": float(state.x[0]),
                "y": float(state.x[1]),
                "runtime": float(chosen["runtime"]),
                "inner_iterations": int(chosen["inner_iterations"]),
                "execution": str(chosen.get("execution", "na")),
                "candidate_type": str(chosen.get("candidate_type", "na")),
                "benchmark_gap": float(benchmark_gap) if np.isfinite(benchmark_gap) else np.nan,
                "psi_value": float(psi_val) if np.isfinite(psi_val) else np.nan,
                **metrics,
            }
            slot_rows.append(slot_record)
        slot_df = pd.DataFrame(slot_rows)
        summary = slot_df.drop(columns=["slot", "x", "y"]).mean(numeric_only=True).to_dict()
        summary["controller"] = controller
        summary["seed"] = seed
        summary["horizon"] = horizon
        summary_df = pd.DataFrame([summary])
        return summary_df, slot_df

    def simulate_gradient_diagnostic(self, horizon: int, seed: int = 0) -> pd.DataFrame:
        state = self.reset(seed)
        rows: List[Dict[str, float]] = []
        for t in range(horizon):
            exo = self.build_exogenous(state)
            grad_num, _, nit_num = self.numerical_gradient_regularized_value(state, exo, warm_start=state.last_reg_action, use_true_weights=True)
            grad_an, _, nit_an = self.analytic_gradient_regularized_value(state, exo, warm_start=state.last_reg_action, use_true_weights=True)
            err_abs = float(np.linalg.norm(grad_num - grad_an))
            err_rel = err_abs / max(float(np.linalg.norm(grad_an)), 1e-9)
            rows.append({
                "slot": t,
                "seed": seed,
                "grad_num_x": float(grad_num[0]),
                "grad_num_y": float(grad_num[1]),
                "grad_an_x": float(grad_an[0]),
                "grad_an_y": float(grad_an[1]),
                "grad_abs_error": err_abs,
                "grad_rel_error": float(err_rel),
                "num_grad_norm": float(np.linalg.norm(grad_num)),
                "an_grad_norm": float(np.linalg.norm(grad_an)),
                "solver_nit_num": int(nit_num),
                "solver_nit_an": int(nit_an),
            })
            chosen = self.choose_action("proposed_full", state, exo)
            self.apply_action(state, exo, chosen)
        return pd.DataFrame(rows)


def aggregate_runs(dfs: List[pd.DataFrame], group_key: str = "controller") -> pd.DataFrame:
    merged = pd.concat(dfs, ignore_index=True)
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    grouped = merged.groupby(group_key)[numeric_cols].agg(["mean", "std"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    return grouped.reset_index()
