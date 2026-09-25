"""Problem instance (fleet, component types, shop capacity, spare parts) and a seeded generator.

Time is discrete: week ``t`` in ``0..horizon-1``. Maintaining a component in week ``t``
grounds its aircraft for that week (one hangar visit, shared by every component of the
same aircraft serviced that week), consumes one spare part of the component's type and
the type's labour hours, and renews the component.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Sequence

import numpy as np

from .risk import cumulative_hazard, hazard_rate, optimal_interval

Schedule = tuple[tuple[int, ...], ...]
"""Maintenance weeks per component, index-aligned with ``Instance.components``, strictly increasing."""


@dataclass(frozen=True)
class ComponentType:
    name: str
    beta: float  # Weibull shape (> 1 means wear-out)
    eta: float  # Weibull scale, weeks
    cost_pm: float  # preventive task cost, $
    cost_cm: float  # corrective (failure) repair cost, $, excluding aircraft downtime
    downtime_cm: float  # aircraft weeks lost per failure
    labor_pm: float  # labour hours per preventive task
    min_interval: int  # weeks between consecutive preventive tasks
    max_interval: int


@dataclass(frozen=True)
class Component:
    aircraft: int
    type: int
    initial_age: int  # weeks since the last preventive task at t = 0


@dataclass(frozen=True, eq=False)
class Instance:
    name: str
    horizon: int
    n_aircraft: int
    types: tuple[ComponentType, ...]
    components: tuple[Component, ...]
    downtime_cost: float  # $ per aircraft-week on the ground
    bays: np.ndarray  # (T,) aircraft the hangar can take each week
    labor: np.ndarray  # (T,) labour hours available each week
    initial_stock: np.ndarray  # (K,) spare parts on hand at t = 0, one part number per type
    deliveries: np.ndarray  # (K, T) parts arriving at the start of each week

    def __post_init__(self):
        T, K = self.horizon, len(self.types)
        if self.bays.shape != (T,) or self.labor.shape != (T,):
            raise ValueError("bays and labor must have shape (horizon,)")
        if self.initial_stock.shape != (K,) or self.deliveries.shape != (K, T):
            raise ValueError("initial_stock must be (K,) and deliveries (K, horizon)")
        for c in self.components:
            ct = self.types[c.type]
            if not 0 <= c.initial_age <= ct.max_interval or not 0 <= c.aircraft < self.n_aircraft:
                raise ValueError(f"invalid component {c}")
            if ct.min_interval > ct.max_interval:
                raise ValueError(f"{ct.name}: min_interval > max_interval")

    @property
    def n_components(self) -> int:
        return len(self.components)

    def failure_cost(self, k: int) -> float:
        """Full cost of one failure of type ``k``: repair plus aircraft downtime."""
        ct = self.types[k]
        return ct.cost_cm + ct.downtime_cm * self.downtime_cost

    @cached_property
    def part_availability(self) -> np.ndarray:
        """(K, T) cumulative parts available up to and including week t."""
        return self.initial_stock[:, None] + np.cumsum(self.deliveries, axis=1)

    @cached_property
    def hazard_tables(self) -> tuple[np.ndarray, ...]:
        """Cumulative hazard H_k(age) for every integer age a schedule can produce."""
        longest = self.horizon + max(c.initial_age for c in self.components) + 1
        ages = np.arange(longest)
        return tuple(cumulative_hazard(ages, ct.beta, ct.eta) for ct in self.types)

    @cached_property
    def priority_order(self) -> tuple[int, ...]:
        """Components by descending current expected failure-cost rate.

        The schedule decoder hands out scarce hangar slots, labour and parts in this
        order, so higher-risk components are served first.
        """
        urgency = [
            float(hazard_rate(c.initial_age, self.types[c.type].beta, self.types[c.type].eta))
            * self.failure_cost(c.type)
            for c in self.components
        ]
        return tuple(sorted(range(self.n_components), key=lambda i: (-urgency[i], i)))

    def optimal_intervals(self) -> tuple[int, ...]:
        """Per-type cost-optimal fixed interval (standalone visit), clipped to its bounds."""
        out = []
        for k, ct in enumerate(self.types):
            i_star = optimal_interval(ct.beta, ct.eta, ct.cost_pm + self.downtime_cost, self.failure_cost(k))
            out.append(int(np.clip(round(i_star), ct.min_interval, ct.max_interval)))
        return tuple(out)

    def with_failure_cost_scale(self, scale: float) -> Instance:
        types = tuple(replace(ct, cost_cm=ct.cost_cm * scale) for ct in self.types)
        return replace(self, name=f"{self.name}-fc{scale:g}", types=types)


def fixed_interval_events(initial_age: int, interval: int, horizon: int) -> list[int]:
    """Weeks at which a component is maintained every ``interval`` weeks of age."""
    t = max(0, interval - initial_age)
    events = []
    while t < horizon:
        events.append(t)
        t += interval
    return events


# Illustrative (not OEM) parameters for four line-replaceable units of a regional jet.
FLEET_TYPES = (
    ComponentType("APU", beta=2.5, eta=40, cost_pm=15_000, cost_cm=60_000, downtime_cm=1.0,
                  labor_pm=30, min_interval=8, max_interval=36),
    ComponentType("Landing gear", beta=3.0, eta=60, cost_pm=25_000, cost_cm=150_000, downtime_cm=2.0,
                  labor_pm=50, min_interval=12, max_interval=52),
    ComponentType("Hydraulic pump", beta=1.8, eta=30, cost_pm=5_000, cost_cm=60_000, downtime_cm=0.5,
                  labor_pm=12, min_interval=4, max_interval=26),
    ComponentType("Brakes", beta=3.5, eta=26, cost_pm=8_000, cost_cm=40_000, downtime_cm=0.5,
                  labor_pm=16, min_interval=6, max_interval=20),
)


def generate_instance(
    *,
    n_aircraft: int = 10,
    types: Sequence[ComponentType] = FLEET_TYPES,
    horizon: int = 52,
    downtime_cost: float = 20_000,
    bays: int = 2,
    labor: float = 120,
    blackouts: Sequence[tuple[int, int, int]] = ((22, 26, 1), (50, 52, 0)),
    parts_margin: float = 1.15,
    delivery_period: int = 13,
    age_fraction: float = 0.6,
    seed: int = 0,
    name: str | None = None,
) -> Instance:
    """Synthetic fleet: every aircraft carries one component of each type.

    ``blackouts`` are ``(start, end, bays)`` week ranges with reduced hangar capacity
    (peak season, holidays). Spare parts arrive every ``delivery_period`` weeks, sized to
    the demand of a fixed-interval plan plus ``parts_margin``, so parts bind whenever a
    schedule maintains much more often than that plan.
    """
    rng = np.random.default_rng(seed)
    types = tuple(types)
    components = tuple(
        Component(a, k, int(rng.integers(0, int(age_fraction * ct.max_interval) + 1)))
        for a in range(n_aircraft)
        for k, ct in enumerate(types)
    )
    bays_arr = np.full(horizon, bays, dtype=int)
    for start, end, cap in blackouts:
        bays_arr[start:end] = cap
    labor_arr = np.full(horizon, float(labor))

    # Size deliveries to the more part-hungry of the two fixed-interval baselines.
    draft = Instance(name or "draft", horizon, n_aircraft, types, components, downtime_cost, bays_arr,
                     labor_arr, np.zeros(len(types), dtype=int), np.zeros((len(types), horizon), dtype=int))
    planning = [min(i_opt, math.floor(0.8 * ct.max_interval))
                for i_opt, ct in zip(draft.optimal_intervals(), types)]
    demand = np.zeros((len(types), horizon), dtype=int)
    for c in components:
        for t in fixed_interval_events(c.initial_age, planning[c.type], horizon):
            demand[c.type, t] += 1
    deliveries = np.zeros_like(demand)
    for start in range(0, horizon, delivery_period):
        window = demand[:, start:start + delivery_period].sum(axis=1)
        deliveries[:, start] = np.ceil(parts_margin * window).astype(int)
    initial_stock = np.ones(len(types), dtype=int)

    return replace(draft, name=name or f"fleet{n_aircraft}x{len(types)}-T{horizon}-s{seed}",
                   initial_stock=initial_stock, deliveries=deliveries)


def default_instance(seed: int = 0) -> Instance:
    """The main case study: 10 aircraft x 4 component types over one year."""
    return generate_instance(seed=seed)


def small_instance(seed: int = 0) -> Instance:
    """3 aircraft x 3 types over 26 weeks, small enough for CP-SAT to prove optimality."""
    return generate_instance(
        n_aircraft=3, types=(FLEET_TYPES[0], FLEET_TYPES[2], FLEET_TYPES[3]), horizon=26,
        bays=1, labor=60, blackouts=((12, 14, 0),), seed=seed, name=f"small3x3-T26-s{seed}",
    )
