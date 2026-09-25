"""Single source of truth for scoring a schedule: costs, risk, downtime and constraint violations.

Every method (baselines, GA, NSGA-II, CP-SAT) is re-scored here, so comparisons are like for like.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .instance import Instance, Schedule

VIOLATION_KEYS = ("format", "min_interval", "max_interval", "bays", "labor", "parts")


@dataclass(frozen=True)
class Evaluation:
    pm_cost: float  # preventive tasks
    visit_cost: float  # aircraft-weeks grounded for preventive visits x downtime cost
    failure_cost: float  # expected repairs + failure downtime
    expected_failures: float
    downtime_weeks: float  # preventive visits + expected failure downtime
    n_events: int
    n_visits: int
    violations: dict[str, float] = field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        return self.pm_cost + self.visit_cost + self.failure_cost

    @property
    def total_violation(self) -> float:
        return float(sum(self.violations.values()))

    @property
    def feasible(self) -> bool:
        return self.total_violation <= 1e-9

    def objective(self, w_cost: float = 1.0, w_risk: float = 0.0) -> float:
        """Weighted objective: expected total cost plus a risk-aversion price per expected failure."""
        return w_cost * self.total_cost + w_risk * self.expected_failures

    def as_dict(self) -> dict[str, float]:
        return {
            "total_cost": float(self.total_cost), "pm_cost": float(self.pm_cost),
            "visit_cost": float(self.visit_cost), "failure_cost": float(self.failure_cost),
            "expected_failures": float(self.expected_failures), "downtime_weeks": float(self.downtime_weeks),
            "n_events": self.n_events, "n_visits": self.n_visits,
            "feasible": self.feasible, "total_violation": self.total_violation,
        }


def evaluate(inst: Instance, schedule: Schedule) -> Evaluation:
    """Score ``schedule``.

    Violations, all zero for a feasible schedule:
      format        events out of range, duplicated or unsorted (scored after sorting)
      min_interval  weeks short of the minimum interval between consecutive tasks
      max_interval  weeks beyond the maximum interval, counting the initial age and the
                    age at the end of the horizon
      bays          aircraft over hangar capacity, summed over weeks
      labor         labour hours over capacity, in units of an average task
      parts         worst cumulative spare-part shortfall, summed over part numbers
    """
    if len(schedule) != inst.n_components:
        raise ValueError(f"schedule has {len(schedule)} components, instance has {inst.n_components}")
    T = inst.horizon
    viol = dict.fromkeys(VIOLATION_KEYS, 0.0)
    visits: set[tuple[int, int]] = set()
    labor_used = np.zeros(T)
    consumption = np.zeros((len(inst.types), T), dtype=int)
    pm_cost = failure_cost = failures = failure_downtime = 0.0
    n_events = 0

    for comp, events in zip(inst.components, schedule):
        k = comp.type
        ct = inst.types[k]
        H = inst.hazard_tables[k]
        clean = sorted({t for t in events if 0 <= t < T})
        if list(events) != clean:
            viol["format"] += max(1, len(events) - len(clean))

        prev = -comp.initial_age  # week of the last renewal
        comp_failures = 0.0
        for t in clean:
            gap = t - prev
            if gap < ct.min_interval:
                viol["min_interval"] += ct.min_interval - gap
            elif gap > ct.max_interval:
                viol["max_interval"] += gap - ct.max_interval
            comp_failures += H[gap] - H[max(0, -prev)]
            prev = t
            visits.add((comp.aircraft, t))
            labor_used[t] += ct.labor_pm
            consumption[k, t] += 1
        tail = T - prev
        if tail > ct.max_interval:
            viol["max_interval"] += tail - ct.max_interval
        comp_failures += H[tail] - H[max(0, -prev)]

        n_events += len(clean)
        pm_cost += len(clean) * ct.cost_pm
        failures += comp_failures
        failure_cost += comp_failures * inst.failure_cost(k)
        failure_downtime += comp_failures * ct.downtime_cm

    visits_per_week = np.bincount([t for _, t in visits], minlength=T)
    viol["bays"] = float(np.maximum(visits_per_week - inst.bays, 0).sum())
    mean_task_hours = np.mean([ct.labor_pm for ct in inst.types])
    viol["labor"] = float(np.maximum(labor_used - inst.labor, 0).sum() / mean_task_hours)
    shortfall = np.cumsum(consumption, axis=1) - inst.part_availability
    viol["parts"] = float(np.maximum(shortfall, 0).max(axis=1).sum())

    n_visits = len(visits)
    return Evaluation(
        pm_cost=float(pm_cost),
        visit_cost=float(n_visits * inst.downtime_cost),
        failure_cost=float(failure_cost),
        expected_failures=float(failures),
        downtime_weeks=float(n_visits + failure_downtime),
        n_events=n_events,
        n_visits=n_visits,
        violations=viol,
    )
