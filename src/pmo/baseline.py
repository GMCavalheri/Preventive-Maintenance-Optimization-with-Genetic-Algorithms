"""Fixed-interval baselines, staggered first-fit around hangar, labour and parts capacity."""

from __future__ import annotations

import math

from .decoder import decode
from .instance import Instance, Schedule, fixed_interval_events


def type_intervals(inst: Instance, rule: str = "optimal", fraction: float = 0.8) -> tuple[int, ...]:
    """``optimal``: per-type cost-optimal interval. ``fraction``: ``fraction`` x max interval."""
    if rule == "optimal":
        return inst.optimal_intervals()
    if rule == "fraction":
        return tuple(max(ct.min_interval, math.floor(fraction * ct.max_interval)) for ct in inst.types)
    raise ValueError(f"unknown rule {rule!r}")


def desired_fixed_interval(inst: Instance, intervals: tuple[int, ...]) -> list[list[int]]:
    return [fixed_interval_events(c.initial_age, intervals[c.type], inst.horizon) for c in inst.components]


def fixed_interval_schedule(inst: Instance, rule: str = "optimal", fraction: float = 0.8) -> Schedule:
    return decode(inst, desired_fixed_interval(inst, type_intervals(inst, rule, fraction)))


BASELINES = {
    "Fixed interval (80% of max)": dict(rule="fraction", fraction=0.8),
    "Fixed interval (cost-optimal)": dict(rule="optimal"),
}
