"""Greedy decoder: desired maintenance weeks -> interval-feasible, capacity-aware schedule.

Components are processed in ``Instance.priority_order`` (highest current risk first) and
claim hangar bays, labour and spare parts as they go. For each component the decoder
walks forward in time:

* the next desired week inside the allowed window ``[last + min_interval, last + max_interval]``
  becomes a task, moved to the nearest week that still has capacity (dropped if none has);
* desired weeks too close to the previous task are dropped;
* if no desired week falls inside the window but the component would exceed its maximum
  interval (including at the end of the horizon), a task is forced as late as capacity allows.

The output always satisfies the interval constraints. Capacity is only violated when a
window has no week with room left, and ``evaluate`` reports it.
"""

from __future__ import annotations

from typing import Sequence

from .instance import Instance, Schedule


def decode(inst: Instance, desired: Sequence[Sequence[int]], order: Sequence[int] | None = None) -> Schedule:
    T = inst.horizon
    bays = inst.bays.tolist()
    labor_cap = inst.labor.tolist()
    bays_used = [0] * T
    labor_used = [0.0] * T
    busy = [[False] * T for _ in range(inst.n_aircraft)]
    slack = inst.part_availability.tolist()  # parts left to allocate, cumulative per week
    out: list[tuple[int, ...]] = [()] * inst.n_components

    for c in order if order is not None else inst.priority_order:
        comp = inst.components[c]
        ct = inst.types[comp.type]
        mn, mx, hours = ct.min_interval, ct.max_interval, ct.labor_pm
        aircraft_busy = busy[comp.aircraft]
        part_slack = slack[comp.type]

        def fits(s: int) -> bool:
            if not aircraft_busy[s] and bays_used[s] >= bays[s]:
                return False
            if labor_used[s] + hours > labor_cap[s] + 1e-9:
                return False
            return min(part_slack[s:]) >= 1

        def nearest(target: int, lo: int, hi: int) -> int | None:
            if fits(target):
                return target
            for d in range(1, max(target - lo, hi - target) + 1):
                if target - d >= lo and fits(target - d):
                    return target - d
                if target + d <= hi and fits(target + d):
                    return target + d
            return None

        wanted = sorted({t for t in desired[c] if 0 <= t < T})
        i = 0
        prev = -comp.initial_age
        placed = []
        while True:
            lo, hi = max(0, prev + mn), min(prev + mx, T - 1)
            while i < len(wanted) and wanted[i] < lo:
                i += 1
            if i < len(wanted) and wanted[i] <= hi:
                s = nearest(wanted[i], lo, hi)
                i += 1
                if s is None:  # no room for this optional task; a forced one follows if needed
                    continue
            elif T - prev > mx:  # would run past max_interval: force a task, as late as possible
                s = nearest(hi, lo, hi)
                if s is None:  # no room anywhere in the window: accept the violation
                    s = hi
            else:
                break
            if not aircraft_busy[s]:
                aircraft_busy[s] = True
                bays_used[s] += 1
            labor_used[s] += hours
            for t in range(s, T):
                part_slack[t] -= 1
            placed.append(s)
            prev = s
        out[c] = tuple(placed)
    return tuple(out)
