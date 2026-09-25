"""Best-response local search: re-plan one component with everyone else held fixed.

Uses the same time-indexed DAG as the exact model (``exact.py``), restricted to a single
component, so the optimal plan is a shortest path solved by dynamic programming. A week
costs no extra hangar visit when the aircraft is already in for another component, which
is what lets the GA consolidate visits. Weeks without bay, labour or spare-part room
(given the other components) are excluded; parts are checked per task, so a plan that
needs several parts can still overdraw, and the decoder repairs that.
"""

from __future__ import annotations

import math
from typing import Sequence

from .instance import Instance


def best_response(inst: Instance, plan: Sequence[Sequence[int]], c: int,
                  w_cost: float = 1.0, w_risk: float = 0.0) -> list[int] | None:
    T = inst.horizon
    comp = inst.components[c]
    ct = inst.types[comp.type]
    H = inst.hazard_tables[comp.type]

    aircraft_in = [False] * T  # aircraft already in the hangar that week for another component
    aircraft_weeks: dict[int, set[int]] = {}
    labor_used = [0.0] * T
    used_parts = [0] * T
    for other, events in enumerate(plan):
        if other == c:
            continue
        o = inst.components[other]
        hours = inst.types[o.type].labor_pm
        for t in events:
            if 0 <= t < T:
                aircraft_weeks.setdefault(o.aircraft, set()).add(t)
                labor_used[t] += hours
                if o.type == comp.type:
                    used_parts[t] += 1
    for t in aircraft_weeks.get(comp.aircraft, ()):
        aircraft_in[t] = True
    bays_used = [0] * T
    for weeks in aircraft_weeks.values():
        for t in weeks:
            bays_used[t] += 1

    # Parts: week t is usable if one more part fits at t and in every later week.
    availability = inst.part_availability[comp.type]
    slack, cum = [0] * T, 0
    for t in range(T):
        cum += used_parts[t]
        slack[t] = availability[t] - cum
    min_future = [0] * T
    running = math.inf
    for t in range(T - 1, -1, -1):
        running = min(running, slack[t])
        min_future[t] = running

    allowed = [
        (aircraft_in[t] or bays_used[t] < inst.bays[t])
        and labor_used[t] + ct.labor_pm <= inst.labor[t] + 1e-9
        and min_future[t] >= 1
        for t in range(T)
    ]
    task_cost = [
        w_cost * (ct.cost_pm + (0.0 if aircraft_in[t] else inst.downtime_cost)) for t in range(T)
    ]
    per_failure = w_cost * inst.failure_cost(comp.type) + w_risk
    mn, mx = ct.min_interval, ct.max_interval

    # Nodes: index 0 = START (week -age), 1..T = weeks 0..T-1, T+1 = END (week T).
    week = [-comp.initial_age, *range(T), T]
    dist = [math.inf] * (T + 2)
    parent = [-1] * (T + 2)
    dist[0] = 0.0
    for u in range(T + 1):
        if dist[u] == math.inf:
            continue
        uw = week[u]
        base_h = H[max(0, -uw)]
        for vw in range(max(0, uw + mn), min(T - 1, uw + mx) + 1):
            if not allowed[vw]:
                continue
            d = dist[u] + task_cost[vw] + per_failure * (H[vw - uw] - base_h)
            if d < dist[vw + 1]:
                dist[vw + 1], parent[vw + 1] = d, u
        if T - uw <= mx:
            d = dist[u] + per_failure * (H[T - uw] - base_h)
            if d < dist[T + 1]:
                dist[T + 1], parent[T + 1] = d, u

    if dist[T + 1] == math.inf:
        return None
    events, node = [], parent[T + 1]
    while node > 0:
        events.append(week[node])
        node = parent[node]
    return events[::-1]
