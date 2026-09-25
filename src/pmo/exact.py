"""Exact model with OR-Tools CP-SAT.

Each component's schedule is a path in a time-indexed DAG:
``START (week -initial_age) -> t1 -> t2 -> ... -> END (week T)``. An arc ``s -> t`` means
"consecutive renewals at s and t" and exists only when the gap respects the component's
min/max interval (the END arc only checks the max). The expected failures on a segment
depend only on its two endpoints, so each arc carries a constant cost and the whole
objective is linear, with no linearisation error. Hangar visits, bays, labour and cumulative
spare-part constraints couple the paths.

CP-SAT needs integer coefficients: costs are multiplied by ``scale`` and rounded, and the
returned schedule is re-scored with ``evaluate`` so the reported cost is exact.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass

from ortools.sat.python import cp_model

from .evaluate import Evaluation, evaluate
from .instance import Instance, Schedule


@dataclass
class ExactResult:
    schedule: Schedule | None
    evaluation: Evaluation | None
    status: str
    objective: float  # the evaluated objective of the returned schedule
    solver_objective: float  # CP-SAT's objective (rounded coefficients)
    bound: float  # proven lower bound on the optimal objective
    optimal: bool
    runtime: float
    n_arcs: int

    @property
    def gap(self) -> float:
        return (self.solver_objective - self.bound) / max(abs(self.solver_objective), 1e-9)


def solve_exact(
    inst: Instance,
    *,
    w_cost: float = 1.0,
    w_risk: float = 0.0,
    time_limit: float = 60.0,
    workers: int = 8,
    scale: float = 10.0,
    hint: Schedule | None = None,
    log: bool = False,
) -> ExactResult:
    start_time = time.perf_counter()
    T = inst.horizon
    model = cp_model.CpModel()
    objective = []
    arcs_of: list[dict[tuple[int, int], cp_model.IntVar]] = []
    task: dict[tuple[int, int], list] = {}  # (c, t) -> incoming arc vars (their sum is x[c, t])
    START, END = "start", T  # START is keyed apart from week 0 (a component can have age 0)

    for c, comp in enumerate(inst.components):
        ct = inst.types[comp.type]
        H = inst.hazard_tables[comp.type]
        per_failure = w_cost * inst.failure_cost(comp.type) + w_risk
        arcs: dict[tuple, cp_model.IntVar] = {}
        out_arcs, in_arcs = defaultdict(list), defaultdict(list)

        def add_arc(u, v: int):
            u_week = -comp.initial_age if u == START else u
            gap = v - u_week
            failures = H[gap] - H[max(0, -u_week)]
            cost = per_failure * failures + (0.0 if v == END else w_cost * ct.cost_pm)
            var = model.NewBoolVar(f"a{c}_{u}_{v}")
            arcs[u, v] = var
            out_arcs[u].append(var)
            in_arcs[v].append(var)
            objective.append((round(scale * cost), var))

        for u in [START, *range(T)]:
            u_week = -comp.initial_age if u == START else u
            for v in range(max(0, u_week + ct.min_interval), min(T - 1, u_week + ct.max_interval) + 1):
                add_arc(u, v)
            if END - u_week <= ct.max_interval:
                add_arc(u, END)

        model.AddExactlyOne(out_arcs[START])
        model.AddExactlyOne(in_arcs[END])
        for t in range(T):
            if in_arcs[t] or out_arcs[t]:
                model.Add(sum(in_arcs[t]) == sum(out_arcs[t]))
                if in_arcs[t]:
                    task[c, t] = in_arcs[t]
        arcs_of.append(arcs)

    # Hangar visits: aircraft a is in the hangar in week t iff any of its components is serviced.
    visits_by_week = defaultdict(list)
    for a in range(inst.n_aircraft):
        comps = [c for c, comp in enumerate(inst.components) if comp.aircraft == a]
        for t in range(T):
            xs = [sum(task[c, t]) for c in comps if (c, t) in task]
            if not xs:
                continue
            v = model.NewBoolVar(f"visit{a}_{t}")
            for x in xs:
                model.Add(x <= v)
            model.Add(v <= sum(xs))
            visits_by_week[t].append(v)
            objective.append((round(scale * w_cost * inst.downtime_cost), v))

    for t in range(T):
        if visits_by_week[t]:
            model.Add(sum(visits_by_week[t]) <= int(inst.bays[t]))
        load = [(round(100 * inst.types[inst.components[c].type].labor_pm), sum(task[c, t]))
                for c in range(inst.n_components) if (c, t) in task]
        if load:
            model.Add(sum(h * x for h, x in load) <= round(100 * inst.labor[t]))

    availability = inst.part_availability
    for k in range(len(inst.types)):
        used = []
        for t in range(T):
            used.extend(sum(task[c, t]) for c, comp in enumerate(inst.components)
                        if comp.type == k and (c, t) in task)
            if used:
                model.Add(sum(used) <= int(availability[k, t]))

    model.Minimize(sum(coef * var for coef, var in objective))

    if hint is not None:
        for c, events in enumerate(hint):
            path = list(zip([START, *events], [*events, END]))
            chosen = set(path)
            for key, var in arcs_of[c].items():
                model.AddHint(var, key in chosen)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.log_search_progress = log
    status = solver.Solve(model)
    runtime = time.perf_counter() - start_time
    n_arcs = sum(len(a) for a in arcs_of)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ExactResult(None, None, solver.StatusName(status), float("nan"), float("nan"),
                           solver.BestObjectiveBound() / scale, False, runtime, n_arcs)

    schedule = []
    for c, arcs in enumerate(arcs_of):
        succ = {u: v for (u, v), var in arcs.items() if solver.Value(var)}
        events, node = [], START
        while succ[node] != END:
            node = succ[node]
            events.append(node)
        schedule.append(tuple(events))
    schedule = tuple(schedule)
    ev = evaluate(inst, schedule)
    return ExactResult(
        schedule=schedule, evaluation=ev, status=solver.StatusName(status),
        objective=ev.objective(w_cost, w_risk), solver_objective=solver.ObjectiveValue() / scale,
        bound=solver.BestObjectiveBound() / scale, optimal=status == cp_model.OPTIMAL,
        runtime=runtime, n_arcs=n_arcs,
    )
