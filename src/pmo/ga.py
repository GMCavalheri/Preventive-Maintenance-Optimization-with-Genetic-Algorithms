"""Weighted-objective genetic algorithm (DEAP).

Encoding: a flat integer vector split into one block per component. Block ``c`` holds
``J_c`` genes, each a desired maintenance week or ``-1`` (no task). ``J_c`` is the most
tasks the component can fit in the horizon, so the number of tasks is itself evolved.

Evaluation decodes the genes into a feasible-by-construction schedule (``decoder.decode``)
and writes that schedule back into the individual (Lamarckian repair). Anything the
decoder cannot fix is priced with a large penalty per unit of violation.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Sequence

from deap import algorithms, base, creator, tools

from .baseline import desired_fixed_interval, type_intervals
from .decoder import decode
from .evaluate import Evaluation, evaluate
from .instance import Instance, Schedule
from .local_search import best_response

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

SKIP = -1


class Encoding:
    def __init__(self, inst: Instance):
        self.inst = inst
        self.lengths = [
            (inst.horizon - 1) // inst.types[c.type].min_interval + 1 for c in inst.components
        ]
        self.offsets = [0]
        for n in self.lengths:
            self.offsets.append(self.offsets[-1] + n)

    @property
    def size(self) -> int:
        return self.offsets[-1]

    def block(self, c: int) -> slice:
        return slice(self.offsets[c], self.offsets[c + 1])

    def split(self, genes: Sequence[int]) -> list[list[int]]:
        return [[g for g in genes[self.block(c)] if g != SKIP] for c in range(len(self.lengths))]

    def encode(self, schedule: Sequence[Sequence[int]]) -> list[int]:
        genes: list[int] = []
        for events, n in zip(schedule, self.lengths):
            events = sorted(events)[:n]
            genes.extend(events + [SKIP] * (n - len(events)))
        return genes

    @cached_property
    def aircraft_components(self) -> list[list[int]]:
        groups = [[] for _ in range(self.inst.n_aircraft)]
        for i, comp in enumerate(self.inst.components):
            groups[comp.aircraft].append(i)
        return groups


# ---------------------------------------------------------------- variation operators

def random_desired(inst: Instance, c: int) -> list[int]:
    """A jittered fixed-interval plan with a random interval: plausible but diverse."""
    comp = inst.components[c]
    ct = inst.types[comp.type]
    interval = random.randint(ct.min_interval, ct.max_interval)
    t = max(0, interval - comp.initial_age) + random.randint(-2, 2)
    events = []
    while t < inst.horizon:
        events.append(min(max(t + random.randint(-2, 2), 0), inst.horizon - 1))
        t += interval
    return events


def cx_blocks(ind1, ind2, enc: Encoding, indpb: float = 0.5):
    """Uniform crossover over whole component blocks or, half the time, whole aircraft.

    Swapping entire blocks keeps each parent's per-component timing intact; swapping
    aircraft also keeps the grouping of tasks into shared hangar visits.
    """
    if random.random() < 0.5:
        units = [[c] for c in range(len(enc.lengths))]
    else:
        units = enc.aircraft_components
    for unit in units:
        if random.random() < indpb:
            for c in unit:
                sl = enc.block(c)
                ind1[sl], ind2[sl] = ind2[sl], ind1[sl]
    return ind1, ind2


OPERATORS = ("shift", "add", "remove", "align", "resample", "move_visit", "merge", "reoptimize")
OPERATOR_WEIGHTS = (4, 2, 2, 3, 1, 2, 2, 2)


def mutate(ind, enc: Encoding, w_cost: float = 1.0, w_risk: float | Callable[[], float] = 0.0,
           max_shift: int = 3, align_window: int = 4):
    """Apply 1-3 random edits to the desired weeks.

    shift     move one task by up to ``max_shift`` weeks
    add       add a task (fills a skip gene)
    remove    drop a task
    align     move nearby tasks of the same aircraft into one week (shared visit)
    resample  replace one component's plan with a fresh random fixed-interval plan
    move_visit  move every task of one hangar visit (same aircraft, same week) by a few weeks
    merge       fold one hangar visit into the aircraft's nearest other visit
    reoptimize  best response: the optimal plan for one component given all the others
                (``local_search.best_response``)

    ``w_risk`` may be a callable drawing a weight per call, which NSGA-II uses to push
    different offspring toward different parts of the front.
    """
    inst = enc.inst
    T = inst.horizon
    for _ in range(random.randint(1, 3)):
        op = random.choices(OPERATORS, weights=OPERATOR_WEIGHTS)[0]
        c = random.randrange(len(enc.lengths))
        sl = enc.block(c)
        idx = list(range(sl.start, sl.stop))
        used = [i for i in idx if ind[i] != SKIP]
        if op == "shift" and used:
            i = random.choice(used)
            ind[i] = min(max(ind[i] + random.choice((-1, 1)) * random.randint(1, max_shift), 0), T - 1)
        elif op == "add":
            free = [i for i in idx if ind[i] == SKIP]
            if free:
                ind[random.choice(free)] = random.randrange(T)
        elif op == "remove" and used:
            ind[random.choice(used)] = SKIP
        elif op == "align" and used:
            anchor = ind[random.choice(used)]
            for other in enc.aircraft_components[inst.components[c].aircraft]:
                for i in range(enc.offsets[other], enc.offsets[other + 1]):
                    if ind[i] != SKIP and 0 < abs(ind[i] - anchor) <= align_window and random.random() < 0.7:
                        ind[i] = anchor
        elif op == "resample":
            events = random_desired(inst, c)[: len(idx)]
            ind[sl] = events + [SKIP] * (len(idx) - len(events))
        elif op in ("move_visit", "merge") and used:
            aircraft = enc.aircraft_components[inst.components[c].aircraft]
            week = ind[random.choice(used)]
            if op == "move_visit":
                target = min(max(week + random.choice((-1, 1)) * random.randint(1, max_shift), 0), T - 1)
            else:
                others = {ind[i] for o in aircraft for i in range(enc.offsets[o], enc.offsets[o + 1])
                          if ind[i] not in (SKIP, week)}
                if not others:
                    continue
                target = min(others, key=lambda t: (abs(t - week), t))
            for o in aircraft:
                for i in range(enc.offsets[o], enc.offsets[o + 1]):
                    if ind[i] == week:
                        ind[i] = target
        elif op == "reoptimize":
            risk = w_risk() if callable(w_risk) else w_risk
            events = best_response(inst, enc.split(ind), c, w_cost, risk)
            if events is not None:
                events = events[: len(idx)]
                ind[sl] = events + [SKIP] * (len(idx) - len(events))
    return (ind,)


# ---------------------------------------------------------------------------- run

@dataclass
class GAConfig:
    pop_size: int = 100
    generations: int = 500
    cx_prob: float = 0.8
    mut_prob: float = 0.6
    tournament: int = 3
    elite: int = 2
    seed_fraction: float = 0.1  # share of the initial population built from fixed-interval plans
    penalty: float = 1e6  # $ per unit of constraint violation
    w_cost: float = 1.0
    w_risk: float = 0.0  # $ of risk aversion per expected failure
    seed: int = 0


@dataclass
class GAResult:
    schedule: Schedule
    evaluation: Evaluation
    objective: float
    history: list[dict] = field(default_factory=list)
    runtime: float = 0.0
    config: GAConfig | None = None


def seed_plans(inst: Instance, n: int) -> list[list[list[int]]]:
    """Fixed-interval plans at both baseline rules and a spread of interval multipliers."""
    plans = [desired_fixed_interval(inst, type_intervals(inst, "optimal")),
             desired_fixed_interval(inst, type_intervals(inst, "fraction"))]
    base_iv = type_intervals(inst, "optimal")
    multipliers = [0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 0.6, 1.4]
    for m in multipliers:
        if len(plans) >= n:
            break
        iv = tuple(int(min(max(round(m * i), ct.min_interval), ct.max_interval))
                   for i, ct in zip(base_iv, inst.types))
        plans.append(desired_fixed_interval(inst, iv))
    return plans[:n]


def make_evaluator(inst: Instance, enc: Encoding, score: Callable[[Evaluation], tuple[float, ...]]):
    """Returns ``evaluate_individual(ind)``: decodes, writes the repair back, sets fitness."""

    def evaluate_individual(ind) -> tuple[Schedule, Evaluation]:
        schedule = decode(inst, enc.split(ind))
        ev = evaluate(inst, schedule)
        ind[:] = enc.encode(schedule)
        ind.fitness.values = score(ev)
        ind.feasible = ev.feasible
        return schedule, ev

    return evaluate_individual


def initial_population(inst: Instance, enc: Encoding, n: int, seed_fraction: float, individual_cls):
    n_seeded = max(2, math.ceil(seed_fraction * n))
    pop = [individual_cls(enc.encode(plan)) for plan in seed_plans(inst, n_seeded)]
    while len(pop) < n:
        pop.append(individual_cls(enc.encode([random_desired(inst, c) for c in range(inst.n_components)])))
    return pop


def run_ga(inst: Instance, cfg: GAConfig | None = None) -> GAResult:
    cfg = cfg or GAConfig()
    random.seed(cfg.seed)
    start = time.perf_counter()
    enc = Encoding(inst)

    def score(ev: Evaluation) -> tuple[float]:
        return (ev.objective(cfg.w_cost, cfg.w_risk) + cfg.penalty * ev.total_violation,)

    evaluate_individual = make_evaluator(inst, enc, score)
    toolbox = base.Toolbox()
    toolbox.register("mate", cx_blocks, enc=enc)
    toolbox.register("mutate", mutate, enc=enc, w_cost=cfg.w_cost, w_risk=cfg.w_risk)
    toolbox.register("select", tools.selTournament, tournsize=cfg.tournament)

    best: tuple[float, Schedule, Evaluation] | None = None
    n_evals = 0

    def evaluate_all(individuals):
        nonlocal best, n_evals
        for ind in individuals:
            if ind.fitness.valid:
                continue
            schedule, ev = evaluate_individual(ind)
            n_evals += 1
            if best is None or ind.fitness.values[0] < best[0]:
                best = (ind.fitness.values[0], schedule, ev)

    pop = initial_population(inst, enc, cfg.pop_size, cfg.seed_fraction, creator.Individual)
    evaluate_all(pop)
    history = []

    def record(gen):
        fits = [ind.fitness.values[0] for ind in pop]
        feasible = [ind.fitness.values[0] for ind in pop if ind.feasible]
        history.append({
            "generation": gen, "evaluations": n_evals, "best": best[0],
            "pop_best": min(fits), "pop_mean_feasible": sum(feasible) / len(feasible) if feasible else math.nan,
            "feasible_share": len(feasible) / len(fits),
        })

    record(0)
    for gen in range(1, cfg.generations + 1):
        elites = [toolbox.clone(ind) for ind in tools.selBest(pop, cfg.elite)]
        offspring = toolbox.select(pop, len(pop) - cfg.elite)
        offspring = algorithms.varAnd(offspring, toolbox, cfg.cx_prob, cfg.mut_prob)
        evaluate_all(offspring)
        pop = elites + offspring
        record(gen)

    objective, schedule, ev = best
    return GAResult(schedule=schedule, evaluation=ev, objective=ev.objective(cfg.w_cost, cfg.w_risk),
                    history=history, runtime=time.perf_counter() - start, config=cfg)


def _run_job(job: tuple[Instance, GAConfig]) -> GAResult:
    return run_ga(*job)


def run_ga_many(jobs: Sequence[tuple[Instance, GAConfig]], workers: int = 1) -> list[GAResult]:
    """Run independent GA jobs, in parallel processes when ``workers > 1``. Order is preserved."""
    if workers <= 1:
        return [_run_job(job) for job in jobs]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_run_job, jobs))
