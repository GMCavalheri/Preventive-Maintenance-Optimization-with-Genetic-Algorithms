"""NSGA-II on (expected total cost, expected failures), reusing the GA's encoding and operators.

Minimising expected cost alone already prices failures, so the front runs from the
cost-optimal schedule toward ever more preventive work that buys fewer failures at a
rising marginal cost, until spare parts and hangar capacity cap it. Every feasible
schedule evaluated goes through a non-dominated archive, which is the reported front.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np
from deap import algorithms, base, creator, tools

from .baseline import desired_fixed_interval, type_intervals
from .evaluate import Evaluation
from .ga import Encoding, cx_blocks, make_evaluator, mutate, random_desired
from .instance import Instance, Schedule

if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    creator.create("IndividualMulti", list, fitness=creator.FitnessMulti)


@dataclass
class NSGA2Config:
    pop_size: int = 100  # multiple of 4 (selTournamentDCD)
    generations: int = 400
    cx_prob: float = 0.8
    mut_prob: float = 0.8
    penalty_cost: float = 1e6  # per unit of violation, on each objective
    penalty_failures: float = 1e3
    risk_weight_range: tuple[float, float] = (1e4, 3e6)  # best-response weights, log-uniform
    seed: int = 0


@dataclass
class NSGA2Result:
    front: list[tuple[Schedule, Evaluation]]  # feasible, non-dominated, sorted by cost
    history: list[dict] = field(default_factory=list)
    reference_point: tuple[float, float] = (math.nan, math.nan)
    runtime: float = 0.0


def hypervolume_2d(points, reference) -> float:
    """Area dominated by ``points`` (both objectives minimised) inside ``reference``."""
    pts = sorted(p for p in points if p[0] < reference[0] and p[1] < reference[1])
    area, best_f2 = 0.0, reference[1]
    for f1, f2 in pts:
        if f2 < best_f2:
            area += (reference[0] - f1) * (best_f2 - f2)
            best_f2 = f2
    return area


class ParetoArchive:
    def __init__(self):
        self.items: list[tuple[float, float, Schedule, Evaluation]] = []

    def add(self, schedule: Schedule, ev: Evaluation) -> None:
        f = (ev.total_cost, ev.expected_failures)
        for c, r, _, _ in self.items:
            if c <= f[0] and r <= f[1]:
                return  # dominated or duplicate
        self.items = [it for it in self.items if not (f[0] <= it[0] and f[1] <= it[1])]
        self.items.append((*f, schedule, ev))

    def points(self) -> list[tuple[float, float]]:
        return [(c, r) for c, r, _, _ in self.items]

    def sorted(self) -> list[tuple[Schedule, Evaluation]]:
        return [(s, e) for _, _, s, e in sorted(self.items, key=lambda it: it[0])]


def spread_plans(inst: Instance, n: int) -> list[list[list[int]]]:
    """Fixed-interval plans from very frequent to very sparse maintenance."""
    base_iv = type_intervals(inst, "optimal")
    plans = [desired_fixed_interval(inst, type_intervals(inst, "fraction"))]
    for m in np.linspace(0.35, 1.4, n - 1):
        iv = tuple(int(min(max(round(m * i), ct.min_interval), ct.max_interval))
                   for i, ct in zip(base_iv, inst.types))
        plans.append(desired_fixed_interval(inst, iv))
    return plans


def run_nsga2(inst: Instance, cfg: NSGA2Config | None = None) -> NSGA2Result:
    cfg = cfg or NSGA2Config()
    if cfg.pop_size % 4:
        raise ValueError("pop_size must be a multiple of 4")
    random.seed(cfg.seed)
    start = time.perf_counter()
    enc = Encoding(inst)
    archive = ParetoArchive()

    def score(ev: Evaluation) -> tuple[float, float]:
        v = ev.total_violation
        return ev.total_cost + cfg.penalty_cost * v, ev.expected_failures + cfg.penalty_failures * v

    evaluate_individual = make_evaluator(inst, enc, score)
    lo, hi = map(math.log10, cfg.risk_weight_range)

    def draw_risk_weight() -> float:
        return 0.0 if random.random() < 0.25 else 10 ** random.uniform(lo, hi)

    toolbox = base.Toolbox()
    toolbox.register("mate", cx_blocks, enc=enc)
    toolbox.register("mutate", mutate, enc=enc, w_cost=1.0, w_risk=draw_risk_weight)

    def evaluate_all(individuals):
        for ind in individuals:
            if not ind.fitness.valid:
                schedule, ev = evaluate_individual(ind)
                if ev.feasible:
                    archive.add(schedule, ev)

    pop = [creator.IndividualMulti(enc.encode(p)) for p in spread_plans(inst, 12)]
    while len(pop) < cfg.pop_size:
        pop.append(creator.IndividualMulti(enc.encode([random_desired(inst, c) for c in range(inst.n_components)])))
    evaluate_all(pop)
    pop = tools.selNSGA2(pop, cfg.pop_size)  # assigns crowding distances

    pts = archive.points()
    reference = (1.05 * max(p[0] for p in pts), 1.05 * max(p[1] for p in pts))
    history = []

    def record(gen):
        history.append({"generation": gen, "hypervolume": hypervolume_2d(archive.points(), reference),
                        "archive_size": len(archive.items)})

    record(0)
    for gen in range(1, cfg.generations + 1):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = algorithms.varAnd(offspring, toolbox, cfg.cx_prob, cfg.mut_prob)
        evaluate_all(offspring)
        pop = tools.selNSGA2(pop + offspring, cfg.pop_size)
        record(gen)

    return NSGA2Result(front=archive.sorted(), history=history, reference_point=reference,
                       runtime=time.perf_counter() - start)
