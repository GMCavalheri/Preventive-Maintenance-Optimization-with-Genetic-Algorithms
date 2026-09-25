"""Sensitivity of the GA solution to the risk weight and to the price of failures."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import pandas as pd

from .ga import GAConfig, GAResult, run_ga_many
from .instance import Instance, Schedule

RISK_WEIGHTS = (0.0, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6)  # $ of risk aversion per expected failure
FAILURE_COST_SCALES = (0.5, 1.0, 2.0, 4.0)  # multiplier on corrective repair cost


def schedule_distance(a: Schedule, b: Schedule) -> int:
    """Tasks present in one schedule but not the other (same component and week)."""
    return sum(len(set(x) ^ set(y)) for x, y in zip(a, b))


def _row(result: GAResult, **keys) -> dict:
    return {**keys, "objective": result.objective, **result.evaluation.as_dict(), "runtime_s": result.runtime}


def risk_weight_sweep(inst: Instance, weights: Sequence[float] = RISK_WEIGHTS, seeds: Sequence[int] = (0, 1, 2),
                      cfg: GAConfig | None = None, workers: int = 1) -> tuple[pd.DataFrame, dict]:
    """One GA run per (risk weight, seed). ``tasks_changed`` compares with weight 0, same seed.

    Read ``tasks_changed`` against the distance between two seeds at the same weight: many
    distinct schedules are near-optimal, so most of it is run-to-run variation, not the weight.
    """
    cfg = cfg or GAConfig()
    keys = [(w, s) for w in weights for s in seeds]
    results = run_ga_many([(inst, replace(cfg, w_risk=w, seed=s)) for w, s in keys], workers)
    by_key = dict(zip(keys, results))
    rows = []
    for (w, s), r in by_key.items():
        reference = by_key.get((weights[0], s))
        rows.append(_row(r, w_risk=w, seed=s,
                         tasks_changed=schedule_distance(r.schedule, reference.schedule) if reference else None))
    return pd.DataFrame(rows), by_key


def failure_cost_sweep(inst: Instance, scales: Sequence[float] = FAILURE_COST_SCALES, seeds: Sequence[int] = (0, 1, 2),
                       cfg: GAConfig | None = None, workers: int = 1) -> pd.DataFrame:
    """Re-optimise with corrective repair costs scaled; capacity and parts stay as generated."""
    cfg = cfg or GAConfig()
    keys = [(f, s) for f in scales for s in seeds]
    results = run_ga_many([(inst.with_failure_cost_scale(f), replace(cfg, seed=s)) for f, s in keys], workers)
    rows = []
    for (f, s), r in zip(keys, results):
        per_type = {f"tasks_{t.name}": 0 for t in inst.types}
        for comp, events in zip(inst.components, r.schedule):
            per_type[f"tasks_{inst.types[comp.type].name}"] += len(events)
        rows.append({**_row(r, failure_cost_scale=f, seed=s), **per_type})
    return pd.DataFrame(rows)
