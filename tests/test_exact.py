import itertools

import pytest

from pmo.evaluate import evaluate
from pmo.exact import solve_exact
from pmo.ga import GAConfig, run_ga
from pmo.instance import small_instance


def _brute_force(inst, w_risk=0.0):
    T = inst.horizon
    options = [
        [tuple(w for w, bit in zip(range(T), bits) if bit) for bits in itertools.product((0, 1), repeat=T)]
        for _ in inst.components
    ]
    best = None
    for schedule in itertools.product(*options):
        ev = evaluate(inst, schedule)
        if ev.feasible:
            obj = ev.objective(1.0, w_risk)
            if best is None or obj < best:
                best = obj
    return best


@pytest.mark.parametrize("w_risk", [0.0, 5_000.0])
def test_cpsat_matches_brute_force(tiny, w_risk):
    r = solve_exact(tiny, w_risk=w_risk, time_limit=30, workers=4)
    assert r.optimal and r.evaluation.feasible
    assert r.objective == pytest.approx(_brute_force(tiny, w_risk), rel=1e-9)


def test_cpsat_objective_matches_evaluator():
    inst = small_instance(1)
    r = solve_exact(inst, time_limit=60, workers=4)
    assert r.optimal and r.evaluation.feasible
    # coefficients are rounded to 1/scale dollars; one rounding per arc/visit variable at most
    assert r.solver_objective == pytest.approx(r.objective, abs=0.05 * (r.evaluation.n_events * 2 + 50))
    assert r.bound <= r.solver_objective + 1e-6


def test_ga_close_to_optimum_on_small_instance():
    inst = small_instance(0)
    opt = solve_exact(inst, time_limit=60, workers=4).objective
    ga = run_ga(inst, GAConfig(generations=150, seed=0))
    assert ga.objective >= opt - 1e-6
    assert ga.objective <= 1.05 * opt
