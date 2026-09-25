import numpy as np
import pytest

from pmo.evaluate import evaluate
from pmo.instance import ComponentType, generate_instance
from pmo.risk import cumulative_hazard, optimal_interval


def test_optimal_interval_minimises_cost_rate():
    beta, eta, cp, cf = 2.5, 40.0, 35_000, 80_000
    i_star = optimal_interval(beta, eta, cp, cf)
    grid = np.linspace(5, 80, 2000)
    rate = (cp + cf * cumulative_hazard(grid, beta, eta)) / grid
    assert abs(grid[rate.argmin()] - i_star) < 0.1


def test_hand_computed_single_component():
    ct = ComponentType("X", beta=2.0, eta=10, cost_pm=100, cost_cm=1_000, downtime_cm=2.0,
                       labor_pm=5, min_interval=2, max_interval=8)
    inst = generate_instance(n_aircraft=1, types=(ct,), horizon=10, downtime_cost=50, bays=1, labor=10,
                             blackouts=(), seed=0)
    age = inst.components[0].initial_age
    ev = evaluate(inst, ((4,),))
    # segments: age -> 4 + age (before the task), then 0 -> 6 (week 4 to the horizon end)
    failures = ((4 + age) / 10) ** 2 - (age / 10) ** 2 + (6 / 10) ** 2
    assert ev.expected_failures == pytest.approx(failures)
    assert ev.pm_cost == 100 and ev.visit_cost == 50 and ev.n_visits == 1
    assert ev.failure_cost == pytest.approx(failures * (1_000 + 2.0 * 50))
    assert ev.downtime_weeks == pytest.approx(1 + 2.0 * failures)
    assert ev.total_cost == pytest.approx(100 + 50 + failures * 1_100)


def test_shared_visit_is_charged_once(tiny):
    both_week_5 = evaluate(tiny, ((5,), (5,)))
    split = evaluate(tiny, ((5,), (6,)))
    assert both_week_5.n_visits == 1 and split.n_visits == 2
    assert split.visit_cost - both_week_5.visit_cost == tiny.downtime_cost


@pytest.mark.parametrize("schedule, key", [
    (((1, 2), (5,)), "min_interval"),  # tasks 1 week apart, min is 2
    (((), (5,)), "max_interval"),  # component A never maintained over 7 weeks + its age
    (((3,), (5,)), "bays"),  # week 3 has zero bays
    (((5, 5), (5,)), "format"),
])
def test_violations_are_detected(tiny, schedule, key):
    ev = evaluate(tiny, schedule)
    assert ev.violations[key] > 0 and not ev.feasible


def test_labor_and_parts_violations():
    ct = ComponentType("X", beta=2.0, eta=10, cost_pm=100, cost_cm=1_000, downtime_cm=1.0,
                       labor_pm=8, min_interval=1, max_interval=10)
    inst = generate_instance(n_aircraft=3, types=(ct,), horizon=10, bays=3, labor=10, blackouts=(), seed=0)
    ev = evaluate(inst, ((2,), (2,), ()))
    assert ev.violations["labor"] > 0
    stock = int(inst.part_availability[0, -1])
    greedy = tuple(tuple(range(0, 10)) for _ in range(3))  # 30 tasks, far above the planned supply
    assert 30 > stock and evaluate(inst, greedy).violations["parts"] > 0
