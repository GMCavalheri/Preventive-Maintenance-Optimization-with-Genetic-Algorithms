import numpy as np
import pytest

from pmo.baseline import BASELINES, fixed_interval_schedule
from pmo.evaluate import evaluate
from pmo.instance import default_instance, small_instance


def test_generator_is_deterministic():
    a, b = default_instance(7), default_instance(7)
    assert a.components == b.components
    assert np.array_equal(a.deliveries, b.deliveries) and np.array_equal(a.bays, b.bays)
    assert default_instance(8).components != a.components


@pytest.mark.parametrize("maker", [default_instance, small_instance])
@pytest.mark.parametrize("seed", range(10))
def test_baselines_are_feasible(maker, seed):
    inst = maker(seed)
    for kw in BASELINES.values():
        ev = evaluate(inst, fixed_interval_schedule(inst, **kw))
        assert ev.feasible, ev.violations
