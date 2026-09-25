import numpy as np
import pytest

from pmo.instance import ComponentType, generate_instance


@pytest.fixture
def tiny():
    """1 aircraft, 2 component types, 7 weeks: small enough to enumerate every schedule."""
    types = (
        ComponentType("A", beta=2.5, eta=5, cost_pm=1_000, cost_cm=6_000, downtime_cm=0.5,
                      labor_pm=10, min_interval=2, max_interval=5),
        ComponentType("B", beta=3.0, eta=6, cost_pm=1_500, cost_cm=9_000, downtime_cm=1.0,
                      labor_pm=12, min_interval=2, max_interval=6),
    )
    return generate_instance(n_aircraft=1, types=types, horizon=7, downtime_cost=2_000, bays=1, labor=15,
                             blackouts=((3, 4, 0),), delivery_period=7, seed=3, name="tiny")
