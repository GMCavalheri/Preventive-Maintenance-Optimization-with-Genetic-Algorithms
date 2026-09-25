"""Weibull failure model under minimal repair.

A failed component is repaired to its condition just before the failure (minimal
repair), so its failures follow a non-homogeneous Poisson process with the Weibull
hazard as intensity. Preventive maintenance renews the component (as good as new).
The expected number of failures while a component ages from ``a`` to ``b`` weeks is
therefore ``H(b) - H(a)``, where ``H(t) = (t / eta) ** beta`` is the cumulative hazard.
"""

from __future__ import annotations

import numpy as np


def cumulative_hazard(age, beta: float, eta: float) -> np.ndarray:
    return (np.asarray(age, dtype=float) / eta) ** beta


def hazard_rate(age, beta: float, eta: float) -> np.ndarray:
    return (beta / eta) * (np.asarray(age, dtype=float) / eta) ** (beta - 1)


def optimal_interval(beta: float, eta: float, cost_pm: float, cost_failure: float) -> float:
    """Periodic PM interval minimising the long-run cost rate under minimal repair.

    Minimises ``(cost_pm + cost_failure * H(I)) / I``, which gives
    ``I* = eta * (cost_pm / (cost_failure * (beta - 1))) ** (1 / beta)``.
    Without wear-out (``beta <= 1``) preventive maintenance never pays off.
    """
    if beta <= 1:
        return float("inf")
    return eta * (cost_pm / (cost_failure * (beta - 1))) ** (1 / beta)
