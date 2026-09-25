from pmo.instance import small_instance
from pmo.nsga2 import NSGA2Config, ParetoArchive, hypervolume_2d, run_nsga2


def test_hypervolume_2d():
    assert hypervolume_2d([(1, 3), (2, 2), (3, 1)], (4, 4)) == 1 * 3 + 1 * 2 + 1 * 1  # staircase: 3 + 2 + 1
    assert hypervolume_2d([], (4, 4)) == 0


def test_front_is_feasible_and_non_dominated():
    res = run_nsga2(small_instance(0), NSGA2Config(pop_size=40, generations=30, seed=1))
    pts = [(e.total_cost, e.expected_failures) for _, e in res.front]
    assert pts and all(e.feasible for _, e in res.front)
    for i, (c1, f1) in enumerate(pts):
        for j, (c2, f2) in enumerate(pts):
            assert i == j or not (c2 <= c1 and f2 <= f1)
    hv = [h["hypervolume"] for h in res.history]
    assert all(b >= a - 1e-9 for a, b in zip(hv, hv[1:]))  # archive only improves
