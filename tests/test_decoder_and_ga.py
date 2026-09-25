import random

from pmo.baseline import BASELINES, fixed_interval_schedule
from pmo.decoder import decode
from pmo.evaluate import evaluate
from pmo.ga import Encoding, GAConfig, mutate, run_ga
from pmo.instance import default_instance, small_instance
from pmo.local_search import best_response


def test_decoder_always_satisfies_intervals():
    inst = default_instance(0)
    rnd = random.Random(0)
    for _ in range(300):
        desired = [[rnd.randrange(-3, inst.horizon + 3) for _ in range(rnd.randrange(0, 9))]
                   for _ in inst.components]
        ev = evaluate(inst, decode(inst, desired))
        assert ev.violations["min_interval"] == ev.violations["max_interval"] == ev.violations["format"] == 0


def test_decoder_keeps_feasible_schedules_unchanged():
    inst = default_instance(0)
    schedule = fixed_interval_schedule(inst)
    assert decode(inst, schedule) == schedule


def test_encoding_round_trip_and_mutation_bounds():
    inst = default_instance(0)
    enc = Encoding(inst)
    schedule = fixed_interval_schedule(inst)
    genes = enc.encode(schedule)
    assert len(genes) == enc.size
    assert tuple(tuple(e) for e in enc.split(genes)) == schedule
    random.seed(0)
    for _ in range(200):
        mutate(genes, enc)
        assert len(genes) == enc.size and all(-1 <= g < inst.horizon for g in genes)


def test_best_response_never_worse_for_its_component():
    inst = default_instance(0)
    schedule = list(fixed_interval_schedule(inst))
    before = evaluate(inst, tuple(schedule)).total_cost
    for c in range(inst.n_components):
        events = best_response(inst, schedule, c)
        if events is None:
            continue
        candidate = schedule[:c] + [tuple(events)] + schedule[c + 1:]
        ev = evaluate(inst, tuple(candidate))
        if ev.feasible:
            assert ev.total_cost <= before + 1e-6
            schedule, before = candidate, ev.total_cost


def test_ga_is_feasible_beats_baselines_and_is_reproducible():
    inst = small_instance(0)
    cfg = GAConfig(pop_size=30, generations=40, seed=5)
    r1, r2 = run_ga(inst, cfg), run_ga(inst, cfg)
    assert r1.evaluation.feasible
    assert r1.schedule == r2.schedule and r1.objective == r2.objective
    for kw in BASELINES.values():
        assert r1.objective <= evaluate(inst, fixed_interval_schedule(inst, **kw)).total_cost
    best = [h["best"] for h in r1.history]
    assert all(b2 <= b1 for b1, b2 in zip(best, best[1:]))  # elitism: never gets worse
