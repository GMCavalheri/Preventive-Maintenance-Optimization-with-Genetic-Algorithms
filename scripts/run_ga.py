"""Run the weighted GA once and compare with the baselines: python scripts/run_ga.py [--small] [--risk-weight 1e5]."""

import argparse

from pmo.baseline import BASELINES, fixed_interval_schedule
from pmo.evaluate import evaluate
from pmo.ga import GAConfig, run_ga
from pmo.instance import default_instance, small_instance

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--small", action="store_true", help="3 aircraft x 3 types x 26 weeks")
    ap.add_argument("--instance-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generations", type=int, default=GAConfig.generations)
    ap.add_argument("--pop-size", type=int, default=GAConfig.pop_size)
    ap.add_argument("--risk-weight", type=float, default=0.0, help="$ per expected failure")
    args = ap.parse_args()

    inst = (small_instance if args.small else default_instance)(args.instance_seed)
    for name, kw in BASELINES.items():
        ev = evaluate(inst, fixed_interval_schedule(inst, **kw))
        print(f"{name:32s} cost {ev.total_cost:12,.0f}  failures {ev.expected_failures:6.2f}  visits {ev.n_visits}")
    r = run_ga(inst, GAConfig(pop_size=args.pop_size, generations=args.generations,
                              w_risk=args.risk_weight, seed=args.seed))
    ev = r.evaluation
    print(f"{'GA':32s} cost {ev.total_cost:12,.0f}  failures {ev.expected_failures:6.2f}  visits {ev.n_visits}"
          f"  feasible {ev.feasible}  ({r.runtime:.1f}s)")
