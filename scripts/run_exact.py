"""Solve an instance with CP-SAT: python scripts/run_exact.py [--small] [--time-limit 60]."""

import argparse

from pmo.exact import solve_exact
from pmo.instance import default_instance, small_instance

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--small", action="store_true", help="3 aircraft x 3 types x 26 weeks")
    ap.add_argument("--instance-seed", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--risk-weight", type=float, default=0.0, help="$ per expected failure")
    ap.add_argument("--log", action="store_true", help="print the CP-SAT search log")
    args = ap.parse_args()

    inst = (small_instance if args.small else default_instance)(args.instance_seed)
    r = solve_exact(inst, w_risk=args.risk_weight, time_limit=args.time_limit, workers=args.workers, log=args.log)
    print(f"{inst.name}: {r.status} in {r.runtime:.1f}s, {r.n_arcs} arc variables")
    print(f"objective {r.objective:,.2f}  lower bound {r.bound:,.2f}  gap {100 * r.gap:.2f}%")
    if r.evaluation:
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in r.evaluation.as_dict().items()})
