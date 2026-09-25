"""Regenerate every result table and figure: python scripts/run_all.py --seed 0 [--quick]."""

import argparse
import os
from pathlib import Path

from pmo.experiments import run_all

if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0, help="seed of the main instance")
    ap.add_argument("--quick", action="store_true", help="tiny budgets, for a smoke test (~1 min)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--out", type=Path, default=root / "results")
    ap.add_argument("--figures", type=Path, default=root / "figures")
    args = ap.parse_args()
    print(run_all(seed=args.seed, quick=args.quick, workers=args.workers, out=args.out, figs=args.figures))
