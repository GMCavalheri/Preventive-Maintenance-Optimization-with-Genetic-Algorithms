"""Experiment pipeline: every table and figure in the README comes from here.

Each ``run_*`` function writes CSV/JSON into ``out`` and figures into ``figs`` and returns
its data so ``run_all`` can assemble ``summary.md`` and the acceptance checks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import viz
from .baseline import BASELINES, fixed_interval_schedule
from .evaluate import evaluate
from .exact import solve_exact
from .ga import GAConfig, run_ga_many
from .instance import default_instance, small_instance
from .nsga2 import NSGA2Config, run_nsga2
from .sensitivity import failure_cost_sweep, risk_weight_sweep


@dataclass
class Budget:
    ga: GAConfig
    ga_seeds: int
    small_instances: int
    small_ga_seeds: int
    exact_time_small: float
    exact_time_full: float
    risk_weights: tuple[float, ...]
    failure_scales: tuple[float, ...]
    sweep_seeds: int
    nsga2: NSGA2Config

    @staticmethod
    def full() -> Budget:
        return Budget(GAConfig(), ga_seeds=10, small_instances=5, small_ga_seeds=5, exact_time_small=60,
                      exact_time_full=300, risk_weights=(0.0, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6),
                      failure_scales=(0.5, 1.0, 2.0, 4.0), sweep_seeds=3, nsga2=NSGA2Config())

    @staticmethod
    def quick() -> Budget:
        return Budget(GAConfig(pop_size=40, generations=60), ga_seeds=2, small_instances=2, small_ga_seeds=2,
                      exact_time_small=20, exact_time_full=20, risk_weights=(0.0, 1e5, 1e6),
                      failure_scales=(1.0, 2.0), sweep_seeds=1, nsga2=NSGA2Config(pop_size=40, generations=40))


def _schedule_json(schedule) -> list[list[int]]:
    return [list(map(int, events)) for events in schedule]


def run_baseline_vs_ga(seed: int, budget: Budget, out: Path, figs: Path, workers: int) -> dict:
    """Baselines vs the weighted GA (w_risk = 0) on the main instance, over several GA seeds."""
    inst = default_instance(seed)
    rows, schedules = [], {}
    for name, kw in BASELINES.items():
        schedules[name] = fixed_interval_schedule(inst, **kw)
        rows.append({"method": name, "seed": None, "objective": evaluate(inst, schedules[name]).total_cost,
                     **evaluate(inst, schedules[name]).as_dict(), "runtime_s": 0.0})
    results = run_ga_many([(inst, replace(budget.ga, seed=s)) for s in range(budget.ga_seeds)], workers)
    for s, r in enumerate(results):
        rows.append({"method": "GA", "seed": s, "objective": r.objective, **r.evaluation.as_dict(),
                     "runtime_s": r.runtime})
    df = pd.DataFrame(rows)
    df.to_csv(out / "baseline_vs_ga.csv", index=False)
    pd.DataFrame([{**h, "seed": s} for s, r in enumerate(results) for h in r.history]).to_csv(
        out / "ga_history.csv", index=False)
    best = min(results, key=lambda r: r.objective)
    (out / "best_schedule.json").write_text(json.dumps({
        "instance": inst.name, "objective": best.objective, "evaluation": best.evaluation.as_dict(),
        "schedule": _schedule_json(best.schedule)}, indent=1))
    return {"inst": inst, "table": df, "results": results, "best": best, "schedules": schedules}


def run_exact_small(budget: Budget, out: Path, workers: int) -> pd.DataFrame:
    """GA and baselines vs the proven CP-SAT optimum on small instances."""
    rows = []
    for i in range(budget.small_instances):
        inst = small_instance(i)
        exact = solve_exact(inst, time_limit=budget.exact_time_small, workers=workers)
        opt = exact.objective
        rows.append({"instance_seed": i, "method": "CP-SAT", "seed": None, "objective": opt,
                     "gap_pct": 0.0, "optimal": exact.optimal, "bound": exact.bound, "runtime_s": exact.runtime})
        for name, kw in BASELINES.items():
            obj = evaluate(inst, fixed_interval_schedule(inst, **kw)).total_cost
            rows.append({"instance_seed": i, "method": name, "seed": None, "objective": obj,
                         "gap_pct": 100 * (obj / opt - 1), "optimal": None, "bound": None, "runtime_s": 0.0})
        results = run_ga_many([(inst, replace(budget.ga, seed=s)) for s in range(budget.small_ga_seeds)], workers)
        for s, r in enumerate(results):
            rows.append({"instance_seed": i, "method": "GA", "seed": s, "objective": r.objective,
                         "gap_pct": 100 * (r.objective / opt - 1), "optimal": None, "bound": None,
                         "runtime_s": r.runtime})
    df = pd.DataFrame(rows)
    df.to_csv(out / "exact_small.csv", index=False)
    return df


def run_exact_full(main: dict, budget: Budget, out: Path, workers: int) -> dict:
    """CP-SAT on the main instance under a time limit: best schedule and proven lower bound."""
    inst = main["inst"]
    exact = solve_exact(inst, time_limit=budget.exact_time_full, workers=workers)
    info = {"status": exact.status, "objective": exact.objective, "bound": exact.bound,
            "optimal": exact.optimal, "runtime_s": exact.runtime, "n_arcs": exact.n_arcs,
            "evaluation": exact.evaluation.as_dict() if exact.evaluation else None,
            "schedule": _schedule_json(exact.schedule) if exact.schedule else None}
    (out / "exact_full.json").write_text(json.dumps(info, indent=1))
    info["schedule_obj"] = exact.schedule
    return info


def run_sensitivity(main: dict, budget: Budget, out: Path, figs: Path, workers: int) -> dict:
    inst = main["inst"]
    seeds = range(budget.sweep_seeds)
    risk, _ = risk_weight_sweep(inst, budget.risk_weights, seeds, budget.ga, workers)
    risk.to_csv(out / "risk_sweep.csv", index=False)
    fc = failure_cost_sweep(inst, budget.failure_scales, seeds, budget.ga, workers)
    fc.to_csv(out / "failure_cost_sweep.csv", index=False)
    viz.plot_risk_sensitivity(risk, figs / "sensitivity_risk_weight.png")
    viz.plot_failure_cost_sensitivity(fc, inst, figs / "sensitivity_failure_cost.png")
    return {"risk": risk, "failure_cost": fc}


def run_nsga2_experiment(main: dict, budget: Budget, out: Path, figs: Path) -> dict:
    inst = main["inst"]
    res = run_nsga2(inst, budget.nsga2)
    front = pd.DataFrame([e.as_dict() for _, e in res.front])
    front.to_csv(out / "nsga2_front.csv", index=False)
    pd.DataFrame(res.history).to_csv(out / "nsga2_history.csv", index=False)
    viz.plot_nsga2_hypervolume(res.history, figs / "nsga2_hypervolume.png")
    return {"front": front, "runtime": res.runtime}


# ---------------------------------------------------------------------- reporting

def _fmt_table(df: pd.DataFrame, floatfmt: dict[str, str] | None = None) -> str:
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("—" if np.isnan(v) else format(v, floatfmt.get(c, ",.0f")))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def acceptance_checks(main, small, full, sens, nsga) -> list[tuple[str, bool, str]]:
    t = main["table"]
    ga = t[t.method == "GA"]
    base_best = t[t.method != "GA"].total_cost.min()
    checks = [(
        "GA beats both baselines on every seed", bool((ga.total_cost < base_best).all()),
        f"GA worst {ga.total_cost.max():,.0f} vs best baseline {base_best:,.0f}",
    )]
    g = small[small.method == "GA"].gap_pct
    checks.append(("Median GA gap to the CP-SAT optimum < 5% (small instances)", bool(g.median() < 5),
                   f"median {g.median():.2f}%, max {g.max():.2f}%"))
    if full["bound"] and np.isfinite(full["bound"]):
        gap = 100 * (ga.objective.median() / full["bound"] - 1)
        checks.append(("Median GA gap to the CP-SAT lower bound (main instance) reported", True,
                       f"{gap:.2f}% (CP-SAT {full['status'].lower()}, own gap "
                       f"{100 * (full['objective'] / full['bound'] - 1):.2f}%)"))
    med = sens["risk"].groupby("w_risk")[["total_cost", "expected_failures"]].median()
    mono = bool(med.total_cost.is_monotonic_increasing and med.expected_failures.is_monotonic_decreasing)
    checks.append(("Raising λ raises cost and lowers expected failures (medians)", mono,
                   "cost " + " → ".join(f"{v / 1e6:.2f}M" for v in med.total_cost)
                   + "; failures " + " → ".join(f"{v:.1f}" for v in med.expected_failures)))
    front = nsga["front"][["total_cost", "expected_failures"]].to_numpy()
    worse = []
    for c, f in med[["total_cost", "expected_failures"]].to_numpy():
        cheaper = front[front[:, 1] <= f + 1e-9, 0]
        worse.append(100 * (cheaper.min() / c - 1) if len(cheaper) else np.nan)
    worse = np.array(worse)
    checks.append(("NSGA-II front within 1% of (or better than) every weighted-sum median point",
                   bool(np.nanmax(worse) <= 1.0),
                   "extra cost at equal risk: " + ", ".join("—" if np.isnan(w) else f"{w:+.1f}%" for w in worse)))
    return checks


def write_summary(main, small, full, sens, nsga, checks, out: Path) -> str:
    t = main["table"]
    by_method = t.groupby("method", sort=False).agg(
        runs=("objective", "size"), total_cost=("total_cost", "median"), best_cost=("total_cost", "min"),
        expected_failures=("expected_failures", "median"), visits=("n_visits", "median"),
        tasks=("n_events", "median"), downtime_weeks=("downtime_weeks", "median"),
        runtime_s=("runtime_s", "median")).reset_index()
    ref = t[t.method == "Fixed interval (cost-optimal)"].total_cost.iloc[0]
    by_method["vs cost-optimal baseline"] = [f"{100 * (c / ref - 1):+.1f}%" for c in by_method.total_cost]
    s = small.groupby("method", sort=False).gap_pct.agg(["median", "max"]).reset_index()
    s.columns = ["method", "median gap %", "max gap %"]
    risk = sens["risk"].groupby("w_risk").agg(
        total_cost=("total_cost", "median"), expected_failures=("expected_failures", "median"),
        visits=("n_visits", "median"), tasks=("n_events", "median")).reset_index()
    parts = [
        f"# Results — {main['inst'].name}\n",
        "## Baselines vs GA (median over GA seeds)\n",
        _fmt_table(by_method, {"expected_failures": ".2f", "visits": ".0f", "tasks": ".0f",
                               "downtime_weeks": ".1f", "runtime_s": ".1f"}),
        "\n## Exact comparison, small instances (3 aircraft × 3 types × 26 weeks)\n",
        _fmt_table(s, {"median gap %": ".2f", "max gap %": ".2f"}),
        f"\n## CP-SAT on the main instance ({full['runtime_s']:.0f}s limit)\n",
        f"Status {full['status']}, best {full['objective']:,.0f}, proven lower bound {full['bound']:,.0f}.\n",
        "## Risk-weight sweep (median over seeds)\n",
        _fmt_table(risk, {"w_risk": ",.0f", "expected_failures": ".2f", "visits": ".0f", "tasks": ".0f"}),
        f"\n## NSGA-II\n\n{len(nsga['front'])} non-dominated schedules in {nsga['runtime']:.0f}s, from "
        f"{nsga['front'].total_cost.min():,.0f} / {nsga['front'].expected_failures.max():.2f} failures to "
        f"{nsga['front'].total_cost.max():,.0f} / {nsga['front'].expected_failures.min():.2f} failures.\n",
        "## Acceptance checks\n",
        "\n".join(f"- [{'x' if ok else ' '}] {name} — {detail}" for name, ok, detail in checks),
    ]
    text = "\n".join(parts) + "\n"
    (out / "summary.md").write_text(text)
    return text


def run_all(seed: int = 0, quick: bool = False, workers: int = 8, out: Path = Path("results"),
            figs: Path = Path("figures")) -> str:
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    budget = Budget.quick() if quick else Budget.full()
    (out / "budget.json").write_text(json.dumps(asdict(budget), indent=1, default=str))

    print("[1/5] baselines vs GA", flush=True)
    main = run_baseline_vs_ga(seed, budget, out, figs, workers)
    print("[2/5] CP-SAT: small instances, then the main instance", flush=True)
    small = run_exact_small(budget, out, workers)
    full = run_exact_full(main, budget, out, workers)
    print("[3/5] sensitivity sweeps", flush=True)
    sens = run_sensitivity(main, budget, out, figs, workers)
    print("[4/5] NSGA-II", flush=True)
    nsga = run_nsga2_experiment(main, budget, out, figs)

    print("[5/5] figures and summary", flush=True)
    inst = main["inst"]
    ga_best = main["best"]
    schedules = {"Fixed interval (cost-optimal)": main["schedules"]["Fixed interval (cost-optimal)"],
                 "Genetic algorithm (best seed)": ga_best.schedule}
    viz.plot_gantt(inst, schedules, figs / "gantt_baseline_vs_ga.png")
    html_schedules = dict(schedules)
    if full["schedule_obj"] is not None:
        html_schedules["CP-SAT"] = full["schedule_obj"]
    viz.gantt_html(inst, html_schedules, figs / "schedule.html")
    refs = {"Fixed interval (cost-optimal)": main["table"].query("method == 'Fixed interval (cost-optimal)'")
            .total_cost.iloc[0]}
    if full["objective"] == full["objective"]:
        refs["CP-SAT best found"] = full["objective"]
        refs["CP-SAT lower bound"] = full["bound"]
    viz.plot_convergence([r.history for r in main["results"]], refs, figs / "ga_convergence.png")
    points = {name: (evaluate(inst, s).total_cost, evaluate(inst, s).expected_failures)
              for name, s in main["schedules"].items()}
    if full["evaluation"]:
        points["CP-SAT (λ = 0)"] = (full["evaluation"]["total_cost"], full["evaluation"]["expected_failures"])
    front = list(zip(nsga["front"].total_cost, nsga["front"].expected_failures))
    viz.plot_tradeoff(front, sens["risk"], points, figs / "tradeoff.png")

    checks = acceptance_checks(main, small, full, sens, nsga)
    return write_summary(main, small, full, sens, nsga, checks, out)
