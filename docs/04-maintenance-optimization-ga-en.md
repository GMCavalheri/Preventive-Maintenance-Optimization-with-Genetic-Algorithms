# Preventive Maintenance Optimization with Genetic Algorithms

## Overview
A combinatorial optimization project that formulates preventive maintenance scheduling for a fleet (or for components of an aircraft) as an optimization problem, solved with genetic algorithms — directly reusing evolutionary algorithm/neuroevolution knowledge from the thesis, now applied to operational decision-making instead of network training.

## Objective
Find a preventive maintenance schedule that minimizes cost/downtime and failure risk, while respecting real operational constraints (maintenance windows, parts and labor availability).

## Why This Project
- Addresses the part of the job about "seeking structuring solutions for analytical decision-making processes combining data and indicators" — focused on decisions, not just predictions.
- Complements the prediction projects (RUL, root cause): after knowing *when* a component might fail, this project decides *when to schedule* the maintenance.
- Reuses and generalizes the thesis's evolutionary algorithms expertise into a combinatorial optimization domain, showing versatility beyond deep learning.

## Problem Formulation
- **Decision variables**: maintenance date/order for each component/aircraft.
- **Objective function**: minimize total cost (maintenance + risk of unscheduled failure) and/or fleet downtime.
- **Constraints**:
  - Available maintenance windows (hangars, crews)
  - Spare parts availability
  - Minimum/maximum intervals between maintenance by component type
  - Prioritization of higher-risk components (can use the RUL project's output as input)

## Tech Stack
- Python
- DEAP (evolutionary algorithms framework) or a custom GA implementation
- OR-Tools or PuLP (for comparison against exact optimization methods, when the problem is small enough)
- Matplotlib/Plotly to visualize schedules (Gantt) and algorithm convergence

## Approach
1. Define a simplified problem instance (e.g., N aircraft, M components, a 1-year time horizon).
2. Encode solutions as chromosomes (e.g., a vector of maintenance dates per component).
3. Implement genetic operators (crossover, mutation) respecting the problem's constraints.
4. Compare the evolved solution against simple heuristics (e.g., maintenance at fixed intervals) and, on small instances, against an exact solver.

## Execution Plan
- [x] **Phase 1 — Formalization**: formally define the variables, objective function, and constraints of the scheduling problem.
- [x] **Phase 2 — Scenario generation**: create a representative synthetic instance (fleet, components, maintenance windows).
- [x] **Phase 3 — Baseline**: implement a simple heuristic (fixed-interval maintenance) as a point of comparison.
- [x] **Phase 4 — Genetic algorithm**: implement encoding, genetic operators, and the fitness function.
- [x] **Phase 5 — Comparison against an exact solver**: on small instances, validate the GA solution's quality against OR-Tools/PuLP.
- [x] **Phase 6 — Sensitivity analysis**: test how the solution changes with different cost vs. risk weights.
- [x] **Phase 7 — Visualization and documentation**: Gantt chart of the optimized schedule, GA convergence plot, README explaining the formulation.

## Deliverables
- Repository with the problem formulation, GA implementation, and comparison against baseline/exact solver
- Visualization of the optimized maintenance schedule (Gantt chart)
- Cost vs. failure-risk trade-off analysis

## Portfolio Differentiators
- Demonstrates mastery of combinatorial optimization and evolutionary algorithms applied to a real decision problem, not just model training — strongly complements the prediction projects with a prescriptive one.
- Connects directly back to the thesis, reinforcing a coherent specialization track (neuroevolution) instead of disconnected projects.

## Possible Extensions
- Integrate the RUL project's output (Project 1) as the per-component risk input
- Multi-objective optimization (NSGA-II) to explicitly expose the cost-vs-risk Pareto front, instead of a single weighted objective function
