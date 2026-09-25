# Problem formulation

Preventive maintenance (PM) scheduling for a fleet of aircraft over a finite horizon, with shared
hangar, labour and spare-part capacity. The code mirrors this document:
[`instance.py`](../src/pmo/instance.py) holds the data, [`evaluate.py`](../src/pmo/evaluate.py) computes
every quantity below, and [`exact.py`](../src/pmo/exact.py) is the same model written for CP-SAT.

## Data

| Symbol | Meaning | Default instance |
|---|---|---|
| $t \in \{0,\dots,T-1\}$ | planning weeks | $T = 52$ |
| $a \in \mathcal{A}$ | aircraft | 10 |
| $k \in \mathcal{K}$ | component types (APU, landing gear, hydraulic pump, brakes) | 4 |
| $c \in \mathcal{C}$ | components; component $c$ is of type $k(c)$ on aircraft $a(c)$ | 40 |
| $\alpha_c$ | age of $c$ at $t=0$ (weeks since its last PM) | uniform in $[0, 0.6\,\bar I_k]$ |
| $\beta_k, \eta_k$ | Weibull shape and scale (weeks) | e.g. brakes $\beta=3.5, \eta=26$ |
| $\underline I_k, \bar I_k$ | min and max weeks between consecutive PM tasks | e.g. brakes 6 and 20 |
| $p_k$ | cost of a PM task | $5\text{k}$ to $25\text{k}$ |
| $f_k$ | cost of a failure: repair $+\ d_k \cdot D$ | repair $40\text{k}$ to $150\text{k}$ |
| $d_k$ | aircraft weeks lost per failure | 0.5 to 2 |
| $D$ | cost of one aircraft-week on the ground | $20\text{k}$ |
| $h_k$ | labour hours per PM task | 12 to 50 |
| $B_t$ | hangar bays in week $t$ | 2 (1 in weeks 22–25, 0 in weeks 50–51) |
| $L_t$ | labour hours in week $t$ | 120 |
| $S_{k,t}$ | spare parts of type $k$ available up to week $t$ (stock + deliveries) | quarterly deliveries |

All parameters are illustrative, not OEM data. The generator is seeded, so every instance is reproducible.

## Failure model

A PM task renews the component (as good as new). Between tasks, failures are repaired minimally
(as bad as old), so they follow a non-homogeneous Poisson process with Weibull intensity. The expected
number of failures while a component ages from $u$ to $v$ weeks is

$$\Delta H_k(u, v) = H_k(v) - H_k(u), \qquad H_k(x) = (x/\eta_k)^{\beta_k}.$$

Because this depends only on the two endpoints of a segment, a schedule's expected failures are a sum
over its segments. That is what makes the exact model below linear.

## Decisions

$x_{c,t} \in \{0,1\}$: component $c$ gets a PM task in week $t$.
$y_{a,t} \in \{0,1\}$: aircraft $a$ is in the hangar in week $t$ (1 when any of its components is serviced).

Write component $c$'s tasks as $t_1 < \dots < t_n$, with $t_0 = -\alpha_c$ (its last renewal) and
$t_{n+1} = T$ (the end of the horizon).

## Objective

Expected cost over the horizon, plus an optional risk-aversion price $\lambda$ per expected failure:

$$\min\; \underbrace{\sum_{c,t} p_{k(c)}\,x_{c,t}}_{\text{PM tasks}} + \underbrace{D\sum_{a,t} y_{a,t}}_{\text{hangar visits}} + \sum_{c} \big(f_{k(c)} + \lambda\big) \underbrace{\sum_{i=0}^{n} \Delta H_{k(c)}\big(\max(0,-t_i),\; t_{i+1}-t_i\big)}_{\text{expected failures of }c}$$

The first segment starts at age $\alpha_c$ rather than 0, which is the $\max(0,-t_i)$ term. With
$\lambda = 0$ this is plain expected cost. Increasing $\lambda$ trades cost for fewer failures. The
multi-objective variant (NSGA-II) keeps the two terms apart and minimises
(expected total cost, expected failures).

Grouping tasks on one aircraft into the same week saves $D$ per extra component. This coupling is what
makes the problem combinatorial rather than a set of independent per-component renewal problems.

## Constraints

1. **Intervals.** $\underline I_k \le t_{i+1} - t_i \le \bar I_k$ for consecutive tasks, including the
   first gap ($t_1 + \alpha_c$). The final segment only has the upper bound, $T - t_n \le \bar I_k$: no
   component may end the year overdue.
2. **Hangar.** $\sum_a y_{a,t} \le B_t$ and $x_{c,t} \le y_{a(c),t}$.
3. **Labour.** $\sum_c h_{k(c)}\,x_{c,t} \le L_t$.
4. **Spare parts.** $\sum_{c:k(c)=k}\sum_{\tau \le t} x_{c,\tau} \le S_{k,t}$ for all $k, t$, because a
   part cannot be used before it is delivered.
5. **Risk priority.** When capacity is short, higher-risk components are served first. This is
   enforced by the order in which the decoder allocates capacity: by current hazard rate times
   failure cost. `Instance.priority_order` is where per-component RUL estimates can be plugged in.

## Exact model (CP-SAT)

Each component's schedule is a path START → $t_1$ → … → $t_n$ → END in a DAG whose arcs $(s,t)$
exist only when $t - s$ respects constraint 1. The arc's cost is $p_k$ (if $t$ is a task)
plus $(f_k+\lambda)\,\Delta H_k$ of the segment, a constant. Flow conservation gives
$x_{c,t} = \sum_{s} z_{c,s,t}$, and constraints 2–4 act on the $x$ and $y$ variables. The main
instance has about 33,500 arc variables. The model has no linearisation error. Costs are scaled to
integers (dimes) for CP-SAT, and every returned schedule is re-scored by `evaluate`.

## Genetic algorithm

* **Genotype.** One block of genes per component, each a desired week or "none". The block length
  is the most tasks the component can fit, so the GA also evolves how many tasks each component gets.
* **Decoder (repair).** Walks each component forward in time, in risk-priority order. It keeps
  desired weeks that respect the intervals, moves each to the nearest week with free bay, labour
  and parts capacity, drops tasks that no week can take, and forces a task before a maximum
  interval would be exceeded. The decoded schedule is written back into the genotype (Lamarckian).
  Anything still infeasible pays $10^6$ per unit of violation.
* **Operators.** Uniform crossover over whole component or aircraft blocks. Mutation applies one to
  three edits: shift a task, add one, remove one, align an aircraft's tasks into one week, move or
  merge whole hangar visits, re-sample a component's plan, or **best response**. Best response
  replaces one component's plan with its optimal plan given all the others: a shortest path in the
  same DAG as the exact model, where weeks the aircraft already visits carry no visit cost.
* **Selection.** Tournament (size 3) with two elites. The initial population is 10% fixed-interval
  plans (both baselines and interval multiples) and 90% randomised fixed-interval plans.

## Baselines

Every component is maintained every $I_k$ weeks of age, and the decoder staggers tasks around
capacity (first fit, nearest week):

* **Cost-optimal interval**: $I_k^\* = \eta_k\big(\tfrac{p_k + D}{f_k(\beta_k-1)}\big)^{1/\beta_k}$,
  the classic periodic-PM optimum under minimal repair for one component on its own.
* **80% of the maximum interval**: a typical conservative rule of thumb.

## Known simplifications

* One week of downtime per hangar visit, whatever the number of tasks.
* The horizon end is handled by requiring $T - t_n \le \bar I_k$. There is no terminal value for the
  age a component ends the year at.
* One spare-part number per component type. Deliveries are sized from a fixed-interval plan plus a
  15% margin, so the parts constraint binds for schedules that maintain much more often.
