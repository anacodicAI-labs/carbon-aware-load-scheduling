"""Greedy carbon-aware scheduler.

Assumptions (v1):
- Resolution: carbon intensity is hourly, so jobs schedule on whole-hour slots.
- Non-preemptible: each job runs as one contiguous block, never split.
- Shared-capacity constraint is OPTIONAL (schedule's capacity_kw). Default None
  keeps the original behavior: jobs may overlap freely in the same hour. When a
  cap is given, cumulative power_kw per hour may not exceed it.
- Placement cost for a start hour is the sum over the job's duration hours of
  power_kw * carbon[hour], i.e. kWh-weighted gCO2.

Notes on what v1 is:
- With no capacity constraint (capacity_kw=None) this is exhaustive per-job
  optimal, not a heuristic: the jobs never interact, so placing each at its own
  cheapest feasible slot is the global optimum, and the ascending-slack sort has
  no effect on the result. Once capacity_kw binds, the jobs DO interact (each
  placement consumes hourly headroom) and the ascending-slack sort becomes the
  actual greedy heuristic: tighter-deadline jobs get first pick of cheap hours.
- The cost model bills ceil(energy_kwh / power_kw) whole hours at full power_kw,
  so a job whose energy is not a whole multiple of power_kw slightly overcounts
  its final partial hour. schedule() and fifo_baseline() use the same
  convention, so the greedy-vs-fifo comparison stays fair; only the absolute
  total_gco2 is a small overestimate.

schedule() sorts jobs by ascending slack (tightest window first) and gives each
the lowest-cost feasible start. fifo_baseline() ignores carbon and places each
job at its earliest start; it is the naive control for the Task 7 comparison.
"""
from __future__ import annotations

import pandas as pd

from cuad.scheduler.jobs import Job, duration_h, slack_h


def _assert_on_grid(ts: pd.Timestamp, origin: pd.Timestamp, job_id: str, field: str) -> None:
    """Raise if ts is not on the carbon series' hourly grid (fail loud, not silent)."""
    try:
        offset = ts - origin
    except TypeError as exc:  # tz-naive vs tz-aware, or otherwise incomparable
        raise ValueError(
            f"job {job_id} {field} {ts} is not comparable to the carbon index (check timezone)"
        ) from exc
    if offset % pd.Timedelta(hours=1) != pd.Timedelta(0):
        raise ValueError(
            f"job {job_id} {field} {ts} is not aligned to the carbon series hourly grid"
        )


def _check_inputs(jobs: list[Job], carbon: pd.Series) -> None:
    """Reject inputs that would otherwise fail silently or with a cryptic error."""
    if not carbon.index.is_unique:
        raise ValueError("carbon index has duplicate timestamps; expected a unique hourly grid")
    origin = carbon.index[0]
    for job in jobs:
        _assert_on_grid(job.earliest_start, origin, job.job_id, "earliest_start")
        _assert_on_grid(job.deadline, origin, job.job_id, "deadline")


def _window_cost(job: Job, carbon: pd.Series, start: pd.Timestamp, dur: int) -> float | None:
    """kWh-weighted gCO2 of running the job for dur hours from start.

    Returns None when the carbon series does not cover the whole window, or when
    any hour in it has a NaN carbon value (carbon_intensity returns NaN for a
    zero-generation hour). Either way the window cannot be priced, so it is not a
    valid placement.
    """
    cost = 0.0
    for h in range(dur):
        hour = start + pd.Timedelta(hours=h)
        if hour not in carbon.index:
            return None
        value = float(carbon.loc[hour])
        if pd.isna(value):
            return None
        cost += job.power_kw * value
    return cost


def _fits_capacity(
    job: Job,
    load: dict[pd.Timestamp, float],
    start: pd.Timestamp,
    dur: int,
    capacity_kw: float | None,
) -> bool:
    """True if adding this job over [start, start+dur) keeps every hour <= cap."""
    if capacity_kw is None:
        return True
    for h in range(dur):
        hour = start + pd.Timedelta(hours=h)
        if load.get(hour, 0.0) + job.power_kw > capacity_kw:
            return False
    return True


def _best_placement(
    job: Job,
    carbon: pd.Series,
    load: dict[pd.Timestamp, float],
    capacity_kw: float | None,
) -> tuple[pd.Timestamp, float] | str | None:
    """Lowest-cost feasible start for the job.

    Returns (start, cost) when a slot is found, the string "capacity" when
    priceable windows exist but every one would breach capacity_kw, or None when
    the job has no priceable window at all (bad slack / uncovered / NaN carbon).
    The "capacity" case is only reachable when capacity_kw is not None, so with a
    None cap the return is exactly (start, cost) or None as before.
    """
    if slack_h(job) < 0:
        return None
    dur = duration_h(job)
    last_start = job.deadline - pd.Timedelta(hours=dur)
    best_start: pd.Timestamp | None = None
    best_cost: float | None = None
    saw_priceable = False
    start = job.earliest_start
    while start <= last_start:
        cost = _window_cost(job, carbon, start, dur)
        if cost is not None:
            saw_priceable = True
            if _fits_capacity(job, load, start, dur, capacity_kw) and (
                best_cost is None or cost < best_cost
            ):
                best_cost = cost
                best_start = start
        start += pd.Timedelta(hours=1)
    if best_start is not None and best_cost is not None:
        return best_start, best_cost
    if saw_priceable and capacity_kw is not None:
        return "capacity"
    return None


def schedule(
    jobs: list[Job],
    carbon: pd.Series,
    *,
    capacity_kw: float | None = None,
    baseline_hours: dict[str, pd.Timestamp] | None = None,
) -> dict:
    """Greedy carbon-minimising placement.

    capacity_kw is an optional per-hour cap on cumulative power_kw across all
    jobs. Default None preserves the original behavior exactly (jobs may overlap
    without limit). When set, a job is placed only in a window that keeps every
    covered hour at or below the cap; jobs are considered in ascending-slack
    order, so tighter-deadline jobs claim scarce cheap hours first.

    baseline_hours maps job_id -> its physical do-nothing hour (EV plug-in time,
    HVAC observed run hour). When provided, a job the optimizer cannot place is
    NOT dropped: it runs at its baseline hour, charged full carbon there, and
    gets zero benefit (it counts toward emissions, it just does not help). This
    is the physically correct accounting -- the car still charges, the AC still
    runs. Default None keeps the legacy drop behavior for callers that do not
    supply a baseline.

    The capacity cap is a SHIFTING limit, not a physical breaker: it bounds where
    the optimizer may move load. Fallback load lands at its baseline hour
    unconditionally and does NOT consume the optimizer's headroom, so it never
    appears in the capacity check for other jobs. That is why an hour's total
    load (optimizer-placed plus fallback) may exceed the cap; callers that report
    a peak should report both the shifted-only peak (<= cap) and the overall peak
    (may exceed) rather than clipping.

    Baseline floor (only active with baseline_hours): a job is never placed at a
    cost STRICTLY greater than doing nothing. If the cheapest feasible slot is
    dirtier than the baseline hour -- which can happen once a cap binds and an
    earlier job has consumed the job's own cheap baseline hour -- the job stays at
    baseline (cause "no_improvement") instead of being shifted somewhere worse.
    This makes each job's realised cost at most its baseline cost, so per-job and
    hence aggregate savings are non-negative BY CONSTRUCTION, not just in
    practice. Uncapped, the baseline hour is always a feasible candidate so the
    optimum is never strictly worse than baseline and the floor never fires.

    Returns a dict with keys:
        assignments:         job_id -> chosen start Timestamp (shifted jobs at
                             their optimizer hour, fallback jobs at baseline)
        total_gco2:          summed kWh-weighted gCO2 of every assigned job
        infeasible:          job_ids the optimizer could not place (diagnostic;
                             with baseline_hours these still run as fallback)
        capacity_infeasible: the subset of infeasible whose only obstacle was the
                             cap (a priceable window existed but all breached it).
                             Distinct from infeasible_window / nan_carbon causes;
                             always empty when capacity_kw is None.
        shifted:             job_ids the optimizer actively placed (not fallback)
        fallback:            job_id -> cause for jobs run at their baseline hour:
                             "capacity" / "window" (no feasible slot at all) or
                             "no_improvement" (a feasible slot existed but was
                             costlier than doing nothing). Empty when
                             baseline_hours is None. "window" covers the
                             bad-slack / uncovered / NaN cases (refine via
                             _classify_unplaced if needed).
        unaccountable:       job_ids that could not be priced even at baseline
                             (baseline hour off the series or NaN); reported apart
                             because they are genuinely unrepresentable, not saved.
    """
    _check_inputs(jobs, carbon)
    assignments: dict[str, pd.Timestamp] = {}
    infeasible: list[str] = []
    capacity_infeasible: list[str] = []
    shifted: list[str] = []
    fallback: dict[str, str] = {}
    unaccountable: list[str] = []
    load: dict[pd.Timestamp, float] = {}  # hour -> cumulative SHIFTED power_kw
    total = 0.0
    for job in sorted(jobs, key=slack_h):  # tightest window first
        best = _best_placement(job, carbon, load, capacity_kw)
        # do-nothing cost for this job (its baseline hour), when a baseline exists
        bh = baseline_hours[job.job_id] if baseline_hours is not None else None
        bcost = _window_cost(job, carbon, bh, duration_h(job)) if bh is not None else None

        if isinstance(best, tuple):  # optimizer found a feasible placement
            start, cost = best
            # Baseline floor: never shift a job to a cost that exceeds doing
            # nothing. When it would, leave it at baseline (zero benefit, not a
            # loss). Fallback load does not consume the optimizer's headroom.
            if bcost is not None and cost > bcost:
                assignments[job.job_id] = bh
                fallback[job.job_id] = "no_improvement"
                total += bcost
                continue
            assignments[job.job_id] = start
            shifted.append(job.job_id)
            for h in range(duration_h(job)):
                hour = start + pd.Timedelta(hours=h)
                load[hour] = load.get(hour, 0.0) + job.power_kw
            total += cost
            continue

        # optimizer could not place it at all: record the distinct cause
        cause = "capacity" if best == "capacity" else "window"
        infeasible.append(job.job_id)
        if cause == "capacity":
            capacity_infeasible.append(job.job_id)
        if baseline_hours is None:
            continue  # legacy: no baseline to fall back to, so drop
        # fall back to the baseline hour, charged full carbon, zero benefit.
        # NOTE: fallback load is physical but does not consume the optimizer's
        # capacity headroom (cap is a shifting limit), so it is not added to load.
        if bcost is None:
            unaccountable.append(job.job_id)  # cannot even price the baseline
            continue
        assignments[job.job_id] = bh
        fallback[job.job_id] = cause
        total += bcost
    return {
        "assignments": assignments,
        "total_gco2": float(total),
        "infeasible": infeasible,
        "capacity_infeasible": capacity_infeasible,
        "shifted": shifted,
        "fallback": fallback,
        "unaccountable": unaccountable,
    }


def schedule_optimal(
    jobs: list[Job],
    carbon: pd.Series,
    *,
    capacity_kw: float | None = None,
    baseline_hours: dict[str, pd.Timestamp] | None = None,
) -> dict:
    """Globally optimal placement under a per-hour capacity cap, via MILP.

    schedule() is a greedy heuristic: it fixes each job in ascending-slack order,
    so once the cap binds an early job can claim a cheap hour that a later job
    needed more. This function solves the same placement problem to global
    optimality instead, so schedule_optimal() - schedule() is exactly the cost of
    the greedy heuristic once a physical limit binds. greedy stays the auditable
    baseline; this is the yardstick.

    Formulation (scipy.optimize.milp / HiGHS). Each job gets one binary variable
    per priceable shift hour plus, when baseline_hours is given, one "do-nothing"
    variable that runs it at its baseline hour. Exactly one variable per job is 1.
    The per-hour capacity constraint sums power_kw over the SHIFT variables that
    cover that hour (<= cap); the do-nothing variable is cap-exempt, mirroring the
    Option A / baseline-floor semantics of schedule() -- doing nothing is always
    available at baseline cost and never consumes the optimizer's headroom. So the
    optimum, like greedy+floor, never places a job above its baseline cost, and
    the two are compared under identical rules (pure arrangement quality).

    Why MILP and not linear_sum_assignment or min-cost flow: those need either one
    job per hour slot or capacity discretised into equal slots, which is only
    exact when job power is uniform. HVAC power is heterogeneous (many distinct
    kW), so the per-hour cap is a real power budget with indivisible demands -- a
    bin-packing-with-costs that MILP solves exactly with no discretisation. Jobs
    may be multi-hour (a shift variable covers duration_h consecutive hours), so
    this also handles non-1h workloads; it is not preemptible (Fix 3).

    With capacity_kw=None there are no capacity constraints, so each job
    independently takes its cheapest priceable hour -- identical to greedy, which
    is provably optimal when nothing binds.

    Returns a dict with keys:
        assignments:  job_id -> chosen start Timestamp (shift hour, or baseline
                      hour for do-nothing jobs)
        total_gco2:   optimal total kWh-weighted gCO2 (the MILP objective)
        do_nothing:   job_ids the optimum left at their baseline hour
        unaccountable: job_ids with no priceable option at all (excluded)
        status:       solver status string
    """
    from scipy.optimize import Bounds, LinearConstraint, milp  # lazy: keep greedy dep-free

    import numpy as np

    _check_inputs(jobs, carbon)

    costs: list[float] = []
    var_job: list[int] = []               # job row each variable belongs to
    var_kind: list[tuple[str, pd.Timestamp]] = []
    var_cover: list[list[pd.Timestamp]] = []  # hours a shift variable occupies
    var_power: list[float] = []
    jobs_with_vars: set[int] = set()
    unaccountable: list[str] = []

    for ji, job in enumerate(jobs):
        dur = duration_h(job)
        had_var = False
        if slack_h(job) >= 0:  # enumerate priceable shift starts
            last = job.deadline - pd.Timedelta(hours=dur)
            start = job.earliest_start
            while start <= last:
                c = _window_cost(job, carbon, start, dur)
                if c is not None:
                    costs.append(c)
                    var_job.append(ji)
                    var_kind.append(("shift", start))
                    var_cover.append([start + pd.Timedelta(hours=h) for h in range(dur)])
                    var_power.append(job.power_kw)
                    had_var = True
                start += pd.Timedelta(hours=1)
        if baseline_hours is not None:  # cap-exempt do-nothing option
            bh = baseline_hours[job.job_id]
            bcost = _window_cost(job, carbon, bh, dur)
            if bcost is not None:
                costs.append(bcost)
                var_job.append(ji)
                var_kind.append(("nothing", bh))
                var_cover.append([])
                var_power.append(0.0)
                had_var = True
        if had_var:
            jobs_with_vars.add(ji)
        else:
            unaccountable.append(job.job_id)

    nvar = len(costs)
    if nvar == 0:
        return {"assignments": {}, "total_gco2": 0.0, "do_nothing": [],
                "unaccountable": unaccountable, "status": "empty"}

    # one-variable-per-job equality constraints
    rows = sorted(jobs_with_vars)
    row_of = {ji: r for r, ji in enumerate(rows)}
    a_eq = np.zeros((len(rows), nvar))
    for v, ji in enumerate(var_job):
        a_eq[row_of[ji], v] = 1.0
    constraints = [LinearConstraint(a_eq, lb=1, ub=1)]

    # per-hour capacity on shift variables only
    if capacity_kw is not None:
        hour_row: dict[pd.Timestamp, int] = {}
        for cover in var_cover:
            for hr in cover:
                hour_row.setdefault(hr, len(hour_row))
        if hour_row:
            a_ub = np.zeros((len(hour_row), nvar))
            for v, cover in enumerate(var_cover):
                for hr in cover:
                    a_ub[hour_row[hr], v] = var_power[v]
            constraints.append(LinearConstraint(a_ub, ub=capacity_kw))

    res = milp(
        c=np.asarray(costs, dtype=float),
        constraints=constraints,
        integrality=np.ones(nvar),
        bounds=Bounds(0, 1),
    )
    if not res.success or res.x is None:
        raise RuntimeError(f"schedule_optimal: MILP did not solve ({res.message})")

    assignments: dict[str, pd.Timestamp] = {}
    do_nothing: list[str] = []
    for v in range(nvar):
        if res.x[v] > 0.5:
            job = jobs[var_job[v]]
            kind, hour = var_kind[v]
            assignments[job.job_id] = hour
            if kind == "nothing":
                do_nothing.append(job.job_id)
    return {
        "assignments": assignments,
        "total_gco2": float(res.fun),
        "do_nothing": do_nothing,
        "unaccountable": unaccountable,
        "status": str(res.status),
    }


def schedule_preemptible(
    jobs: list[Job],
    carbon: pd.Series,
    *,
    capacity_kw: float | None = None,
    baseline_hours: dict[str, pd.Timestamp] | None = None,
) -> dict:
    """Optimal PREEMPTIBLE placement: a job may occupy its duration_h hours in any
    feasible hours within its window, not necessarily adjacent.

    Same MILP machinery and semantics as schedule_optimal (per-hour capacity on
    active load, cap-exempt do-nothing at baseline, so the floor holds), with one
    change: instead of one variable per contiguous block start, there is one
    binary occupancy variable per (job, priceable-window-hour), and a job takes
    exactly duration_h of them (sum_h x = dur when running, 0 when do-nothing).
    Dropping the adjacency constraint is the whole point -- schedule_optimal()
    minus schedule_preemptible() is exactly the cost of forcing jobs to run in one
    unbroken block (the contiguity penalty), the mirror of Fix 2's heuristic gap.

    Preemption can only help multi-hour jobs; a duration_h == 1 job has one
    occupancy variable and reduces to the same single-hour choice as
    schedule_optimal.

    Returns a dict with keys:
        assignments:  job_id -> sorted list of occupied hour Timestamps (the
                      baseline block for do-nothing jobs)
        total_gco2:   optimal total kWh-weighted gCO2 (the MILP objective)
        do_nothing:   job_ids the optimum left at their baseline block
        unaccountable: job_ids with too few priceable hours and no priceable
                      baseline (excluded)
        status:       HiGHS status code (0 == proven optimal)
        message:      HiGHS status message
        n_vars:       number of MILP variables
    """
    from scipy.optimize import Bounds, LinearConstraint, milp  # lazy: keep greedy dep-free

    import numpy as np

    _check_inputs(jobs, carbon)

    costs: list[float] = []
    var_job: list[int] = []
    var_hour: list[pd.Timestamp | None] = []   # occupied hour, or None for do-nothing
    var_power: list[float] = []
    var_is_nothing: list[bool] = []
    job_dur: dict[int, int] = {}
    jobs_with_vars: set[int] = set()
    unaccountable: list[str] = []
    nothing_block: dict[int, list[pd.Timestamp]] = {}

    for ji, job in enumerate(jobs):
        dur = duration_h(job)
        job_dur[ji] = dur
        n_hours = 0
        if slack_h(job) >= 0:  # one occupancy variable per priceable window hour
            hour = job.earliest_start
            while hour < job.deadline:
                c = _window_cost(job, carbon, hour, 1)  # single-hour price
                if c is not None:
                    costs.append(c)
                    var_job.append(ji)
                    var_hour.append(hour)
                    var_power.append(job.power_kw)
                    var_is_nothing.append(False)
                    n_hours += 1
                hour += pd.Timedelta(hours=1)
        do_nothing_ok = False
        if baseline_hours is not None:
            bh = baseline_hours[job.job_id]
            bcost = _window_cost(job, carbon, bh, dur)
            if bcost is not None:
                costs.append(bcost)
                var_job.append(ji)
                var_hour.append(None)
                var_power.append(0.0)
                var_is_nothing.append(True)
                nothing_block[ji] = [bh + pd.Timedelta(hours=h) for h in range(dur)]
                do_nothing_ok = True
        # a job is schedulable if it has >= dur priceable hours, or can do nothing
        if n_hours >= dur or do_nothing_ok:
            jobs_with_vars.add(ji)
        else:
            unaccountable.append(job.job_id)

    nvar = len(costs)
    if nvar == 0:
        return {"assignments": {}, "total_gco2": 0.0, "do_nothing": [],
                "unaccountable": unaccountable, "status": 0, "message": "empty", "n_vars": 0}

    # per-job demand: sum of occupancy vars + dur * do_nothing == dur
    rows = sorted(jobs_with_vars)
    row_of = {ji: r for r, ji in enumerate(rows)}
    a_eq = np.zeros((len(rows), nvar))
    b_eq = np.array([float(job_dur[ji]) for ji in rows])
    for v, ji in enumerate(var_job):
        if ji in row_of:
            a_eq[row_of[ji], v] = job_dur[ji] if var_is_nothing[v] else 1.0
    constraints = [LinearConstraint(a_eq, lb=b_eq, ub=b_eq)]

    if capacity_kw is not None:  # per-hour cap on active occupancy
        hour_row: dict[pd.Timestamp, int] = {}
        for hr in var_hour:
            if hr is not None:
                hour_row.setdefault(hr, len(hour_row))
        if hour_row:
            a_ub = np.zeros((len(hour_row), nvar))
            for v, hr in enumerate(var_hour):
                if hr is not None:
                    a_ub[hour_row[hr], v] = var_power[v]
            constraints.append(LinearConstraint(a_ub, ub=capacity_kw))

    res = milp(
        c=np.asarray(costs, dtype=float),
        constraints=constraints,
        integrality=np.ones(nvar),
        bounds=Bounds(0, 1),
    )
    if res.x is None:
        raise RuntimeError(f"schedule_preemptible: MILP returned no solution ({res.message})")

    assignments: dict[str, list[pd.Timestamp]] = {}
    do_nothing: list[str] = []
    for v in range(nvar):
        if res.x[v] > 0.5:
            job = jobs[var_job[v]]
            if var_is_nothing[v]:
                assignments[job.job_id] = list(nothing_block[var_job[v]])
                do_nothing.append(job.job_id)
            else:
                assignments.setdefault(job.job_id, []).append(var_hour[v])
    for jid in assignments:
        assignments[jid] = sorted(assignments[jid])
    return {
        "assignments": assignments,
        "total_gco2": float(res.fun),
        "do_nothing": do_nothing,
        "unaccountable": unaccountable,
        "status": int(res.status),
        "message": str(res.message),
        "n_vars": nvar,
    }


# greedy and fifo may mark different jobs infeasible; evaluate() compares only over the set both arms place.
def fifo_baseline(jobs: list[Job], carbon: pd.Series) -> dict:
    """Naive control: place every feasible job at its earliest_start, ignoring carbon.

    Same return shape as schedule(); the comparison baseline for Task 7.
    """
    _check_inputs(jobs, carbon)
    assignments: dict[str, pd.Timestamp] = {}
    infeasible: list[str] = []
    total = 0.0
    for job in jobs:
        if slack_h(job) < 0:
            infeasible.append(job.job_id)
            continue
        cost = _window_cost(job, carbon, job.earliest_start, duration_h(job))
        if cost is None:
            infeasible.append(job.job_id)
            continue
        assignments[job.job_id] = job.earliest_start
        total += cost
    return {"assignments": assignments, "total_gco2": float(total), "infeasible": infeasible}
