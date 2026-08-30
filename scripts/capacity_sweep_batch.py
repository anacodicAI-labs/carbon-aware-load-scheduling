"""Generator for results/capacity_sweep_batch.csv -- the batch (Alibaba GPU) capacity sweep.

Same k x median_active_kw grid as scripts/capacity_sweep.py (HVAC) and
scripts/capacity_sweep_ev.py, so the three arms share a cap convention.

GREEDY ONLY, and that is a hard limit, not a shortcut. schedule_optimal() builds
DENSE constraint matrices (greedy.py: np.zeros((n_jobs, nvar)) and
np.zeros((n_hours, nvar))). Measured on this machine (Apple M5, 17.2 GB): the MILP
solves at n=16,000 batch jobs (41 s) and is OOM-killed (exit 137) at n=32,000.
The batch arm is 227,529 jobs, 7-14x past that wall, so no exact optimum exists
for this arm and the greedy/MILP gap that Figure 7 reports for HVAC cannot be
computed here. optimal_pct / gap_pp are therefore written as NaN, not omitted.

READ THE CAP ROWS WITH CARE. median_active_kw() is the median JOB power, which
equals the median hourly SITE load only because the HVAC adapter emits exactly one
block per hour. Batch does not: 227,529 jobs occupy 1,642 active hours (~138
jobs/hour), and the median aggregate hourly load is ~253.8 kW against a median job
power of 0.400 kW. A 3x median cap is therefore 1.2 kW -- about 635x tighter than
3x the aggregate load -- so every k x median row here measures forced fallback,
not scheduling. The convention is held for comparability with the other two arms;
the `degenerate_for_fleet` column flags that it does not carry its HVAC meaning.
"""
from __future__ import annotations

import collections
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import numpy as np
import pandas as pd

import nb_utils as U
from cals import schedule
from carbon_sim import CAP_SWEEP_K, median_active_kw
from cuad.scheduler.jobs import duration_h


def main() -> None:
    ci, source = U.get_carbon_intensity()
    jobs, _, label = U.get_ai_jobs(gpu_power_kw=0.4, flex_hours=6, verbose=False)
    run_hours = {j.job_id: j.earliest_start for j in jobs}
    base = U.do_nothing_gco2(jobs, run_hours, ci)
    med = median_active_kw(jobs)
    peak = max(j.power_kw for j in jobs)

    prof = collections.Counter()
    for j in jobs:
        for h in range(duration_h(j)):
            prof[j.earliest_start + pd.Timedelta(hours=h)] += j.power_kw
    agg = np.array(sorted(prof.values()))
    print(f"carbon: {source}  ({len(ci)} hours)")
    print(f"jobs={len(jobs):,}  median active JOB power={med:.4f} kW  peak job={peak:.4f} kW")
    print(f"active hours={len(agg):,}  median AGGREGATE hourly load={np.median(agg):.1f} kW  "
          f"max={agg.max():.1f} kW  (~{len(jobs)/len(agg):.0f} jobs/hour)")
    print(f"3x median job power = {3*med:.3f} kW vs 3x median aggregate = {3*np.median(agg):.1f} kW "
          f"({3*np.median(agg)/(3*med):.0f}x tighter)\n")

    grid = [(f"{k:g}x med", k, round(k * med, 6)) for k in CAP_SWEEP_K]
    grid.append(("uncapped", None, None))

    rows = []
    for lab, k, cap_kw in grid:
        t0 = time.time()
        g = schedule(jobs, ci, capacity_kw=cap_kw, baseline_hours=run_hours)
        t_greedy = time.time() - t0
        gp = U.savings_pct(base, g["total_gco2"])
        rows.append({
            "label": lab, "k": k, "M_kW": cap_kw,
            "greedy_pct": gp, "optimal_pct": float("nan"), "gap_pp": float("nan"),
            "t_greedy_s": t_greedy, "t_milp_s": float("nan"),
            "milp_status": "infeasible_on_this_machine_OOM_above_n~32000",
            "degenerate_for_fleet": k is not None,
        })
        cs = "  n/a  " if cap_kw is None else f"{cap_kw:7.4f}"
        print(f"  {lab:11} M={cs}  greedy={gp:7.4f}%  optimal=   n/a   "
              f"({t_greedy:.0f}s greedy)", flush=True)

    out = ROOT / "results" / "capacity_sweep_batch.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
