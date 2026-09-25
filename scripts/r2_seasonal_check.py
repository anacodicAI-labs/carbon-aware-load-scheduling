"""R2 verification: is the seasonal pattern budget-driven in winter and grid-driven in summer?

Reports, for the headline HVAC job set (building 486202, flex 6 h):
  1. monthly savings capped (k=3 greedy, the published basis; and exact MILP) and uncapped,
     each energy-weighted (aggregate avoided / aggregate baseline) and unweighted (mean of
     per-job percentages), plus the season aggregates on each basis;
  2. the monthly distribution of jobs left at their metered hour under the budget;
  3. per-month job power against the budget, and the mean CI of the metered hour.
Writes results/r2_monthly_savings.csv and results/r2_monthly_binding.csv. Read-only otherwise.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import numpy as np
import pandas as pd

import nb_utils as U
from cals import schedule, schedule_optimal
from carbon_sim import WORKING_CAP_K, median_active_kw

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASONS = (("winter (DJF)", [12, 1, 2]), ("spring (MAM)", [3, 4, 5]),
           ("summer (JJA)", [6, 7, 8]), ("fall (SON)", [9, 10, 11]))


def per_job(jobs, rh, assign, ci):
    """(baseline gCO2, optimized gCO2) per job, as arrays aligned with `jobs`."""
    b = np.array([U.price_at(j, rh[j.job_id], ci) for j in jobs], float)
    o = np.array([U.price_at(j, assign[j.job_id], ci) for j in jobs], float)
    return b, o


def main() -> None:
    t0 = time.time()
    ci, source = U.get_carbon_intensity()
    jobs, rh, _ = U.load_hvac_jobs(flex_hours=6)
    M = round(WORKING_CAP_K * median_active_kw(jobs), 6)
    month = np.array([pd.Timestamp(rh[j.job_id]).month for j in jobs])
    power = np.array([j.power_kw for j in jobs], float)
    print(f"carbon: {source} | jobs={len(jobs)} | budget M = {M:.4f} kW "
          f"({WORKING_CAP_K:g} x annual median active {median_active_kw(jobs):.4f} kW)")

    # ---- arm 1/2: capped greedy and capped MILP, scheduled month by month (months partition
    # the hours, so a per-hour budget makes this identical to scheduling the whole year).
    arms = {}
    for name, solver in (("capped_greedy", lambda js, rm: schedule(js, ci, capacity_kw=M, baseline_hours=rm)),
                         ("capped_milp", lambda js, rm: schedule_optimal(js, ci, capacity_kw=M, baseline_hours=rm)),
                         ("uncapped_greedy", lambda js, rm: schedule(js, ci, baseline_hours=rm))):
        b_all, o_all, m_all = [], [], []
        for m in range(1, 13):
            js = [j for j, mm in zip(jobs, month) if mm == m]
            rm = {j.job_id: rh[j.job_id] for j in js}
            a = solver(js, rm)["assignments"]
            b, o = per_job(js, rm, a, ci)
            b_all.append(b); o_all.append(o); m_all.append(np.full(len(js), m))
        arms[name] = (np.concatenate(b_all), np.concatenate(o_all), np.concatenate(m_all))
        print(f"  {name}: annual EW = {100*(arms[name][0].sum()-arms[name][1].sum())/arms[name][0].sum():.3f}%"
              f"   ({time.time()-t0:.0f}s)")

    rows = []
    for m in range(1, 13):
        r = {"month": m, "name": MONTHS[m - 1]}
        for name, (b, o, mm) in arms.items():
            s = mm == m
            r[f"{name}_ew"] = 100.0 * (b[s].sum() - o[s].sum()) / b[s].sum()
            r[f"{name}_uw"] = float(np.mean(100.0 * (b[s] - o[s]) / b[s]))
        sel = month == m
        r["n_jobs"] = int(sel.sum())
        r["mean_power_kw"] = float(power[sel].mean())
        r["median_power_kw"] = float(np.median(power[sel]))
        r["share_power_gt_M"] = float((power[sel] > M).mean())
        r["mean_ci_metered_hour"] = float(np.mean([ci.loc[rh[j.job_id]] for j, s in zip(jobs, sel) if s]))
        rows.append(r)
    monthly = pd.DataFrame(rows)

    print("\n### monthly savings (%), EW = energy-weighted aggregate, UW = mean of per-job %")
    print(monthly[["name", "n_jobs", "capped_greedy_ew", "capped_greedy_uw", "capped_milp_ew",
                   "capped_milp_uw", "uncapped_greedy_ew", "uncapped_greedy_uw"]].round(2).to_string(index=False))

    print("\n### season aggregates (%)")
    srows = []
    for lbl, ms in SEASONS:
        r = {"season": lbl}
        for name, (b, o, mm) in arms.items():
            s = np.isin(mm, ms)
            r[f"{name}_ew"] = 100.0 * (b[s].sum() - o[s].sum()) / b[s].sum()
            r[f"{name}_uw_jobs"] = float(np.mean(100.0 * (b[s] - o[s]) / b[s]))
            r[f"{name}_uw_months"] = float(monthly.loc[monthly["month"].isin(ms), f"{name}_ew"].mean())
        srows.append(r)
    seasons = pd.DataFrame(srows)
    print(seasons[["season", "capped_greedy_ew", "capped_greedy_uw_months", "capped_greedy_uw_jobs",
                   "capped_milp_ew", "uncapped_greedy_ew"]].round(2).to_string(index=False))
    print("\nsubmitted seasonal figures: spring 11.1, fall 10.7, winter 6.7, summer 7.7; January 3.1")

    # ---- jobs left at their metered hour under the budget, by month (whole-year solve)
    print("\n### unshifted jobs at the budget (whole-year solve)")
    out = {}
    for name, solver in (("milp", lambda: schedule_optimal(jobs, ci, capacity_kw=M, baseline_hours=rh)),
                         ("greedy", lambda: schedule(jobs, ci, capacity_kw=M, baseline_hours=rh))):
        a = solver()["assignments"]
        stay = np.array([a[j.job_id] == rh[j.job_id] for j in jobs])
        out[name] = stay
        print(f"  {name}: {int(stay.sum()):,} of {len(jobs):,} stay put   ({time.time()-t0:.0f}s)")
    binding = pd.DataFrame({
        "month": range(1, 13), "name": MONTHS,
        "n_jobs": [int((month == m).sum()) for m in range(1, 13)],
        "unshifted_milp": [int(out["milp"][month == m].sum()) for m in range(1, 13)],
        "unshifted_greedy": [int(out["greedy"][month == m].sum()) for m in range(1, 13)],
    })
    binding["share_unshifted_milp"] = binding["unshifted_milp"] / binding["n_jobs"]
    binding = binding.merge(monthly[["month", "mean_power_kw", "median_power_kw",
                                     "share_power_gt_M", "mean_ci_metered_hour"]], on="month")
    print(binding.round(3).to_string(index=False))
    print(f"\nbudget M = {M:.4f} kW; months whose MEAN job power exceeds it: "
          f"{', '.join(binding.loc[binding.mean_power_kw > M, 'name'])}")
    print(f"annual mean CI at metered hours = "
          f"{float(np.mean([ci.loc[rh[j.job_id]] for j in jobs])):.1f} gCO2/kWh")

    monthly.to_csv(ROOT / "results/r2_monthly_savings.csv", index=False)
    seasons.to_csv(ROOT / "results/r2_season_savings.csv", index=False)
    binding.to_csv(ROOT / "results/r2_monthly_binding.csv", index=False)
    print(f"\nwrote 3 files to results/  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
