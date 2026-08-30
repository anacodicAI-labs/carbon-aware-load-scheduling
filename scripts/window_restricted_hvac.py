"""Generator for results/window_restricted_hvac.csv -- the HVAC saving restricted to
the batch trace's calendar window.

Reproduces the like-for-like comparison Section 4.4 states: the heat-pump uncapped
ceiling is 12.70% over the full year 2019 but 9.57% over the 69 days the Alibaba
batch trace spans, so the batch arm's 3.21% ceiling and the heat pump's differ by
about a factor of three rather than four.

Run from anywhere:  python scripts/window_restricted_hvac.py
Runtime ~2 min; the full-year k=3 MILP dominates at ~95 s.

WHY THIS FILE EXISTS. Figure 7 (04b) plots the heat-pump and batch capacity sweeps
on one shared axis, but the two loads do not cover the same calendar span: HVAC and
EV run over all of 2019 while the Alibaba trace is anchored to 2019-07-07..09-13.
A reader taking the ratio of the two ceilings straight off the figure would
overstate it by about 25%, because summer is a low-savings season for the heat pump
(Section 4.5). That correction is quoted in the paper, so it needs a generator.

WINDOW IS DERIVED, NOT HARDCODED. The bounds come from the batch job set itself
(min earliest_start to max end), so if the trace anchor in nb_utils.get_ai_jobs
ever changes, this script follows it instead of silently reporting a stale window.

UNCAPPED USES GREEDY. With M -> infinity the jobs do not interact and a greedy
minimum-cost placement is provably the exact optimum (Section 3.3); the full-year
control below reproduces the published 12.6986% on that basis. The capped rows do
call the MILP, since that is where greedy and optimum diverge.

DETERMINISM. Every column except the timing columns is deterministic.
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
from cuad.scheduler.jobs import duration_h

FULL_YEAR_UNCAPPED = 12.6986  # published Section 4.4 figure, reproduced as a control


def arm(label, jobs, run_hours, ci, cap_kw, want_milp):
    """One (span, cap) cell: greedy always, exact MILP only where the cap binds."""
    base = U.do_nothing_gco2(jobs, run_hours, ci)
    t0 = time.time()
    g = schedule(jobs, ci, capacity_kw=cap_kw, baseline_hours=run_hours)
    t_greedy = time.time() - t0
    greedy_pct = U.savings_pct(base, g["total_gco2"])

    optimal_pct, gap, t_milp, status = float("nan"), float("nan"), float("nan"), None
    if want_milp:
        t0 = time.time()
        o = schedule_optimal(jobs, ci, capacity_kw=cap_kw, baseline_hours=run_hours)
        t_milp = time.time() - t0
        optimal_pct = U.savings_pct(base, o["total_gco2"])
        gap = optimal_pct - greedy_pct
        status = o.get("status")
    return {
        "span": label, "n_jobs": len(jobs),
        "cap": "uncapped" if cap_kw is None else f"{WORKING_CAP_K:g}x med",
        "M_kW": cap_kw, "greedy_pct": greedy_pct, "optimal_pct": optimal_pct,
        "gap_pp": gap, "base_gco2": base, "status": status,
        "t_greedy_s": t_greedy, "t_milp_s": t_milp,
    }


def main() -> None:
    ci, source = U.get_carbon_intensity()
    print(f"carbon: {source}  ({len(ci)} hours)")

    # --- the batch trace's own window, read off the batch job set ---
    bj, _, _ = U.get_ai_jobs(gpu_power_kw=0.4, flex_hours=6, verbose=False)
    w0 = min(j.earliest_start for j in bj)
    w1 = max(j.earliest_start + pd.Timedelta(hours=duration_h(j) - 1) for j in bj)
    days = (w1 - w0).days + 1
    print(f"batch trace window: {w0} -> {w1}  ({days} days, {len(bj):,} jobs)")

    hv, rh, _ = U.load_hvac_jobs(flex_hours=6)
    sub = [j for j in hv if w0 <= pd.Timestamp(rh[j.job_id]) <= w1]
    rh_sub = {j.job_id: rh[j.job_id] for j in sub}
    cap = round(WORKING_CAP_K * median_active_kw(hv), 6)
    print(f"HVAC: {len(hv):,} jobs full year, {len(sub):,} in window; "
          f"working cap M={cap:.4f} kW\n")

    rows = [
        arm("full_year_2019", hv, rh, ci, None, False),
        arm("batch_window", sub, rh_sub, ci, None, False),
        arm("full_year_2019", hv, rh, ci, cap, True),
        arm("batch_window", sub, rh_sub, ci, cap, True),
    ]
    for r in rows:
        o = "   n/a  " if np.isnan(r["optimal_pct"]) else f"{r['optimal_pct']:7.4f}%"
        print(f"  {r['span']:15} {r['cap']:9} n={r['n_jobs']:5d}  "
              f"greedy={r['greedy_pct']:7.4f}%  optimal={o}")

    unc = {r["span"]: r["greedy_pct"] for r in rows if r["cap"] == "uncapped"}
    fy, bw = unc["full_year_2019"], unc["batch_window"]
    assert abs(fy - FULL_YEAR_UNCAPPED) < 5e-4, f"control drifted: {fy} vs {FULL_YEAR_UNCAPPED}"
    print(f"\nCONTROL OK: full-year uncapped {fy:.4f}% == published {FULL_YEAR_UNCAPPED}%")
    print(f"WINDOW-RESTRICTED uncapped = {bw:.4f}%  "
          f"({bw - fy:+.4f} pp, {bw / fy:.2f}x the full-year figure)")

    # why: the summer window is dirtier and materially less variable
    ciw = ci[(ci.index >= w0) & (ci.index <= w1)]
    for nm, s in (("full year", ci), ("batch window", ciw)):
        print(f"  CI {nm:13}: median {s.median():6.1f}  IQR {s.quantile(.75) - s.quantile(.25):5.1f}  "
              f"min {s.min():6.1f}  max {s.max():6.1f} gCO2/kWh")

    out = ROOT / "results" / "window_restricted_hvac.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
