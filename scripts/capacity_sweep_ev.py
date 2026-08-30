"""Generator for results/capacity_sweep_ev.csv -- the EV capacity sweep.

Mirrors scripts/capacity_sweep.py (HVAC) so the two arms are directly comparable:
same k x median_active_kw cap grid, same greedy-vs-MILP pair, same columns.

EV differs from HVAC in one way that matters: the scheduling window is OBSERVED
(floor(connectionTime) .. floor(disconnectTime)), not assigned, so there is no
flex_hours knob. The baseline hour is the plug-in hour, which makes a
first-come-first-served baseline and an unshifted baseline coincide exactly.

The real Caltech site cap is 150 kW; it is carried as an extra row because it is
~47.5x median and effectively non-binding, which is itself the finding.

Runtime is dominated by the tightest caps (branch-and-bound is hardest near the
binding threshold), exactly as k=4 dominates the HVAC sweep.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import pandas as pd

import nb_utils as U
from cals import schedule, schedule_optimal
from carbon_sim import CAP_SWEEP_K, CAP_SWEEP_K_REPORTED, median_active_kw

SITE_CAP_KW = 150.0  # real Caltech ACN site rating


def main() -> None:
    ci, source = U.get_carbon_intensity()
    jobs, label = U.get_ev_jobs(verbose=False)
    run_hours = {j.job_id: j.earliest_start for j in jobs}  # plug-in hour == unshifted
    base = U.do_nothing_gco2(jobs, run_hours, ci)
    med = median_active_kw(jobs)
    peak = max(j.power_kw for j in jobs)
    print(f"carbon: {source}  ({len(ci)} hours)")
    print(f"jobs={len(jobs)}  median active={med:.4f} kW  observed peak={peak:.4f} kW")
    print(f"site cap {SITE_CAP_KW:g} kW = {SITE_CAP_KW/med:.1f}x median")

    grid: list[tuple[str, float | None, float | None]] = [
        (f"{k:g}x med", k, round(k * med, 6)) for k in CAP_SWEEP_K
    ]
    grid.append(("site 150kW", SITE_CAP_KW / med, SITE_CAP_KW))
    grid.append(("uncapped", None, None))

    rows = []
    for lab, k, cap_kw in grid:
        t0 = time.time()
        g = schedule(jobs, ci, capacity_kw=cap_kw, baseline_hours=run_hours)
        t_greedy = time.time() - t0
        t0 = time.time()
        o = schedule_optimal(jobs, ci, capacity_kw=cap_kw, baseline_hours=run_hours)
        t_milp = time.time() - t0

        gp = U.savings_pct(base, g["total_gco2"])
        op = U.savings_pct(base, o["total_gco2"])
        rows.append({
            "label": lab, "k": k, "M_kW": cap_kw,
            "greedy_pct": gp, "optimal_pct": op, "gap_pp": op - gp,
            "do_nothing": len(o.get("do_nothing", [])), "status": o.get("status"),
            "t_greedy_s": t_greedy, "t_milp_s": t_milp,
            "degenerate": k is not None and k in CAP_SWEEP_K and k not in CAP_SWEEP_K_REPORTED,
        })
        cs = "  n/a  " if cap_kw is None else f"{cap_kw:7.4f}"
        print(f"  {lab:11} M={cs}  greedy={gp:7.4f}%  optimal={op:7.4f}%  "
              f"gap={op-gp:6.4f} pp  do_nothing={rows[-1]['do_nothing']:5d}  "
              f"({t_milp:.0f}s MILP)", flush=True)

    out = ROOT / "results" / "capacity_sweep_ev.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
