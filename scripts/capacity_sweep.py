"""Generator for results/capacity_sweep.csv -- the HVAC capacity sweep.

Reproduces the committed CSV that Section 4.4's headline numbers come from:
the exact capped optimum (8.29%), the greedy heuristic (7.28%), and the
1.01 pp gap between them all read off the `3x med` row.

Run from anywhere:  python scripts/capacity_sweep.py
Runtime ~15 min; the k=4 MILP dominates at ~8 min.

WHY THIS FILE EXISTS. The CSV was committed in 11f2803 without a generator, so
for several rounds the paper's most-cited pair of numbers could not be
regenerated from the repository. Notebook 04 cell `ea5e85bd` computes the same
sweep but rounds to two decimals and does not emit do_nothing / status /
timings, so it is not a substitute.

DETERMINISM. Every column except `t_greedy_s` and `t_milp_s` is deterministic:
same inputs give bit-identical values. The two timing columns are wall clock and
will differ on every run and every machine -- they are diagnostics, not results,
and no reported number depends on them.
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


def main() -> None:
    ci, source = U.get_carbon_intensity()
    print(f"carbon: {source}  ({len(ci)} hours)")

    jobs, run_hours, _ = U.load_hvac_jobs(flex_hours=6)
    base_run = U.do_nothing_gco2(jobs, run_hours, ci)
    med = median_active_kw(jobs)
    peak = max(j.power_kw for j in jobs)
    print(f"jobs={len(jobs)}  median active={med:.4f} kW  observed peak={peak:.4f} kW")

    # k x median, then the observed peak as the loosest swept cap. k <= 1 is
    # DEGENERATE (at cap = 1x median half the blocks exceed the cap on their own),
    # so those rows measure forced fallback rather than scheduling; the flag is
    # carried in the CSV so a consumer cannot report them by accident.
    grid: list[tuple[str, float | None, float]] = [
        (f"{k:g}x med", k, round(k * med, 6)) for k in CAP_SWEEP_K
    ]
    grid.append(("peak", None, round(peak, 6)))

    rows = []
    for label, k, cap_kw in grid:
        t0 = time.time()
        g = schedule(jobs, ci, capacity_kw=cap_kw, baseline_hours=run_hours)
        t_greedy = time.time() - t0

        t0 = time.time()
        o = schedule_optimal(jobs, ci, capacity_kw=cap_kw, baseline_hours=run_hours)
        t_milp = time.time() - t0

        greedy_pct = U.savings_pct(base_run, g["total_gco2"])
        optimal_pct = U.savings_pct(base_run, o["total_gco2"])
        rows.append(
            {
                "label": label,
                "k": k,
                "M_kW": cap_kw,
                "greedy_pct": greedy_pct,
                "optimal_pct": optimal_pct,
                "gap_pp": optimal_pct - greedy_pct,
                "do_nothing": len(o.get("do_nothing", [])),
                "status": o.get("status"),
                "t_greedy_s": t_greedy,
                "t_milp_s": t_milp,
                "degenerate": k is not None and k not in CAP_SWEEP_K_REPORTED,
            }
        )
        print(
            f"  {label:10} M={cap_kw:7.4f}  greedy={greedy_pct:7.4f}%  "
            f"optimal={optimal_pct:7.4f}%  gap={optimal_pct - greedy_pct:6.4f} pp  "
            f"do_nothing={rows[-1]['do_nothing']:5d}  ({t_milp:.0f}s MILP)"
        )

    out = ROOT / "results" / "capacity_sweep.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
