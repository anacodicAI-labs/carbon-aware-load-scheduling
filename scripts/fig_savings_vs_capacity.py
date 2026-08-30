"""Generator for figures/04b_savings_vs_capacity.png -- capacity sweep, HVAC and batch,
combined into a SINGLE panel.

WHY ONE PANEL. The two arms were previously drawn side by side, which made them
non-comparable: HVAC's budget axis spans 0.21-6.67 kW and batch's spans 127-1523 kW, a
600x difference driven entirely by fleet size, not by behaviour. Both sweeps are
parameterised identically as M = k x median hourly AGGREGATE load, so plotting against
the dimensionless k puts them on one axis and makes the substantive comparison visible:
batch reaches its uncapped ceiling exactly at k=4, while HVAC is still climbing at
k=15.8 (its peak-load cap). That contrast is the point of the figure and the two-panel
layout concealed it.

ENCODING. Colour = load class (HVAC dark, batch orange). Linestyle/marker = solver
(solid+filled marker = greedy, dashed+open circle = exact MILP). The shortfall band is
the HVAC greedy-to-optimum gap. This differs from the two-panel version, where colour
carried the solver; with four series on one axes colour has to carry the arm.

CAP BASIS. Budgets are k x the median hourly AGGREGATE load, for both arms. For HVAC
this is identical to the old k x median JOB power basis, because the adapter emits
exactly one block per hour (1.00 jobs/hour, median aggregate == median job power ==
0.4230 kW); HVAC's published 7.28 / 8.29 / 1.01 pp numbers at k=3 are verified
unchanged. For batch the two differ by 635x: median job power is 0.400 kW but the
median aggregate is 253.8 kW, so a job-power basis would make every capped point
measure forced fallback. The fleet runs ~664 concurrent jobs per active hour (mean;
median 697) across 1,642 active hours of the year. The 635x is the ratio of the two
cap bases, not the concurrency; the two are close but not equal because job powers
vary (26 distinct values, 0.004-3.2 kW).

BATCH HAS NO MILP ARM. schedule_optimal builds dense constraint matrices and is
OOM-killed (exit 137) above ~32,000 jobs on this machine; batch is 227,529 jobs.
Greedy only for that arm, stated on the panel.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nb_utils as U
from carbon_sim import WORKING_CAP_K

HVAC_UNCAPPED = 12.6986  # uncapped greedy == exact optimum (see fig_savings_vs_flex.py)

C_HVAC = "#264653"   # dark teal  -- HVAC heat pump
C_BATCH = "#e76f51"  # orange     -- batch GPU


def main() -> None:
    hv = pd.read_csv(ROOT / "results/capacity_sweep.csv")
    ba = pd.read_csv(ROOT / "results/capacity_sweep_batch.csv")

    # --- put both arms on the shared dimensionless budget axis k = M / median aggregate ---
    hv_med = float(hv.loc[hv["k"] == 1.0, "M_kW"].iloc[0])          # 0.4230 kW
    hv = hv.copy()
    hv["k_eff"] = hv["M_kW"].to_numpy(float) / hv_med               # 'peak' row -> k = 15.76
    hk = hv["k_eff"].to_numpy(float)
    gy = hv["greedy_pct"].to_numpy(float)
    oy = hv["optimal_pct"].to_numpy(float)

    ba_k = ba[ba["label"].str.contains("med")].copy()
    bk = ba_k["k"].to_numpy(float)
    gb = ba_k["greedy_pct"].to_numpy(float)
    ba_unc = float(ba.loc[ba["label"] == "uncapped", "greedy_pct"].iloc[0])

    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    # --- degenerate region: cap at or below the median aggregate load (k <= 1) ---
    ax.axvspan(hk.min() * 0.86, 1.0, color="0.85", alpha=0.55, zorder=0,
               label=r"degenerate (cap $\leq$ median aggregate)")

    # --- HVAC: greedy, exact MILP, and the shortfall between them ---
    ax.fill_between(hk, gy, oy, color=C_HVAC, alpha=0.16, zorder=1,
                    label="HVAC greedy shortfall")
    ax.plot(hk, gy, "s-", color=C_HVAC, lw=1.8, ms=6, zorder=4,
            label="HVAC heat pump, greedy (n=8,363)")
    ax.plot(hk, oy, "o--", color=C_HVAC, lw=1.8, ms=6, zorder=4,
            mfc="white", mew=1.5, label="HVAC heat pump, exact MILP")
    ax.axhline(HVAC_UNCAPPED, ls=":", color=C_HVAC, lw=1.5, alpha=0.85, zorder=2,
               label=f"HVAC uncapped ceiling ({HVAC_UNCAPPED:.2f}%)")

    # --- batch: greedy only ---
    ax.plot(bk, gb, "^-", color=C_BATCH, lw=1.8, ms=7, zorder=4,
            label="Batch GPU, greedy only (n=227,529; MILP OOM-infeasible)")
    ax.axhline(ba_unc, ls=":", color=C_BATCH, lw=1.5, alpha=0.85, zorder=2,
               label=f"Batch uncapped ceiling ({ba_unc:.2f}%)")

    # --- working cap ---
    w = hv[hv["k"] == WORKING_CAP_K].iloc[0]
    ax.axvline(WORKING_CAP_K, ls="--", color="0.35", lw=1.1, zorder=2)
    ax.annotate(f"working cap $k$={WORKING_CAP_K:g} ({float(w['M_kW']):.2f} kW for HVAC)\n"
                f"greedy {float(w['greedy_pct']):.2f}% / opt {float(w['optimal_pct']):.2f}%"
                f" / gap {float(w['gap_pp']):.2f} pp",
                xy=(WORKING_CAP_K, float(w["optimal_pct"])), xytext=(3.5, 5.2),
                fontsize=8.2, arrowprops=dict(arrowstyle="->", color="0.25", lw=1.0))

    # --- the comparison the shared axis exists to make ---
    # grey, straight, and shrunk clear of the line: an orange curved arrow here reads
    # as a continuation of the batch series and implies a decline that is not in the data
    ax.annotate("batch reaches its ceiling at $k$=4;\nHVAC has not at $k$=16",
                xy=(4.3, ba_unc), xytext=(6.2, 1.15), fontsize=8.4, style="italic",
                color="0.3", ha="left",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=1.0,
                                shrinkA=2, shrinkB=6))

    ticks = [0.5, 1, 2, 3, 4, 6, float(hv["k_eff"].max())]
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0.5", "1", "2", "3", "4", "6", "15.8\n(HVAC peak)"], fontsize=8.4)
    ax.minorticks_off()
    ax.set_xlabel(r"per-hour budget $M$, as a multiple $k$ of each load's median hourly "
                  r"aggregate draw (log)")
    ax.set_ylabel("carbon saved (%)")
    ax.set_ylim(0, HVAC_UNCAPPED * 1.16)
    ax.grid(alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=8.0,
              framealpha=0.95, ncol=2, columnspacing=1.6, handlelength=2.4,
              borderaxespad=0.0)

    fig.tight_layout()
    U.savefig(fig, "04b_savings_vs_capacity.png")

    print("[04b] combined panel, shared axis k = M / median hourly aggregate load")
    print("  HVAC (k x median aggregate == k x median job power for this arm):")
    for _, r in hv.iterrows():
        print(f"    {r['label']:9} k={r['k_eff']:6.2f}  M={r['M_kW']:8.4f} kW  "
              f"greedy={r['greedy_pct']:7.4f}%  opt={r['optimal_pct']:7.4f}%  "
              f"gap={r['gap_pp']:6.4f} pp{'   [degenerate]' if r['degenerate'] else ''}")
    print(f"    uncapped ceiling = {HVAC_UNCAPPED:.4f}%")
    print("  batch (k x median aggregate = 253.83 kW), greedy only:")
    for _, r in ba.iterrows():
        k = "  n/a " if pd.isna(r["k"]) else f"{r['k']:5.2f}"
        m = "   n/a  " if pd.isna(r["M_kW"]) else f"{r['M_kW']:8.2f}"
        print(f"    {r['label']:9} k={k}  M={m} kW  greedy={r['greedy_pct']:7.4f}%  "
              f"opt=n/a (OOM)")


if __name__ == "__main__":
    main()
