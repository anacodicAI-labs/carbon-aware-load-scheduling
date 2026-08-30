"""Generator for figures/04f_savings_vs_capacity_ev_batch.png.

The Figure-7 (04b) equivalent for the EV and batch arms, same k x median_active_kw
cap convention so all three arms are comparable.

LEFT (EV): greedy and exact MILP, as for HVAC. The real Caltech site rating of
150 kW is marked; it sits at 47.5x median and is effectively non-binding
(greedy == optimum to 0.0000 pp, and 99.0% of the uncapped saving).

RIGHT (batch): greedy ONLY. schedule_optimal builds dense constraint matrices and
is OOM-killed above ~32,000 jobs on this machine; the batch arm is 227,529 jobs, so
no exact optimum exists for it and the greedy/MILP gap cannot be drawn. The k x
median caps are also degenerate here: median_active_kw is the median JOB power,
which equals median hourly SITE load only for HVAC (one block per hour). Batch runs
~139 jobs/hour, so a 3x median cap of 1.2 kW is ~635x tighter than 3x the median
aggregate load (253.8 kW) and admits about three jobs per hour out of the fleet.
Those rows measure forced fallback, not scheduling, and are shaded as such.
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

SITE_CAP_KW = 150.0


def main() -> None:
    ev = pd.read_csv(ROOT / "results/capacity_sweep_ev.csv")
    ba = pd.read_csv(ROOT / "results/capacity_sweep_batch.csv")

    ev_unc = float(ev[ev.label == "uncapped"]["greedy_pct"].iloc[0])
    ba_unc = float(ba[ba.label == "uncapped"]["greedy_pct"].iloc[0])
    ev_k = ev[ev.label.str.contains("med")].copy()
    ev_site = ev[ev.label == "site 150kW"].iloc[0]
    ba_k = ba[ba.label.str.contains("med")].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.1))

    # ---- EV -----------------------------------------------------------------
    ax = axes[0]
    xs = np.append(ev_k["M_kW"].to_numpy(float), SITE_CAP_KW)
    gy = np.append(ev_k["greedy_pct"].to_numpy(float), float(ev_site["greedy_pct"]))
    oy = np.append(ev_k["optimal_pct"].to_numpy(float), float(ev_site["optimal_pct"]))
    deg = ev_k[ev_k["k"] <= 1.0]["M_kW"].max()
    ax.axvspan(xs.min() * 0.7, deg, color="0.85", alpha=0.55, zorder=0,
               label=r"degenerate (cap $\leq$ median)")
    ax.fill_between(xs, gy, oy, color="#e76f51", alpha=0.15, zorder=1, label="greedy shortfall")
    ax.plot(xs, gy, "s-", color="#264653", lw=1.8, ms=6, zorder=3, label="greedy heuristic")
    ax.plot(xs, oy, "o--", color="#e76f51", lw=1.8, ms=6, zorder=3, label="MILP optimum")
    ax.axhline(ev_unc, ls=":", color="0.4", lw=1.4, zorder=2,
               label=f"uncapped ceiling ({ev_unc:.2f}%)")
    ax.axvline(SITE_CAP_KW, ls="--", color="#2a6f97", lw=1.3, zorder=2)
    ax.annotate(f"real site cap 150 kW\n($47.5\\times$ median)\nnon-binding: gap 0.00 pp",
                xy=(SITE_CAP_KW, float(ev_site["greedy_pct"])), xytext=(21, 0.34),
                fontsize=8.2, color="#1c55a7",
                arrowprops=dict(arrowstyle="->", color="#2a6f97", lw=1.0))
    ax.set_xscale("log"); ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{v:.2f}" if v < 100 else f"{v:.0f}" for v in xs],
                       fontsize=8, rotation=45, ha="right")
    ax.minorticks_off()
    ax.set_xlabel(r"per-hour cap $M$ (kW, log; ticks = $k\times$median, then site cap)")
    ax.set_ylabel("carbon saved (%)")
    ax.set_ylim(0, ev_unc * 1.25)
    ax.grid(alpha=0.22, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.text(0.98, 0.03, "EV charging (n=8,507)", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9.5, fontweight="bold", color="0.25")

    # ---- batch --------------------------------------------------------------
    ax = axes[1]
    xb = ba_k["M_kW"].to_numpy(float)
    gb = ba_k["greedy_pct"].to_numpy(float)
    ax.axvspan(xb.min() * 0.7, xb.max() * 1.35, color="0.85", alpha=0.55, zorder=0,
               label="degenerate for a fleet\n(cap $\\ll$ aggregate load)")
    ax.plot(xb, gb, "s-", color="#264653", lw=1.8, ms=6, zorder=3, label="greedy heuristic")
    ax.axhline(ba_unc, ls=":", color="0.4", lw=1.4, zorder=2,
               label=f"uncapped ceiling ({ba_unc:.2f}%)")
    ax.set_xscale("log"); ax.set_xticks(list(xb))
    ax.set_xticklabels([f"{v:.2f}" for v in xb], fontsize=8.5); ax.minorticks_off()
    ax.set_xlabel(r"per-hour cap $M$ (kW, log; ticks = $k\times$median job power)")
    ax.set_ylabel("carbon saved (%)")
    ax.set_ylim(0, ba_unc * 1.25)
    ax.grid(alpha=0.22, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.text(0.98, 0.03, "Batch GPU (n=227,529)", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9.5, fontweight="bold", color="0.25")
    ax.text(0.60, 0.46,
            "no MILP arm: the exact optimum is\nOOM-infeasible above ~32,000 jobs\n"
            "median aggregate load is 253.8 kW,\nso every cap shown is ~635$\\times$ too tight",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.2, style="italic",
            color="0.3", bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.7"))

    fig.tight_layout()
    U.savefig(fig, "04f_savings_vs_capacity_ev_batch.png")

    print("[04f-left] EV capacity sweep:")
    for _, r in ev.iterrows():
        m = "  n/a  " if pd.isna(r["M_kW"]) else f"{r['M_kW']:8.4f}"
        print(f"      {r['label']:10} M={m} kW  greedy={r['greedy_pct']:7.4f}%  "
              f"opt={r['optimal_pct']:7.4f}%  gap={r['gap_pp']:6.4f} pp  "
              f"do_nothing={int(r['do_nothing']):5d}")
    print(f"      150 kW retains {100*float(ev_site['greedy_pct'])/ev_unc:.1f}% of the uncapped saving")
    print("[04f-right] batch capacity sweep (greedy only):")
    for _, r in ba.iterrows():
        m = "  n/a  " if pd.isna(r["M_kW"]) else f"{r['M_kW']:8.4f}"
        print(f"      {r['label']:10} M={m} kW  greedy={r['greedy_pct']:7.4f}%  opt=n/a")


if __name__ == "__main__":
    main()
