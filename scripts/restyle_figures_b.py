"""Regenerate 04c and 04d WITHOUT in-figure titles (04b is no longer generated here) (MDPI: text goes in the caption).

04b is redrawn from results/capacity_sweep.csv (the committed HVAC MILP sweep) rather
than re-solving it -- same numbers, ~15 min cheaper. 04c and 04d are recomputed live;
both are capped-greedy only, so both are seconds.
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
from cals import schedule
from carbon_sim import WORKING_CAP_K, median_active_kw
from cuad.carbon.factors import FACTORS

OTH_VALS = [130, 200, 230, 300, 420, 490, 600, 700]


def fig04b(uncapped_pct):
    d = pd.read_csv(ROOT / "results/capacity_sweep.csv")
    xs = d["M_kW"].to_numpy(float)
    gy = d["greedy_pct"].to_numpy(float)
    oy = d["optimal_pct"].to_numpy(float)
    deg = d["degenerate"].to_numpy(bool)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.axvspan(xs.min() * 0.86, xs[deg].max(), color="0.85", alpha=0.55, zorder=0,
               label=r"degenerate (cap $\leq$ median)")
    ax.fill_between(xs, gy, oy, color="#e76f51", alpha=0.15, zorder=1, label="greedy shortfall")
    ax.plot(xs, gy, "s-", color="#264653", lw=1.8, ms=6, zorder=3, label="greedy heuristic")
    ax.plot(xs, oy, "o--", color="#e76f51", lw=1.8, ms=6, zorder=3, label="MILP optimum")
    ax.axhline(uncapped_pct, ls=":", color="0.4", lw=1.4, zorder=2,
               label=f"uncapped ceiling ({uncapped_pct:.2f}%)")
    w = d[d["k"] == WORKING_CAP_K].iloc[0]
    ax.axvline(float(w["M_kW"]), ls="--", color="0.35", lw=1.1, zorder=2)
    ax.annotate(f"working cap k={WORKING_CAP_K:g}\n({float(w['M_kW']):.2f} kW)\n"
                f"greedy {float(w['greedy_pct']):.2f}% / opt {float(w['optimal_pct']):.2f}%",
                xy=(float(w["M_kW"]), float(w["optimal_pct"])), xytext=(1.55, 5.6),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", color="0.25", lw=1.0))
    ax.set_xscale("log"); ax.set_xticks(xs)
    ax.set_xticklabels([f"{v:.2f}" for v in xs]); ax.minorticks_off()
    ax.set_xlabel(r"per-hour capacity cap $M$ (kW, log scale; ticks = $k\times$median, then peak)")
    ax.set_ylabel("carbon saved (%)")
    ax.grid(alpha=0.22, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout(); U.savefig(fig, "04b_savings_vs_capacity.png")
    print("[04b] HVAC capacity sweep (from results/capacity_sweep.csv):")
    for _, r in d.iterrows():
        print(f"      {r['label']:9} M={r['M_kW']:8.4f} kW  greedy={r['greedy_pct']:7.4f}%  "
              f"opt={r['optimal_pct']:7.4f}%  gap={r['gap_pp']:6.4f} pp"
              f"{'   [degenerate]' if r['degenerate'] else ''}")


def fig04c(ci, hv, rh):
    cap = round(WORKING_CAP_K * median_active_kw(hv), 6)
    month = np.array([pd.Timestamp(rh[j.job_id]).month for j in hv])
    sav = []
    for m in range(1, 13):
        jm = [j for j, mm in zip(hv, month) if mm == m]
        rm = {j.job_id: rh[j.job_id] for j in jm}
        fb = U.do_nothing_gco2(jm, rm, ci)
        op = schedule(jm, ci, capacity_kw=cap, baseline_hours=rm)["total_gco2"]
        sav.append(U.savings_pct(fb, op))
    labels = list("JFMAMJJASOND")
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.bar(range(12), sav, color="#4c7fa3", width=0.7)
    ax.set_xticks(range(12)); ax.set_xticklabels(labels)
    ax.set_xlabel("month"); ax.set_ylabel("carbon saved (%)")
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); U.savefig(fig, "04c_savings_by_month.png")
    print(f"[04c] monthly greedy saving at cap {cap:.4f} kW (k={WORKING_CAP_K:g}x median):")
    print("      " + ", ".join(f"{l}={v:.2f}%" for l, v in zip(labels, sav)))
    print(f"      min={min(sav):.2f}% ({labels[int(np.argmin(sav))]})  "
          f"max={max(sav):.2f}% ({labels[int(np.argmax(sav))]})")
    return sav


def fig04d(mix, hv, rh):
    cap = round(WORKING_CAP_K * median_active_kw(hv), 6)
    sav = []
    for v in OTH_VALS:
        f = dict(FACTORS); f["OTH"] = v
        ci_v = U.ci_from_factors(mix, f)
        fb = U.do_nothing_gco2(hv, rh, ci_v)
        op = schedule(hv, ci_v, capacity_kw=cap, baseline_hours=rh)["total_gco2"]
        sav.append(U.savings_pct(fb, op))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(OTH_VALS, sav, "^-", color="#e76f51", lw=1.8, ms=7)
    ax.axvline(230, ls="--", color="gray", label="default (230, AR5 biomass)")
    ax.axvline(700, ls=":", color="gray", label="CarbonCast 'Other' (700)")
    ax.set_xlabel("OTH emission factor (gCO2/kWh)"); ax.set_ylabel("carbon saved (%)")
    ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout(); U.savefig(fig, "04d_savings_vs_oth_factor.png")
    print(f"[04d] OTH sensitivity, greedy at cap {cap:.4f} kW:")
    print("      " + ", ".join(f"{v}={s:.2f}%" for v, s in zip(OTH_VALS, sav)))
    print(f"      range {min(sav):.2f}%-{max(sav):.2f}% across OTH {OTH_VALS[0]}-{OTH_VALS[-1]}")
    return sav


def main():
    ci, _ = U.get_carbon_intensity()
    mix, _ = U.get_fuel_mix()
    hv, rh, _ = U.load_hvac_jobs(flex_hours=6)
    base = U.do_nothing_gco2(hv, rh, ci)
    uncapped = U.savings_pct(base, schedule(hv, ci, baseline_hours=rh)["total_gco2"])
    # fig04b() intentionally NOT called: 04b is now the combined single-panel
    # HVAC+batch figure, on the shared k = M / median-aggregate axis, owned by
    # scripts/fig_savings_vs_capacity.py. Retained for reference.
    fig04c(ci, hv, rh); fig04d(mix, hv, rh)


if __name__ == "__main__":
    main()
