"""Restore 02, 04a and 04e to their COMMITTED content, with the in-figure title removed.

This reverts the session's content changes -- the three-load slack ECDF, the merged
HVAC+batch flexibility figure, and the single-panel 04e -- and keeps only the MDPI
title stripping. Plot code below is a faithful transcription of the originating
notebook cells (02_loads.ipynb and 04_sweeps.ipynb) minus every set_title call;
figsize, colours, markers, annotations and limits are unchanged.

NOTE. The original 02 title carried real content -- the explanation of why HVAC and
batch are absent (their slack is constant by construction, so they would be single
degenerate bars). That sentence must move into the caption; it is not decoration.
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
from carbon_sim import MA_REDATE
from cuad.scheduler.adapters import nrel_to_jobs
from cuad.scheduler.jobs import duration_h, slack_h


def fig02(ev_jobs, hvac_jobs):
    ev_slack = np.array([slack_h(j) for j in ev_jobs])
    hvac_slack = np.array([slack_h(j) for j in hvac_jobs])
    assert len(set(hvac_slack.tolist())) == 1, "HVAC slack should be constant by construction"
    n_zero = int((ev_slack == 0).sum())
    n_tail = int((ev_slack > 24).sum())
    counts = [int((ev_slack == k).sum()) for k in range(25)]

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.bar(range(25), counts, width=0.85, edgecolor="white",
           color=["#c1121f"] + ["#4878a8"] * 24)
    ax.set_xlim(-0.8, 25.2)
    ax.set_xticks(range(0, 26, 2))
    ax.set_xlabel("slack (hours)  =  dwell $-$ charge duration")
    ax.set_ylabel("EV charging jobs")
    ax.annotate(
        f"{n_zero:,} jobs ({100 * n_zero / len(ev_jobs):.0f}%) have ZERO slack\n"
        "— they cannot be shifted at all and\nemit at their plug-in hour regardless",
        xy=(0.45, counts[0]), xytext=(3.1, counts[0] * 0.99), fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fdecec", ec="#c1121f", lw=1.2),
        arrowprops=dict(arrowstyle="-", color="#c1121f", lw=1.2))
    ax.text(0.985, 0.74, f"{n_tail} jobs with slack > 24 h not shown\n"
                         f"(tail runs to {int(ev_slack.max())} h)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6"))
    fig.tight_layout(); U.savefig(fig, "02_slack_histograms.png")

    print(f"[02] EV-only slack histogram, n={len(ev_jobs):,}, median "
          f"{np.median(ev_slack):.0f} h, max {int(ev_slack.max())} h")
    print(f"     HVAC constant slack = {int(hvac_slack[0])} h (omitted, as in the committed version)")
    print(f"     zero-slack bar (red) = {n_zero:,} ({100*n_zero/len(ev_jobs):.1f}%)   "
          f"beyond 24 h, not shown = {n_tail}")
    print("     bar heights, slack 0..24 h:")
    for k in range(0, 25, 5):
        print("       " + "  ".join(f"s={kk:2d}:{counts[kk]:5,}" for kk in range(k, min(k + 5, 25))))
    return counts


def fig04a(ci, hvac_df):
    flex_vals = list(range(2, 9))
    sav_flex = []
    for f in flex_vals:
        jobs, rh = nrel_to_jobs(hvac_df, utc_offset_hours=-5, flex_hours=f)
        fb = U.do_nothing_gco2(jobs, rh, ci)
        op = schedule(jobs, ci, baseline_hours=rh)["total_gco2"]
        sav_flex.append(U.savings_pct(fb, op))

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(flex_vals, sav_flex, "o-", color="#2a9d8f")
    ax.set_xlabel("flex_hours (slack granted each HVAC hour)")
    ax.set_ylabel("carbon saved (%)")
    fig.tight_layout(); U.savefig(fig, "04a_savings_vs_flex.png")

    print("[04a] HVAC-only, uncapped greedy, x-axis is flex_hours (NOT slack):")
    for f, s in zip(flex_vals, sav_flex):
        print(f"      flex_hours={f}  saving={s:7.4f}%")
    return sav_flex


def fig04e():
    ai_sweep = pd.read_csv(ROOT / "results/ai_sweep.csv")
    pct = ai_sweep.groupby("flex_hours")["savings_pct"].mean().sort_index()
    spread = float(ai_sweep.groupby("flex_hours")["savings_pct"]
                   .agg(lambda s: s.max() - s.min()).max())
    assert spread < 1e-6, f"savings_pct must not depend on gpu_power_kw (spread {spread:g})"

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(pct.index, pct.to_numpy(), "o-", color="#2a9d8f", lw=1.8, ms=6)
    for x, y in zip(pct.index, pct.to_numpy()):
        axes[0].annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                         ha="center", fontsize=9)
    powers = "/".join(f"{p:g}" for p in sorted(ai_sweep["gpu_power_kw"].unique()))
    axes[0].text(0.985, 0.05,
                 f"identical for all\ngpu_power_kw ({powers}):\npower cancels in the ratio",
                 transform=axes[0].transAxes, ha="right", va="bottom", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6"))
    axes[0].set_ylim(-0.7, float(pct.max()) * 1.20)
    axes[0].set_xlabel("assigned flexibility (hours)")
    axes[0].set_ylabel("carbon saved (%)")

    for gpu_power_kw, group in ai_sweep.groupby("gpu_power_kw", sort=True):
        group = group.sort_values("flex_hours")
        axes[1].plot(group["flex_hours"], group["avoided_gco2"] / 1e6, "o-",
                     label=f"{gpu_power_kw:.1f} kW/GPU")
    axes[1].set_xlabel("assigned flexibility (hours)")
    axes[1].set_ylabel("avoided emissions (tCO2)")
    axes[1].legend(title="assumed power")
    fig.tight_layout(); U.savefig(fig, "04e_ai_savings_sweep.png")

    print("[04e] TWO panels restored.")
    print("      (e1) savings %% vs flexibility (single curve, power cancels; spread %.2e):" % spread)
    print("           " + ", ".join(f"flex{int(x)}={y:.4f}%" for x, y in pct.items()))
    print("      (e2) avoided emissions (tCO2) by assumed per-GPU power:")
    for gp, g in ai_sweep.groupby("gpu_power_kw", sort=True):
        g = g.sort_values("flex_hours")
        print(f"           {gp:.1f} kW/GPU: " + ", ".join(
            f"flex{int(f)}={v/1e6:.4f}" for f, v in zip(g["flex_hours"], g["avoided_gco2"])))


def main():
    ci, _ = U.get_carbon_intensity()
    hvac_jobs, _, _ = U.load_hvac_jobs(flex_hours=6)
    ev_jobs, _ = U.get_ev_jobs(verbose=False)
    hvac_df = pd.read_parquet(ROOT / "data/hvac/bldg486202_MA_year.parquet").copy()
    hvac_df["timestamp"] = pd.to_datetime(hvac_df["timestamp"]) + MA_REDATE

    fig02(ev_jobs, hvac_jobs)
    print()
    fig04a(ci, hvac_df)
    print()
    fig04e()


if __name__ == "__main__":
    main()
