"""Reviewer-1 regeneration of figures 02, 03, 04b and 05 from the R1 rerun CSVs.

Every plotted number is read from results/r1_*.csv (written by scripts/r1_reruns.py);
styles are copied from the original generators:
  02  scripts/fig_slack_ecdf.py        EV curve = continuous real slack (valid sessions)
  03  scripts/restyle_figures_a.py     EV bar = exact continuous-time saving
  04b scripts/fig_savings_vs_capacity.py  batch curve = one 732,691-task population
  05  scripts/restyle_figures_a.py     adds the capped oracle/forecast pair
HVAC in 04b still comes from results/capacity_sweep.csv (it has the degenerate k=0.5 and
k=1 points); main() asserts it matches results/r1_hvac_capacity.csv where both exist.
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

R = ROOT / "results"
C_HVAC, C_BATCH, C_EV_BAR, C_HVAC_BAR = "#264653", "#e76f51", "#e9c46a", "#2a9d8f"
HVAC_FLEX = 6            # HVAC window is +-flex around a 1 h block -> slack 2*flex
BATCH_FLEX = 6           # batch deadline = start + dur + flex -> slack flex
XMAX = 24
ONE_MIN_H = 1 / 60
EPS_H = 1e-9             # CSV stores 16 sig. digits: exactly-60 s slack reads back just under 1/60


def load():
    audit = pd.read_csv(R / "r1_ev_audit.csv")
    ev = pd.read_csv(R / "r1_ev_exact_summary.csv").iloc[0]
    hvac = pd.read_csv(R / "r1_hvac_capacity.csv").set_index("config")
    batch = pd.read_csv(R / "r1_batch_unified.csv")
    fc = pd.read_csv(R / "r1_forecast_capped.csv").set_index(["cap", "arm"])
    return audit, ev, hvac, batch, fc


def fig02(audit, ev, batch):
    s = audit.loc[audit["reason"] == "valid", "real_slack_h"].to_numpy(float)
    assert len(s) == int(ev["n_priceable"])
    n_lt = int((s < ONE_MIN_H - EPS_H).sum())
    f_lt = n_lt / len(s)
    assert abs(f_lt - float(ev["share_slack_lt_1min"])) < 1e-12, (f_lt, ev["share_slack_lt_1min"])
    med = float(np.median(s))
    n_batch = int(batch["n_jobs"].iloc[0])
    n_hvac = int(pd.read_csv(R / "r1_hvac_capacity.csv")["n_jobs"].iloc[0])
    hv_c, ai_c = 2 * HVAC_FLEX, BATCH_FLEX

    x = np.sort(s); y = np.arange(1, len(x) + 1) / len(x)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xs = np.concatenate(([0.0], np.repeat(x, 2), [XMAX]))
    ys = np.concatenate(([0.0, 0.0], np.repeat(y, 2)[:-1], [y[-1]]))
    ax.plot(xs, ys, color=C_HVAC, lw=2.0, solid_joinstyle="miter",
            label=f"EV — measured (n={len(s):,}, median {med:.1f} h)", zorder=3)
    for c, col, lab, n in ((hv_c, C_HVAC_BAR, "HVAC", n_hvac), (ai_c, C_BATCH, "Batch", n_batch)):
        ax.plot([c, c], [0, 1], color=col, lw=2.0, ls="--", zorder=2,
                label=f"{lab} — assigned (n={n:,}, constant {c} h)")
        ax.plot([c, XMAX], [1, 1], color=col, lw=2.0, ls="--", zorder=2)
        ax.plot([0, c], [0, 0], color=col, lw=2.0, ls="--", zorder=2)
    ax.plot([0], [f_lt], "o", color=C_HVAC, ms=7, zorder=4)
    ax.annotate(f"{n_lt:,} of {len(s):,} EV sessions ({100*f_lt:.1f}%)\n"
                f"have less than 1 minute of slack",
                xy=(0, f_lt), xytext=(13.0, 0.30), fontsize=8.5, color=C_HVAC,
                arrowprops=dict(arrowstyle="->", color=C_HVAC, lw=1.0),
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_HVAC, alpha=0.92))
    ax.set_xlabel("deadline slack (hours)")
    ax.set_ylabel("fraction of jobs with slack $\\leq$ s")
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.15)
    ax.set_xticks(range(0, XMAX + 1, 2)); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax.text(0.015, 0.995, "solid = measured behaviour    dashed = window we assign",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5, style="italic", color="0.35")
    fig.tight_layout(); U.savefig(fig, "02_slack_histograms.png")
    print(f"[02] EV n={len(s):,} median={med:.4f} h  <1 min: {n_lt:,} ({100*f_lt:.2f}%)  "
          f">{XMAX} h: {int((s > XMAX).sum())}  max={s.max():.2f} h | HVAC n={n_hvac:,} {hv_c} h | "
          f"batch n={n_batch:,} {ai_c} h")


def fig03(ev, hvac, batch):
    unc = batch[batch["k"].isna()].iloc[0]
    vals = [float(hvac.loc["uncapped", "optimal_pct"]), float(ev["savings_pct"]), float(unc["greedy_pct"])]
    ns = [int(hvac.loc["uncapped", "n_jobs"]), int(ev["n_priceable"]), int(unc["n_jobs"])]
    pretty = ["HVAC\n(heat pump)", "EV\n(charging)", "Batch\n(AI compute)"]
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.bar(range(3), vals, width=0.6, color=[C_HVAC_BAR, C_EV_BAR, C_BATCH])
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v:.2f}%", ha="center", fontweight="bold", fontsize=11)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{p}\nn={n:,}" for p, n in zip(pretty, ns)])
    ax.set_ylim(0, max(vals) * 1.22); ax.set_ylabel("carbon saved (%)")
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); U.savefig(fig, "03_savings_by_load.png")
    print(f"[03] HVAC {vals[0]:.4f}% (n={ns[0]:,})  EV {vals[1]:.4f}% (n={ns[1]:,})  "
          f"Batch {vals[2]:.4f}% (n={ns[2]:,})  [all uncapped]")


def fig04b(hvac, batch):
    hv = pd.read_csv(R / "capacity_sweep.csv")
    # HVAC unchanged: check the committed sweep against the R1 rerun wherever both exist
    pairs = [("2x med", "M=2x median"), ("3x med", "M=3x median"), ("4x med", "M=4x median"),
             ("6x med", "M=6x median"), ("peak", "M=observed peak")]
    for old, new in pairs:
        o = hv[hv["label"] == old].iloc[0]
        for col in ("greedy_pct", "optimal_pct"):
            assert abs(float(o[col]) - float(hvac.loc[new, col])) < 1e-6, (old, col)
    hvac_unc = float(hvac.loc["uncapped", "optimal_pct"])
    hv_med = float(hv.loc[hv["k"] == 1.0, "M_kW"].iloc[0])
    hv = hv.assign(k_eff=hv["M_kW"].to_numpy(float) / hv_med)
    hk, gy, oy = hv["k_eff"].to_numpy(float), hv["greedy_pct"].to_numpy(float), hv["optimal_pct"].to_numpy(float)

    bk_rows = batch[batch["k"].notna()].sort_values("k")
    bk, gb = bk_rows["k"].to_numpy(float), bk_rows["greedy_pct"].to_numpy(float)
    unc = batch[batch["k"].isna()].iloc[0]
    ba_unc, n_batch = float(unc["greedy_pct"]), int(unc["n_jobs"])
    k_ceiling = float(bk[np.abs(gb - ba_unc) < 1e-6].min())

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    ax.axvspan(hk.min() * 0.86, 1.0, color="0.85", alpha=0.55, zorder=0,
               label=r"degenerate (cap $\leq$ median aggregate)")
    ax.fill_between(hk, gy, oy, color=C_HVAC, alpha=0.16, zorder=1, label="HVAC greedy shortfall")
    ax.plot(hk, gy, "s-", color=C_HVAC, lw=1.8, ms=6, zorder=4, label="HVAC heat pump, greedy (n=8,363)")
    ax.plot(hk, oy, "o--", color=C_HVAC, lw=1.8, ms=6, zorder=4, mfc="white", mew=1.5,
            label="HVAC heat pump, exact MILP")
    ax.axhline(hvac_unc, ls=":", color=C_HVAC, lw=1.5, alpha=0.85, zorder=2,
               label=f"HVAC uncapped ceiling ({hvac_unc:.2f}%)")
    ax.plot(bk, gb, "^-", color=C_BATCH, lw=1.8, ms=7, zorder=4,
            label=f"Batch GPU, greedy only (n={n_batch:,})")
    ax.axhline(ba_unc, ls=":", color=C_BATCH, lw=1.5, alpha=0.85, zorder=2,
               label=f"Batch uncapped ceiling ({ba_unc:.2f}%)")
    w = hvac.loc["M=3x median"]
    ax.axvline(WORKING_CAP_K, ls="--", color="0.35", lw=1.1, zorder=2)
    ax.annotate(f"working cap $k$={WORKING_CAP_K:g} ({float(w['M_kW']):.2f} kW for HVAC)\n"
                f"greedy {float(w['greedy_pct']):.2f}% / opt {float(w['optimal_pct']):.2f}%"
                f" / gap {float(w['gap_pp']):.2f} pp",
                xy=(WORKING_CAP_K, float(w["optimal_pct"])), xytext=(3.5, 5.2),
                fontsize=8.2, arrowprops=dict(arrowstyle="->", color="0.25", lw=1.0))
    ax.annotate(f"batch reaches its ceiling at $k$={k_ceiling:g};\nHVAC has not at $k$=16",
                xy=(k_ceiling * 1.02, ba_unc), xytext=(6.8, 1.15), fontsize=8.4, style="italic",
                color="0.3", ha="left",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=1.0, shrinkA=2, shrinkB=6))
    ticks = [0.5, 1, 2, 3, 4, 6, float(hv["k_eff"].max())]
    ax.set_xscale("log"); ax.set_xticks(ticks)
    ax.set_xticklabels(["0.5", "1", "2", "3", "4", "6", "15.8\n(HVAC peak)"], fontsize=8.4)
    ax.minorticks_off()
    ax.set_xlabel(r"per-hour budget $M$, as a multiple $k$ of each load's median hourly "
                  r"aggregate draw (log)")
    ax.set_ylabel("carbon saved (%)"); ax.set_ylim(0, hvac_unc * 1.16)
    ax.grid(alpha=0.22, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=8.0, framealpha=0.95,
              ncol=2, columnspacing=1.6, handlelength=2.4, borderaxespad=0.0)
    fig.tight_layout(); U.savefig(fig, "04b_savings_vs_capacity.png")
    print(f"[04b] HVAC matches r1_hvac_capacity.csv at k=2,3,4,6,peak; uncapped {hvac_unc:.4f}%")
    print("      batch (n={:,}): ".format(n_batch) + ", ".join(f"k={k:g} {g:.4f}%" for k, g in zip(bk, gb))
          + f", uncapped {ba_unc:.4f}%  -> first k at ceiling = {k_ceiling:g}")


def fig05(fc):
    labels = ["no cap", f"M = {1.269:.3f} kW"]
    oracle = [float(fc.loc[("uncapped", "milp"), "oracle_pct"]), float(fc.loc[("M=3x median", "milp"), "oracle_pct"])]
    fcast = [float(fc.loc[("uncapped", "milp"), "forecast_pct"]), float(fc.loc[("M=3x median", "milp"), "forecast_pct"])]
    x = np.arange(2); wdt = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    b1 = ax.bar(x - wdt / 2, oracle, wdt, color=C_HVAC_BAR, label="oracle (true CI)")
    b2 = ax.bar(x + wdt / 2, fcast, wdt, color=C_EV_BAR, label="forecast (climatology)")
    for bars, vals in ((b1, oracle), (b2, fcast)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}%", ha="center", va="bottom",
                    fontweight="bold", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("carbon saved (%)"); ax.set_ylim(0, max(oracle) * 1.18)
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout(); U.savefig(fig, "05_oracle_vs_forecast.png")
    print(f"[05] uncapped oracle {oracle[0]:.4f}% forecast {fcast[0]:.4f}% | "
          f"M=3x median (milp) oracle {oracle[1]:.4f}% forecast {fcast[1]:.4f}%")


def main() -> None:
    audit, ev, hvac, batch, fc = load()
    assert abs(float(hvac.loc["M=3x median", "M_kW"]) - 1.269) < 1e-9  # label in fig05
    fig02(audit, ev, batch); fig03(ev, hvac, batch); fig04b(hvac, batch); fig05(fc)


if __name__ == "__main__":
    main()
