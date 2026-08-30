"""Regenerate 01, 03 and 05 WITHOUT in-figure titles (MDPI: text goes in the caption).

Only the title is removed; every plotted value is recomputed from the same sources
the notebooks use, so the figures are numerically identical to the committed ones.
 04e is NOT generated here. It was briefly rebuilt as a single panel during the merge
experiment; that change has been reverted and 04e is now owned by
scripts/restore_figures.py, which restores its committed two-panel form. The fig04e()
below is retained for reference only and is deliberately not called by main().
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
from cals.forecast import climatology_forecast, forecast_penalty


def fig01(ci):
    c = ci[(ci.index >= "2019-01-01") & (ci.index < "2020-01-01")]
    c_std = c.copy(); c_std.index = c.index - pd.Timedelta(hours=5)
    grid = (pd.DataFrame({"ci": c_std.values, "month": c_std.index.month,
                          "hour": c_std.index.hour})
            .groupby(["month", "hour"])["ci"].mean().unstack("hour"))
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    im = ax.imshow(grid.values, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, 24, 0.5, 12.5])
    ax.set_yticks(range(1, 13))
    ax.set_yticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("hour of day (standard time, UTC$-$5)"); ax.set_ylabel("month")
    fig.colorbar(im, ax=ax, label="gCO2/kWh")
    fig.tight_layout(); U.savefig(fig, "01_ci_heatmap.png")
    print(f"[01] month x hour mean CI: min={grid.values.min():.1f} "
          f"max={grid.values.max():.1f} gCO2/kWh over {len(c):,} hours")
    return grid


def fig03(ci, hv, rh_hv, ev):
    b_hv = U.do_nothing_gco2(hv, rh_hv, ci)
    p_hv = U.savings_pct(b_hv, schedule(hv, ci, baseline_hours=rh_hv)["total_gco2"])
    rh_ev = {j.job_id: j.earliest_start for j in ev}
    b_ev = U.do_nothing_gco2(ev, rh_ev, ci)
    p_ev = U.savings_pct(b_ev, schedule(ev, ci, baseline_hours=rh_ev)["total_gco2"])
    sw = pd.read_csv(ROOT / "results/ai_sweep.csv")
    row = sw[(sw.gpu_power_kw == 0.4) & (sw.flex_hours == 6)].iloc[0]
    p_ai, n_ai = float(row["savings_pct"]), int(row["completed_tasks"])

    vals = [p_hv, p_ev, p_ai]; ns = [len(hv), len(ev), n_ai]
    pretty = ["HVAC\n(heat pump)", "EV\n(charging)", "Batch\n(AI compute)"]
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    ax.bar(range(3), vals, width=0.6, color=["#2a9d8f", "#e9c46a", "#e76f51"])
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v:.2f}%", ha="center", fontweight="bold", fontsize=11)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{p}\nn={n:,}" for p, n in zip(pretty, ns)])
    ax.set_ylim(0, max(vals) * 1.22); ax.set_ylabel("carbon saved (%)")
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); U.savefig(fig, "03_savings_by_load.png")
    print(f"[03] HVAC {p_hv:.2f}% (n={len(hv):,})  EV {p_ev:.2f}% (n={len(ev):,})  "
          f"Batch {p_ai:.2f}% (n={n_ai:,})  [all uncapped]")


def fig04e():
    sw = pd.read_csv(ROOT / "results/ai_sweep.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    print("[04e] avoided emissions (tCO2) by assumed per-GPU power:")
    for gp, g in sw.groupby("gpu_power_kw", sort=True):
        g = g.sort_values("flex_hours")
        t = g["avoided_gco2"] / 1e6
        ax.plot(g["flex_hours"], t, "o-", lw=1.8, ms=6, label=f"{gp:.1f} kW/GPU")
        print("      %.1f kW/GPU: " % gp + ", ".join(
            f"flex{int(f)}={v:.2f}" for f, v in zip(g["flex_hours"], t)))
    ax.set_xlabel("assigned flexibility (hours)"); ax.set_ylabel("avoided emissions (tCO2)")
    ax.grid(alpha=0.25, lw=0.6); ax.legend(title="assumed power", fontsize=9)
    fig.tight_layout(); U.savefig(fig, "04e_ai_savings_sweep.png")


def fig05(ci, hv, rh_hv):
    fc = climatology_forecast(ci)
    res = forecast_penalty(hv, ci, fc, baseline_hours=rh_hv)
    v = [res["oracle_savings_pct"], res["forecast_savings_pct"]]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bars = ax.bar(["oracle\n(true CI)", "forecast\n(climatology)"], v,
                  color=["#2a9d8f", "#e9c46a"])
    for b, x in zip(bars, v):
        ax.text(b.get_x() + b.get_width()/2, x + 0.05, f"{x:.2f}%", ha="center", fontweight="bold")
    ax.set_ylabel("carbon saved (%)"); ax.set_ylim(0, max(v) * 1.18)
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); U.savefig(fig, "05_oracle_vs_forecast.png")
    print(f"[05] oracle {v[0]:.2f}%  forecast {v[1]:.2f}%  penalty "
          f"{res['forecast_penalty_pp']:.2f} pp  (n={res['n_jobs']:,} jobs, retains "
          f"{100*v[1]/v[0]:.1f}%)")


def main():
    ci, source = U.get_carbon_intensity()
    hv, rh_hv, _ = U.load_hvac_jobs(flex_hours=6)
    ev, _ = U.get_ev_jobs(verbose=False)
    fig01(ci); fig03(ci, hv, rh_hv, ev); fig05(ci, hv, rh_hv)
    # fig04e() intentionally NOT called -- see module docstring; 04e is owned
    # by scripts/restore_figures.py (committed two-panel form).


if __name__ == "__main__":
    main()
