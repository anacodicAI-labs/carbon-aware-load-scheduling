"""Generator for figures/04a_savings_vs_flex.png -- savings vs deadline slack, all three loads.

Merges the old 04a (HVAC only) with panel (e1) of the old 04e (batch only), and adds
EV as a third curve.

AXIS. The arms are NOT comparable on `flex_hours`: the HVAC adapter grants a
SYMMETRIC window [t-flex, t+1h+flex], so slack = 2*flex_hours, while the batch
adapter extends the deadline ONE-SIDED, so slack = flex_hours. The shared axis is
therefore DEADLINE SLACK.

EV IS MEASURED, NOT SWEPT. acn_to_jobs() takes no flex_hours parameter; the window
is the observed [floor(connectionTime), floor(disconnectTime)) interval. So the EV
curve is built by CONDITIONING: for each slack value s, the saving is recomputed over
the subset of sessions that actually have slack s. It is drawn with markers and a
dashed connector to keep that distinction visible.

DO NOT pair the aggregate EV saving (1.87%) with EV's median slack (2 h). The
aggregate is carried by a small right tail: sessions at exactly 2 h of slack save
0.886%, and the 1.87% figure corresponds to a conditioned slack of about 4.6 h.
The aggregate is drawn as a horizontal reference line only, with no slack value
attached to it.

HVAC is swept down to flex_hours=0 so that all three arms meet at slack 0, which is
the sanity check that the three pipelines agree where no shifting is possible.

No in-figure title and no provenance string (MDPI: that text belongs in the caption).
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
from cuad.scheduler.jobs import slack_h

HVAC_FLEX = range(0, 9)     # slack 0..16
EV_EXACT = range(0, 9)      # exact slack values resolved individually
EV_TAIL = 9                 # sessions with slack >= 9 collapsed into one bin


def saving(jobs, ci):
    if not jobs:
        return float("nan")
    rh = {j.job_id: j.earliest_start for j in jobs}
    base = U.do_nothing_gco2(jobs, rh, ci)
    opt = schedule(jobs, ci, baseline_hours=rh)["total_gco2"]
    return U.savings_pct(base, opt)


def main() -> None:
    ci, _ = U.get_carbon_intensity()

    # ---- HVAC: swept, symmetric window -> slack = 2 * flex ------------------
    df = pd.read_parquet(ROOT / "data/hvac/bldg486202_MA_year.parquet").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]) + MA_REDATE
    hv_slack, hv_pct, n_hv = [], [], None
    print("HVAC -- swept, uncapped greedy (== exact optimum), building 486202:")
    for f in HVAC_FLEX:
        jobs, rh = nrel_to_jobs(df, utc_offset_hours=-5, flex_hours=f)
        n_hv = len(jobs)
        base = U.do_nothing_gco2(jobs, rh, ci)
        pct = U.savings_pct(base, schedule(jobs, ci, baseline_hours=rh)["total_gco2"])
        hv_slack.append(2 * f); hv_pct.append(pct)
        print(f"   flex={f} h -> slack={2*f:2d} h   saving={pct:7.4f}%   (n={len(jobs):,})")

    # ---- batch: swept, one-sided window -> slack = flex ---------------------
    sw = pd.read_csv(ROOT / "results/ai_sweep.csv")
    pct_ai = sw.groupby("flex_hours")["savings_pct"].mean().sort_index()
    spread = float(sw.groupby("flex_hours")["savings_pct"].agg(lambda s: s.max() - s.min()).max())
    assert spread < 1e-6, f"batch savings_pct must not depend on gpu_power_kw (spread {spread:g})"
    n_ai = int(sw["completed_tasks"].iloc[0])
    ai_slack = [int(f) for f in pct_ai.index]
    ai_pct = [float(v) for v in pct_ai.to_numpy()]
    print(f"\nbatch -- swept, uncapped greedy, Alibaba GPU v2020, n={n_ai:,} tasks:")
    for f, v in zip(ai_slack, ai_pct):
        print(f"   flex={f:2d} h -> slack={f:2d} h   saving={v:7.4f}%")
    print(f"   (savings_pct identical across gpu_power_kw 0.3/0.4/0.5; max spread {spread:.2e})")

    # ---- EV: measured, conditioned on observed slack ------------------------
    ev, _ = U.get_ev_jobs(verbose=False)
    s = np.array([slack_h(j) for j in ev])
    ev_agg = saving(list(ev), ci)
    q1, med, q3 = (float(np.percentile(s, p)) for p in (25, 50, 75))
    print(f"\nEV -- MEASURED, conditioned on observed slack (n={len(ev):,}):")
    print(f"   quartiles Q1={q1:.0f} h  median={med:.0f} h  Q3={q3:.0f} h   "
          f"mean={s.mean():.3f} h  max={s.max()} h  zero-slack={100*(s==0).mean():.2f}%")
    ev_x, ev_y, ev_n = [], [], []
    for v in EV_EXACT:
        sub = [j for j, x in zip(ev, s) if x == v]
        p = saving(sub, ci)
        ev_x.append(v); ev_y.append(p); ev_n.append(len(sub))
        print(f"   slack={v:2d} h        saving={p:7.4f}%   (n={len(sub):,})")
    tail = [j for j, x in zip(ev, s) if x >= EV_TAIL]
    tail_s = s[s >= EV_TAIL]
    tail_x = float(np.median(tail_s))
    tail_y = saving(tail, ci)
    print(f"   slack>={EV_TAIL} h  (bin)  saving={tail_y:7.4f}%   (n={len(tail):,}, "
          f"median slack {tail_x:.0f} h, max {tail_s.max()} h)")
    print(f"   AGGREGATE over all sessions: {ev_agg:.4f}%  "
          f"(equivalent to a conditioned slack of about 4.6 h, NOT the 2 h median)")

    # ---- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(hv_slack, hv_pct, "o-", color="#2a9d8f", lw=1.9, ms=6, zorder=3,
            label=f"HVAC heat pump — swept (n={n_hv:,})")
    ax.plot(ai_slack, ai_pct, "s--", color="#e76f51", lw=1.9, ms=6, zorder=3,
            label=f"Batch GPU — swept (n={n_ai:,})")
    ax.plot(ev_x, ev_y, "D:", color="#264653", lw=1.6, ms=6, zorder=4,
            label=f"EV charging — measured (n={len(ev):,})")
    ax.plot([tail_x], [tail_y], "D", color="#264653", ms=7, mfc="white", mew=1.6, zorder=4)
    ax.plot([ev_x[-1], tail_x], [ev_y[-1], tail_y], ":", color="#264653", lw=1.2,
            alpha=0.55, zorder=3)
    ax.annotate(f"EV slack $\\geq$9 h bin\n(n={len(tail):,}, median {tail_x:.0f} h)",
                xy=(tail_x, tail_y), xytext=(tail_x + 6.6, tail_y - 1.5), fontsize=7.8,
                color="#264653",
                arrowprops=dict(arrowstyle="->", color="#264653", lw=0.9))

    ax.axhline(ev_agg, ls="-.", color="#264653", lw=1.2, alpha=0.75, zorder=2)
    ax.text(24.6, ev_agg + 0.22, f"EV aggregate {ev_agg:.2f}%\n(all slacks pooled)",
            ha="right", va="bottom", fontsize=7.8, color="#264653")

    ax.plot([0], [0], "o", ms=13, mfc="none", mec="0.45", mew=1.2, zorder=5)
    ax.annotate("all three agree at zero slack", xy=(0.35, 0.0), xytext=(3.0, -1.35),
                fontsize=7.8, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.9))

    ax.set_xlabel("deadline slack (hours)")
    ax.set_ylabel("carbon saved (%)")
    ax.set_xlim(-1.2, 25.6); ax.set_ylim(-2.2, 17.4)
    ax.set_xticks(range(0, 25, 2))
    ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    ax.legend(loc="upper left", fontsize=8.6, framealpha=0.95)
    ax.text(0.985, 0.055,
            "solid / dashed = swept parameter      dotted = measured distribution",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.8,
            style="italic", color="0.35")
    fig.tight_layout()
    U.savefig(fig, "04a_savings_vs_flex.png")


if __name__ == "__main__":
    main()
