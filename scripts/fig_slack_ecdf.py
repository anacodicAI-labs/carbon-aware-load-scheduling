"""Generator for figures/02_slack_histograms.png -- deadline slack, all three loads.

Replaces the EV-only histogram. Why an ECDF and not three histograms: only the EV
slack is a DISTRIBUTION. HVAC and batch slack are constants fixed by the adapters,
so a three-panel histogram would be one real panel plus two single bars.

  EV     observed. window = [floor(connectionTime), floor(disconnectTime)); slack
         is dwell minus ceil(energy/power). 47 distinct values, median 2 h.
  HVAC   assigned. nrel_to_jobs emits [t-flex, t+1h+flex] and sets power=energy/1h,
         so duration is exactly 1 h and slack == 2*flex_hours for every block.
  batch  assigned. alibaba_to_jobs sets deadline = start + dur + flex and duration
         recomputes to the same dur, so slack == flex_hours for every job.

No in-figure title (MDPI wants that text in the caption) and no provenance string.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import matplotlib.pyplot as plt
import numpy as np

import nb_utils as U
from cuad.scheduler.jobs import slack_h

FLEX = 6
XMAX = 24


def ecdf(v: np.ndarray):
    x = np.sort(v)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def main() -> None:
    ev, _ = U.get_ev_jobs(verbose=False)
    hv, _, _ = U.load_hvac_jobs(flex_hours=FLEX)
    ai, _, _ = U.get_ai_jobs(gpu_power_kw=0.4, flex_hours=FLEX, verbose=False)

    s_ev = np.array([slack_h(j) for j in ev])
    s_hv = np.array([slack_h(j) for j in hv])
    s_ai = np.array([slack_h(j) for j in ai])
    assert s_hv.std() == 0 and s_ai.std() == 0, "HVAC/batch slack must be constant"
    hv_c, ai_c = int(s_hv[0]), int(s_ai[0])

    n0 = int((s_ev == 0).sum())
    f0 = n0 / len(s_ev)
    x, y = ecdf(s_ev)

    print(f"EV    n={len(s_ev):,}  min={s_ev.min()} median={np.median(s_ev):.0f} "
          f"mean={s_ev.mean():.3f} max={s_ev.max()} distinct={len(np.unique(s_ev))}")
    print(f"      zero-slack sessions = {n0:,} ({100*f0:.2f}%)")
    print(f"HVAC  n={len(s_hv):,}  slack = {hv_c} h for every job (= 2 x flex_hours={FLEX})")
    print(f"batch n={len(s_ai):,}  slack = {ai_c} h for every job (= flex_hours={FLEX})")
    print("\nEV ECDF F(s) at integer slack hours (plotted curve):")
    for s in range(0, XMAX + 1):
        print(f"   s={s:3d} h   F={100*(s_ev <= s).mean():6.2f}%   count<= {int((s_ev<=s).sum()):,}")
    tail = int((s_ev > XMAX).sum())
    print(f"   beyond {XMAX} h: {tail} sessions ({100*tail/len(s_ev):.2f}%), max {s_ev.max()} h")

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    # EV: measured distribution (solid)
    xs = np.concatenate(([0.0], np.repeat(x, 2), [XMAX]))
    ys = np.concatenate(([0.0, 0.0], np.repeat(y, 2)[:-1], [y[-1]]))
    ax.plot(xs, ys, color="#264653", lw=2.0, solid_joinstyle="miter",
            label=f"EV — measured (n={len(s_ev):,}, median {int(np.median(s_ev))} h)", zorder=3)

    # HVAC / batch: assigned constants (dashed step at the constant)
    for c, col, lab, n in ((hv_c, "#2a9d8f", "HVAC", len(s_hv)),
                           (ai_c, "#e76f51", "Batch", len(s_ai))):
        ax.plot([c, c], [0, 1], color=col, lw=2.0, ls="--", zorder=2,
                label=f"{lab} — assigned (n={n:,}, constant {c} h)")
        ax.plot([c, XMAX], [1, 1], color=col, lw=2.0, ls="--", zorder=2)
        ax.plot([0, c], [0, 0], color=col, lw=2.0, ls="--", zorder=2)

    # the 29.3% zero-slack mass: the mechanism behind the low EV ceiling
    ax.plot([0], [f0], "o", color="#264653", ms=7, zorder=4)
    ax.annotate(f"{n0:,} of {len(s_ev):,} EV sessions ({100*f0:.1f}%)\nhave zero slack "
                f"— they cannot move at all",
                xy=(0, f0), xytext=(4.6, 0.26), fontsize=8.5, color="#264653",
                arrowprops=dict(arrowstyle="->", color="#264653", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#264653", alpha=0.92))

    ax.set_xlabel("deadline slack (hours)")
    ax.set_ylabel("fraction of jobs with slack $\\leq$ s")
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 1.15)
    ax.set_xticks(range(0, XMAX + 1, 2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax.text(0.015, 0.995, "solid = measured behaviour    dashed = window we assign",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            style="italic", color="0.35")
    fig.tight_layout()
    U.savefig(fig, "02_slack_histograms.png")


if __name__ == "__main__":
    main()
