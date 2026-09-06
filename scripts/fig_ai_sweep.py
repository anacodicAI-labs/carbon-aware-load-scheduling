"""Generator for figures/04e_ai_savings_sweep.png -- the batch (AI) sweep.

Two panels, both a function of assigned deadline flexibility, from the committed
results/ai_sweep.csv (no API key needed):

  Left  : batch carbon savings (%) vs flexibility. The percentage is identical
          for every assumed per-GPU power, because power scales the baseline and
          the scheduled emissions equally and cancels in the ratio. The flex-6
          headline operating point (3.97%) is marked.
  Right : absolute avoided emissions (tCO2) vs flexibility, one line per assumed
          per-GPU power (0.3, 0.4, 0.5 kW). Only the tonnage depends on that
          assumption; the percentage does not.

Run:  python scripts/fig_ai_sweep.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "ai_sweep.csv"
OUT = ROOT / "figures" / "04e_ai_savings_sweep.png"


def _series(rows: list[dict], gpu: str, col: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for r in rows:
        if r["gpu_power_kw"] == gpu:
            xs.append(float(r["flex_hours"]))
            ys.append(float(r[col]))
    return xs, ys


def main() -> None:
    rows = list(csv.DictReader(CSV.open()))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: savings % vs flexibility (identical across gpu; use 0.4).
    fx, fpct = _series(rows, "0.4", "savings_pct")
    axL.plot(fx, fpct, marker="o", color="#c0504d", lw=2, ms=6)
    i6 = fx.index(6.0)
    axL.scatter([6], [fpct[i6]], s=120, facecolors="none",
                edgecolors="#333333", lw=1.6, zorder=5)
    axL.annotate("headline: flex 6 h = 3.97%", xy=(6, fpct[i6]), xytext=(8.5, 2.2),
                 fontsize=9, color="#333333",
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=1))
    axL.set_xlabel("Assigned flexibility (hours)")
    axL.set_ylabel("Batch carbon savings (%)")
    axL.set_xticks([0, 2, 6, 12, 24])
    axL.set_ylim(0, 10.5)
    axL.grid(True, alpha=0.3)

    # Right: absolute avoided tCO2 vs flexibility, by per-GPU power.
    shades = {"0.3": "#9ecae1", "0.4": "#4292c6", "0.5": "#08519c"}
    for gpu in ("0.3", "0.4", "0.5"):
        gx, gg = _series(rows, gpu, "avoided_gco2")
        axR.plot(gx, [v / 1e6 for v in gg], marker="s", lw=2, ms=5,
                 color=shades[gpu], label=f"{gpu} kW / GPU")
    axR.set_xlabel("Assigned flexibility (hours)")
    axR.set_ylabel("Avoided emissions (tCO$_2$)")
    axR.set_xticks([0, 2, 6, 12, 24])
    axR.grid(True, alpha=0.3)
    axR.legend(title="Per-GPU power", frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
