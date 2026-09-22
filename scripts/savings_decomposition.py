"""Decomposition of the per-job saving into volatility, timing, and slack.

The manuscript reports that savings are seasonal and attributes the pattern to
an anti-correlation between demand peaks and clean hours, but never separates
the quantities that produce it. For a one-hour HVAC job under the uncapped
optimum the saving admits an exact identity rather than a regression fit:

    saving_j = position_j * spread_j / CI(b_j)

over job j's feasible window W_j and metered run hour b_j,

    spread_j   = max CI over W_j - min CI over W_j     grid carbon volatility
    position_j = (CI(b_j) - min CI over W_j) / spread_j    original load timing
    CI(b_j)                                                    baseline level

position_j is 0 when the job already ran in the cleanest hour it could reach
and 1 when it ran in the dirtiest. Deadline slack enters by setting W_j, hence
both spread and position -- which is why the three cannot be varied
independently and a regression on them would be reading correlated inputs.

The identity is exact because a one-hour job is placed at the window minimum,
so CI(b_j) - CI(s_j) = position_j * spread_j. Aggregate savings are the
energy-weighted mean, which is what the identity_check column verifies.

Run:  python scripts/savings_decomposition.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import numpy as np
import pandas as pd

import nb_utils as U

HEADLINE_FLEX = 6
FLEX_SWEEP = (1, 2, 4, 6, 8)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def emit(title, value) -> None:
    print(f"\n### {title}")
    print(value.to_string(index=False) if isinstance(value, pd.DataFrame) else value)


def per_job_terms(jobs, run_hours, ci: pd.Series) -> pd.DataFrame:
    """One row per job: the three terms of the identity and the realized saving."""
    one_h = pd.Timedelta(hours=1)
    rows = []
    for job in jobs:
        start = pd.Timestamp(job.earliest_start)
        end = pd.Timestamp(job.deadline) - one_h  # last feasible start for a 1 h job
        window = ci.loc[start:end].dropna()
        base_hour = pd.Timestamp(run_hours[job.job_id])
        if window.empty or base_hour not in ci.index or pd.isna(ci.loc[base_hour]):
            continue

        ci_base = float(ci.loc[base_hour])
        ci_min = float(window.min())
        spread = float(window.max()) - ci_min
        headroom = ci_base - ci_min
        # A flat window offers nothing to move into; position is undefined there
        # rather than zero-over-zero, so it is reported as 0 and counted apart.
        position = headroom / spread if spread > 1e-9 else 0.0

        rows.append({
            "job_id": job.job_id,
            "month": base_hour.month,
            "power_kw": job.power_kw,
            "ci_base": ci_base,
            "spread": spread,
            "position": position,
            "headroom": headroom,
            "saving_frac": headroom / ci_base,
            "flat_window": spread <= 1e-9,
        })
    return pd.DataFrame(rows)


def weighted(df: pd.DataFrame) -> dict:
    """Energy-weighted aggregates. Energy is power_kw x 1 h for these jobs."""
    e = df["power_kw"].to_numpy()
    base = float((e * df["ci_base"]).sum())
    avoided = float((e * df["headroom"]).sum())
    return {
        "n_jobs": len(df),
        "mean_spread": float(np.average(df["spread"], weights=e)),
        "mean_position": float(np.average(df["position"], weights=e)),
        "mean_ci_base": float(np.average(df["ci_base"], weights=e)),
        "saving_pct": 100.0 * avoided / base if base else 0.0,
    }


def main() -> None:
    t0 = time.time()
    ci, source = U.get_carbon_intensity()
    print("carbon source:", source)

    jobs, run_hours = U.load_hvac_jobs(flex_hours=HEADLINE_FLEX)[:2]
    df = per_job_terms(jobs, run_hours, ci)
    print(f"jobs priced: {len(df)} of {len(jobs)}")

    # ------------------------------------------------- 1. identity holds exactly
    agg = weighted(df)
    check = 100.0 * float((df["power_kw"] * df["position"] * df["spread"]).sum()
                          / (df["power_kw"] * df["ci_base"]).sum())
    emit("identity_check_flex6", {
        "saving_pct_from_headroom": round(agg["saving_pct"], 4),
        "saving_pct_from_position_x_spread": round(check, 4),
        "max_abs_residual_gco2": round(
            float((df["headroom"] - df["position"] * df["spread"]).abs().max()), 9),
    })

    # ------------------------------------------------------- 2. monthly breakdown
    mrows = []
    for m in range(1, 13):
        sub = df[df["month"] == m]
        r = {"month": m, "name": MONTHS[m - 1]}
        r.update(weighted(sub))
        r["flat_windows"] = int(sub["flat_window"].sum())
        mrows.append(r)
    monthly = pd.DataFrame(mrows)
    emit("monthly_decomposition_flex6", monthly[
        ["name", "n_jobs", "mean_spread", "mean_position", "mean_ci_base",
         "saving_pct"]].round(3))

    # Which term explains the spread between the best and worst months? Hold one
    # term at its annual level and recompute, so each counterfactual isolates the
    # other. These are diagnostics, not schedules: no job is re-placed.
    ann = weighted(df)
    cf = []
    for _, r in monthly.iterrows():
        # saving ~ position * spread / ci_base, evaluated at monthly means
        actual = r["mean_position"] * r["mean_spread"] / r["mean_ci_base"]
        fix_pos = ann["mean_position"] * r["mean_spread"] / r["mean_ci_base"]
        fix_spr = r["mean_position"] * ann["mean_spread"] / r["mean_ci_base"]
        cf.append({
            "name": r["name"],
            "approx_saving_pct": 100 * actual,
            "if_annual_position": 100 * fix_pos,
            "if_annual_spread": 100 * fix_spr,
        })
    counter = pd.DataFrame(cf)
    counter["timing_effect_pp"] = counter["approx_saving_pct"] - counter["if_annual_position"]
    counter["volatility_effect_pp"] = counter["approx_saving_pct"] - counter["if_annual_spread"]
    emit("monthly_counterfactuals_flex6", counter.round(3))

    # --------------------------------------------------- 3. how slack moves them
    srows = []
    for f in FLEX_SWEEP:
        j, rh = U.load_hvac_jobs(flex_hours=f)[:2]
        d = per_job_terms(j, rh, ci)
        r = {"flex_h": f}
        r.update(weighted(d))
        r["flat_windows"] = int(d["flat_window"].sum())
        srows.append(r)
    slack = pd.DataFrame(srows)
    emit("slack_sweep_decomposition", slack[
        ["flex_h", "mean_spread", "mean_position", "saving_pct", "flat_windows"]].round(3))

    out = ROOT / "results"
    monthly.to_csv(out / "decomposition_monthly.csv", index=False)
    counter.to_csv(out / "decomposition_counterfactual.csv", index=False)
    slack.to_csv(out / "decomposition_slack.csv", index=False)
    df.to_csv(out / "decomposition_per_job.csv", index=False)
    print(f"\nwrote 4 files to {out}")
    print(f"total runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
