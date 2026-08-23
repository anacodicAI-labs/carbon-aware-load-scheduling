"""Section 4.9 driver: average- vs marginal-basis emissions accounting.

Reproduces every number in notes/carbon-scheduler-handoff.md, section 4.9.
Run from anywhere:  python scripts/avg_vs_marginal.py

Both arms price the SAME real EIA ISO-NE 2019 fuel mix, the SAME 8363 HVAC jobs
(ResStock 486202), against the SAME observed-run-hour baseline, under the SAME
fixed cap (3 x median active load -- a physical power limit, so it is NOT
re-derived per basis). Only the accounting basis changes. Nothing is tuned.

Runtime ~4 min; the flex-6 average-basis MILP dominates at ~95 s.

SIGN CONVENTION: delta_pp = Savings_average - Savings_marginal, matching the
manuscript (Section 4.9) and notebook 04 cell `ma-average-marginal-run`. A
POSITIVE delta therefore means the marginal basis LOWERS the attainable saving.
This file previously used the opposite sign; only the sign changed, never a
magnitude.
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
from cals import schedule, schedule_optimal
from carbon_sim import WORKING_CAP_K, median_active_kw


def emit(title, value) -> None:
    print(f"\n### {title}")
    print(value.to_string(index=False) if isinstance(value, pd.DataFrame) else value)


def main() -> None:
    t0 = time.time()
    ci_avg, source = U.get_carbon_intensity()
    mix, _ = U.get_fuel_mix()
    ci_mar = U.marginal_ci(mix)
    print("carbon source:", source)
    assert ci_avg.index.equals(ci_mar.index), "both bases must share the hourly index"
    assert int(ci_avg.isna().sum()) == int(ci_mar.isna().sum()) == 0

    # ------------------------------------------------------------ 1. hour split
    cal19 = (ci_avg.index >= "2019-01-01") & (ci_avg.index < "2020-01-01")
    oil_mask = U.oil_generating_hours(mix).reindex(ci_avg.index, fill_value=False)
    n_oil, n_tot = int(oil_mask[cal19].sum()), int(cal19.sum())
    emit("hour_split", {
        "calendar_2019_hours": n_tot,
        "oil_marginal_hours_650": n_oil,
        "gas_marginal_hours_490": n_tot - n_oil,
        "oil_share_pct": round(100.0 * n_oil / n_tot, 2),
        "fetched_window_hours": len(ci_avg),
        "fetched_window_oil_hours": int(oil_mask.sum()),
    })
    om = pd.Series(oil_mask[cal19].to_numpy(), index=ci_avg.index[cal19])
    emit("oil_hours_by_month", {int(m): int(v) for m, v in om.groupby(om.index.month).sum().items()})

    # Threshold DIAGNOSTIC only. The headline rule is "> 0", as specified; this
    # is reported so a reader can see how much the count depends on what
    # "generating" means, and is never used to produce a headline number.
    emit("oil_threshold_sensitivity_hours", {
        f">{t} MWh": int(U.oil_generating_hours(mix, threshold_mwh=t)
                         .reindex(ci_avg.index, fill_value=False)[cal19].sum())
        for t in (0.0, 1.0, 5.0, 10.0, 50.0, 100.0)})

    # --------------------------------------------------------- 2. signal stats
    def stats(s, label):
        x = s[cal19]
        return {"basis": label, "min": round(float(x.min()), 1),
                "median": round(float(x.median()), 1), "mean": round(float(x.mean()), 1),
                "max": round(float(x.max()), 1), "spread": round(float(x.max() - x.min()), 1),
                "ratio_max_min": round(float(x.max() / x.min()), 3),
                "std": round(float(x.std()), 1)}

    emit("signal_stats_calendar_2019",
         pd.DataFrame([stats(ci_avg, "average"), stats(ci_mar, "marginal")]))
    emit("signal_correlation_pearson", round(float(np.corrcoef(
        ci_avg[cal19].to_numpy(), ci_mar[cal19].to_numpy())[0, 1]), 4))
    emit("avg_ci_within_marginal_classes", {
        "oil_hours_avg_ci_median": round(float(ci_avg[cal19][om.to_numpy()].median()), 1),
        "gas_hours_avg_ci_median": round(float(ci_avg[cal19][~om.to_numpy()].median()), 1)})

    # -------------------------------------------------------- 3. savings sweep
    bases = (("average", ci_avg), ("marginal", ci_mar))
    flex_vals = (1, 2, 4, 6)
    jobsets = {f: U.load_hvac_jobs(flex_hours=f)[:2] for f in flex_vals}
    jobs6, rh6 = jobsets[6]
    med = median_active_kw(jobs6)
    cap_kw = round(WORKING_CAP_K * med, 6)
    print(f"\nmedian active = {med:.4f} kW ; cap = {WORKING_CAP_K:g}x median = {cap_kw:.4f} kW")
    print(f"jobs = {len(jobs6)} (the job set does not depend on flex_hours)")

    rows = []
    for cap_label, cap in ((f"cap={WORKING_CAP_K:g}x median", cap_kw), ("uncapped", None)):
        for f in flex_vals:
            j, rh = jobsets[f]
            cell = {"cap": cap_label, "flex_h": f}
            for label, ci in bases:
                b = U.do_nothing_gco2(j, rh, ci)
                r = schedule(j, ci, capacity_kw=cap, baseline_hours=rh)
                cell[f"base_gco2_{label}"] = b
                cell[f"sched_gco2_{label}"] = r["total_gco2"]
                cell[f"avoided_gco2_{label}"] = b - r["total_gco2"]
                cell[f"sav_{label}"] = U.savings_pct(b, r["total_gco2"])
                cell[f"assign_{label}"] = r["assignments"]
            # Basis-mismatch cost: price the AVERAGE-derived schedule on the
            # marginal metric. Gap to sav_marginal is what an operator who
            # optimises on the wrong basis leaves on the table.
            cross = sum(U.price_at(job, cell["assign_average"][job.job_id], ci_mar) for job in j)
            cell["sav_marginal_of_avg_schedule"] = U.savings_pct(cell["base_gco2_marginal"], cross)
            rows.append(cell)

    sweep = pd.DataFrame(rows)
    sweep["delta_pp"] = sweep["sav_average"] - sweep["sav_marginal"]
    emit("greedy_savings_sweep", sweep[
        ["cap", "flex_h", "sav_average", "sav_marginal", "delta_pp",
         "sav_marginal_of_avg_schedule"]].round(3))
    emit("greedy_savings_absolute", sweep[
        ["cap", "flex_h", "base_gco2_average", "avoided_gco2_average",
         "base_gco2_marginal", "avoided_gco2_marginal"]].round(1))

    # Exact MILP at the working cap. Uncapped MILP is omitted: it provably equals
    # greedy when nothing binds (carbon_sim.check_anchors, anchor 4).
    milp = []
    for f in flex_vals:
        j, rh = jobsets[f]
        r = {"flex_h": f}
        for label, ci in bases:
            b = U.do_nothing_gco2(j, rh, ci)
            t = time.time()
            o = schedule_optimal(j, ci, capacity_kw=cap_kw, baseline_hours=rh)
            r[f"sav_{label}"] = U.savings_pct(b, o["total_gco2"])
            r[f"status_{label}"] = o.get("status")
            print(f"  MILP flex{f} {label}: {r[f'sav_{label}']:.2f}%  ({time.time() - t:.0f}s)")
        r["delta_pp"] = r["sav_average"] - r["sav_marginal"]
        milp.append(r)
    emit(f"milp_savings_cap{WORKING_CAP_K:g}x", pd.DataFrame(milp).round(3))

    # ------------------------------------------------ 4. delta by month/season
    # Operating point of Section 4.5: greedy, flex 6, cap = 3x ANNUAL median. The
    # cap is a fixed physical limit and must not be re-derived per month, or the
    # months stop being comparable.
    j, rh = jobsets[6]
    mon = np.array([pd.Timestamp(rh[job.job_id]).month for job in j])
    mrows = []
    for m in range(1, 13):
        jm = [job for job, mm in zip(j, mon) if mm == m]
        rhm = {job.job_id: rh[job.job_id] for job in jm}
        row = {"month": m, "n_jobs": len(jm)}
        for label, ci in bases:
            b = U.do_nothing_gco2(jm, rhm, ci)
            tot = schedule(jm, ci, capacity_kw=cap_kw, baseline_hours=rhm)["total_gco2"]
            row[f"base_{label}"] = b
            row[f"avoided_{label}"] = b - tot
            row[f"sav_{label}"] = U.savings_pct(b, tot) if b else 0.0
        row["delta_pp"] = row["sav_average"] - row["sav_marginal"]
        row["oil_hours"] = int(om[om.index.month == m].sum())
        mrows.append(row)
    mo = pd.DataFrame(mrows)
    mo.insert(1, "name", ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    emit("monthly_cap3x_flex6", mo[["month", "name", "n_jobs", "oil_hours",
                                    "sav_average", "sav_marginal", "delta_pp"]].round(3))

    # Seasons are ENERGY-WEIGHTED aggregate ratios (total avoided / total
    # baseline), matching notebook 04 cell (c) -- NOT means of monthly percents.
    srows = []
    for lbl, idx in (("winter (DJF)", [11, 0, 1]), ("spring (MAM)", [2, 3, 4]),
                     ("summer (JJA)", [5, 6, 7]), ("fall (SON)", [8, 9, 10])):
        sub = mo.iloc[idx]
        r = {"season": lbl, "oil_hours": int(sub["oil_hours"].sum())}
        for label, _ in bases:
            r[f"sav_{label}"] = 100.0 * sub[f"avoided_{label}"].sum() / sub[f"base_{label}"].sum()
        r["delta_pp"] = r["sav_average"] - r["sav_marginal"]
        srows.append(r)
    emit("seasonal_cap3x_flex6_energy_weighted", pd.DataFrame(srows).round(3))

    # ---------------------------------------------- 5. mechanism diagnostics
    # A two-valued signal makes most alternative placements EXACT TIES, so a raw
    # count of "jobs placed differently" hugely overstates real disagreement.
    for cap_label, cap in ((f"cap={WORKING_CAP_K:g}x median", cap_kw), ("uncapped", None)):
        a = schedule(j, ci_avg, capacity_kw=cap, baseline_hours=rh)["assignments"]
        m2 = schedule(j, ci_mar, capacity_kw=cap, baseline_hours=rh)["assignments"]
        diff = [job for job in j if a[job.job_id] != m2[job.job_id]]
        ties = sum(1 for job in diff
                   if abs(U.price_at(job, a[job.job_id], ci_mar)
                          - U.price_at(job, m2[job.job_id], ci_mar)) < 1e-9)
        print(f"{cap_label}: differ={len(diff)} ({100 * len(diff) / len(j):.1f}%)  "
              f"ties={ties} ({100 * ties / len(diff):.1f}% of differing)  "
              f"genuine={len(diff) - ties} ({100 * (len(diff) - ties) / len(j):.1f}% of all jobs)")

    # The load-bearing statistic: how much gradient each job can actually reach.
    for label, ci in bases:
        spans = np.array([
            float(ci.loc[job.earliest_start:job.deadline - pd.Timedelta(hours=1)].max()
                  - ci.loc[job.earliest_start:job.deadline - pd.Timedelta(hours=1)].min())
            for job in j])
        print(f"{label}: per-job in-window CI spread mean={spans.mean():.1f} "
              f"median={np.median(spans):.1f} p10={np.percentile(spans, 10):.1f} "
              f"p90={np.percentile(spans, 90):.1f} gCO2/kWh | "
              f"flat windows = {int((spans < 1e-9).sum())} "
              f"({100 * (spans < 1e-9).mean():.1f}%)")

    out = ROOT / "results"
    sweep.drop(columns=["assign_average", "assign_marginal"]).to_csv(
        out / "avg_vs_marginal_sweep.csv", index=False)
    mo.to_csv(out / "avg_vs_marginal_monthly.csv", index=False)
    print(f"\nwrote {out}/avg_vs_marginal_sweep.csv and avg_vs_marginal_monthly.csv")
    print(f"total runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
