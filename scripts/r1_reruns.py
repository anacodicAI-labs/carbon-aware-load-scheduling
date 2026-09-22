"""Reviewer 1 (round 1) reruns. One subcommand per comment; each writes results/r1_<name>.csv.

  python scripts/r1_reruns.py hvac        # R1-2 total-demand cap, R1-6 MILP status/gap/bound, unshifted counts
  python scripts/r1_reruns.py forecast    # R1-4 climatology forecast under the SAME cap and job set as 8.29%
  python scripts/r1_reruns.py ev          # R1-3 exclusion audit + exact continuous-time EV saving
  python scripts/r1_reruns.py batch       # R1-6 one batch population (732,691 tasks) for Table 3 AND Fig. 7
  python scripts/r1_reruns.py marginal-iso --iso-csv PATH   # R1-5 ISO-NE 2019 hourly marginal CO2 reference

Real runs need the same inputs as the paper (EIA key/cache, ACN cache, Alibaba trace).
Add ``--synthetic --quick`` for an offline smoke test: synthetic carbon, one month of HVAC,
demo EV/batch data; output goes to /tmp/r1_smoke and the numbers MEAN NOTHING.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import nb_utils as U  # noqa: E402
from cals import schedule, schedule_optimal  # noqa: E402
from cals.forecast import climatology_forecast  # noqa: E402
from carbon_sim import CAP_SWEEP_K_REPORTED, WORKING_CAP_K, median_active_kw  # noqa: E402
from cuad.carbon.intensity import carbon_intensity  # noqa: E402
from cuad.scheduler.jobs import duration_h  # noqa: E402

LB_PER_MWH_TO_G_PER_KWH = 453.59237 / 1000.0


# ------------------------------------------------------------------ helpers
def load_mix_and_ci(synthetic: bool):
    if synthetic:
        mix = U.demo_fuel_mix()
        return mix, carbon_intensity(mix), "SYNTHETIC demo fuel mix"
    mix, label = U.get_fuel_mix()
    return mix, carbon_intensity(mix), label


def hvac_jobs(quick: bool):
    jobs, rh, _ = U.load_hvac_jobs(flex_hours=6)
    if quick:  # one winter month keeps the smoke test fast
        jobs = [j for j in jobs if rh[j.job_id].month == 1]
        rh = {j.job_id: rh[j.job_id] for j in jobs}
    return jobs, rh


def hourly_total(jobs, assign) -> dict:
    tot: dict = {}
    for j in jobs:
        s = assign[j.job_id]
        for h in range(duration_h(j)):
            t = s + pd.Timedelta(hours=h)
            tot[t] = tot.get(t, 0.0) + j.power_kw
    return tot


def shifted_total(jobs, assign, rh) -> dict:
    return hourly_total([j for j in jobs if assign[j.job_id] != rh[j.job_id]], assign)


def priced(jobs, assign, ci) -> float:
    return float(sum(U.price_at(j, assign[j.job_id], ci) for j in jobs))


def out_path(args, name: str) -> Path:
    d = Path("/tmp/r1_smoke") if args.synthetic else ROOT / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"r1_{name}.csv"


# ------------------------------------------------------------ R1-2 and R1-6
def cmd_hvac(args) -> None:
    _, ci, src = load_mix_and_ci(args.synthetic)
    jobs, rh = hvac_jobs(args.quick)
    base = U.do_nothing_gco2(jobs, rh, ci)
    med = median_active_kw(jobs)
    peak_total = max(hourly_total(jobs, rh).values())
    print(f"carbon: {src} | jobs={len(jobs)} median={med:.4f} kW observed total peak={peak_total:.4f} kW")

    cfgs = [(f"M={k:g}x median", k * med, None) for k in CAP_SWEEP_K_REPORTED]
    cfgs += [("M=observed peak", peak_total, None),
             ("uncapped", None, None),
             ("total<=observed peak (no M)", None, peak_total),
             (f"M={WORKING_CAP_K:g}x median AND total<=observed peak", WORKING_CAP_K * med, peak_total)]
    rows = []
    for label, m, tot in cfgs:
        t0 = time.time()
        o = schedule_optimal(jobs, ci, capacity_kw=m, baseline_hours=rh, total_capacity_kw=tot,
                             mip_rel_gap=args.mip_rel_gap, time_limit=args.milp_time_limit,
                             allow_suboptimal=args.milp_time_limit is not None)
        t_milp = time.time() - t0
        row = {"config": label, "M_kW": m, "total_cap_kW": tot, "n_jobs": len(jobs),
               "optimal_pct": U.savings_pct(base, o["total_gco2"]),
               # dual bound -> the best saving ANY schedule could reach: brackets the "exact" claim
               "optimal_pct_upper_bound": U.savings_pct(base, o["mip_dual_bound"]),
               "mip_status": o["status"], "mip_message": o["message"], "mip_gap": o["mip_gap"],
               "mip_rel_gap_tol": o["mip_rel_gap_tol"], "mip_nodes": o["mip_node_count"],
               "n_vars": o["n_vars"], "t_milp_s": round(t_milp, 1)}
        a = o["assignments"]
        row["milp_jobs_at_own_hour"] = sum(a[j.job_id] == rh[j.job_id] for j in jobs)
        row["milp_do_nothing_var"] = len(o["do_nothing"])
        tt = hourly_total(jobs, a)
        row["milp_total_peak_kW"] = max(tt.values())
        st = shifted_total(jobs, a, rh)
        row["milp_shifted_peak_kW"] = max(st.values(), default=0.0)
        row["milp_hours_shifted_at_M"] = sum(v >= m - 1e-6 for v in st.values()) if m else np.nan
        row["milp_hours_total_gt_M"] = sum(v > m + 1e-9 for v in tt.values()) if m else np.nan
        if tot is None:  # the greedy has no total-demand option; report it where comparable
            g = schedule(jobs, ci, capacity_kw=m, baseline_hours=rh)
            row["greedy_pct"] = U.savings_pct(base, g["total_gco2"])
            row["gap_pp"] = row["optimal_pct"] - row["greedy_pct"]
            row["greedy_jobs_at_own_hour"] = sum(g["assignments"][j.job_id] == rh[j.job_id] for j in jobs)
            for cause in ("capacity", "window", "no_improvement"):
                row[f"greedy_fallback_{cause}"] = sum(1 for c in g.get("fallback", {}).values() if c == cause)
            gt = hourly_total(jobs, g["assignments"])
            row["greedy_total_peak_kW"] = max(gt.values())
            row["greedy_hours_total_gt_M"] = sum(v > m + 1e-9 for v in gt.values()) if m else np.nan
        rows.append(row)
        print(f"  {label:48s} opt={row['optimal_pct']:.3f}% (bound {row['optimal_pct_upper_bound']:.3f}%) "
              f"gap={row['mip_gap']:.2e} status={row['mip_status']}  ({t_milp:.0f}s)")
    p = out_path(args, "hvac_capacity")
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"wrote {p}")


# ------------------------------------------------------------------- R1-4
def cmd_forecast(args) -> None:
    _, ci, src = load_mix_and_ci(args.synthetic)
    jobs, rh = hvac_jobs(args.quick)
    fc, tiers = climatology_forecast(ci, return_tiers=True)
    base = U.do_nothing_gco2(jobs, rh, ci)
    cap = WORKING_CAP_K * median_active_kw(jobs)

    # Information set, stated per job. Tier 1/2 predict hour h from the same hour-of-day on
    # EARLIER days only (h-24, h-48, ...). A job released at r_j = run - flex whose window ends
    # by r_j + 2*flex therefore uses no observation later than r_j + 2*flex - 24 < r_j for flex<12.
    tier3_hours = set(tiers.index[tiers.isin([0, 3])])
    jobs_touching_tier3 = sum(
        any((j.earliest_start + pd.Timedelta(hours=h)) in tier3_hours
            for h in range(int((j.deadline - j.earliest_start) / pd.Timedelta(hours=1))))
        for j in jobs)
    flex = int((rh[jobs[0].job_id] - jobs[0].earliest_start) / pd.Timedelta(hours=1))
    info = {"issue_time": "job release r_j = metered hour - flex",
            "latest_observation_used": f"tiers 1-2: h - 24 h, i.e. at least {24 - 2 * flex} h before release",
            "tier_share_2019": tiers[tiers.index.year == 2019].value_counts(normalize=True).round(5).to_dict(),
            "jobs_whose_window_touches_tier0_or_tier3": jobs_touching_tier3}
    print("information set:", info)

    rows = []
    for cap_label, m in (("uncapped", None), (f"M={WORKING_CAP_K:g}x median", cap)):
        for arm in ("greedy", "milp"):
            solve = (lambda c: schedule(jobs, c, capacity_kw=m, baseline_hours=rh)) if arm == "greedy" else \
                    (lambda c: schedule_optimal(jobs, c, capacity_kw=m, baseline_hours=rh))
            o, f = solve(ci), solve(fc)
            shared = [j for j in jobs if j.job_id in o["assignments"] and j.job_id in f["assignments"]]
            b = U.do_nothing_gco2(shared, rh, ci)
            po, pf = priced(shared, o["assignments"], ci), priced(shared, f["assignments"], ci)
            worse = sum(U.price_at(j, f["assignments"][j.job_id], ci) > U.price_at(j, rh[j.job_id], ci) + 1e-9
                        for j in shared)
            so, sf = U.savings_pct(b, po), U.savings_pct(b, pf)
            rows.append({"cap": cap_label, "arm": arm, "n_shared": len(shared),
                         "oracle_pct": so, "forecast_pct": sf, "penalty_pp": so - sf,
                         "retention_pct": 100 * sf / so if so else np.nan,
                         "forecast_jobs_dirtier_than_baseline_on_true_CI": worse,
                         "forecast_issue_time": info["issue_time"]})
            print(f"  {cap_label:16s} {arm:6s} oracle={so:.3f}% forecast={sf:.3f}% "
                  f"penalty={so - sf:.3f} pp retention={rows[-1]['retention_pct']:.1f}%")
    p = out_path(args, "forecast_capped")
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"wrote {p}  (baseline over all jobs = {base:.1f} gCO2)")


# ------------------------------------------------------------------- R1-3
def cmd_ev(args) -> None:
    from cals.ev_exact import audit_sessions, ev_exact_savings
    from cuad.scheduler.adapters import acn_to_jobs

    _, ci, src = load_mix_and_ci(args.synthetic)
    if args.synthetic:
        sessions = U.demo_ev_sessions(n=400, origin=pd.Timestamp("2019-07-01", tz="UTC"))
    else:
        from cals.acn_sessions import fetch_acn_sessions
        sessions = fetch_acn_sessions(dt.date(2019, 1, 1), dt.date(2019, 12, 31), cache_dir=U.DATA / "ev")
    valid, audit = audit_sessions(sessions)
    counts = audit["reason"].value_counts().to_dict()
    disc_only = int(((audit["reason"] == "valid") & ~audit["hourly_adapter_kept"]).sum())
    kept = int(audit["hourly_adapter_kept"].sum())
    print(f"raw={len(sessions)} reasons={counts} hourly_kept={kept} "
          f"valid_but_rejected_by_hour_grid={disc_only}")
    zero_hourly = int((audit["hourly_slack_h"] == 0).sum())
    real_lt_1min = int((audit.loc[audit["hourly_slack_h"] == 0, "real_slack_h"] < 1 / 60).sum())
    print(f"hourly zero-slack sessions={zero_hourly}, of which real slack < 1 min: {real_lt_1min}")

    summary, per = ev_exact_savings(valid, ci)
    jobs = acn_to_jobs(sessions)
    rh = {j.job_id: j.earliest_start for j in jobs}
    g = schedule(jobs, ci, baseline_hours=rh)
    hourly_pct = U.savings_pct(U.do_nothing_gco2(jobs, rh, ci), g["total_gco2"])
    summary.update({"n_raw": len(sessions), "n_hourly_kept": kept, "hourly_grid_savings_pct": hourly_pct,
                    "valid_but_rejected_by_hour_grid": disc_only,
                    "hourly_zero_slack": zero_hourly, "hourly_zero_slack_real_lt_1min": real_lt_1min,
                    **{f"excluded_{k}": v for k, v in counts.items() if k != "valid"}})
    print(f"exact continuous-time saving={summary['savings_pct']:.3f}% over {summary['n_priceable']} "
          f"sessions | paper's hourly method={hourly_pct:.3f}% over {len(jobs)}")
    pd.DataFrame([summary]).to_csv(out_path(args, "ev_exact_summary"), index=False)
    audit.to_csv(out_path(args, "ev_audit"), index=False)
    per.to_csv(out_path(args, "ev_exact_per_session"), index=False)
    print(f"wrote {out_path(args, 'ev_exact_summary')}")


# ------------------------------------------------------------------- R1-6 batch
def cmd_batch(args) -> None:
    from cals.ai_loads import alibaba_to_jobs, load_pai_task_table, parse_alibaba_trace
    _, ci, src = load_mix_and_ci(args.synthetic)
    if args.synthetic:
        parsed = parse_alibaba_trace(U.demo_alibaba_trace(), origin=U.DEMO_ORIGIN)
    else:
        parsed = parse_alibaba_trace(load_pai_task_table(U.DATA / "ai" / "pai_task_table.csv"),
                                     origin=U.DEMO_ORIGIN)
    # min_kwh=0: SAME population as Table 3 / Fig. 4 (ai_sweep.csv, 732,691 tasks). The old 0.1 kWh
    # filter depended on the assumed per-GPU power and silently changed the job set.
    jobs = alibaba_to_jobs(parsed, gpu_power_kw=0.4, flex_hours=6, min_kwh=0.0)
    rh = {j.job_id: j.earliest_start for j in jobs}
    base = U.do_nothing_gco2(jobs, rh, ci)
    agg = np.array(list(hourly_total(jobs, rh).values()))
    med_agg = float(np.median(agg))
    fleet = {"active_hours": len(agg), "peak_over_median": float(agg.max() / med_agg),
             "mean_concurrency": float(sum(duration_h(j) for j in jobs) / len(agg))}
    print(f"batch jobs={len(jobs):,} median aggregate={med_agg:.1f} kW peak/median={agg.max() / med_agg:.2f}")
    rows = []
    for k in (*CAP_SWEEP_K_REPORTED, None):
        m = None if k is None else k * med_agg
        t0 = time.time()
        g = schedule(jobs, ci, capacity_kw=m, baseline_hours=rh)
        row = {"k": k, "M_kW": m, "n_jobs": len(jobs), "greedy_pct": U.savings_pct(base, g["total_gco2"]),
               "t_greedy_s": round(time.time() - t0, 1), **fleet}
        if args.milp_time_limit and k == WORKING_CAP_K:
            o = schedule_optimal(jobs, ci, capacity_kw=m, baseline_hours=rh,
                                 time_limit=args.milp_time_limit, allow_suboptimal=True)
            row.update({"milp_pct": U.savings_pct(base, o["total_gco2"]),
                        "milp_pct_upper_bound": U.savings_pct(base, o["mip_dual_bound"]),
                        "mip_status": o["status"], "mip_message": o["message"], "mip_gap": o["mip_gap"]})
        rows.append(row)
        print(f"  k={k}  greedy={row['greedy_pct']:.4f}%  ({row['t_greedy_s']}s)")
    p = out_path(args, "batch_unified")
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"wrote {p}  -- uncapped row must equal ai_sweep.csv flex-6 (3.9692%) on real data")


# ------------------------------------------------------------------- R1-5
def load_iso_marginal(path: Path, tz: str = "America/New_York", hour_beginning: bool = True) -> pd.Series:
    """ISO-NE 2019 hourly marginal CO2 file -> hourly marginal intensity (gCO2/kWh, combustion basis).

    Expects one row per (date, hour, fuel type) with a marginal share and a CO2 rate in lb/MWh,
    as in ISO-NE's 2019 Real-Time Marginal CO2 Emissions release (load-weighted sheet). Hourly
    rate = sum(share * rate). CHECK the file's hour convention and time zone and set tz /
    hour_beginning accordingly; DST-ambiguous hours are resolved as standard time.
    """
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    cols = {c.lower().strip(): c for c in df.columns}
    pick = lambda *names: next(cols[n] for n in names if n in cols)  # noqa: E731
    c_date, c_hour = pick("date", "local_date"), pick("hour", "he", "hour_ending")
    c_share = pick("percent marginal", "percent_marginal", "pct_marginal", "share")
    c_rate = pick("co2 rate", "co2_rate", "co2_rate_lb_mwh")
    d = pd.DataFrame({"date": pd.to_datetime(df[c_date].astype(str), errors="coerce"),
                      "hour": pd.to_numeric(df[c_hour], errors="coerce"),
                      "share": pd.to_numeric(df[c_share], errors="coerce"),
                      "rate": pd.to_numeric(df[c_rate].astype(str).str.replace(",", ""), errors="coerce")}).dropna()
    if d["date"].dt.year.min() < 2000:  # "1-Jan" style dates parse without a year
        d["date"] = d["date"].apply(lambda x: x.replace(year=2019))
    d["hour"] = d["hour"] - (0 if hour_beginning else 1)
    d["local"] = d["date"] + pd.to_timedelta(d["hour"], unit="h")
    d["w"] = d["share"] * d["rate"]
    hourly = d.groupby("local")["w"].sum() / d.groupby("local")["share"].sum()
    idx = hourly.index.tz_localize(tz, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC")
    s = pd.Series(hourly.to_numpy() * LB_PER_MWH_TO_G_PER_KWH, index=idx).groupby(level=0).mean()
    return s.sort_index().rename("iso_ne_marginal_gco2_kwh")


def cmd_marginal_iso(args) -> None:
    mix, ci_avg, src = load_mix_and_ci(args.synthetic)
    ci_proxy = U.marginal_ci(mix)
    if args.synthetic:
        rng = np.random.default_rng(3)
        ci_iso = pd.Series(rng.uniform(380, 520, len(ci_avg)), index=ci_avg.index)
    else:
        ci_iso = load_iso_marginal(Path(args.iso_csv), hour_beginning=not args.hour_ending)
    ci_iso = ci_iso.reindex(ci_avg.index)
    ci_iso = ci_iso.fillna(ci_iso.rolling(3, center=True, min_periods=1).mean())  # isolated DST/missing hours

    yr = ci_avg.index.year == 2019
    oil = U.oil_generating_hours(mix).reindex(ci_avg.index, fill_value=False)
    valid = pd.DataFrame({"avg": ci_avg[yr], "proxy": ci_proxy[yr], "iso": ci_iso[yr]}).dropna()
    val = {"hours_compared": len(valid),
           "proxy_oil_marginal_share": float(oil[yr].mean()),
           "r_proxy_vs_iso": float(valid["proxy"].corr(valid["iso"])),
           "r_avg_vs_iso": float(valid["avg"].corr(valid["iso"])),
           "iso_mean_gco2_kwh": float(valid["iso"].mean()),
           "iso_mean_proxy_oil_hours": float(valid.loc[oil[yr].reindex(valid.index).to_numpy(), "iso"].mean()),
           "iso_mean_proxy_gas_hours": float(valid.loc[~oil[yr].reindex(valid.index).to_numpy(), "iso"].mean())}
    print("proxy validation:", {k: round(v, 3) for k, v in val.items()})

    jobs, rh = hvac_jobs(args.quick)
    # Shared job set for every schedule: the metered hour AND every hour of the window must have
    # an ISO value, else a schedule could move a job into an unpriceable (year-boundary) hour.
    iso_ok = ci_iso.notna()
    has_iso = lambda t: bool(iso_ok.get(t, False))  # noqa: E731
    ok = [j for j in jobs if has_iso(rh[j.job_id]) and
          all(has_iso(j.earliest_start + pd.Timedelta(hours=h))
              for h in range(int((j.deadline - j.earliest_start) / pd.Timedelta(hours=1))))]
    print(f"jobs={len(jobs)} kept={len(ok)} dropped (window reaches an hour without ISO value)={len(jobs) - len(ok)}")
    rows = []
    for cap_label, m in ((f"M={WORKING_CAP_K:g}x median", WORKING_CAP_K * median_active_kw(jobs)), ("uncapped", None)):
        base_iso = U.do_nothing_gco2(ok, rh, ci_iso)
        row = {"cap": cap_label, "n_jobs": len(ok), "n_dropped": len(jobs) - len(ok), **val}
        for label, sig in (("avg_optimized", ci_avg), ("proxy_optimized", ci_proxy), ("iso_optimized", ci_iso)):
            a = schedule(ok, sig, capacity_kw=m, baseline_hours=rh)["assignments"]
            row[f"{label}_on_iso_ruler_pct"] = U.savings_pct(base_iso, priced(ok, a, ci_iso))
            print(f"  {cap_label} {label}: {row[f'{label}_on_iso_ruler_pct']:.3f}%", flush=True)
        row["avg_retained_of_iso_opt_pct"] = 100 * row["avg_optimized_on_iso_ruler_pct"] / row["iso_optimized_on_iso_ruler_pct"]
        rows.append(row)
        print(f"  {cap_label}: baseline=0 | avg-opt={row['avg_optimized_on_iso_ruler_pct']:.3f}% "
              f"proxy-opt={row['proxy_optimized_on_iso_ruler_pct']:.3f}% "
              f"iso-opt={row['iso_optimized_on_iso_ruler_pct']:.3f}% (all on the ISO-NE ruler)")
    p = out_path(args, "marginal_iso")
    pd.DataFrame(rows).to_csv(p, index=False)
    print(f"wrote {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["hvac", "forecast", "ev", "batch", "marginal-iso"])
    ap.add_argument("--synthetic", action="store_true", help="offline smoke test; numbers meaningless")
    ap.add_argument("--quick", action="store_true", help="January HVAC jobs only")
    ap.add_argument("--mip-rel-gap", type=float, default=None, help="HiGHS mip_rel_gap (default 1e-4)")
    ap.add_argument("--milp-time-limit", type=float, default=None,
                    help="seconds per MILP; hvac: cap long total-demand solves; batch: try sparse MILP at k=3")
    ap.add_argument("--iso-csv", type=str, default=None, help="ISO-NE 2019 hourly marginal CO2 file")
    ap.add_argument("--hour-ending", action="store_true", help="ISO file hours are hour-ending (1-24)")
    args = ap.parse_args()
    if args.cmd == "marginal-iso" and not args.synthetic and not args.iso_csv:
        ap.error("marginal-iso needs --iso-csv (download from ISO-NE; see docstring of load_iso_marginal)")
    {"hvac": cmd_hvac, "forecast": cmd_forecast, "ev": cmd_ev, "batch": cmd_batch,
     "marginal-iso": cmd_marginal_iso}[args.cmd](args)


if __name__ == "__main__":
    main()
