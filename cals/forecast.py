"""Climatology CI forecast + forecast-penalty evaluation (notebook 05, Path 2).

Oracle scheduling (what the paper's headline reports) sees the TRUE carbon curve.
A real operator sees only a FORECAST. The *forecast penalty* is how much of the
saving is lost to imperfect foresight.

    climatology_forecast : predict CI(t) as the mean over all hours sharing the
                           same (month, hour-of-day) — the "typical" curve. Blind
                           to today's weather, but a cheap, honest baseline.

    forecast_penalty     : schedule each job on the FORECAST, price it on the
                           TRUE CI, and compare against the oracle (scheduled on
                           TRUE). Reports oracle vs forecast savings and the gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cuad.scheduler.jobs import Job, duration_h


def climatology_forecast(ci: pd.Series) -> pd.Series:
    """CI forecast = mean CI for the same (month, hour-of-day) across the series."""
    idx = pd.DatetimeIndex(ci.index)
    grp = pd.Series(ci.to_numpy(dtype=float), index=pd.MultiIndex.from_arrays([idx.month, idx.hour]))
    clim = grp.groupby(level=[0, 1]).transform("mean")
    out = pd.Series(clim.to_numpy(), index=ci.index, name="ci_forecast_climatology")
    return out


def _cheapest_start(job: Job, ci: pd.Series, dur: int):
    """Start hour in [earliest_start, deadline-dur] minimizing summed CI over dur hours."""
    one_h = pd.Timedelta(hours=1)
    r, d = pd.Timestamp(job.earliest_start), pd.Timestamp(job.deadline)
    best_s, best_c = None, np.inf
    s = r
    while s + dur * one_h <= d:
        hrs = [s + i * one_h for i in range(dur)]
        if all(h in ci.index for h in hrs):
            c = float(sum(ci.loc[h] for h in hrs))
            if c < best_c:
                best_c, best_s = c, s
        s += one_h
    return best_s


def forecast_penalty(
    jobs: list[Job],
    true_ci: pd.Series,
    fcst_ci: pd.Series,
    baseline_hours: dict | None = None,
) -> dict:
    """Oracle vs forecast savings vs a do-nothing baseline, over the shared set.

    do-nothing baseline = ``baseline_hours[job_id]`` when given, else
    earliest_start; oracle = cheapest window under TRUE ci; forecast = cheapest
    window under FORECAST ci, but PRICED on true ci.

    PASS baseline_hours FOR HVAC. Its window is [run - flex, run + 1 + flex), so
    an earliest_start baseline silently advances every job by `flex` hours onto
    cleaner grid hours, making the control a function of the flex knob and
    understating the saving. adapters.py:66-69 warns about exactly this, and the
    rest of the pipeline prices HVAC at its metered run hour. EV is unaffected:
    its earliest_start IS the plug-in hour, so the two coincide.
    """
    one_h = pd.Timedelta(hours=1)
    rows = []
    for j in jobs:
        dur = duration_h(j)
        base_s = pd.Timestamp(
            j.earliest_start if baseline_hours is None else baseline_hours[j.job_id]
        )
        oracle_s = _cheapest_start(j, true_ci, dur)
        fcst_s = _cheapest_start(j, fcst_ci, dur)
        if oracle_s is None or fcst_s is None or base_s not in true_ci.index:
            continue

        def price(s):
            hrs = [s + i * one_h for i in range(dur)]
            if not all(h in true_ci.index for h in hrs):
                return None
            return j.power_kw * float(sum(true_ci.loc[h] for h in hrs))

        b, o, f = price(base_s), price(oracle_s), price(fcst_s)
        if None in (b, o, f):
            continue
        rows.append((b, o, f))

    if not rows:
        return {}
    base = sum(r[0] for r in rows)
    orc = sum(r[1] for r in rows)
    fc = sum(r[2] for r in rows)
    return {
        "n_jobs": len(rows),
        "baseline_gco2": base,
        "oracle_gco2": orc,
        "forecast_gco2": fc,
        "oracle_savings_pct": 100 * (base - orc) / base if base else 0.0,
        "forecast_savings_pct": 100 * (base - fc) / base if base else 0.0,
        "forecast_penalty_pp": 100 * (fc - orc) / base if base else 0.0,
    }
