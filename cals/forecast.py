"""Climatology CI forecast + forecast-penalty evaluation (notebook 05, Path 2).

Oracle scheduling (what the paper's headline reports) sees the TRUE carbon curve.
A real operator sees only a FORECAST. The *forecast penalty* is how much of the
saving is lost to imperfect foresight.

    climatology_forecast : CAUSAL expanding-window climatology. Predicts CI(t)
                           from the mean of EARLIER hours sharing the same
                           (month, hour-of-day), never from t itself or later
                           hours. Blind to today's weather, and now genuinely
                           out-of-sample in time.

    forecast_penalty     : schedule each job on the FORECAST, price it on the
                           TRUE CI, and compare against the oracle (scheduled on
                           TRUE). Reports oracle vs forecast savings and the gap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cuad.scheduler.jobs import Job, duration_h


def climatology_forecast(ci: pd.Series, *, return_tiers: bool = False):
    """CAUSAL expanding-window climatology: predict CI(t) from EARLIER hours only.

    For each hour t the prediction uses only observations at positions STRICTLY
    BEFORE t in the series. No hour ever contributes to its own prediction, and
    no later hour contributes either, so this is a forecast an operator standing
    at time t could actually have produced.

    Why this replaced the previous implementation. The old version computed the
    (month, hour-of-day) mean with ``groupby(...).transform("mean")`` over the
    WHOLE series and then evaluated on that same series. Every hour's prediction
    therefore contained that hour plus every later hour sharing its (month,
    hour-of-day) cell -- a retrospective average, not a forecast. It made the
    reported forecast penalty an optimistic lower bound of unknown size. The
    leaky version is retained as ``climatology_forecast_insample`` for the sole
    purpose of reproducing the superseded figure; do not use it for a result.

    Predictor, with a strictly causal fallback chain. For hour t, in order:
      tier 1  mean of earlier hours with the SAME (month, hour-of-day)  -- the
              intended predictor, and what almost every hour ends up using;
      tier 2  mean of earlier hours with the same HOUR-OF-DAY, any month -- used
              on the first occurrence of each (month, hour) cell, i.e. the first
              day of each month, when tier 1 has no history yet;
      tier 3  mean of ALL earlier hours -- used only on the first day of the
              series, when even tier 2 is empty for that hour-of-day;
      tier 0  NaN -- only the very first hour of the series, which has no history
              of any kind. It is left NaN rather than filled, because any fill
              would have to look forward and would reintroduce the leak.

    A NaN forecast hour is not a hole in the evaluation: ``_cheapest_start``
    compares candidate window costs with ``<``, and a NaN cost never wins, so a
    NaN hour is simply never chosen as a placement. A job is dropped only if
    EVERY window in its range is NaN, which one leading NaN hour cannot cause for
    any realistic flex setting.

    Parameters
    ----------
    ci:
        Observed hourly carbon intensity, time-ordered.
    return_tiers:
        When True, return ``(forecast, tiers)`` where ``tiers`` is an int Series
        recording which fallback tier produced each hour (0-3 as above), so a
        caller can report how much of the series relied on a fallback.
    """
    idx = pd.DatetimeIndex(ci.index)
    if not idx.is_monotonic_increasing:
        raise ValueError(
            "climatology_forecast requires a time-ordered series: a causal "
            "expanding window is meaningless if the index is not sorted"
        )
    values = ci.to_numpy(dtype=float)
    months = idx.month.to_numpy()
    hours = idx.hour.to_numpy()

    out = np.full(len(values), np.nan, dtype=float)
    tiers = np.zeros(len(values), dtype=int)

    # Running sums over PAST observations only; each is updated after the
    # prediction for the current hour has been written.
    sum_mh: dict[int, float] = {}
    cnt_mh: dict[int, int] = {}
    sum_h: dict[int, float] = {}
    cnt_h: dict[int, int] = {}
    sum_all = 0.0
    cnt_all = 0

    for i in range(len(values)):
        cell = int(months[i]) * 100 + int(hours[i])
        hour = int(hours[i])
        if cnt_mh.get(cell, 0) > 0:
            out[i] = sum_mh[cell] / cnt_mh[cell]
            tiers[i] = 1
        elif cnt_h.get(hour, 0) > 0:
            out[i] = sum_h[hour] / cnt_h[hour]
            tiers[i] = 2
        elif cnt_all > 0:
            out[i] = sum_all / cnt_all
            tiers[i] = 3
        # else: tier 0, stays NaN

        v = values[i]
        if not np.isnan(v):  # a NaN observation teaches nothing; skip the update
            sum_mh[cell] = sum_mh.get(cell, 0.0) + v
            cnt_mh[cell] = cnt_mh.get(cell, 0) + 1
            sum_h[hour] = sum_h.get(hour, 0.0) + v
            cnt_h[hour] = cnt_h.get(hour, 0) + 1
            sum_all += v
            cnt_all += 1

    forecast = pd.Series(out, index=ci.index, name="ci_forecast_climatology_causal")
    if return_tiers:
        return forecast, pd.Series(tiers, index=ci.index, name="forecast_tier")
    return forecast


def climatology_forecast_insample(ci: pd.Series) -> pd.Series:
    """LEAKY same-(month, hour-of-day) mean over the WHOLE series. DO NOT REPORT.

    This is the superseded implementation. Each hour's prediction is the mean of
    every hour sharing its (month, hour-of-day) cell, INCLUDING that hour itself
    and every later one, so scoring it on the same series it was fitted to is
    in-sample and yields an optimistic penalty. It is kept only so the previously
    published figure remains reproducible and the correction is auditable.
    Use ``climatology_forecast`` for anything reported.
    """
    idx = pd.DatetimeIndex(ci.index)
    grp = pd.Series(ci.to_numpy(dtype=float), index=pd.MultiIndex.from_arrays([idx.month, idx.hour]))
    clim = grp.groupby(level=[0, 1]).transform("mean")
    return pd.Series(clim.to_numpy(), index=ci.index, name="ci_forecast_climatology_insample")


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
