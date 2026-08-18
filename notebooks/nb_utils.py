"""Shared helpers for the five paper notebooks.

Nothing here is novel science -- it is glue that (a) keeps every reported number
sourced from REAL data, failing loudly rather than substituting a plausible
stand-in, and (b) keeps illustrative demo data clearly LABELLED so a reader never
confuses a synthetic curve for a measured one.
All the real modelling lives in ``cals`` (imported below, never reimplemented).

Every data path RAISES rather than faking it -- there is no synthetic fallback
left anywhere in this module, and no flag to re-enable one:
  * carbon signal   -> EIA Open Data API   (needs EIA_API_KEY, or a real cache)
  * EV charging      -> Caltech ACN-Data    (needs a real cache, or acnportal +
                        ACN_API_TOKEN)
  * AI load          -> Alibaba GPU v2020 trace (drop a real CSV in data/ai/);
                        recorded real-trace results live in results/ai_sweep.csv
The two offline-real paths:
  * HVAC load        -> committed NREL ResStock parquet in data/hvac/
  * emission factors -> cals.FACTORS

The demo_* helpers (demo_fuel_mix, demo_ev_sessions, demo_alibaba_trace) are
retained for clearly-labelled ILLUSTRATION only and must never reach a result.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cals import (
    carbon_intensity,
    fetch_eia_fuel_mix,
    nrel_to_jobs,
    acn_to_jobs,
    alibaba_to_jobs,
    parse_alibaba_trace,
    load_pai_task_table,
    duration_h,
)

# --- paths -------------------------------------------------------------------
HERE = Path(__file__).resolve().parent        # .../code/notebooks
ROOT = HERE.parent                            # .../code
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# The committed HVAC parquet is calendar-year 2018 in local standard time. We
# build/fetch carbon over a slightly wider window so every job's flex window is
# fully priceable (a job near Jan 1 reaches back into Dec 31).
# Real EIA ISO-NE fuel mix begins 2019-01-01, so the whole study aligns to 2019.
CI_START = "2019-01-01"
CI_END = "2020-01-02"  # +2-day buffer so job windows near year-end stay priceable
DEMO_ORIGIN = pd.Timestamp("2019-06-01", tz="UTC")  # EV/AI demo loads land in 2019 too

# ResStock amy2018 -> 2019 carbon year. +365d for a FULL YEAR (exact day-for-day
# bijection); the AL one-week slice uses +364d (preserves weekday). Full
# justification in load_hvac_jobs(); mirrored in the .provenance.json sidecars.
MA_REDATE = pd.Timedelta(days=365)


# --- carbon signal -----------------------------------------------------------
def demo_fuel_mix(start: str = CI_START, end: str = CI_END, seed: int = 0) -> pd.DataFrame:
    """A clearly-labelled SYNTHETIC ISO-NE-shaped fuel mix, hourly, long format.

    Columns [timestamp, fueltype, gen_mwh] -- exactly what ``carbon_intensity``
    consumes, so the demo curve still flows through the REAL intensity math
    ( CI(t) = sum(gen*EF) / sum(gen) ). The shape is only meant to be plausible:
      * SUN peaks midday  -> pulls CI down when the sun is up,
      * NG ramps into the evening peak -> pushes CI up at night,
      * a mild winter-heavy seasonal term on the fossil/hydro fuels.
    It is NOT real data; use a real EIA_API_KEY for anything reported.
    """
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    h = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    rng = np.random.default_rng(seed)

    solar_shape = np.clip(np.sin((h - 6) / 12 * np.pi), 0, None)          # daylight bump
    evening_peak = np.clip(np.sin((h - 18) / 12 * np.pi), 0, None)        # ~6pm ramp
    season = 1 + 0.3 * np.cos((doy - 15) / 365 * 2 * np.pi)               # winter heavier

    def gen(base, shape):
        return np.clip(base * shape * (1 + 0.05 * rng.standard_normal(len(idx))), 0, None)

    fuels = {
        "NG": gen(4000, season * (1 + 0.3 * evening_peak)),
        "NUC": gen(3000, np.ones(len(idx))),   # flat baseload, keeps every hour > 0
        "WAT": gen(800, season),
        "WND": gen(700, 1 + 0.4 * np.cos(h / 24 * 2 * np.pi)),
        "SUN": gen(1200, solar_shape),
        "COL": gen(150, season),
        "OIL": gen(50, season * (h > 16)),
        "OTH": gen(200, np.ones(len(idx))),
    }
    frames = [pd.DataFrame({"timestamp": idx, "fueltype": ft, "gen_mwh": g}) for ft, g in fuels.items()]
    return pd.concat(frames, ignore_index=True)


def _eia_cache_status(start: str, end: str, region: str) -> tuple[Path, bool]:
    """(cache path, is it a REAL eia pull?) for this window.

    Mirrors fetch_eia_fuel_mix's cache naming. A cache counts as real only when
    its provenance sidecar records source=="eia"; a synthetic-sourced cache is
    treated as absent so it can never stand in for the real signal.
    """
    respondent = {"ISO-NE": "ISNE", "ISONE": "ISNE"}.get(region, region)
    cache = (DATA / "carbon" / "eia" /
             f"fuelmix_{respondent}_{pd.Timestamp(start).date()}_{pd.Timestamp(end).date()}.csv")
    meta = cache.with_suffix(cache.suffix + ".meta.json")
    if not cache.is_file() or not meta.is_file():
        return cache, False
    try:
        return cache, json.loads(meta.read_text()).get("source") == "eia"
    except Exception:
        return cache, False


def get_fuel_mix(start: str = CI_START, end: str = CI_END, region: str = "ISO-NE",
                 verbose: bool = True) -> tuple[pd.DataFrame, str]:
    """Hourly fuel mix [timestamp, fueltype, gen_mwh], REAL ONLY. Returns (mix, label).

    Requires either EIA_API_KEY (loaded from ../.env) or a previously cached REAL
    EIA pull for this window, and RAISES with neither. A cache whose provenance
    sidecar says "synthetic" does not count. There is deliberately no synthetic
    fallback and no flag to re-enable one: this
    function feeds every reported number, and a fabricated carbon curve produces
    plausible-looking savings that are indistinguishable from real ones in a
    notebook's stored output. Failing loudly is the only honest default.

    ``demo_fuel_mix`` still exists for clearly-labelled illustration only (see
    notebook 01's single-day derivation figure); it must never reach a result.

    Exposing the *mix* (not just the CI) lets the emission-factor sensitivity
    sweep re-price it under alternative factors.
    """
    key = os.environ.get("EIA_API_KEY", "").strip()
    cache, cache_is_real = _eia_cache_status(start, end, region)
    if not key and not cache_is_real:
        raise RuntimeError(
            "EIA_API_KEY is not set and there is no real cached EIA fuel mix at "
            f"{cache}, so the ISO-NE carbon signal cannot be built. Copy .env.example "
            "to .env and add your key (EIA_API_KEY=...), or export it in the "
            "environment. This function has NO synthetic fallback on purpose -- every "
            "reported number must come from the real EIA signal."
        )
    try:
        mix = fetch_eia_fuel_mix(
            pd.Timestamp(start).date(), pd.Timestamp(end).date(),
            region, raw_dir=DATA / "carbon", api_key=key,
            use_synthetic_if_missing=False,  # never invent a curve
        )
    except Exception as exc:  # bad key, network, cache clash -- never fall back
        raise RuntimeError(
            f"EIA fuel-mix fetch failed for {region} {start}..{end} "
            f"({type(exc).__name__}: {exc}). Refusing to substitute a synthetic curve."
        ) from exc
    if mix is None or len(mix) == 0:
        raise RuntimeError(f"EIA returned no fuel-mix rows for {region} {start}..{end}.")
    return mix, f"EIA {region} ({'LIVE API' if key else 'cached real pull'})"


def get_carbon_intensity(start: str = CI_START, end: str = CI_END, region: str = "ISO-NE",
                         verbose: bool = True) -> tuple[pd.Series, str]:
    """Hourly carbon intensity (gCO2/kWh), guarded. Returns (ci, source_label)."""
    mix, label = get_fuel_mix(start, end, region, verbose=verbose)
    return carbon_intensity(mix), label


def ci_from_factors(mix: pd.DataFrame, factors: dict) -> pd.Series:
    """Re-price a fuel mix under an ALTERNATIVE emission-factor table.

    Same generation-weighted math as ``carbon_intensity`` -- CI(t) = sum(gen*EF)
    / sum(gen) -- but with a caller-supplied {fueltype: gCO2/kWh} dict, so the
    OTH (and any other) sensitivity can be swept without touching cals.
    """
    df = mix[["timestamp", "fueltype", "gen_mwh"]].copy()
    df["gen_mwh"] = pd.to_numeric(df["gen_mwh"], errors="coerce").fillna(0.0).clip(lower=0.0)
    default = factors.get("OTH", 0.0)
    df["w"] = df["gen_mwh"] * df["fueltype"].map(lambda c: factors.get(c, default))
    g = df.groupby("timestamp")
    ci = g["w"].sum() / g["gen_mwh"].sum().where(g["gen_mwh"].sum() > 0)
    ci.name = "carbon_intensity_gco2_kwh"
    return ci.sort_index()


# --- HVAC load (real, offline) ----------------------------------------------
def load_hvac_jobs(fname: str = "bldg486202_MA_year.parquet", *, flex_hours: int = 6,
                   utc_offset_hours: int = -5):
    """Load one committed NREL ResStock building and turn it into deferrable Jobs.

    Returns (jobs, run_hours, raw_df). run_hours maps job_id -> the UTC hour the
    HVAC ACTUALLY ran (the do-nothing baseline hour).

    RE-DATE (+365 days) -- canonical statement; the provenance sidecars point here.
    ResStock profiles are amy2018 (real 2018 weather); EIA's ISO-NE hourly
    fuel-type series begins 2019-01-01, so the load must be moved onto the carbon
    year. Two rules are defensible and this repo uses BOTH, for different windows:

      +365d  FULL-YEAR MA buildings (this function). 2018 and 2019 are both
             non-leap, so +365d is an exact day-for-day bijection of the whole
             year: Jan 1 -> Jan 1, Dec 31 -> Dec 31, nothing dropped or
             duplicated. Preserves DATE (hence season), shifts weekday by one.
      +364d  the AL one-WEEK slice (52 weeks; preserves season AND weekday).
             Correct for a 7-day window, wrong for a year: it maps 2018-01-01
             onto 2018-12-31, leaving 2019-12-31 uncovered and spilling a day
             back into 2018.

    +365d is chosen here because for both the carbon signal and the HVAC load the
    seasonal/date component dwarfs the weekday/weekend component, so preserving
    the date is worth more than preserving the weekday. This matches upstream
    carbon_sim.py, which defines REDATE=364 for the AL week and MA_REDATE=365 for
    the MA full year. The pairing remains a cross-year join (2018 weather priced
    against 2019 grid carbon) and must be disclosed as such in the manuscript.
    """
    df = pd.read_parquet(DATA / "hvac" / fname)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]) + MA_REDATE
    jobs, run_hours = nrel_to_jobs(df, utc_offset_hours=utc_offset_hours, flex_hours=flex_hours)
    return jobs, run_hours, df


# --- EV load (guarded: Caltech ACN-Data) ------------------------------------
def demo_ev_sessions(n: int = 200, seed: int = 1, origin: pd.Timestamp = DEMO_ORIGIN) -> list[dict]:
    """LABELLED synthetic ACN-Data-shaped charging sessions (RFC-1123 GMT strings).

    Same field names/format the real API returns, so ``acn_to_jobs`` parses them
    unchanged. NOT real driver behaviour.
    """
    rng = np.random.default_rng(seed)
    fmt = "%a, %d %b %Y %H:%M:%S GMT"
    out = []
    for i in range(n):
        connect = origin + pd.Timedelta(hours=int(rng.integers(0, 24 * 30)))
        charge_h = float(rng.uniform(1.5, 5.0))                 # time actually charging
        plugged_h = charge_h + float(rng.uniform(1.0, 8.0))     # dwell after full = slack
        done = connect + pd.Timedelta(hours=charge_h)
        disconnect = connect + pd.Timedelta(hours=plugged_h)
        out.append({
            "sessionID": f"demo_{i}",
            "connectionTime": connect.strftime(fmt),
            "doneChargingTime": done.strftime(fmt),
            "disconnectTime": disconnect.strftime(fmt),
            "kWhDelivered": float(rng.uniform(4.0, 30.0)),
        })
    return out


def get_ev_jobs(*, n: int = 200, verbose: bool = True):
    """REAL Caltech ACN-Data EV Jobs for 2019. Returns (jobs, source_label). RAISES.

    Served from the committed cache in data/ev/ when present, else fetched live
    (needs acnportal + ACN_API_TOKEN). Like the carbon signal, this no longer
    degrades to a synthetic set: the demo sessions are not real driver behaviour,
    and silently substituting them produced an EV saving that looked reportable
    but was not. ``demo_ev_sessions`` remains for illustration only.
    """
    import datetime as _dt

    from cals.acn_sessions import fetch_acn_sessions, read_provenance

    start, end = _dt.date(2019, 1, 1), _dt.date(2019, 12, 31)
    cache = DATA / "ev" / f"caltech_{start}_{end}.json"
    try:
        sessions = fetch_acn_sessions(start, end, cache_dir=DATA / "ev")
    except Exception as exc:
        raise RuntimeError(
            f"Real Caltech ACN-Data sessions are unavailable ({type(exc).__name__}: {exc}). "
            f"Expected a cached pull at {cache}, or install acnportal and set "
            "ACN_API_TOKEN to fetch live. Refusing to substitute synthetic EV sessions."
        ) from exc
    prov = read_provenance(cache) or {}
    label = "ACN-Data caltech 2019 (%s)" % (
        "cached real pull" if cache.is_file() else "LIVE API")
    if verbose and prov:
        print(f"EV: {prov.get('session_count', len(sessions))} real sessions "
              f"from {prov.get('endpoint', 'acn-data')}")
    return acn_to_jobs(sessions), label


# --- AI load (download-gated: Alibaba GPU v2020) ----------------------------
def demo_alibaba_trace(n: int = 300, seed: int = 2, span_days: int = 20) -> pd.DataFrame:
    """LABELLED synthetic Alibaba-pai_job-shaped frame: [job_name, start_time,
    end_time, plan_gpu]. start/end in seconds; plan_gpu in hundredths of a GPU
    (the trace's native unit). NOT the real trace -- download the real one for
    anything reported (see the markdown cell in notebook 02)."""
    rng = np.random.default_rng(seed)
    start_s = rng.integers(0, 3600 * 24 * span_days, n)
    dur_s = rng.integers(600, 3600 * 6, n)               # 10 min .. 6 h jobs
    return pd.DataFrame({
        "job_name": [f"j{i}" for i in range(n)],
        "start_time": start_s,
        "end_time": start_s + dur_s,
        "plan_gpu": rng.choice([100, 200, 400, 800], n),  # 1,2,4,8 GPUs
    })


def get_ai_jobs(*, gpu_power_kw: float = 0.4, flex_hours: int = 6,
                origin: pd.Timestamp = DEMO_ORIGIN, verbose: bool = True):
    """REAL Alibaba GPU v2020 AI-compute Jobs. Returns (jobs, parsed_df, label). RAISES.

    Requires a real trace CSV in data/ai/ and raises without one. Like the carbon
    signal and the EV sessions, this no longer degrades to a synthetic frame: a
    made-up job mix produces a savings number that looks reportable but measures
    nothing. ``demo_alibaba_trace`` remains for illustration only.

    gpu_power_kw and flex_hours are the two swept assumptions. NOTE that
    gpu_power_kw cancels out of the savings PERCENTAGE (it scales the baseline
    and the schedule identically) and moves only absolute gCO2 -- see
    results/ai_sweep.csv and figure 04e.
    """
    csvs = sorted((DATA / "ai").glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No Alibaba GPU v2020 trace in {DATA / 'ai'}. Download "
            "cluster-trace-gpu-v2020 (pai_task_table.csv) from "
            "https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020 "
            "and drop it there. Refusing to substitute a synthetic trace; the "
            "recorded real-trace results are in results/ai_sweep.csv."
        )
    try:
        raw = (load_pai_task_table(csvs[0]) if csvs[0].name == "pai_task_table.csv"
               else pd.read_csv(csvs[0]))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read the Alibaba trace at {csvs[0]} ({type(exc).__name__}: "
            f"{exc}). Refusing to substitute a synthetic trace."
        ) from exc
    label = f"Alibaba GPU v2020 ({csvs[0].name})"
    parsed = parse_alibaba_trace(raw, origin=origin)
    jobs = alibaba_to_jobs(parsed, gpu_power_kw=gpu_power_kw, flex_hours=flex_hours)
    return jobs, parsed, label


# --- small scheduling / plotting conveniences -------------------------------
def savings_pct(baseline_gco2: float, opt_gco2: float) -> float:
    """Percent carbon saved: (baseline - optimal) / baseline * 100."""
    return 100.0 * (baseline_gco2 - opt_gco2) / baseline_gco2 if baseline_gco2 else 0.0


def bootstrap_savings_ci(
    baseline_daily: pd.Series,
    scheduled_daily: pd.Series,
    *,
    n_boot: int = 1_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Return a 95% bootstrap CI for savings by resampling whole days."""
    if not baseline_daily.index.equals(scheduled_daily.index):
        raise ValueError("baseline and scheduled daily totals must share an index")
    if baseline_daily.empty or n_boot < 1:
        raise ValueError("daily totals must be nonempty and n_boot must be positive")

    values = np.column_stack((baseline_daily.to_numpy(float), scheduled_daily.to_numpy(float)))
    picks = np.random.default_rng(seed).integers(0, len(values), size=(n_boot, len(values)))
    totals = values[picks].sum(axis=1)
    savings = 100.0 * (totals[:, 0] - totals[:, 1]) / totals[:, 0]
    lower, upper = np.quantile(savings, [0.025, 0.975])
    return float(lower), float(upper)


def price_at(job, start, ci: pd.Series) -> float:
    """kWh-weighted gCO2 of running ``job`` for its whole duration from ``start``.

    Mirrors the scheduler's cost model: power_kw * sum(CI over the run hours).
    """
    total = 0.0
    for k in range(duration_h(job)):
        total += job.power_kw * float(ci.loc[start + pd.Timedelta(hours=k)])
    return total


def do_nothing_gco2(jobs, run_hours, ci: pd.Series) -> float:
    """Physical do-nothing cost: every job priced at its metered run hour.

    Used where dropping capacity-infeasible jobs would bias a fifo ratio upward.
    """
    return float(sum(price_at(j, run_hours[j.job_id], ci) for j in jobs))


def savefig(fig, name: str) -> Path:
    """Save a figure into ../figures/ at print resolution and report the path."""
    path = FIG / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved figure -> {path}")
    return path
