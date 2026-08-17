"""Shared helpers for the five paper notebooks.

Nothing here is novel science -- it is glue that (a) GUARDS every live-API /
download path so the notebooks always run offline, and (b) keeps the demo data
clearly LABELLED so a reader never confuses a synthetic curve for a measured one.
All the real modelling lives in ``cals`` (imported below, never reimplemented).

The two guarded, network/key-gated paths:
  * carbon signal   -> EIA Open Data API   (needs EIA_API_KEY)
  * EV charging      -> Caltech ACN-Data    (needs acnportal + ACN_API_TOKEN)
The two offline-real paths:
  * HVAC load        -> committed NREL ResStock parquet in data/hvac/
  * emission factors -> cals.FACTORS
The download-gated path:
  * AI load          -> Alibaba GPU v2020 trace (drop a CSV in data/ai/); a
                        clearly-labelled synthetic Alibaba-shaped frame stands in.
"""
from __future__ import annotations

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


def get_fuel_mix(start: str = CI_START, end: str = CI_END, region: str = "ISO-NE",
                 verbose: bool = True) -> tuple[pd.DataFrame, str]:
    """Hourly fuel mix [timestamp, fueltype, gen_mwh], guarded. Returns (mix, label).

    If EIA_API_KEY is set (loaded from ../.env), fetch the real ISO-NE mix.
    Otherwise print a clear warning and return a LABELLED demo mix -- so nothing
    ever crashes on a missing key. Exposing the *mix* (not just the CI) lets the
    emission-factor sensitivity sweep re-price it under alternative factors.
    """
    key = os.environ.get("EIA_API_KEY", "").strip()
    if key:
        try:
            mix = fetch_eia_fuel_mix(
                pd.Timestamp(start).date(), pd.Timestamp(end).date(),
                region, raw_dir=DATA / "carbon", api_key=key,
            )
            return mix, f"EIA {region} (LIVE API)"
        except Exception as exc:  # network hiccup, bad key, cache clash -> stay runnable
            if verbose:
                print(f"EIA fetch failed ({type(exc).__name__}: {exc}); using DEMO mix instead.")
    elif verbose:
        print("⚠️  set EIA_API_KEY in .env to fetch real carbon; "
              "using a labelled DEMO CI curve instead")
    return demo_fuel_mix(start, end), "DEMO synthetic CI (offline)"


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
    """
    df = pd.read_parquet(DATA / "hvac" / fname)
    # ResStock is amy2018 (2018 weather); real EIA ISO-NE carbon starts 2019.
    # Re-date +365 days -- both are non-leap years, so it is an exact day-for-day
    # shift -- so the 2018 load lines up with the 2019 carbon curve (Rajan's rule).
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]) + pd.Timedelta(days=365)
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
    """EV charging Jobs, guarded. Returns (jobs, source_label).

    Tries live Caltech ACN-Data (needs acnportal + ACN_API_TOKEN). If either is
    absent it skips gracefully to a LABELLED demo set rather than erroring.
    """
    try:
        from cals.acn_sessions import fetch_acn_sessions  # imports acnportal
        import datetime as _dt
        sessions = fetch_acn_sessions(_dt.date(2019, 1, 1), _dt.date(2019, 12, 31),
                                      cache_dir=DATA / "ev")
        return acn_to_jobs(sessions), "ACN-Data caltech (LIVE API)"
    except Exception as exc:
        if verbose:
            print(f"EV: real ACN-Data unavailable ({type(exc).__name__}); using a labelled "
                  "DEMO EV set. Install acnportal + set ACN_API_TOKEN for real sessions.")
        return acn_to_jobs(demo_ev_sessions(n=n)), "DEMO synthetic EV (offline)"


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
    """AI-compute Jobs, guarded. Returns (jobs, parsed_df, source_label).

    Uses a real Alibaba GPU v2020 CSV if one is present in data/ai/; otherwise a
    LABELLED synthetic Alibaba-shaped frame. gpu_power_kw and flex_hours are the
    two swept assumptions (see notebook 04 style sweeps).
    """
    csvs = sorted((DATA / "ai").glob("*.csv"))
    raw, label = None, None
    if csvs:
        try:
            if csvs[0].name == "pai_task_table.csv":
                raw = load_pai_task_table(csvs[0])
            else:
                raw = pd.read_csv(csvs[0])
            label = f"Alibaba GPU v2020 ({csvs[0].name})"
        except Exception as exc:
            if verbose:
                print(f"AI: failed to read {csvs[0].name} ({exc}); using synthetic.")
            raw = None
    if raw is None:
        if verbose:
            print("AI: real Alibaba GPU v2020 trace not found in data/ai/; using a "
                  "labelled SYNTHETIC Alibaba-shaped trace.")
        raw = demo_alibaba_trace()
        label = "DEMO synthetic Alibaba (offline)"
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
