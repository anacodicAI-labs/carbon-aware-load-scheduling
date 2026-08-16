from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def generate_synthetic_load(
    start: datetime,
    hours: int = 8760,
    region: str = "ISO-NE",
    seed: int = 42,
) -> pd.DataFrame:
    """Demo hourly load when ISO-NE credentials or pulls are unavailable."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=hours, freq="h", tz="UTC")
    hour = idx.hour
    dow = idx.dayofweek
    base = 12000 + 2500 * np.sin(2 * np.pi * (hour / 24)) + 800 * (dow >= 5)
    temp = 10 + 12 * np.sin(2 * np.pi * (idx.dayofyear / 365)) + rng.normal(0, 2, hours)
    load = base + 180 * temp + rng.normal(0, 150, hours)
    return pd.DataFrame({"timestamp": idx, "region": region, "load_mw": load.astype(float)})


def generate_synthetic_weather(index: pd.DatetimeIndex, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    n = len(index)
    temp = 10 + 12 * np.sin(2 * np.pi * (index.dayofyear / 365)) + rng.normal(0, 2, n)
    return pd.DataFrame(
        {
            "timestamp": index,
            "temp_c": temp,
            "wind_ms": np.clip(rng.normal(5, 2, n), 0, None),
            "humidity": np.clip(rng.normal(60, 15, n), 0, 100),
            "precip": np.clip(rng.exponential(0.05, n), 0, None),
        }
    )


def generate_synthetic_fuel_mix(
    start: datetime,
    hours: int = 8760,
    seed: int = 42,
) -> pd.DataFrame:
    """Demo hourly generation by fuel type (MWh) when an EIA key is unavailable.

    Returns long-format rows [timestamp, fueltype, gen_mwh]; all gen >= 0.
    """
    rng = np.random.default_rng(seed + 2)
    idx = pd.date_range(start=start, periods=hours, freq="h", tz="UTC")
    hour = idx.hour
    diurnal = np.sin(2 * np.pi * hour / 24)
    solar = np.clip(np.sin(2 * np.pi * (hour - 6) / 24), 0, None)
    profiles = {
        "NG": 5000 + 1500 * diurnal + rng.normal(0, 200, hours),
        "NUC": 3200 + rng.normal(0, 40, hours),
        "WAT": 1200 + 300 * diurnal + rng.normal(0, 80, hours),
        "WND": 600 + 500 * rng.normal(0, 1, hours),
        "SUN": 900 * solar,
        "COL": 150 + rng.normal(0, 50, hours),
        "OIL": 40 + rng.normal(0, 20, hours),
        "OTH": 250 + rng.normal(0, 40, hours),
    }
    frames = [
        pd.DataFrame(
            {"timestamp": idx, "fueltype": code, "gen_mwh": np.clip(vals, 0, None).astype(float)}
        )
        for code, vals in profiles.items()
    ]
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["timestamp", "fueltype"]).reset_index(drop=True)
