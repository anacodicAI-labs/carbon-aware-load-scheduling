"""EIA Open Data API v2 — hourly electricity demand by Balancing Authority.

Endpoint: https://api.eia.gov/v2/electricity/rto/region-data/data/
Free API key: https://www.eia.gov/opendata/register.php

Supported region codes (``respondent`` in EIA terminology):
    ISO-NE  → ISNE
    PJM     → PJM
    MISO    → MISO
    CAISO   → CISO
    NYISO   → NYIS
    ERCOT   → ERCO
    SPP     → SPP
    WECC    → NWMT  (NorthWestern Montana, example)
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from cuad.data.cache import load_cached_csv, save_cached_csv
from cuad.data.sources.synthetic import generate_synthetic_load

# Mapping from friendly region names → EIA respondent codes
EIA_REGION_MAP: dict[str, str] = {
    "ISO-NE": "ISNE",
    "ISNE":   "ISNE",
    "PJM":    "PJM",
    "MISO":   "MISO",
    "CAISO":  "CISO",
    "CISO":   "CISO",
    "NYISO":  "NYIS",
    "NYIS":   "NYIS",
    "ERCOT":  "ERCO",
    "ERCO":   "ERCO",
    "SPP":    "SPP",
}

_EIA_BASE = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
_PAGE_SIZE = 5000  # max rows per EIA API page


def _respondent(region: str) -> str:
    """Convert friendly region name to EIA respondent code."""
    return EIA_REGION_MAP.get(region.upper(), region.upper())


def _fetch_page(
    respondent: str,
    start_dt: str,
    end_dt: str,
    api_key: str,
    offset: int = 0,
) -> list[dict]:
    """Fetch one page of hourly demand from EIA API v2."""
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": respondent,
        "facets[type][]": "D",          # D = demand (actual load)
        "start": start_dt,
        "end": end_dt,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": _PAGE_SIZE,
        "offset": offset,
    }
    resp = requests.get(_EIA_BASE, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("response", {}).get("data", [])


def fetch_eia_load(
    start: date,
    end: date,
    region: str = "PJM",
    *,
    raw_dir: Path,
    api_key: str = "",
    use_synthetic_if_missing: bool = True,
) -> pd.DataFrame:
    """Fetch hourly actual demand (MWh) from EIA Open Data API v2.

    Parameters
    ----------
    start, end:
        Inclusive date range.
    region:
        Friendly region name (e.g. ``"PJM"``, ``"MISO"``, ``"CAISO"``).
        Also accepts EIA respondent codes directly (e.g. ``"CISO"``).
    raw_dir:
        Directory for CSV cache.
    api_key:
        EIA API key. Falls back to synthetic data when empty.
    use_synthetic_if_missing:
        When True and no API key is provided, returns synthetic load.
    """
    respondent = _respondent(region)
    cache = raw_dir / "eia" / f"load_{respondent}_{start}_{end}.csv"

    # --- cache hit ---
    cached = load_cached_csv(cache)
    if cached is not None and not cached.empty:
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
        return cached[["timestamp", "load_mw", "region"]].sort_values("timestamp").reset_index(drop=True)

    # --- live fetch ---
    if api_key:
        # EIA expects UTC datetime strings: "2024-01-01T00"
        start_str = datetime.combine(start, datetime.min.time()).strftime("%Y-%m-%dT%H")
        end_str   = (datetime.combine(end, datetime.min.time()) + timedelta(hours=23)).strftime("%Y-%m-%dT%H")
        rows: list[dict] = []
        offset = 0
        while True:
            try:
                page = _fetch_page(respondent, start_str, end_str, api_key, offset)
            except Exception as exc:
                if not rows and use_synthetic_if_missing:
                    break  # fall through to synthetic
                raise RuntimeError(f"EIA API error at offset {offset}: {exc}") from exc
            if not page:
                break
            rows.extend(page)
            if len(page) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE

        if rows:
            df = pd.DataFrame(rows)
            # EIA period format: "2024-01-01T00" (local ET for most BAs)
            df["timestamp"] = pd.to_datetime(df["period"], utc=False, errors="coerce")
            # Localise to UTC (EIA reports in local prevailing time for each BA;
            # for simplicity we label as UTC — consistent with the rest of the pipeline)
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC", ambiguous="infer", nonexistent="shift_forward")
            df["load_mw"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["timestamp", "load_mw"])
            df["region"] = region
            out = df[["timestamp", "load_mw", "region"]].sort_values("timestamp").reset_index(drop=True)
            save_cached_csv(out, cache)
            return out

    # --- synthetic fallback ---
    if use_synthetic_if_missing:
        hours = int(
            (
                datetime.combine(end, datetime.min.time())
                - datetime.combine(start, datetime.min.time())
            ).total_seconds() // 3600
        ) + 24
        hours = max(hours, 8760)
        start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
        df = generate_synthetic_load(start_dt, hours=hours, region=region)
        save_cached_csv(df, cache)
        return df

    raise RuntimeError(
        f"EIA API key missing and synthetic fallback disabled for region={region}"
    )
