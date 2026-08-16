"""EIA Open Data API v2 - hourly generation by fuel type for a Balancing Authority.

Endpoint: https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/
Free API key: https://www.eia.gov/opendata/register.php  (env var EIA_API_KEY)

Mirrors cuad.data.sources.eia (the demand fetch): same auth, 5000-row offset
pagination, CSV cache, and UTC labelling. Output columns:
[timestamp, fueltype, gen_mwh].

ISO-NE (respondent ISNE) reports these fuel type codes:
    COL coal, NG natural gas, NUC nuclear, OIL petroleum,
    OTH other (refuse / biomass / landfill gas), SUN solar, WAT hydro, WND wind
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from cuad.data.cache import load_cached_csv, save_cached_csv
from cuad.data.sources.eia import _respondent  # reuse the respondent-code mapping
from cuad.data.sources.synthetic import generate_synthetic_fuel_mix

_EIA_FUEL_BASE = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
_PAGE_SIZE = 5000  # max rows per EIA API page

# EIA fuel type codes reported by ISO-NE (ISNE).
FUELTYPES: list[str] = ["COL", "NG", "NUC", "OIL", "OTH", "SUN", "WAT", "WND"]


def _meta_path(cache: Path) -> Path:
    """Sidecar that records where a cached CSV came from ('eia' or 'synthetic')."""
    return cache.with_suffix(cache.suffix + ".meta.json")


def _write_provenance(cache: Path, source: str) -> None:
    """Stamp the cache's provenance so synthetic data is distinguishable at rest."""
    _meta_path(cache).write_text(json.dumps({"source": source}))


def _read_provenance(cache: Path) -> str | None:
    """Recorded source for a cached CSV, or None when no sidecar exists (legacy)."""
    meta = _meta_path(cache)
    if not meta.is_file():
        return None
    try:
        return json.loads(meta.read_text()).get("source")
    except (ValueError, OSError):
        return None


def _fetch_page(
    respondent: str,
    start_dt: str,
    end_dt: str,
    api_key: str,
    offset: int = 0,
) -> list[dict]:
    """Fetch one page of hourly generation-by-fuel-type from EIA API v2."""
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": respondent,
        "facets[fueltype][]": FUELTYPES,
        "start": start_dt,
        "end": end_dt,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": _PAGE_SIZE,
        "offset": offset,
    }
    resp = requests.get(_EIA_FUEL_BASE, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("response", {}).get("data", [])


def fetch_eia_fuel_mix(
    start: date,
    end: date,
    region: str = "ISO-NE",
    *,
    raw_dir: Path,
    api_key: str = "",
    use_synthetic_if_missing: bool = True,
) -> pd.DataFrame:
    """Fetch hourly generation by fuel type (MWh) from EIA Open Data API v2.

    Parameters
    ----------
    start, end:
        Inclusive date range.
    region:
        Friendly region name (e.g. ``"ISO-NE"``) or an EIA respondent code
        (e.g. ``"ISNE"``).
    raw_dir:
        Directory for CSV cache.
    api_key:
        EIA API key. Falls back to synthetic data when empty.
    use_synthetic_if_missing:
        When True and no API key is provided, returns a synthetic fuel mix.

    Returns
    -------
    Long-format dataframe with columns ``[timestamp, fueltype, gen_mwh]``.
    """
    respondent = _respondent(region)
    cache = raw_dir / "eia" / f"fuelmix_{respondent}_{start}_{end}.csv"

    # --- cache hit ---
    cached = load_cached_csv(cache)
    if cached is not None and not cached.empty:
        # Never serve synthetic-sourced cache to a caller who has a real key.
        if api_key and _read_provenance(cache) == "synthetic":
            raise RuntimeError(
                f"cached fuel mix at {cache} was written from SYNTHETIC data, but an "
                f"api_key is present. Delete {cache} and {_meta_path(cache)}, then "
                f"re-fetch to get real EIA data."
            )
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
        return (
            cached[["timestamp", "fueltype", "gen_mwh"]]
            .sort_values(["timestamp", "fueltype"])
            .reset_index(drop=True)
        )

    # --- live fetch ---
    if api_key:
        # EIA expects UTC datetime strings: "2024-01-01T00"
        start_str = datetime.combine(start, datetime.min.time()).strftime("%Y-%m-%dT%H")
        end_str = (datetime.combine(end, datetime.min.time()) + timedelta(hours=23)).strftime("%Y-%m-%dT%H")
        rows: list[dict] = []
        offset = 0
        while True:
            try:
                page = _fetch_page(respondent, start_str, end_str, api_key, offset)
            except Exception as exc:
                if not rows and use_synthetic_if_missing:
                    break  # fall through to synthetic
                raise RuntimeError(f"EIA fuel-mix API error at offset {offset}: {exc}") from exc
            if not page:
                break
            rows.extend(page)
            if len(page) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE

        if rows:
            df = pd.DataFrame(rows)
            # EIA period format: "2024-01-01T00"; label as UTC (UTC has no DST
            # transitions, so duplicate hourly timestamps localise cleanly).
            df["timestamp"] = pd.to_datetime(df["period"], errors="coerce").dt.tz_localize("UTC")
            df["gen_mwh"] = pd.to_numeric(df["value"], errors="coerce")
            df["fueltype"] = df["fueltype"].astype(str)
            df = df.dropna(subset=["timestamp", "gen_mwh", "fueltype"])
            out = (
                df[["timestamp", "fueltype", "gen_mwh"]]
                .sort_values(["timestamp", "fueltype"])
                .reset_index(drop=True)
            )
            save_cached_csv(out, cache)
            _write_provenance(cache, "eia")
            return out

    # --- synthetic fallback ---
    if use_synthetic_if_missing:
        hours = int(
            (
                datetime.combine(end, datetime.min.time())
                - datetime.combine(start, datetime.min.time())
            ).total_seconds()
            // 3600
        ) + 24
        start_dt = datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc)
        out = generate_synthetic_fuel_mix(start_dt, hours=hours)
        save_cached_csv(out, cache)
        _write_provenance(cache, "synthetic")
        return out

    raise RuntimeError(
        f"EIA API key missing and synthetic fallback disabled for region={region}"
    )
