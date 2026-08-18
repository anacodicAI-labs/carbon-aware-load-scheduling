#!/usr/bin/env python3
"""Select candidate ResStock HVAC buildings for the carbon-aware scheduler.

Regenerates the heat-pump shortlist that the MA sensitivity set came from, so the
building choice is reproducible from code instead of being an unexplained pick.

Why this exists: the original AL building (bldg 1) is GAS heated, so its electric
HVAC series carries cooling only -- a full-year run would put no winter heating in
front of the scheduler. To get a dual-season electric HVAC load we need a building
whose heating is electric AND has no fossil heating at all (MA has a lot of oil
heat, so gas alone is not a sufficient filter).

Selection criterion (see CRITERION below, recorded verbatim in each building's
provenance sidecar):
  1. Single-Family Detached
  2. heating type is a heat pump (Ducted or Non-Ducted)
  3. heating fuel is Electricity
  4. ZERO fossil heating energy -- natural gas, fuel oil, AND propane, including
     the heat-pump-backup variants of each
  5. both seasons materially present (annual electric heating and cooling each
     above --min-kwh) so the building is not effectively single-season
  6. ranked by heating/cooling ratio ASCENDING, i.e. most balanced first, rather
     than heating-dominated

Usage (from code/):
    python scripts/select_hvac_building.py                 # MA, top 5
    python scripts/select_hvac_building.py --state AL --top 3
    python scripts/select_hvac_building.py --breakdown     # also print the
                                                           # heating-fuel mix

The metadata parquet is downloaded from the public OEDI data lake (no
credentials) and cached under --cache-dir. Nothing else is fetched: this reads
metadata only and never downloads a per-building timeseries.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

OEDI_BASE = (
    "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock"
)
RELEASE = "2024/resstock_amy2018_release_2"

CRITERION = (
    "MA single-family detached, heat-pump heating (Ducted or Non-Ducted), heating "
    "fuel Electricity, ZERO fossil heating (natural gas + fuel oil + propane, incl. "
    "hp_bkup variants), annual electric heating and cooling both > 200 kWh, ranked by "
    "heating/cooling ratio ascending (most balanced first). "
    "Reproduce with scripts/select_hvac_building.py."
)

HEAT_PUMP_TYPES = ("Ducted Heat Pump", "Non-Ducted Heat Pump")

ELEC_HEAT_COLS = (
    "out.electricity.heating.energy_consumption.kwh",
    "out.electricity.heating_fans_pumps.energy_consumption.kwh",
    "out.electricity.heating_hp_bkup.energy_consumption.kwh",
    "out.electricity.heating_hp_bkup_fa.energy_consumption.kwh",
)
BACKUP_COLS = (
    "out.electricity.heating_hp_bkup.energy_consumption.kwh",
    "out.electricity.heating_hp_bkup_fa.energy_consumption.kwh",
)
# Every fossil heating end use. Gas alone is NOT enough -- MA is ~40% oil heat.
FOSSIL_HEAT_COLS = (
    "out.natural_gas.heating.energy_consumption.kwh",
    "out.natural_gas.heating_hp_bkup.energy_consumption.kwh",
    "out.fuel_oil.heating.energy_consumption.kwh",
    "out.fuel_oil.heating_hp_bkup.energy_consumption.kwh",
    "out.propane.heating.energy_consumption.kwh",
    "out.propane.heating_hp_bkup.energy_consumption.kwh",
)
COOL_COL = "out.electricity.cooling.energy_consumption.kwh"

META_COLS = (
    "in.heating_fuel",
    "in.hvac_heating_type",
    "in.hvac_heating_efficiency",
    "in.hvac_cooling_type",
    "in.hvac_cooling_efficiency",
    "in.county_name",
    "in.geometry_building_type_recs",
    "in.sqft",
    "in.vintage",
    "in.ashrae_iecc_climate_zone_2004",
    "weight",
)


def metadata_url(state: str, release: str = RELEASE) -> str:
    """OEDI URL of the baseline (upgrade 0) metadata + annual results for a state."""
    return (
        f"{OEDI_BASE}/{release}/metadata_and_annual_results/by_state/"
        f"state={state}/parquet/{state}_baseline_metadata_and_annual_results.parquet"
    )


def timeseries_url(building_id: int, state: str, *, upgrade: int = 0, release: str = RELEASE) -> str:
    """OEDI URL of one building's 15-minute timeseries (NOT downloaded here)."""
    return (
        f"{OEDI_BASE}/{release}/timeseries_individual_buildings/by_state/"
        f"upgrade={upgrade}/state={state}/{building_id}-{upgrade}.parquet"
    )


def load_metadata(state: str, *, cache_dir: Path, release: str = RELEASE) -> pd.DataFrame:
    """Download (once) and read a state's baseline metadata parquet."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{state}_baseline_metadata.parquet"
    if not cache.is_file():
        url = metadata_url(state, release)
        print(f"downloading {url}")
        # requests (not urllib) so the bundled CA store is used; the macOS stdlib
        # SSL context has no local issuer certificate and fails on this host.
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
        cache.write_bytes(resp.content)
    cols = list(META_COLS) + list(ELEC_HEAT_COLS) + list(FOSSIL_HEAT_COLS) + [COOL_COL]
    # bldg_id is the parquet INDEX, not a column, so reset_index() to surface it.
    return pd.read_parquet(cache, columns=cols).reset_index()


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived energy totals the criterion is expressed in."""
    out = df.copy()
    out["elec_heat_kwh"] = out[list(ELEC_HEAT_COLS)].astype(float).sum(axis=1)
    out["backup_kwh"] = out[list(BACKUP_COLS)].astype(float).sum(axis=1)
    out["fossil_heat_kwh"] = out[list(FOSSIL_HEAT_COLS)].astype(float).sum(axis=1)
    out["cool_kwh"] = out[COOL_COL].astype(float)
    out["sqft_n"] = pd.to_numeric(out["in.sqft"], errors="coerce")
    return out


def select(df: pd.DataFrame, *, top: int = 5, min_kwh: float = 200.0) -> pd.DataFrame:
    """Apply the criterion and rank most-balanced-first."""
    hit = (
        (df["in.geometry_building_type_recs"] == "Single-Family Detached")
        & (df["in.hvac_heating_type"].isin(HEAT_PUMP_TYPES))
        & (df["in.heating_fuel"] == "Electricity")
        & (df["fossil_heat_kwh"] == 0.0)          # no gas, no oil, no propane
        & (df["elec_heat_kwh"] > min_kwh)
        & (df["cool_kwh"] > min_kwh)
    )
    cand = df[hit].copy()
    cand["heat_cool_ratio"] = cand["elec_heat_kwh"] / cand["cool_kwh"]
    cand["backup_pct_of_heat"] = cand["backup_kwh"] / cand["elec_heat_kwh"] * 100.0
    return cand.sort_values("heat_cool_ratio").head(top)


def heating_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Single-family-detached heating mix, for representativeness reporting."""
    sfd = df[df["in.geometry_building_type_recs"] == "Single-Family Detached"].copy()

    def category(row: pd.Series) -> str:
        if row["in.hvac_heating_type"] in HEAT_PUMP_TYPES:
            return "Heat Pump (electric)"
        fuel = row["in.heating_fuel"]
        if fuel == "Electricity":
            return "Electric Resistance"
        return str(fuel)

    sfd["category"] = sfd.apply(category, axis=1)
    grouped = sfd.groupby("category").agg(count=("bldg_id", "size"))
    grouped["pct"] = (grouped["count"] / len(sfd) * 100.0).round(1)
    return grouped.sort_values("count", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--state", default="MA", help="two-letter state (default MA)")
    parser.add_argument("--top", type=int, default=5, help="candidates to print (default 5)")
    parser.add_argument("--min-kwh", type=float, default=200.0,
                        help="minimum annual electric heating AND cooling kWh (default 200)")
    parser.add_argument("--release", default=RELEASE, help=f"ResStock release (default {RELEASE})")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/raw/nrel/_metadata_cache"),
                        help="where to cache the downloaded metadata parquet")
    parser.add_argument("--breakdown", action="store_true",
                        help="also print the single-family-detached heating-fuel mix")
    args = parser.parse_args()

    df = annotate(load_metadata(args.state, cache_dir=args.cache_dir, release=args.release))
    cand = select(df, top=args.top, min_kwh=args.min_kwh)

    print(f"\nCRITERION: {CRITERION}\n")
    print(f"=== {args.state}: top {len(cand)} candidates (most balanced first) ===")
    view = cand[[
        "bldg_id", "in.county_name", "in.ashrae_iecc_climate_zone_2004",
        "in.hvac_heating_type", "in.hvac_heating_efficiency", "sqft_n", "in.vintage",
        "elec_heat_kwh", "cool_kwh", "heat_cool_ratio", "backup_kwh", "backup_pct_of_heat",
    ]].rename(columns={
        "in.county_name": "county", "in.ashrae_iecc_climate_zone_2004": "cz",
        "in.hvac_heating_type": "heat_type", "in.hvac_heating_efficiency": "heat_eff",
        "sqft_n": "sqft", "in.vintage": "vintage",
    })
    for col in ("elec_heat_kwh", "cool_kwh", "backup_kwh"):
        view[col] = view[col].round(0)
    for col in ("heat_cool_ratio", "backup_pct_of_heat"):
        view[col] = view[col].round(2)
    print(view.to_string(index=False))

    print("\ntimeseries URLs (not downloaded by this script):")
    for bid in cand["bldg_id"]:
        print(f"  {bid}: {timeseries_url(int(bid), args.state, release=args.release)}")

    if args.breakdown:
        print(f"\n=== {args.state} single-family-detached heating mix ===")
        print(heating_breakdown(df).to_string())


if __name__ == "__main__":
    main()
