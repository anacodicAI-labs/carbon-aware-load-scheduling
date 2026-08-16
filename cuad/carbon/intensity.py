"""Hourly carbon intensity (gCO2/kWh) from a fuel mix."""
from __future__ import annotations

import warnings

import pandas as pd

from cuad.carbon.factors import FACTORS, OTHER_CODE


def carbon_intensity(mix_df: pd.DataFrame) -> pd.Series:
    """Generation-weighted hourly carbon intensity from a fuel mix.

    Parameters
    ----------
    mix_df:
        Long-format dataframe with columns ``[timestamp, fueltype, gen_mwh]``
        (e.g. the output of ``fetch_eia_fuel_mix``).

    Returns
    -------
    A Series indexed by timestamp, named ``carbon_intensity_gco2_kwh``, giving
    the generation-weighted mean lifecycle emission factor for each hour.

    Behaviour
    ---------
    * Negative ``gen_mwh`` is clipped to 0 before computing shares (negative
      generation appears occasionally in raw EIA data and must not create
      negative shares).
    * Unknown fuel type codes emit a warning and fall back to the ``OTH``
      factor; the function never raises on an unrecognised code.
    * Hours whose total (clipped) generation is 0 yield ``NaN``.
    """
    df = mix_df[["timestamp", "fueltype", "gen_mwh"]].copy()
    df["gen_mwh"] = (
        pd.to_numeric(df["gen_mwh"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )

    unknown = sorted(c for c in df["fueltype"].unique() if c not in FACTORS)
    if unknown:
        warnings.warn(
            f"Unknown fuel type code(s) {unknown}; treating as "
            f"'{OTHER_CODE}' ({FACTORS[OTHER_CODE]} gCO2/kWh)",
            stacklevel=2,
        )
    default = FACTORS[OTHER_CODE]
    df["factor"] = df["fueltype"].map(lambda code: FACTORS.get(code, default))
    df["weighted"] = df["gen_mwh"] * df["factor"]

    grouped = df.groupby("timestamp")
    total_gen = grouped["gen_mwh"].sum()
    total_weighted = grouped["weighted"].sum()
    intensity = total_weighted / total_gen.where(total_gen > 0)
    intensity.name = "carbon_intensity_gco2_kwh"
    return intensity.sort_index()
