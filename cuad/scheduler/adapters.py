"""Adapters that turn real public datasets into scheduler Job objects.

Each function maps an external record into the Job contract in jobs.py
(energy_kwh, earliest_start, deadline, power_kw, all times UTC on the hourly
grid the scheduler runs on).

Implemented:
- nrel_to_jobs: NREL End-Use Load Profiles HVAC electricity -> deferrable Jobs.
- acn_to_jobs: Caltech ACN-Data EV charging sessions -> deadline Jobs.
"""
from __future__ import annotations

import pandas as pd

from cuad.scheduler.jobs import Job, slack_h

# Electric HVAC end uses in the NREL ResStock timeseries. Gas heating shows up in
# separate out.natural_gas.* columns and is ignored on purpose: the scheduler
# defers electricity, so only electric HVAC is a movable electrical load.
HVAC_ELECTRIC_COLS = (
    "out.electricity.cooling.energy_consumption",
    "out.electricity.cooling_fans_pumps.energy_consumption",
    "out.electricity.heating.energy_consumption",
    "out.electricity.heating_fans_pumps.energy_consumption",
    "out.electricity.heating_hp_bkup.energy_consumption",
    "out.electricity.heating_hp_bkup_fa.energy_consumption",
)


def nrel_to_jobs(
    df: pd.DataFrame,
    *,
    utc_offset_hours: int,
    flex_hours: int = 2,
    min_kwh: float = 0.05,
    building_id: str = "nrel",
) -> tuple[list[Job], dict[str, pd.Timestamp]]:
    """Turn one NREL ResStock building's HVAC timeseries into deferrable Jobs.

    Input is the raw per-building parquet loaded into a DataFrame: a `timestamp`
    column plus the out.electricity.* HVAC columns. The source is 15-minute and
    stamped at the END of each interval in LOCAL STANDARD TIME (no daylight
    saving), so the value at 17:00 is the energy used from 16:00 to 17:00.

    Steps:
    1. Sum the electric HVAC columns that are present into one 15-minute series.
    2. Aggregate to clock-hour blocks, labeled at the hour START.
    3. Shift by utc_offset_hours to get UTC (local standard time is a fixed
       offset, so a single integer is correct; e.g. -6 for US Central).
    4. Emit one Job per hour whose HVAC energy exceeds min_kwh.

    MODELING ASSUMPTION (needs review): HVAC has no hard deadline the way an EV
    charging session does. Each hour of HVAC energy is treated as a one-hour
    block that may run up to flex_hours earlier or later than it actually did,
    standing in for thermal inertia and comfort slack. The window is therefore
    [hour - flex_hours, hour + 1 + flex_hours], giving 2 * flex_hours of slack.
    flex_hours is a knob, not a physical fact; the default of 2 is a placeholder.

    power_kw is set to the hour's energy divided by one hour, i.e. the average
    HVAC power over that hour, so every block is exactly one hour long and
    reproduces the measured energy. Hours at or below min_kwh are dropped: the
    HVAC is effectively off, there is nothing to defer, and a near-zero power
    would not be meaningful.

    Returns (jobs, run_hours). run_hours maps job_id -> the UTC hour the load
    ACTUALLY ran (the block's own hour, independent of flex_hours). A do-nothing
    HVAC baseline must be priced at that hour, NOT at earliest_start: since
    earliest_start = hour - flex_hours, pricing the baseline at earliest_start
    would make it drift (pre-cool earlier) as flex_hours grows.
    """
    present = [c for c in HVAC_ELECTRIC_COLS if c in df.columns]
    if not present:
        raise ValueError(
            "no NREL electric HVAC columns found; expected some of "
            f"{HVAC_ELECTRIC_COLS}"
        )
    if "timestamp" not in df.columns:
        raise ValueError("expected a 'timestamp' column in the NREL frame")

    ts = pd.to_datetime(df["timestamp"])
    hvac_15min = df[present].sum(axis=1)
    series = pd.Series(hvac_15min.to_numpy(dtype=float), index=pd.DatetimeIndex(ts))

    # End-labeled 15-min -> hour blocks labeled at the hour start.
    hourly = series.resample("1h", label="left", closed="right").sum()
    hour_start_utc = (hourly.index - pd.Timedelta(hours=utc_offset_hours)).tz_localize("UTC")

    flex = pd.Timedelta(hours=flex_hours)
    one_hour = pd.Timedelta(hours=1)
    jobs: list[Job] = []
    run_hours: dict[str, pd.Timestamp] = {}
    for start, energy in zip(hour_start_utc, hourly.to_numpy(dtype=float)):
        energy = float(energy)
        if energy <= min_kwh:
            continue
        power = energy / 1.0  # average HVAC power over the hour; duration is 1h
        job_id = f"{building_id}_{start.strftime('%Y%m%dT%H')}"
        jobs.append(Job(job_id, energy, start - flex, start + one_hour + flex, power))
        run_hours[job_id] = start  # the hour the load actually ran (do-nothing baseline)
    return jobs, run_hours


# Timestamps ACN-Data reports as RFC-1123 strings in GMT (== UTC), e.g.
# "Sun, 01 Sep 2019 02:23:44 GMT". kWhDelivered is the energy; there is no
# charging-rate field, so power_kw is derived (see acn_to_jobs).
def _acn_time(value: object) -> pd.Timestamp | None:
    """Parse one ACN timestamp string to tz-aware UTC, or None if unparseable."""
    if not value:
        return None
    ts = pd.to_datetime(str(value), utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


def acn_to_jobs(sessions: list[dict]) -> list[Job]:
    """Turn Caltech ACN-Data EV charging sessions into deadline Jobs.

    Input is the list of session dicts from the ACN-Data API (the ``_items``
    array). Each session carries ``connectionTime``, ``disconnectTime``,
    ``doneChargingTime`` (RFC-1123 GMT strings) and ``kWhDelivered``.

    Mapping to the Job contract:
      energy_kwh     = kWhDelivered.
      earliest_start = connectionTime floored DOWN to the hour (UTC).
      deadline       = disconnectTime floored DOWN to the hour (UTC). Floored,
                       never ceiled: the car is gone at disconnectTime, so
                       rounding up would invent time the driver did not have.
      power_kw       = kWhDelivered / (doneChargingTime - connectionTime), the
                       average delivered power over the interval the car was
                       actually charging. ACN has no charging-rate field, so this
                       is derived. The charge interval is used rather than the
                       whole plugged-in window so idle time after the battery is
                       full does not dilute the rate. power_kw drives
                       duration = ceil(energy / power), so it sets run length,
                       not just cost.

    A session is dropped (not returned) when:
      - kWhDelivered is missing or <= 0 (no energy to schedule),
      - any of the three timestamps is missing or unparseable (no window or no
        power basis),
      - doneChargingTime <= connectionTime (charge interval non-positive, so
        power cannot be derived),
      - the derived power_kw is <= 0,
      - flooring leaves slack < 0, i.e. ceil(energy / power) hours do not fit in
        the [floor(connect), floor(disconnect)] window (the sub-hour-window case
        that flooring the deadline down can create).
    """
    jobs: list[Job] = []
    for s in sessions:
        energy = s.get("kWhDelivered")
        if energy is None or float(energy) <= 0.0:
            continue
        energy = float(energy)

        connect = _acn_time(s.get("connectionTime"))
        disconnect = _acn_time(s.get("disconnectTime"))
        done = _acn_time(s.get("doneChargingTime"))
        if connect is None or disconnect is None or done is None:
            continue

        charge_h = (done - connect).total_seconds() / 3600.0
        if charge_h <= 0.0:
            continue
        power = energy / charge_h
        if power <= 0.0:
            continue

        earliest_start = connect.floor("h")
        deadline = disconnect.floor("h")
        job_id = str(s.get("sessionID") or s.get("_id"))
        job = Job(job_id, energy, earliest_start, deadline, power)
        if slack_h(job) < 0:  # ceil(energy/power) hours do not fit the floored window
            continue
        jobs.append(job)
    return jobs
