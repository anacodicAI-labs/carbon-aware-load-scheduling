"""Caltech ACN-Data EV charging sessions from the live ACN-Data API.

Endpoint: https://ev.caltech.edu/api/v1/  (acnportal.acndata.DataClient)
Free API token: https://ev.caltech.edu/  (register); read from the ACN_API_TOKEN
setting (cuad.config.settings.Settings.acn_api_token).

This is the fetch path that was missing: `carbon_sim.build_ev` only ever read a
hand-downloaded JSON file, and the ACN token setting was declared but never used.
This module pulls sessions for a site and window, caches them, and records a
provenance sidecar so a cached file's origin is auditable at rest.

Only the fields the scheduler needs are kept (sessionID, connectionTime,
doneChargingTime, disconnectTime, kWhDelivered) and timeseries=False is used --
the per-session /ts charging-rate stream is slow and unused. Timestamps are
stored as RFC-1123 GMT strings (e.g. "Sun, 14 Jul 2019 00:06:26 GMT"), matching
both the raw ACN-Data API format and the existing hand-downloaded caches, so
scheduler.adapters.acn_to_jobs (which parses those strings) is unchanged.

There is deliberately NO synthetic fallback: a missing token raises. Unlike
grid carbon, an invented EV session has no honest default.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# acnportal is imported LAZILY inside fetch_acn_sessions, not at module scope.
# A module-level import made `import cals.acn_sessions` raise ModuleNotFoundError
# without acnportal installed, so callers fell back to synthetic sessions even
# when a real cached pull was sitting on disk. Only the live-fetch path needs the
# library; serving a cache must not.

_ACN_URL = "https://ev.caltech.edu/api/v1/"

# The scheduler-relevant fields; acn_to_jobs reads exactly these.
_FIELDS = ("sessionID", "connectionTime", "doneChargingTime", "disconnectTime", "kWhDelivered")


def _cache_path(cache_dir: Path, site: str, start: date, end: date) -> Path:
    """Cache file for one (site, window), mirroring the existing acn cache names."""
    return cache_dir / f"{site}_{start}_{end}.json"


def _meta_path(cache: Path) -> Path:
    """Sidecar recording where a cached session file came from."""
    return cache.with_suffix(cache.suffix + ".meta.json")


def _write_provenance(cache: Path, *, site: str, start: date, end: date, count: int) -> None:
    """Stamp the cache's provenance (source URL, site, window, count, fetch time)."""
    meta = {
        "source": "acn-data",
        "url": _ACN_URL,
        "endpoint": f"{_ACN_URL}sessions/{site}",
        "site": site,
        "start": str(start),
        "end": str(end),
        "session_count": count,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _meta_path(cache).write_text(json.dumps(meta, indent=2))


def read_provenance(cache: Path) -> dict | None:
    """Recorded provenance for a cached session file, or None when no sidecar exists."""
    meta = _meta_path(cache)
    if not meta.is_file():
        return None
    try:
        return json.loads(meta.read_text())
    except (ValueError, OSError):
        return None


def _row(session: dict) -> dict:
    """Keep only the scheduler fields; store datetimes as RFC-1123 GMT strings.

    DataClient.get_sessions parses connection/disconnect/doneCharging into
    datetime objects (via parse_dates). We serialise them back with http_date so
    the cache matches the raw API / hand-downloaded string format that
    acn_to_jobs already parses.
    """
    out: dict = {}
    for field in _FIELDS:
        value = session.get(field)
        if isinstance(value, datetime):
            from acnportal.acndata.utils import http_date  # lazy
            value = http_date(value)
        out[field] = value
    return out


def _token(api_token: str | None) -> str:
    """The ACN token, from the argument or the ACN_API_TOKEN setting. Raise if absent."""
    token = api_token if api_token is not None else os.environ.get("ACN_API_TOKEN", "")
    if not token:
        raise RuntimeError(
            "ACN-Data API token is not set. Provide api_token= or set ACN_API_TOKEN "
            "in the environment. There is no synthetic fallback for EV sessions."
        )
    return token


def _window(start: date, end: date) -> tuple[datetime, datetime]:
    """[start 00:00, end 23:59:59] UTC. end is inclusive of its whole day, mirroring
    the EIA fetch convention, so fetch(..., 2019-07-14, 2019-07-20) captures every
    session that connected on Jul 20."""
    start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    end_dt = datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(
        hours=23, minutes=59, seconds=59
    )
    return start_dt, end_dt


def count_acn_sessions(
    start: date,
    end: date,
    *,
    site: str = "caltech",
    min_energy: float | None = None,
    api_token: str | None = None,
) -> int:
    """Number of sessions the API holds for (site, window) WITHOUT downloading them.

    Uses get_sessions_by_time(count=True) so a full-year footprint can be reported
    before committing to a download.
    """
    from acnportal.acndata import DataClient  # lazy: cache path needs no acnportal
    client = DataClient(api_token=_token(api_token), url=_ACN_URL)
    start_dt, end_dt = _window(start, end)
    return int(
        client.get_sessions_by_time(
            site, start_dt, end_dt, min_energy=min_energy, timeseries=False, count=True
        )
    )


def fetch_acn_sessions(
    start: date,
    end: date,
    *,
    site: str = "caltech",
    min_energy: float | None = None,
    cache_dir: Path,
    api_token: str | None = None,
    refresh: bool = False,
) -> list[dict]:
    """Fetch Caltech ACN-Data EV sessions for a window, with a local cache.

    Parameters
    ----------
    start, end:
        Inclusive date range on connectionTime (UTC); end covers its whole day.
    site:
        ACN-Data site id ("caltech", "jpl", "office001").
    min_energy:
        Optional kWhDelivered floor passed to the API (None keeps all sessions).
    cache_dir:
        Directory for the JSON cache and its provenance sidecar.
    api_token:
        Override for the ACN token; defaults to the ACN_API_TOKEN setting.
    refresh:
        When True, ignore any cache hit and re-fetch.

    Returns
    -------
    A list of session dicts with keys sessionID, connectionTime, doneChargingTime,
    disconnectTime (RFC-1123 GMT strings), and kWhDelivered (float) -- the shape
    scheduler.adapters.acn_to_jobs consumes.
    """
    cache_dir = Path(cache_dir)
    cache = _cache_path(cache_dir, site, start, end)
    if cache.is_file() and not refresh:
        return json.loads(cache.read_text())

    from acnportal.acndata import DataClient  # lazy: cache path needs no acnportal
    client = DataClient(api_token=_token(api_token), url=_ACN_URL)
    start_dt, end_dt = _window(start, end)
    # get_sessions_by_time returns a generator that pages via _links.next.
    sessions = client.get_sessions_by_time(
        site, start_dt, end_dt, min_energy=min_energy, timeseries=False
    )
    rows = [_row(s) for s in sessions]

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, indent=2))
    _write_provenance(cache, site=site, start=start, end=end, count=len(rows))
    return rows
