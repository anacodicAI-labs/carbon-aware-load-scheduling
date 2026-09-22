"""Exact continuous-time EV charging evaluation (Reviewer 1, Round 1, Comment 3).

Why this exists
---------------
``cuad.scheduler.adapters.acn_to_jobs`` puts every session on the hourly grid:
earliest_start = floor(connect), deadline = floor(disconnect), duration =
ceil(charge hours). Flooring ``connect`` lets a car charge *before it arrived*,
and rounding the duration up can reject sessions that are physically feasible
(e.g. connect 09:10, done 12:40, disconnect 12:50: 3.5 h of charging inside a
3.67 h dwell, but the floored window [09, 12) holds only 3 h < ceil(3.5) = 4 h).

This module removes the grid from the *load* side while keeping the hourly
carbon signal. Each session charges one contiguous block of length
tau = done - connect at constant power P = E / tau (the paper's non-preemptive
convention, Section 3.3), starting at any real time s in [connect, disconnect - tau].

Exactness
---------
CI(t) is piecewise constant on hours, so C(s) = P * integral_s^{s+tau} CI(t) dt
is continuous and piecewise linear in s, with slope P * (CI(s+tau) - CI(s)).
The slope changes only where s or s + tau crosses an hour boundary, so the
minimum over the closed interval [a, b] is attained at a, at b, at an integer
hour k in (a, b), or at k - tau for an integer k with k - tau in (a, b).
Evaluating that finite candidate set is therefore the exact continuous-time
optimum -- no discretisation, no rounding of energy or of availability.
Uncapped, sessions do not interact, so per-session minima are the global optimum.

Baseline = the observed charge, [connect, done] at P (what the car actually did).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

VALID = "valid"
EXCLUSION_REASONS = (
    "missing_energy",           # kWhDelivered absent
    "nonpositive_energy",       # kWhDelivered <= 0
    "missing_timestamp",        # connect / done / disconnect absent or unparseable
    "done_not_after_connect",   # charge interval <= 0: power undefined
    "done_after_disconnect",    # charging ends after unplug: record inconsistent
)


@dataclass(frozen=True)
class EVSession:
    session_id: str
    connect: pd.Timestamp     # tz-aware UTC
    done: pd.Timestamp
    disconnect: pd.Timestamp
    energy_kwh: float

    @property
    def charge_h(self) -> float:
        return (self.done - self.connect).total_seconds() / 3600.0

    @property
    def dwell_h(self) -> float:
        return (self.disconnect - self.connect).total_seconds() / 3600.0

    @property
    def slack_h(self) -> float:
        """Real (continuous) slack: time plugged in but not needed for charging."""
        return (self.disconnect - self.done).total_seconds() / 3600.0

    @property
    def power_kw(self) -> float:
        return self.energy_kwh / self.charge_h


def _ts(value: object) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    ts = pd.to_datetime(str(value), utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


def classify_session(s: dict) -> tuple[str, EVSession | None]:
    """Return (reason, session). reason is VALID or one of EXCLUSION_REASONS."""
    sid = str(s.get("sessionID") or s.get("_id"))
    e = s.get("kWhDelivered")
    if e is None:
        return "missing_energy", None
    try:
        e = float(e)
    except (TypeError, ValueError):
        return "missing_energy", None
    if not e > 0.0:
        return "nonpositive_energy", None
    c, d, x = _ts(s.get("connectionTime")), _ts(s.get("doneChargingTime")), _ts(s.get("disconnectTime"))
    if c is None or d is None or x is None:
        return "missing_timestamp", None
    if d <= c:
        return "done_not_after_connect", None
    if d > x:
        return "done_after_disconnect", None
    return VALID, EVSession(sid, c, d, x, e)


def audit_sessions(sessions: list[dict]) -> tuple[list[EVSession], pd.DataFrame]:
    """Classify every raw session and flag those the HOURLY adapter would drop.

    The audit answers Reviewer 1's request to separate invalid records from
    exclusions caused by discretisation: ``hourly_adapter_kept`` re-runs the
    paper's own ``acn_to_jobs`` on each session, so a row with reason == "valid"
    and hourly_adapter_kept == False is a physically feasible session that the
    hour grid rejected.
    """
    from cuad.scheduler.adapters import acn_to_jobs  # the paper's current adapter
    from cuad.scheduler.jobs import slack_h

    valid: list[EVSession] = []
    rows = []
    for s in sessions:
        reason, sess = classify_session(s)
        kept = acn_to_jobs([s])
        rows.append({
            "session_id": str(s.get("sessionID") or s.get("_id")),
            "reason": reason,
            "hourly_adapter_kept": bool(kept),
            "hourly_slack_h": slack_h(kept[0]) if kept else np.nan,
            "real_slack_h": sess.slack_h if sess else np.nan,
            "charge_h": sess.charge_h if sess else np.nan,
            "dwell_h": sess.dwell_h if sess else np.nan,
        })
        if sess is not None:
            valid.append(sess)
    return valid, pd.DataFrame(rows)


class HourlyIntegral:
    """Exact integral of an hourly, piecewise-constant carbon signal."""

    def __init__(self, ci: pd.Series):
        idx = pd.DatetimeIndex(ci.index)
        if idx.tz is None:
            raise ValueError("carbon index must be tz-aware (UTC)")
        if len(idx) > 1 and not bool(((idx[1:] - idx[:-1]) == pd.Timedelta(hours=1)).all()):
            raise ValueError("carbon index must be a gap-free, sorted hourly grid")
        self.t0 = idx[0]
        v = ci.to_numpy(dtype=float)
        self.n = len(v)
        nan = np.isnan(v)
        self.v = np.where(nan, 0.0, v)
        self.F = np.concatenate([[0.0], np.cumsum(self.v)])      # F[k] = integral over [0, k)
        self.nan_cum = np.concatenate([[0], np.cumsum(nan)])     # NaN hours in [0, k)

    def hours(self, ts: pd.Timestamp) -> float:
        return (ts - self.t0).total_seconds() / 3600.0

    def _F(self, x: np.ndarray) -> np.ndarray:
        k = np.clip(np.floor(x).astype(int), 0, self.n - 1)
        return self.F[k] + (x - k) * self.v[k]

    def integral(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """integral_a^b CI(t) dt in (gCO2/kWh)*h; NaN if outside the series or touching a NaN hour."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        out = np.full(a.shape, np.nan)
        ok = (a >= 0.0) & (b <= self.n) & (b > a)
        if ok.any():
            ka = np.floor(a[ok]).astype(int)
            kb = np.ceil(b[ok]).astype(int)
            clean = (self.nan_cum[kb] - self.nan_cum[ka]) == 0
            val = self._F(b[ok]) - self._F(a[ok])
            out[np.flatnonzero(ok)] = np.where(clean, val, np.nan)
        return out


def best_start_hours(a: float, b: float, tau: float, integ: HourlyIntegral) -> tuple[float, float]:
    """Exact argmin/min of integral_s^{s+tau} CI over s in [a, b] (all in hours since t0)."""
    if b < a:
        raise ValueError("latest start precedes earliest start")
    cand = [a, b]
    cand.extend(range(math.ceil(a), math.floor(b) + 1))
    cand.extend(k - tau for k in range(math.ceil(a + tau), math.floor(b + tau) + 1))
    c = np.unique(np.asarray(cand, dtype=float))
    c = c[(c >= a) & (c <= b)]
    cost = integ.integral(c, c + tau)
    if np.all(np.isnan(cost)):
        return float("nan"), float("nan")
    i = int(np.nanargmin(cost))
    return float(c[i]), float(cost[i])


def ev_exact_savings(sessions: list[EVSession], ci: pd.Series) -> tuple[dict, pd.DataFrame]:
    """Uncapped exact continuous-time saving over the shared priceable set.

    Returns (summary, per_session). Costs are gCO2 (kW * h * gCO2/kWh).
    """
    integ = HourlyIntegral(ci)
    rows = []
    for s in sessions:
        a = integ.hours(s.connect)
        tau = s.charge_h
        b = integ.hours(s.disconnect) - tau
        base = float(integ.integral(np.array([a]), np.array([a + tau]))[0])
        s_opt, opt = best_start_hours(a, b, tau, integ)
        priceable = not (math.isnan(base) or math.isnan(opt))
        rows.append({
            "session_id": s.session_id,
            "priceable": priceable,
            "power_kw": s.power_kw,
            "energy_kwh": s.energy_kwh,
            "real_slack_h": s.slack_h,
            "shift_h": (s_opt - a) if priceable else np.nan,
            "baseline_gco2": s.power_kw * base if priceable else np.nan,
            "optimal_gco2": s.power_kw * opt if priceable else np.nan,
        })
    df = pd.DataFrame(rows)
    ok = df[df["priceable"]]
    b, o = ok["baseline_gco2"].sum(), ok["optimal_gco2"].sum()
    summary = {
        "n_sessions_in": len(sessions),
        "n_priceable": int(len(ok)),
        "baseline_gco2": float(b),
        "optimal_gco2": float(o),
        "savings_pct": float(100.0 * (b - o) / b) if b else 0.0,
        "median_real_slack_h": float(ok["real_slack_h"].median()) if len(ok) else float("nan"),
        "share_slack_lt_1min": float((ok["real_slack_h"] < 1 / 60).mean()) if len(ok) else float("nan"),
        "share_slack_lt_15min": float((ok["real_slack_h"] < 0.25).mean()) if len(ok) else float("nan"),
        "share_shifted": float((ok["shift_h"] > 1e-9).mean()) if len(ok) else float("nan"),
    }
    return summary, df
