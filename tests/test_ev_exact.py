"""Tests for cals.ev_exact (Reviewer 1, Comment 3)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cals.ev_exact import (  # noqa: E402
    HourlyIntegral, audit_sessions, best_start_hours, classify_session, ev_exact_savings,
)

T0 = pd.Timestamp("2019-01-01", tz="UTC")


def _ci(values):
    return pd.Series(values, index=pd.date_range(T0, periods=len(values), freq="h"), dtype=float)


def _raw(connect_h, done_h, disconnect_h, kwh=10.0, sid="s"):
    f = lambda h: (T0 + pd.Timedelta(hours=h)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    return {"sessionID": sid, "connectionTime": f(connect_h), "doneChargingTime": f(done_h),
            "disconnectTime": f(disconnect_h), "kWhDelivered": kwh}


def test_known_optimum():
    integ = HourlyIntegral(_ci([10, 1, 10, 10]))
    s, cost = best_start_hours(0.5, 2.5, 0.5, integ)   # tau = 0.5 h, dwell [0.5, 3.0]
    assert 1.0 <= s <= 1.5 and abs(cost - 0.5) < 1e-12  # half an hour inside the clean hour


def test_matches_brute_force_on_random_instances():
    rng = np.random.default_rng(0)
    ci_vals = rng.uniform(100, 400, 200)
    integ = HourlyIntegral(_ci(ci_vals))
    step = 10 / 3600  # 10-second grid
    for _ in range(300):
        a = rng.uniform(0, 150)
        tau = rng.uniform(0.05, 8)
        b = a + rng.uniform(0, 20)
        s, exact = best_start_hours(a, b, tau, integ)
        grid = np.arange(a, b + 1e-12, step)
        grid = np.append(grid, b)
        brute = np.nanmin(integ.integral(grid, grid + tau))
        assert exact <= brute + 1e-9                                  # never worse than any grid point
        assert brute - exact <= (ci_vals.max() - ci_vals.min()) * step + 1e-9  # and grid converges to it
        assert a - 1e-12 <= s <= b + 1e-12                            # start is inside the real window


def test_hourly_adapter_rejects_a_feasible_session():
    # connect 09:10, done 12:40 (3.5 h), disconnect 12:50: feasible with 10 min real slack,
    # but floor window [09,12) = 3 h < ceil(3.5) = 4 h, so acn_to_jobs drops it.
    raw = _raw(9 + 10 / 60, 12 + 40 / 60, 12 + 50 / 60)
    reason, sess = classify_session(raw)
    assert reason == "valid" and abs(sess.slack_h - 10 / 60) < 1e-9
    _, audit = audit_sessions([raw])
    assert not bool(audit.loc[0, "hourly_adapter_kept"])


def test_invalid_records_are_classified():
    assert classify_session(_raw(1, 1, 2))[0] == "done_not_after_connect"
    assert classify_session(_raw(1, 3, 2))[0] == "done_after_disconnect"
    assert classify_session(_raw(1, 2, 3, kwh=0))[0] == "nonpositive_energy"


def test_savings_nonnegative_and_baseline_is_observed_charge():
    rng = np.random.default_rng(1)
    ci = _ci(rng.uniform(100, 400, 400))
    raws = []
    for i in range(200):
        c = rng.uniform(0, 300)
        ch = rng.uniform(0.2, 6)
        raws.append(_raw(c, c + ch, c + ch + rng.exponential(2), sid=str(i)))
    valid, _ = audit_sessions(raws)
    summary, per = ev_exact_savings(valid, ci)
    assert summary["savings_pct"] >= 0
    ok = per[per["priceable"]]
    assert (ok["optimal_gco2"] <= ok["baseline_gco2"] + 1e-9).all()
