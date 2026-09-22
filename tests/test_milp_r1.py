"""Reviewer-1 MILP changes: sparse == dense objective; total-demand cap honoured; gap reported."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cuad.scheduler.greedy import schedule, schedule_optimal  # noqa: E402
from cuad.scheduler.jobs import Job, duration_h  # noqa: E402

T0 = pd.Timestamp("2019-01-01", tz="UTC")


def _instance(seed, n=60, hours=96, multi_hour=False):
    rng = np.random.default_rng(seed)
    ci = pd.Series(rng.uniform(100, 400, hours), index=pd.date_range(T0, periods=hours, freq="h"))
    jobs, base = [], {}
    for i in range(n):
        run = T0 + pd.Timedelta(hours=int(rng.integers(8, hours - 20)))
        p = float(rng.uniform(0.1, 2.0))
        dur = int(rng.integers(1, 4)) if multi_hour else 1
        flex = pd.Timedelta(hours=int(rng.integers(1, 7)))
        j = Job(f"j{i}", p * dur, run - flex, run + pd.Timedelta(hours=dur) + flex, p)
        jobs.append(j)
        base[j.job_id] = run
    return jobs, base, ci


def _hourly_total(jobs, assign, base=None):
    tot = {}
    for j in jobs:
        s = assign[j.job_id]
        for h in range(duration_h(j)):
            t = s + pd.Timedelta(hours=h)
            tot[t] = tot.get(t, 0.0) + j.power_kw
    return tot


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("multi", [False, True])
def test_sparse_matches_original_dense(seed, multi):
    orig_path = Path("/tmp/greedy_orig.py")
    if not orig_path.exists():
        pytest.skip("original greedy.py snapshot not available")
    spec = importlib.util.spec_from_file_location("greedy_orig", orig_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    jobs, base, ci = _instance(seed, multi_hour=multi)
    cap = 2.0
    new = schedule_optimal(jobs, ci, capacity_kw=cap, baseline_hours=base)
    old = mod.schedule_optimal(jobs, ci, capacity_kw=cap, baseline_hours=base)
    assert abs(new["total_gco2"] - old["total_gco2"]) <= 1e-4 * abs(old["total_gco2"])
    assert new["status"] == "0" and new["mip_gap"] <= new["mip_rel_gap_tol"] + 1e-12


@pytest.mark.parametrize("seed", range(4))
def test_total_capacity_is_respected(seed):
    jobs, base, ci = _instance(seed, n=80)
    peak = max(_hourly_total(jobs, base).values())
    o = schedule_optimal(jobs, ci, baseline_hours=base, total_capacity_kw=peak, mip_rel_gap=0.0)
    tot = _hourly_total(jobs, o["assignments"])
    assert max(tot.values()) <= peak + 1e-9
    unc = schedule_optimal(jobs, ci, baseline_hours=base)
    assert o["total_gco2"] >= unc["total_gco2"] - 1e-6   # a tighter feasible set cannot do better


def test_uncapped_milp_equals_greedy():
    jobs, base, ci = _instance(11, n=50)
    g = schedule(jobs, ci, baseline_hours=base)
    o = schedule_optimal(jobs, ci, baseline_hours=base, mip_rel_gap=0.0)
    assert abs(g["total_gco2"] - o["total_gco2"]) < 1e-6
