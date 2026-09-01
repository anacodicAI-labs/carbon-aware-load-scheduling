"""Shared scheduling configuration, helpers, and a fast regression self-test.

This module is the common backbone the notebooks and the ``scripts/`` import: it
defines the capacity-sweep grid (``CAP_SWEEP_K``, ``WORKING_CAP_K``), the HVAC
re-date offset (``MA_REDATE``), the median-active-load helper, and the
``evaluate`` driver that prices a schedule against the carbon signal.

Run directly (``python carbon_sim.py``) it executes ``check_anchors()``, a fast
regression gate that reproduces load-bearing reference numbers on a small
committed one-week sample, so a wiring change that silently moves a result is
caught before it reaches a figure. The full paper analysis (Massachusetts
heat-pump buildings over all of 2019) lives in the notebooks; this gate uses a
one-week sample purely so it runs in seconds.

The carbon-aware arm is optimal, not merely greedy: with no shared allocation
budget the jobs never interact, so each job's cheapest feasible slot is the
per-job optimum. The do-nothing baseline is workload-specific:
- EV : earliest_start is a real plug-in time, so "run at earliest_start" is the
       honest do-nothing (FIFO) placement.
- HVAC: the honest do-nothing is the hour the load ACTUALLY ran in the source
       data (not the earliest feasible hour, which would drift with the
       flexibility window and move the control arm).
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

from cuad.carbon.intensity import carbon_intensity
from cuad.config.settings import get_settings
from cuad.data.sources.eia import _respondent
from cuad.data.sources.eia_fuel_mix import _read_provenance, fetch_eia_fuel_mix
from cuad.scheduler.adapters import acn_to_jobs, nrel_to_jobs
from cuad.scheduler.greedy import (
    _window_cost,
    schedule,
    schedule_optimal,
    schedule_preemptible,
)
from cuad.scheduler.jobs import Job, duration_h, slack_h

NREL_WEEK = Path("data/hvac/bldg1_AL_week.parquet")
EV_WEEK = Path("data/ev/caltech_2019-07-14_2019-07-20.json")
FLEX_SWEEP = (0, 1, 2, 4, 6)
REDATE = pd.Timedelta(days=364)  # 52 weeks: preserves season and weekday

# Capacity cap definition: k x MEDIAN active load, swept.
#
# The cap is a SHIFTING limit (Option A), so what decides whether it binds is how
# many typical blocks fit under it -- NOT the building's service rating. Defining
# the cap as k x median makes that stacking headroom exactly k, which stays
# comparable across buildings whose peaks have different physical causes.
# Observed peak is NOT a portable cap: on the AL week the peak is a summer
# cooling hour on a flat profile and sits only 2.66x its own median (so it
# binds), whereas the MA heat-pump buildings peak in single January hours that
# are ~85-89% electric-resistance backup and sit 15.8-25.2x above their own
# median, where a cap at the peak is very loose. Peak is retained as the loosest
# swept cap -- NOT as a non-binding endpoint: even there the greedy-vs-optimal gap
# is 0.46 pp and 506 jobs still fall back, so the cap is doing work.
#
# Reconciliation with the existing AL numbers: the old cap (observed peak,
# 2.761 kW) is 2.66x the AL week's median active load, so the working cap sits in
# the same regime; 2.7x median (2.7999 kW) reproduces the flex-6 ISO-NE result
# exactly (7.220531%, saved 4866.4831 gCO2 -- identical to cap=peak).
CAP_SWEEP_K = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
WORKING_CAP_K = 3.0
# k=0.5 and k=1 are DEGENERATE, not data: cap = 1x median means half the blocks
# exceed the cap on their own (that is what a median is), so those cells measure
# forced fallback, not scheduling. Report k=2..6 only.
CAP_SWEEP_K_REPORTED = (2.0, 3.0, 4.0, 6.0)

# --- MA full-year heat-pump buildings (the in-scope workload) -------------------
# 486202 is the headline; 286081 / 274807 are sensitivity arms. All three are
# ResStock amy2018 (real 2018 weather), full year, US Eastern standard time.
MA_HEADLINE = Path("data/hvac/bldg486202_MA_year.parquet")
MA_SENSITIVITY = (
    Path("data/hvac/bldg286081_MA_year.parquet"),
    Path("data/hvac/bldg274807_MA_year.parquet"),
)
MA_UTC_OFFSET = -5  # US Eastern Standard, no DST (ResStock convention)

# 2018 weather -> 2019 carbon, full year. +365d, NOT the week's +364d.
# 2018 and 2019 are both non-leap (365 days), so +365d is an exact day-for-day
# bijection of the whole year: Jan 1 -> Jan 1, Dec 31 -> Dec 31, nothing dropped
# or duplicated. The week's +364d preserved WEEKDAY instead, which is the right
# trade for a 7-day window but breaks over a year: it maps 2018-01-01 onto
# 2018-12-31, leaving 2019-12-31 uncovered and spilling a day back into 2018.
# Justified empirically (see the re-date analysis): for both the carbon signal and
# the HVAC load, the seasonal/date component dwarfs the weekday/weekend component,
# so preserving the DATE is worth more than preserving the weekday.
MA_REDATE = pd.Timedelta(days=365)


def load_carbon(start: date, end: date, region: str = "ISO-NE") -> pd.Series:
    """Real EIA carbon intensity for a region. Raise and stop if it fell back to synthetic."""
    s = get_settings()
    raw = Path("data/carbon")
    mix = fetch_eia_fuel_mix(start, end, region=region, raw_dir=raw, api_key=s.eia_api_key)
    cache = raw / "eia" / f"fuelmix_{_respondent(region)}_{start}_{end}.csv"
    prov = _read_provenance(cache)
    if prov != "eia":
        raise SystemExit(
            f"REFUSING to report CO2: carbon cache {cache} provenance is {prov!r}, not real "
            "EIA. It fell back to synthetic; fix the fetch before quoting a number."
        )
    return carbon_intensity(mix)


def build_ev(path: Path) -> tuple[list[Job], dict[str, pd.Timestamp]]:
    """EV jobs; do-nothing baseline = each job's earliest_start (real plug-in time)."""
    jobs = acn_to_jobs(json.loads(path.read_text()))
    baseline = {j.job_id: j.earliest_start for j in jobs}
    return jobs, baseline


def build_hvac(
    nrel_path: Path,
    flex_hours: int,
    *,
    redate: pd.Timedelta = REDATE,
    utc_offset_hours: int = -6,
) -> tuple[list[Job], dict[str, pd.Timestamp]]:
    """HVAC jobs re-dated onto the carbon year; baseline = the observed run hour.

    Defaults reproduce the AL week exactly (+364d, US Central) so check_anchors()
    keeps gating the original regression numbers. The MA full-year buildings pass
    redate=MA_REDATE (+365d) and utc_offset_hours=MA_UTC_OFFSET (US Eastern);
    see MA_REDATE for why a full year needs a different rule than the week did.
    """
    df = pd.read_parquet(nrel_path).copy()
    df["timestamp"] = df["timestamp"] + redate
    return nrel_to_jobs(df, utc_offset_hours=utc_offset_hours, flex_hours=flex_hours)


def median_active_kw(jobs: list[Job]) -> float:
    """Median power of the emitted hourly blocks, i.e. the median ACTIVE load.

    nrel_to_jobs emits one block per hour whose HVAC energy clears min_kwh, and
    sets power_kw to that hour's kWh, so a block's power IS that hour's load and
    this is the median over active hours (idle hours are not blocks at all).
    """
    if not jobs:
        return 0.0
    return float(statistics.median(j.power_kw for j in jobs))


def cap_grid(jobs: list[Job]) -> list[tuple[str, float]]:
    """Cap sweep for a job pool: k x median active load, plus the peak endpoint.

    Returns (label, cap_kw) pairs in sweep order. The final entry is the observed
    peak. It is NOT non-binding: at the peak cap the greedy-vs-optimal gap is still
    0.46 pp and 506 jobs fall back, so it is reported as the loosest swept cap, not
    as an unconstrained endpoint (see CAP_SWEEP_K).
    """
    med = median_active_kw(jobs)
    grid = [(f"{k:g}x med", round(k * med, 6)) for k in CAP_SWEEP_K]
    grid.append(("peak", round(max(j.power_kw for j in jobs), 6)))
    return grid


def _classify_unplaced(job: Job, carbon: pd.Series) -> str:
    """Why a job could not be placed: 'infeasible_window' or 'nan_carbon'."""
    if slack_h(job) < 0:
        return "infeasible_window"
    dur = duration_h(job)
    last = job.deadline - pd.Timedelta(hours=dur)
    start = job.earliest_start
    any_covered = False
    while start <= last:
        hrs = [start + pd.Timedelta(hours=h) for h in range(dur)]
        if all(h in carbon.index for h in hrs):
            any_covered = True
            if not any(pd.isna(carbon.loc[h]) for h in hrs):
                return "placeable"
        start += pd.Timedelta(hours=1)
    return "nan_carbon" if any_covered else "infeasible_window"


def evaluate(
    jobs: list[Job],
    carbon: pd.Series,
    baseline_hours: dict[str, pd.Timestamp],
    *,
    capacity_kw: float | None = None,
    method: str = "floored",
    arms: tuple[str, ...] = (),
) -> dict:
    """Carbon-aware vs do-nothing baseline, compared over the shared placed set.

    method:
      "floored" (default) passes baseline_hours to schedule(), so the baseline
        floor and the fallback are active: a job the optimizer cannot place (or
        can only place at a cost above doing nothing) runs at its baseline hour
        at full carbon and STAYS in the accounting (it contributes 0 saving but
        keeps its baseline cost in the denominator). This is the current decided
        semantics and what the paper's numbers reproduce from.
      "exclude" is the legacy path: baseline_hours is NOT passed to schedule(),
        so an unplaceable job is dropped from the optimizer arm and therefore
        from the shared set. Kept behind this flag on purpose; not deleted.

    capacity_kw is an optional per-hour cap on cumulative SHIFTED power_kw
    (Option A: fallback load lands at baseline unconditionally and does not
    consume the cap). None means uncapped.

    arms names extra exact-MILP schedulers to run for the SAME jobs / cap /
    baseline and report next to greedy -- any of ("optimal", "preemptible").
    "optimal" is the greedy-vs-optimal gap (appendix D); "preemptible" the
    contiguity cost (appendix E). Both are the cap-exempt do-nothing MILPs, so
    their totals sit under the same baseline floor as greedy and are directly
    comparable. Each arm's saved is measured against the same shared-set baseline
    total as greedy.
    """
    if method == "floored":
        opt = schedule(jobs, carbon, capacity_kw=capacity_kw, baseline_hours=baseline_hours)
    elif method == "exclude":
        opt = schedule(jobs, carbon, capacity_kw=capacity_kw)
    else:
        raise ValueError(f"unknown method {method!r}; use 'floored' or 'exclude'")
    opt_a = opt["assignments"]
    by_id = {j.job_id: j for j in jobs}

    base_a: dict[str, pd.Timestamp] = {}
    for j in jobs:
        if slack_h(j) < 0:
            continue
        bh = baseline_hours[j.job_id]
        if _window_cost(j, carbon, bh, duration_h(j)) is not None:
            base_a[j.job_id] = bh

    shared = set(opt_a) & set(base_a)

    def cost(job: Job, start: pd.Timestamp) -> float:
        c = _window_cost(job, carbon, start, duration_h(job))
        return float(c) if c is not None else float("nan")

    rows = [
        (cost(by_id[jid], opt_a[jid]), cost(by_id[jid], base_a[jid]), slack_h(by_id[jid]))
        for jid in shared
    ]

    def agg(sub: list) -> tuple[float, float, float]:
        b = sum(r[1] for r in sub)
        o = sum(r[0] for r in sub)
        return b, o, b - o

    causes = Counter(_classify_unplaced(by_id[jid], carbon) for jid in opt["infeasible"])
    result = {
        "method": method,
        "capacity_kw": capacity_kw,
        "n_jobs": len(jobs),
        "placed_opt": len(opt_a),
        "placed_base": len(base_a),
        "shared": len(shared),
        "total_gco2": float(opt["total_gco2"]),
        "unplaced_infeasible": causes.get("infeasible_window", 0),
        "unplaced_nan": causes.get("nan_carbon", 0),
        "all": agg(rows),
        "movable": agg([r for r in rows if r[2] >= 1]),
        "n_movable": sum(1 for r in rows if r[2] >= 1),
        "n_pinned": sum(1 for r in rows if r[2] == 0),
        "pinned_delta": agg([r for r in rows if r[2] == 0])[2],
    }

    if arms:
        arm_fns = {"optimal": schedule_optimal, "preemptible": schedule_preemptible}
        b_all = result["all"][0]  # shared-set baseline total (same yardstick as greedy)
        arm_out: dict[str, dict] = {}
        for name in arms:
            if name not in arm_fns:
                raise ValueError(f"unknown arm {name!r}; use 'optimal' or 'preemptible'")
            r = arm_fns[name](
                jobs, carbon, capacity_kw=capacity_kw, baseline_hours=baseline_hours
            )
            saved = b_all - r["total_gco2"]
            arm_out[name] = {
                "total_gco2": r["total_gco2"],
                "saved": saved,
                "pct_all": saved / b_all * 100.0 if b_all else 0.0,
                "status": r.get("status"),
            }
        result["arms"] = arm_out
    return result


def _pct(delta: float, base: float) -> str:
    return f"{(delta / base * 100.0):5.2f}%" if base else " 0.00%"


def sweep_header() -> None:
    print(f"{'flex':>4} {'placed':>6} {'base_gCO2':>11} {'opt_gCO2':>11} {'saved':>9} "
          f"{'%all':>7} {'%movable':>9}")


def sweep_row(flex: int, e: dict) -> None:
    b, o, d = e["all"]
    bm, _, dm = e["movable"]
    print(f"{flex:>4} {e['shared']:>6} {b:>11.1f} {o:>11.1f} {d:>9.1f} "
          f"{_pct(d, b):>7} {_pct(dm, bm):>9}")


# Load-bearing numbers the paper reports. If wiring changes move any of these,
# STOP -- a silent drift here is a wrong reported number. Values are the
# regression-gated reference truth for the reported results.
ANCHOR_HVAC_UNCAPPED = {0: 0.00, 1: 1.85, 2: 3.39, 4: 5.93, 6: 8.67}
ANCHOR_HVAC_F6_CAPPED = {"ISO-NE": 7.22, "CAISO": 15.36}
ANCHOR_PEAK_KW = 2.761


def _pct_all(e: dict) -> float:
    b, _, d = e["all"]
    return d / b * 100.0 if b else 0.0


def check_anchors() -> None:
    """Regression gate: reproduce the pipeline's load-bearing reference numbers or STOP.

    Prints each anchor's computed value next to its expected value and raises
    SystemExit on the first mismatch, so a commit can be gated on `python
    carbon_sim.py` completing this block without exit.
    """
    wk = (date(2019, 7, 12), date(2019, 7, 23))
    isne = load_carbon(*wk, region="ISO-NE")
    caiso = load_carbon(*wk, region="CAISO")
    peak = round(max(j.power_kw for j in build_hvac(NREL_WEEK, 0)[0]), 6)

    print("=== ANCHOR CHECKS (gate; must pass before commit) ===")
    failures: list[str] = []

    ok_peak = abs(peak - ANCHOR_PEAK_KW) <= 0.001
    print(f"[{'PASS' if ok_peak else 'FAIL'}] HVAC observed peak: computed={peak:.3f} kW "
          f"expected={ANCHOR_PEAK_KW:.3f} kW")
    if not ok_peak:
        failures.append(f"peak {peak:.3f} != {ANCHOR_PEAK_KW}")

    # Anchors 1 + 2: flex-6 HVAC, cap = observed peak, floored, both grids.
    for grid, carbon in (("ISO-NE", isne), ("CAISO", caiso)):
        hv, hb = build_hvac(NREL_WEEK, 6)
        got = _pct_all(evaluate(hv, carbon, hb, capacity_kw=peak, method="floored"))
        exp = ANCHOR_HVAC_F6_CAPPED[grid]
        ok = abs(got - exp) <= 0.01
        print(f"[{'PASS' if ok else 'FAIL'}] {grid} HVAC flex6 cap=peak floored: "
              f"computed={got:.2f} expected={exp:.2f}")
        if not ok:
            failures.append(f"{grid} flex6 cap=peak {got:.2f} != {exp:.2f}")

    # Anchor 3: uncapped HVAC flex sweep, ISO-NE (floor-independent invariants).
    for flex, exp in ANCHOR_HVAC_UNCAPPED.items():
        hv, hb = build_hvac(NREL_WEEK, flex)
        got = _pct_all(evaluate(hv, isne, hb, capacity_kw=None, method="floored"))
        ok = abs(got - exp) <= 0.01
        print(f"[{'PASS' if ok else 'FAIL'}] ISO-NE HVAC uncapped flex{flex}: "
              f"computed={got:.2f} expected={exp:.2f}")
        if not ok:
            failures.append(f"ISO-NE uncapped flex{flex} {got:.2f} != {exp:.2f}")

    # Anchor 4: uncapped, exact optimal total == greedy total (greedy provably
    # optimal when nothing binds). Exercises the schedule_optimal call site.
    hv, hb = build_hvac(NREL_WEEK, 6)
    e = evaluate(hv, isne, hb, capacity_kw=None, method="floored", arms=("optimal",))
    diff = abs(e["total_gco2"] - e["arms"]["optimal"]["total_gco2"])
    ok = diff < 3e-11
    print(f"[{'PASS' if ok else 'FAIL'}] uncapped optimal==greedy total (flex6): "
          f"|diff|={diff:.2e} threshold=3e-11")
    if not ok:
        failures.append(f"optimal!=greedy uncapped |diff|={diff:.2e}")

    if failures:
        raise SystemExit("ANCHOR CHECK FAILED: " + "; ".join(failures)
                         + " -- refusing to proceed (wiring changed a paper number).")
    print("=== ALL ANCHORS PASS ===\n")


def main() -> None:
    # Gate first: refuse to print anything if a load-bearing number moved.
    check_anchors()

    # --- verification on the ORIGINAL 3-day window (check b: EV must be unchanged) ---
    print("=== VERIFICATION (3-day window, checks the fix did not move EV) ===")
    carbon3 = load_carbon(date(2019, 7, 13), date(2019, 7, 19))
    ev3, ev3_base = build_ev(Path("data/ev/caltech_2019-07-14_2019-07-16.json"))
    v = evaluate(ev3, carbon3, ev3_base)
    b, o, d = v["all"]
    print(f"EV 3-day: shared={v['shared']}  base={b:.1f}  opt={o:.1f}  saved={d:.1f}  "
          f"(expected 79 / 283840.1 / 278588.8 / 5251.3)")

    # --- WEEK headline ---
    carbon = load_carbon(date(2019, 7, 12), date(2019, 7, 23))
    print(f"\ncarbon (week): provenance=eia  hours={len(carbon)}  gCO2/kWh "
          f"min={carbon.min():.1f} median={carbon.median():.1f} max={carbon.max():.1f}  "
          f"NaN_hours={int(carbon.isna().sum())}  spread={carbon.max() - carbon.min():.1f}")

    # EV-only, same jobs, two carbon signals side by side. CAISO is the grid the
    # Caltech cars actually sit on; ISO-NE is kept for comparison, not replaced.
    ev, ev_base = build_ev(EV_WEEK)
    caiso = load_carbon(date(2019, 7, 12), date(2019, 7, 23), region="CAISO")

    print("\n=== EV-only: ISO-NE vs CAISO (ACN caltech, Jul 14-20 2019) ===")
    print(f"{'region':>7} {'cmin':>6} {'cmed':>6} {'cmax':>6} {'spread':>7} {'jobs':>5} "
          f"{'pinned':>6} {'movable':>7} {'saved':>9} {'%all':>7} {'%movable':>9}")
    for region, c in (("ISO-NE", carbon), ("CAISO", caiso)):
        e = evaluate(ev, c, ev_base)
        b, _, d = e["all"]
        bm, _, dm = e["movable"]
        print(f"{region:>7} {c.min():>6.1f} {c.median():>6.1f} {c.max():>6.1f} "
              f"{c.max() - c.min():>7.1f} {e['n_jobs']:>5} {e['n_pinned']:>6} {e['n_movable']:>7} "
              f"{d:>9.1f} {_pct(d, b):>7} {_pct(dm, bm):>9}")

    print("\n=== HVAC-only flex sweep (NREL re-dated Jul 14-20 2019) ===")
    sweep_header()
    for flex in FLEX_SWEEP:
        hvac, hvac_base = build_hvac(NREL_WEEK, flex)
        sweep_row(flex, evaluate(hvac, carbon, hvac_base))

    print("\n=== Combined (EV + HVAC) flex sweep ===")
    sweep_header()
    for flex in FLEX_SWEEP:
        hvac, hvac_base = build_hvac(NREL_WEEK, flex)
        base = dict(ev_base)
        base.update(hvac_base)
        sweep_row(flex, evaluate(ev + hvac, carbon, base))

    # Capacity sweep: cap = k x median active load, greedy arm, floored.
    # The cap is a shifting limit, so k IS the stacking headroom (see CAP_SWEEP_K).
    base_jobs = build_hvac(NREL_WEEK, 0)[0]
    med = median_active_kw(base_jobs)
    grid = cap_grid(base_jobs)
    print(f"\n=== Capacity sweep, HVAC ISO-NE, cap = k x median active load "
          f"(median={med:.3f} kW) ===")
    print("     " + " ".join(f"{lbl:>19}" for lbl, _ in grid))
    print("     " + " ".join(f"{cap:>18.3f}k" for _, cap in grid))
    for flex in FLEX_SWEEP:
        hvac, hvac_base = build_hvac(NREL_WEEK, flex)
        cells = []
        for _, cap in grid:
            e = evaluate(hvac, carbon, hvac_base, capacity_kw=cap, method="floored")
            cells.append(f"{_pct_all(e):>18.2f}%")
        print(f"f={flex:<3} " + " ".join(cells))

    # Appendix D: greedy vs exact-optimal gap at the WORKING cap, floored both
    # arms. Gap concentrates where the cap binds.
    work_cap = round(WORKING_CAP_K * med, 6)
    print(f"\n=== Greedy vs optimal, HVAC ISO-NE, working cap = {WORKING_CAP_K:g}x median "
          f"= {work_cap} kW (appendix D) ===")
    print(f"{'flex':>4} {'greedy%':>8} {'optimal%':>9} {'gap_pp':>7}")
    for flex in FLEX_SWEEP:
        hvac, hvac_base = build_hvac(NREL_WEEK, flex)
        e = evaluate(hvac, carbon, hvac_base, capacity_kw=work_cap, method="floored",
                     arms=("optimal",))
        gr = _pct_all(e)
        op = e["arms"]["optimal"]["pct_all"]
        print(f"{flex:>4} {gr:>7.2f}% {op:>8.2f}% {op - gr:>+6.3f}")

    # Appendix E: preemptible vs non-preemptible OPTIMAL, EV only, both grids.
    print("\n=== Preemptible vs non-preemptible optimal, EV only (appendix E) ===")
    print(f"{'grid':>7} {'cap':>9} {'np%':>6} {'pre%':>6} {'benefit%':>9}")
    for region, c in (("ISO-NE", carbon), ("CAISO", caiso)):
        for cname, cap in (("uncapped", None), ("150kW", 150.0), ("9.6kW", 9.6)):
            e = evaluate(ev, c, ev_base, capacity_kw=cap, method="floored",
                         arms=("optimal", "preemptible"))
            npd = e["arms"]["optimal"]["saved"]
            prd = e["arms"]["preemptible"]["saved"]
            npp = e["arms"]["optimal"]["pct_all"]
            prp = e["arms"]["preemptible"]["pct_all"]
            benefit = (prd - npd) / npd * 100.0 if npd > 1e-9 else 0.0
            print(f"{region:>7} {cname:>9} {npp:>5.2f}% {prp:>5.2f}% {benefit:>+8.2f}%")


if __name__ == "__main__":
    main()
