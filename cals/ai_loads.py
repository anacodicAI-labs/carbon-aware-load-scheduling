"""Alibaba Cluster Trace GPU v2020 -> deferrable AI-compute Jobs.

Trace: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020
Each job runs on some GPUs for a duration. Batch / training jobs are deferrable
(they must finish, but not at a specific hour) — the third load type in the
paper, alongside EV charging and HVAC.

Mapping to the Job contract (same shape as the EV/HVAC adapters):
    power_kw       = num_gpus * gpu_power_kw           (draw rate while running)
    energy_kwh     = power_kw * duration_h
    earliest_start = submit time, floored to the hour
    deadline       = earliest_start + duration + flex_hours

TWO ASSUMPTIONS THAT MUST BE SWEPT IN THE PAPER (documented, not hidden):
    gpu_power_kw   per-GPU power (A100/V100 ~ 0.3-0.4 kW). Energy scales linearly,
                   so report a sensitivity sweep (like the OTH emission-factor sweep).
    flex_hours     how deferrable a batch job is; it sets the slack. The trace has
                   NO real deadline, so this is an assigned knob (like HVAC flex).
"""
from __future__ import annotations

import math

import pandas as pd

from cuad.scheduler.jobs import Job


def parse_alibaba_trace(
    df: pd.DataFrame,
    *,
    origin: pd.Timestamp,
    gpu_col: str = "plan_gpu",
    gpu_unit: float = 100.0,
    id_col: str | None = None,
) -> pd.DataFrame:
    """Normalize a raw Alibaba pai_job slice to [job_id, submit, duration_h, num_gpus].

    origin
        A tz-aware pd.Timestamp that trace second 0 maps to, so jobs land on the
        real carbon timeline (a re-dating choice, like the HVAC +365-day rule).
    Expects columns ``start_time`` / ``end_time`` (seconds) and a GPU column
    (``plan_gpu`` in hundredths of a GPU by default).
    """
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None:
        origin = origin.tz_localize("UTC")
    idc = id_col or next((c for c in ["job_name", "inst_id", "task_name"] if c in df.columns), df.columns[0])

    start_s = pd.to_numeric(df["start_time"], errors="coerce")
    end_s = pd.to_numeric(df["end_time"], errors="coerce")
    out = pd.DataFrame(
        {
            "job_id": df[idc].astype(str),
            "submit": origin + pd.to_timedelta(start_s, unit="s"),
            "duration_h": (end_s - start_s) / 3600.0,
            "num_gpus": pd.to_numeric(df[gpu_col], errors="coerce") / gpu_unit,
        }
    )
    out = out.dropna(subset=["submit", "duration_h", "num_gpus"])
    out = out[(out["duration_h"] > 0) & (out["num_gpus"] > 0)]
    return out.reset_index(drop=True)


def alibaba_to_jobs(
    df: pd.DataFrame,
    *,
    gpu_power_kw: float = 0.4,
    flex_hours: int = 6,
    min_kwh: float = 0.1,
) -> list[Job]:
    """Turn parsed Alibaba jobs (output of ``parse_alibaba_trace``) into Jobs.

    Returns a list of ``Job`` with the same contract as the EV/HVAC adapters, so
    it drops straight into ``schedule`` / ``schedule_optimal`` / ``fifo_baseline``.
    """
    one_h = pd.Timedelta(hours=1)
    jobs: list[Job] = []
    for _, r in df.iterrows():
        power = float(r["num_gpus"]) * float(gpu_power_kw)
        if power <= 0:
            continue
        energy = power * float(r["duration_h"])
        if energy <= min_kwh:
            continue
        r_start = pd.Timestamp(r["submit"]).floor("h")
        dur = max(1, math.ceil(energy / power))
        deadline = r_start + dur * one_h + pd.Timedelta(hours=flex_hours)
        jobs.append(Job(str(r["job_id"]), energy, r_start, deadline, power))
    return jobs
