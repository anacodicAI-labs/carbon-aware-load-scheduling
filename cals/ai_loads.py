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
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from cuad.scheduler.jobs import Job


PAI_TASK_COLUMNS = (
    "job_name",
    "task_name",
    "inst_num",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
    "plan_gpu",
    "gpu_type",
)


def load_pai_task_table(path: str | Path) -> pd.DataFrame:
    """Load the official headerless Alibaba GPU v2020 task table."""
    return pd.read_csv(path, header=None, names=PAI_TASK_COLUMNS)


def iter_pai_task_table(path: str | Path, *, chunksize: int) -> Iterator[pd.DataFrame]:
    """Read the official headerless task table in bounded row chunks."""
    return pd.read_csv(path, header=None, names=PAI_TASK_COLUMNS, chunksize=chunksize)


def parse_alibaba_trace(
    df: pd.DataFrame,
    *,
    origin: pd.Timestamp,
    gpu_col: str = "plan_gpu",
    gpu_unit: float = 100.0,
    id_col: str | None = None,
    completed_status: str | None = "Terminated",
) -> pd.DataFrame:
    """Normalize a raw Alibaba pai_job slice to [job_id, submit, duration_h, num_gpus].

    origin
        A tz-aware pd.Timestamp that trace second 0 maps to, so jobs land on the
        real carbon timeline (a re-dating choice, like the HVAC +365-day rule).
    Expects columns ``start_time`` / ``end_time`` (seconds) and a GPU column
    (``plan_gpu`` in hundredths of a GPU by default).

    When the official task-table ``status`` column is present, only completed
    tasks are retained by default. Failed, waiting, and still-running entries
    do not represent completed, schedulable work.
    """
    origin = pd.Timestamp(origin)
    if origin.tzinfo is None:
        origin = origin.tz_localize("UTC")
    if completed_status is not None and "status" in df.columns:
        df = df[df["status"] == completed_status]
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
    duplicate_ids = out["job_id"].duplicated(keep=False)
    if duplicate_ids.any():
        row_numbers = pd.Series(range(len(out)), index=out.index)
        out.loc[duplicate_ids, "job_id"] = (
            out.loc[duplicate_ids, "job_id"]
            + "::task-"
            + row_numbers.loc[duplicate_ids].astype(str)
        )
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


def aggregate_alibaba_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """Group parsed tasks with the same hourly start and whole-hour duration."""
    grouped = pd.DataFrame(
        {
            "earliest_start": pd.to_datetime(df["submit"]).dt.floor("h"),
            "duration_h": np.ceil(df["duration_h"]).astype(int),
            "num_gpus": pd.to_numeric(df["num_gpus"], errors="raise"),
        }
    )
    return (
        grouped.groupby(["earliest_start", "duration_h"], as_index=False)
        .agg(task_count=("num_gpus", "size"), total_gpus=("num_gpus", "sum"))
    )


def uncapped_alibaba_costs(
    groups: pd.DataFrame,
    carbon: pd.Series,
    *,
    gpu_power_kw: float,
    flex_hours: int,
) -> dict[str, float | int]:
    """Price aggregated AI tasks exactly as the uncapped FIFO and greedy scheduler do."""
    if carbon.index.has_duplicates:
        raise ValueError("carbon index must contain unique hourly timestamps")
    if carbon.isna().any():
        raise ValueError("carbon series must not contain missing values")

    groups = groups.reset_index(drop=True)
    values = carbon.to_numpy(dtype=float)
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    start_index = carbon.index.get_indexer(pd.to_datetime(groups["earliest_start"]))
    baseline_gco2 = scheduled_gco2 = 0.0
    comparable_tasks = unpriced_baseline_tasks = 0

    for duration_h, duration_groups in groups.groupby("duration_h", sort=False):
        duration = int(duration_h)
        costs = prefix[duration:] - prefix[:-duration]
        positions = start_index[duration_groups.index]
        task_counts = duration_groups["task_count"].to_numpy(dtype=int)
        powers = duration_groups["total_gpus"].to_numpy(dtype=float) * gpu_power_kw
        baseline_valid = (positions >= 0) & (positions < len(costs))
        unpriced_baseline_tasks += int(task_counts[~baseline_valid].sum())
        if not baseline_valid.any():
            continue

        valid_positions = positions[baseline_valid]
        valid_counts = task_counts[baseline_valid]
        valid_powers = powers[baseline_valid]
        baseline_costs = costs[valid_positions]
        best_costs = np.array(
            [costs[position : position + flex_hours + 1].min() for position in valid_positions]
        )
        comparable_tasks += int(valid_counts.sum())
        baseline_gco2 += float(np.dot(valid_powers, baseline_costs))
        scheduled_gco2 += float(np.dot(valid_powers, best_costs))

    return {
        "baseline_gco2": baseline_gco2,
        "scheduled_gco2": scheduled_gco2,
        "comparable_tasks": comparable_tasks,
        "unpriced_baseline_tasks": unpriced_baseline_tasks,
    }
