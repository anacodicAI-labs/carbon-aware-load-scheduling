"""Job data contract for the carbon-aware scheduler."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Job:
    job_id: str
    energy_kwh: float
    earliest_start: pd.Timestamp  # UTC
    deadline: pd.Timestamp        # UTC
    power_kw: float


def duration_h(job: Job) -> int:
    """Whole hours the job must run at power_kw to deliver energy_kwh."""
    return math.ceil(job.energy_kwh / job.power_kw)


def slack_h(job: Job) -> int:
    """Spare hours in [earliest_start, deadline] beyond duration; < 0 is infeasible."""
    window = int((job.deadline - job.earliest_start) / pd.Timedelta(hours=1))
    return window - duration_h(job)


def generate_synthetic_jobs(n: int, seed: int = 42) -> list[Job]:
    """Demo deferrable jobs with varied energy, power, and deadline windows."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01T00:00", tz="UTC")
    jobs: list[Job] = []
    for i in range(n):
        power = float(rng.choice([3.3, 7.2, 11.0, 22.0]))  # kW, typical EV chargers
        energy = float(rng.uniform(power, power * 8))       # 1 to 8 hours of runtime
        duration = math.ceil(energy / power)
        earliest = start + pd.Timedelta(hours=int(rng.integers(0, 12)))
        window = int(rng.integers(duration + 1, 24))        # always leaves >= 1h slack
        deadline = earliest + pd.Timedelta(hours=window)
        jobs.append(Job(f"job_{i:03d}", energy, earliest, deadline, power))
    return jobs
