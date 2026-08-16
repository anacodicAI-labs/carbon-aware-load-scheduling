"""cals — helper layer for the Carbon-Aware Load Scheduling paper.

SELF-CONTAINED: the tested carbon + scheduler core is VENDORED in this repo at
``../cuad`` (copied from the CUAD-STLF / energy-load-forecasting repo; see
README for the source commit). This repo has NO dependency on any other repo.

This layer adds the two pieces unique to this paper:
  - ai_loads.alibaba_to_jobs : Alibaba GPU trace -> deferrable AI compute jobs
  - forecast                 : climatology CI forecast + forecast-penalty eval

Notebooks do:  `from cals import carbon_intensity, schedule, alibaba_to_jobs, ...`
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the VENDORED local `cuad` package importable regardless of cwd, and put it
# first so the local copy always wins even if another cuad is on the path.
_repo_code = Path(__file__).resolve().parents[1]  # .../carbon-aware-load-scheduling/code
if str(_repo_code) not in sys.path:
    sys.path.insert(0, str(_repo_code))

# --- tested core, now imported from the LOCAL vendored cuad -------------------
from cuad.carbon.factors import FACTORS  # noqa: E402,F401
from cuad.carbon.intensity import carbon_intensity  # noqa: E402,F401
from cuad.scheduler.jobs import Job, duration_h, slack_h  # noqa: E402,F401
from cuad.scheduler.greedy import (  # noqa: E402,F401
    schedule,
    schedule_optimal,
    schedule_preemptible,
    fifo_baseline,
)
from cuad.scheduler.adapters import nrel_to_jobs, acn_to_jobs  # noqa: E402,F401
from cuad.data.sources.eia_fuel_mix import fetch_eia_fuel_mix  # noqa: E402,F401

import cuad  # noqa: E402

CUAD_VENDORED = str(Path(cuad.__file__).resolve().parent)  # where the local cuad lives

# --- this paper's own additions ----------------------------------------------
from .ai_loads import (  # noqa: E402,F401
    aggregate_alibaba_tasks,
    alibaba_to_jobs,
    iter_pai_task_table,
    load_pai_task_table,
    parse_alibaba_trace,
    uncapped_alibaba_costs,
)
from .forecast import climatology_forecast, forecast_penalty  # noqa: E402,F401
