# Carbon-Aware Load Scheduling — code

Reproducible pipeline for the ACN paper: **public fuel mix → hourly grid carbon
intensity → schedule flexible loads (AI compute, EV, HVAC) into clean hours →
CO₂ avoided.**

## Layout
```
 cuad/                VENDORED tested core (carbon, schedulers, adapters, EIA fetch)
                      — copied from anacodicAI-labs/energy-load-forecasting @ daa61f9.
                      This makes the repo self-contained; no external repo needed.
 cals/           thin helper layer over the vendored cuad
   __init__.py        imports the LOCAL ./cuad + re-exports the building blocks
   ai_loads.py        NEW: Alibaba GPU trace -> deferrable AI jobs
   forecast.py        NEW: climatology CI forecast + forecast-penalty
   acn_sessions.py    Caltech ACN-Data (EV) fetch client
 notebooks/           the paper, end to end (run these in order)
   01_carbon_intensity  fetch EIA fuel mix -> CI(t) -> heatmap
   02_loads             AI + EV + HVAC -> jobs
   03_schedule          FCFS vs greedy vs optimal -> per-load savings
   04_sweeps            slack / capacity / seasonal / factor-sensitivity figures
   05_forecast          climatology forecast + forecast penalty
 data/hvac/           REAL committed NREL ResStock MA heat-pump homes (run offline)
 data/{ev,carbon,ai}/ filled by the notebooks once keys are set
 figures/             saved figures
 .env.example         copy to .env and add keys
```

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env      # then paste EIA_API_KEY (get it from the team scc/.env.scc)
jupyter lab               # run notebooks 01 -> 05
```

## What runs offline vs needs a key
| Piece | Needs |
|---|---|
| HVAC load profiles (the parquets themselves) | **nothing** — real MA data is committed |
| Scheduling, sweeps, seasonal, OTH, forecast | `EIA_API_KEY` — every one of these prices load against the carbon signal |
| Carbon signal (real EIA fuel mix) | `EIA_API_KEY`, or a cached real pull. **There is no synthetic fallback**: `get_fuel_mix` raises rather than substitute a demo curve, because a fabricated signal produces plausible-looking savings that are indistinguishable from real ones |
| EV loads (Caltech ACN-Data) | `ACN_API_TOKEN` + `acnportal`, or a cached pull in `data/ev/`. That cache is **gitignored**, so a fresh clone does not have it |
| AI loads (Alibaba GPU v2020) | download `pai_task_table.csv` (~34 MB gzipped, 108 MB extracted) into `data/ai/`; see `cluster-trace-gpu-v2020/data/download_data.sh` in `alibaba/clusterdata` |

## What a clean clone can and cannot reproduce

`data/carbon/`, `data/ev/` and `data/ai/*.csv` are gitignored, so a fresh clone holds
only the HVAC parquets. What that means in practice:

| Result | Reproducible from a clone? |
|---|---|
| HVAC headline (12.70 / 8.29 / 7.28), seasonal, OTH sweep, forecast penalty | **yes**, with an `EIA_API_KEY` |
| `results/capacity_sweep.csv` (8.29 %, 1.01 pp) | **yes** — `python scripts/capacity_sweep.py`, ~15 min |
| `results/avg_vs_marginal_*.csv` | **yes** — `python scripts/avg_vs_marginal.py`, ~3 min |
| EV 1.87 %, 8,507 jobs | needs `ACN_API_TOKEN` or the gitignored cache |
| AI 3.97 %, 732,691 tasks | needs the Alibaba trace download |
| Three-building average-vs-marginal table (§4.9) | notebook output only; no committed CSV |
| The PDF itself | **no** — `Definitions/` (the MDPI class) is not in this repo |

Anything that prices load against carbon needs the EIA key, including the sweeps and the
forecast. The regression gate `python carbon_sim.py` also needs it.

## Note on the vendored core
The `cuad/` package (carbon intensity, schedulers, EV/HVAC adapters, EIA fetch)
is **vendored** — a copy of the tested code from
`anacodicAI-labs/energy-load-forecasting` (commit `daa61f9`). It lives in this
repo so the artifact is **fully self-contained and reproducible**: clone this
repo and it runs, with no dependency on any other repository. If you update the
upstream `cuad`, re-copy the same modules to keep them in sync.
