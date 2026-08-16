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
| HVAC loads + scheduling + sweeps + forecast | **nothing** — real committed MA data |
| Carbon signal (real EIA fuel mix) | `EIA_API_KEY` (notebooks fall back to a labelled DEMO curve without it) |
| EV loads (Caltech ACN-Data) | network |
| AI loads (Alibaba GPU v2020) | download the trace (link in nb 02) |

## Note on the vendored core
The `cuad/` package (carbon intensity, schedulers, EV/HVAC adapters, EIA fetch)
is **vendored** — a copy of the tested code from
`anacodicAI-labs/energy-load-forecasting` (commit `daa61f9`). It lives in this
repo so the artifact is **fully self-contained and reproducible**: clone this
repo and it runs, with no dependency on any other repository. If you update the
upstream `cuad`, re-copy the same modules to keep them in sync.
