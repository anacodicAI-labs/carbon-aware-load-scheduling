# R1 TODO fill report

Target manuscript: `paper.tex` (repo root). Session date 2026-09-21. Nothing committed.

Non-comment `\TODO{` count: **86 at start, 0 at end**.

## Setup

- `/tmp/greedy_orig.py` taken from `git show HEAD:cuad/scheduler/greedy.py` before patching.
- `git apply "chat files/reviewer1_code.patch"` (no `--index`): clean.
- `python -m pytest -q -rs tests`: 32 passed, 0 skipped. All 12 `test_sparse_matches_original_dense` cases ran and passed.
  Re-run after the marginal-iso edit: 32 passed, 0 skipped.
- Inputs: the EIA fuel mix was cached at `data/carbon/eia/fuelmix_ISNE_2019-01-01_2020-01-02.csv` (there is no `.env`, so no EIA key). The ACN 2019 cache and `data/ai/pai_task_table.csv` were present.
- Python 3.12.4 (system interpreter, no venv), SciPy 1.18.0, pandas 2.3.3.
- Code edits in `scripts/r1_reruns.py`:
  1. `cmd_hvac`: added `("M=observed peak", peak_total, None)` right after the k-multiple list.
  2. `cmd_marginal_iso` (on request): the loop now calls `schedule(...)` (capped greedy) instead of `schedule_optimal(...)` and prints each result with `flush=True`.
  3. `cmd_marginal_iso` (on request): one shared job set whose metered hour and whole window have ISO values; `n_dropped` is recorded in the CSV.
- `bibs.bib`: appended the 5 entries from `chat files/bibs_R1_additions.bib`. None of the keys were already present, and no existing lines changed. Note: `bibs.bib` is gitignored, so this change does not appear in `git diff`.
- Text edits in `paper.tex` beyond TODO tokens (all requested, all inside the existing `\BRev{}`):
  - ISO table caption: appended the sentence about the 10 excluded boundary jobs (8{,}353 kept).
  - ISO table caption: "exact MILP" was replaced with "the same capped greedy scheduler as Table~\ref{tab:avgmarginal}".
  - Sec 4.9: "which is the most that perfect knowledge ... would allow" was replaced with the requested sentence.
  - An automated word-diff against `chat files/main_R1_revised.tex` confirmed that the first two edits were the only non-TODO changes before Section F was filled. The one other difference is the missing trailing newline that `paper.tex` already had.

## Runtimes

| Run | Wall time |
|---|---|
| ev | 18 s |
| forecast | 438 s |
| batch | 619 s |
| hvac (`--milp-time-limit 1800`) | 1,267 s. Every solve converged well under the limit; the slowest was M=4x at 539 s |
| marginal-iso, MILP version | killed after about 1 h 30 min on request, with no output |
| marginal-iso, greedy version (baseline-hour filter, NaN) | 15 s |
| marginal-iso, greedy version, shared job set (final) | about 15 s |
| batch `--milp-time-limit 3600` (rule 11) | not run, as instructed |

## Sanity checks

| Check | Expected | Observed | Tol. | Result |
|---|---|---|---|---|
| hvac M=3x optimal_pct | 8.29 | 8.2923 | 0.02 | PASS |
| hvac M=3x greedy_pct | 7.28 | 7.2799 | 0.02 | PASS |
| hvac uncapped optimal_pct | 12.70 | 12.6986 | 0.02 | PASS |
| hvac M=2x optimal_pct | 5.61 | 5.6074 | 0.02 | PASS |
| hvac M=4x optimal_pct | 9.13 | 9.1322 | 0.02 | PASS |
| hvac M=6x optimal_pct | 10.44 | 10.4371 | 0.02 | PASS |
| hvac M=observed peak optimal_pct | 12.41 | 12.4147 | 0.02 | PASS |
| forecast uncapped oracle_pct (greedy / milp) | 12.70 | 12.6986 / 12.6986 | 0.02 | PASS |
| forecast uncapped forecast_pct (greedy / milp) | 11.24 | 11.2360 / 11.2360 | 0.02 | PASS |
| forecast M=3x milp oracle_pct | 8.29 | 8.2923 | 0.02 | PASS |
| ev n_raw | 10617 | 10617 | exact | PASS |
| ev n_hourly_kept | 8507 | 8507 | exact | PASS |
| ev hourly_grid_savings_pct | 1.87 | 1.8725 | 0.02 | PASS |
| ev excluded + n_sessions_in | 10617 | 790 + 9827 = 10617 | exact | PASS |
| batch n_jobs | 732691 | 732691 | exact | PASS |
| batch uncapped greedy_pct | 3.97 | 3.96923 | 0.001 | PASS |
| MILP table: first five rows Optimal | Optimal | mip_status 0 for all 8 hvac rows | n/a | PASS |

## Filled TODOs

Line numbers refer to the current `paper.tex`. Abbreviations for sources:
- H = `results/r1_hvac_capacity.csv` (row = `config`)
- E = `results/r1_ev_exact_summary.csv`
- EA = `results/r1_ev_audit.csv` (rows with reason == "valid")
- B = `results/r1_batch_unified.csv`
- F = `results/r1_forecast_capped.csv` (cap "M=3x median")
- FL = `results/r1_logs/forecast.log`
- S = `results/r1_logs/scipy_version.txt`

| Line | Placeholder | Value | Source |
|---|---|---|---|
| 90 | r1_hvac_capacity: total≤peak (no M) | 12.19 | H "total<=observed peak (no M)".optimal_pct = 12.188647 |
| 90 | r1_forecast_capped: retention, capped, milp | 91.5 | F milp.retention_pct = 91.454062 |
| 230 | r1_ev: n_priceable | 9{,}827 | E.n_priceable |
| 230 | median_real_slack_h | 2.0 | E.median_real_slack_h = 2.040556 |
| 230 | median dwell_h | 5.9 | EA median dwell_h = 5.909722 (1 decimal, as for hours) |
| 230 | share_slack_lt_1min | 21.05 | E.share_slack_lt_1min × 100 = 21.05424 |
| 230 | hourly_zero_slack_real_lt_1min | 979 | E.hourly_zero_slack_real_lt_1min |
| 230 | n (slack > 24 h) | 33 | EA count real_slack_h > 24 |
| 230 | max | 139 | EA max real_slack_h = 138.636 |
| 254 | r1_hvac: milp_hours_shifted_at_M | 32 | H "M=3x median" |
| 254 | milp_total_peak_kW | 6.667 | H "M=3x median" |
| 254 | milp_hours_total_gt_M | 1{,}845 | H "M=3x median" |
| 254 | milp_jobs_at_own_hour | 1{,}944 | H "M=3x median" |
| 254 | greedy_jobs_at_own_hour | 1{,}731 | H "M=3x median" |
| 263 | version | 1.18.0 | S |
| 322 | r1_ev: sum of excluded_* | 790 | E: 466 + 317 + 7 (no energy columns, so 0) |
| 322 | excluded_missing_energy + nonpositive | 0 | E: both columns absent, so 0 |
| 322 | excluded_missing_timestamp | 466 | E |
| 322 | excluded_done_not_after_connect | 7 | E |
| 322 | excluded_done_after_disconnect | 317 | E |
| 322 | n_sessions_in | 9{,}827 | E |
| 322 | valid_but_rejected_by_hour_grid | 1{,}479 | E |
| 325 | median_real_slack_h | 2.0 | E |
| 333 | n / x.xx / m | 9{,}827 / 1.81\% / 2.0 | E n_priceable / savings_pct = 1.812813 / median_real_slack_h |
| 350 | HVAC (config table) | 12.19\% | H "total<=observed peak (no M)".optimal_pct |
| 351 | EV (config table) ×2 | 1.81\% / 9{,}827 | E savings_pct / n_priceable |
| 354 | Forecast (config table) | 7.58\% | F milp.forecast_pct = 7.583652 |
| 370 | median_real_slack_h | 2.0 | E |
| 393 | total≤peak (no M), 1st and 3rd | 12.19 | H |
| 393 | M=3x AND total≤peak | 8.29 | H "M=3x median AND total<=observed peak".optimal_pct = 8.289635 |
| 395 | first k at ceiling / k | 6 / 6 | B: smallest k whose greedy_pct is within 1e-6 of uncapped (k=6, difference 0); k=4 is 0.048 below |
| 395 | active_hours | 1{,}642 | B |
| 395 | mean_concurrency | 991 | B = 991.056 |
| 395 | peak_over_median | 2.2 | B = 2.166178 |
| 400 | Fig. 7 clause (rule 11) | deleted | rule 11 skipped as instructed; the `.}}` after it is kept |
| 413–417 | MILP table: gap, upper bound (M=2x/3x/4x/6x/peak) | 8.4e-5, 5.615 / 6.8e-5, 8.299 / 8.5e-5, 9.140 / 7.7e-5, 10.444 / 3.0e-5, 12.417 | H mip_gap, optimal_pct_upper_bound. Printed Saving cells 5.61 / 8.29 / 9.13 / 10.44 / 12.41 match optimal_pct |
| 418 | total≤6.667 kW, no M | Optimal, $5.7\times10^{-5}$, 12.19, 12.194 | H "total<=observed peak (no M)" |
| 419 | M=3x and total≤6.667 kW | Optimal, $9.2\times10^{-5}$, 8.29, 8.298 | H "M=3x median AND total<=observed peak" |
| 458 | jobs touching tier 0/3 | 25 | FL information set: jobs_whose_window_touches_tier0_or_tier3 |
| 460 | forecast_pct, capped, milp | 7.58 | F milp.forecast_pct |
| 460 | penalty_pp | 0.71 | F milp.penalty_pp = 0.708655 |
| 460 | retention_pct (×2) | 91.5 | F milp.retention_pct |
| 460 | forecast_pct, capped, greedy | 6.60 | F greedy.forecast_pct = 6.596971 |
| 460 | forecast_jobs_dirtier_than_baseline | 639 | F milp.forecast_jobs_dirtier_than_baseline_on_true_CI |
| 466 | forecast_pct / retention_pct | 7.58 / 91.5 | F milp |
| 568 | 12.70 minus total-draw result | 0.51 | H: 12.698599 − 12.188647 = 0.509952 |
| 570 | retention_pct / forecast_pct, capped | 91.5 / 7.58 | F milp |
| 596 | total≤peak (no M) | 12.19 | H |
| 596 | retention_pct / forecast_pct, capped | 91.5 / 7.58 | F milp |

The relative gaps are written as `$x.x\times10^{-5}$`. Upper bounds use 3 decimals. The MILP table cells have no `\%` because the column headers carry it.

## Section F: boundary-hour fix, zero-hour check, fills

**Boundary hours: option 1, drop the jobs, applied to all schedules.**
- In `cmd_marginal_iso`, `ok` is now the jobs whose metered hour *and* every hour of `[earliest_start, deadline)` have a non-NaN `ci_iso`. It is built once and shared by all six schedules.
- The CSV rows now record `n_jobs` and `n_dropped`.
- The cap M is still 3 × the median over all 8,363 jobs (1.269 kW), unchanged.
- History: the first greedy run filtered only on the baseline hour and gave NaN in three columns. The cause was the local-2019 vs UTC-2019 mismatch at the year boundary; see `results/r1_logs/marginal_iso_nan_diagnosis.txt`.

**Rerun** (`results/r1_logs/marginal_iso.log`): 10 jobs dropped, 8,353 kept. The results match the earlier diagnostic exactly:

| Schedule, saving on the ISO rates | Capped (full precision) | Uncapped (full precision) | Expected | Match |
|---|---|---|---|---|
| avg-optimized | 3.6240 | 7.0532 | 3.62 / 7.05 | yes |
| ISO-optimized | 28.9346 | 65.7680 | 28.94 / 65.77 | yes, see note |

Note: the capped ISO-optimized value is 28.9346, which rounds to **28.93**, and that is what went in the paper. The "28.94" in the check came from double rounding of my 3-decimal diagnostic print (28.935). It is the same number, not a discrepancy.

**Zero-rate-hour check** (`results/r1_logs/iso_zero_hours.txt`):
- (a) Every one of the 8,755 hours has shares summing to 1.000000; none fall outside 0.99–1.01.
- (b) 228 of 8,755 hours (2.60%) have CI_ISO = 0.
  - Fuels present in those hours (hours present, mean share): Pumped Storage Demand (118, 0.88), Pumped Storage (99, 0.96), Hydro (48, 0.32), Wind (81, 0.16), Demand Response (5, 0.17), Import (1, 0.08). Every row has co2 rate 0.
  - None of these hours has a Gas row; 138 of them sit between two hours that do.
  - The independent 5-minute sheet shows no Gas interval and a zero load-weighted rate in all 228 hours, and has exactly 228 zero hours overall. So the missing gas is in ISO-NE's source data, **not a parsing artifact**, and the zero hours are genuine.
  - The exported CSV matches the xlsx exactly, except for `percent marginal` float round-trip differences of at most 1.1e-16.
- (c) Share of the ISO-optimized saving that comes from jobs placed in zero-rate hours:
  - capped: **15.7%** (943 jobs)
  - uncapped: **30.0%** (1,984 jobs)
- On the scheduling series (after the rolling fill), local-2019 has 229 zero hours out of 8,760. The one extra zero is a filled missing hour. The paper uses the file figure (2.60%).

**Filled** (source `results/r1_marginal_iso.csv`; capped = cap "M=3x median"):

| Line | Placeholder | Value | Column (full precision) |
|---|---|---|---|
| 539 | r_proxy_vs_iso | 0.074 | 0.0739732 |
| 539 | r_avg_vs_iso | 0.189 | 0.1888905 |
| 539 | iso_mean_proxy_oil_hours | 344.2 | 344.2226 |
| 539 | iso_mean_proxy_gas_hours | 320.7 | 320.6557 |
| 541 | avg_retained_of_iso_opt_pct | 12.52 | capped 12.5249 (2-decimal "% of saving" rule) |
| 553 | Average: capped / uncapped | 3.62 / 7.05 | avg_optimized_on_iso_ruler_pct |
| 554 | Proxy: capped / uncapped | 0.67 / 1.17 | proxy_optimized_on_iso_ruler_pct: 0.67096 / 1.16802 |
| 555 | ISO: capped / uncapped | 28.93 / 65.77 | iso_optimized_on_iso_ruler_pct |
| 588 | avg-opt, capped / ISO-opt, capped | 3.62 / 28.93 | as above |

**Text additions:**
- (a) **Made.** Appended to the ISO table caption inside `\BRev{}`: "Jobs whose window reaches an hour without an ISO New England value (10, all at the year boundary) are excluded from every schedule, leaving 8{,}353 jobs."
- (b) **Held back, pending your decision.**
  - The drafted sentence says the ISO-optimized savings are large *because* the optimizer moves load into zero-rate hours. Check (c) shows those jobs supply only 15.7% (capped) and 30.0% (uncapped) of that saving.
  - Most of it comes from the wide spread of nonzero marginal rates: gas 641–2,179 lb/MWh; coal, oil and "Other" up to 3,375, 3,513 and 4,679.
  - The zero-rate units do include hydro and pumped storage, so the "energy-limited" sentence would apply if (b) goes in.

## Surprises and notes

- **Very large ISO-ruler savings.** ISO-optimized saves 28.93% capped and 65.77% uncapped, against 3.62% and 7.05% for average-optimized. Zero-rate hours explain only 15.7% and 30.0% of the ISO-optimized saving (see Section F). This is why addition (b) is on hold.
- **No hvac solve hit the time limit.** Every hvac MILP returned Optimal with a relative gap between 3.0e-5 and 9.2e-5. The total-draw solves took 296 s and 104 s, not the 30 minutes budgeted.
- **ISO file.** The source is `data/marginal/20211102_2019_rt_marginal_co2_emission_rates.xlsx` (ISO-NE document library, SHA-256 b1408bd7…5e256).
  - The exported sheet is "Load-wtd hourly marginal co2", with columns date, hour, type, fuel type, load, percent marginal and co2 rate (lb/MWh).
  - Hours run 0–23, **hour-beginning, prevailing local time** per the README, so the run used no `--hour-ending`, with tz America/New_York.
  - Dates such as "1-Jan" were written out as ISO 2019 dates in `data/marginal/iso_ne_2019_marginal_loadweighted.csv`.
  - The file has 8,755 hours. Besides the spring-forward day, one hour each is missing on 5-Feb, 27-Mar, 14-May and 10-Sep; the script's rolling fill covers those.
- **Rounding choices.** The median dwell (5.9 h) uses the 1-decimal hours rule. The share below 1 minute (21.05%) uses the 2-decimal percentage rule.
- **Compile check skipped** as instructed: there is no `Definitions/mdpi.cls` and no pdflatex.
- **Manual follow-up.** The comment `% TODO(Rajan): regenerate figures/05_oracle_vs_forecast.png ...` is still in place. No figures were regenerated.

## Sec 4.9 addition (b), reworded version (inserted)

Inserted after "attains 12.52\% of that." inside the existing `\BRev{}`. Every number was checked first:
- 15.7 / 30.0: `results/r1_logs/iso_zero_hours.txt` (c)
- 2.60: same file, (b)
- 3.62 / 7.05: `results/r1_marginal_iso.csv`, `avg_optimized_on_iso_ruler_pct`
- The fuel list matches (b).

## Figures regenerated (second pass)

- Generator: new `scripts/fig_r1.py`. It copies the styles of `fig_slack_ecdf.py`, `restyle_figures_a.py` and `fig_savings_vs_capacity.py`, and reads only the R1 CSVs.
- Originals are in `figures/_pre_R1/`.

| Figure | Change | Values (source) | Bytes before → after |
|---|---|---|---|
| 02 | EV curve = empirical CDF of continuous `real_slack_h` over valid sessions; batch n updated; annotation = slack under 1 min | n=9,827, median 2.04 h; 2,069 (21.05%) under 1 min; batch n=732,691 (`r1_ev_audit.csv`, `r1_batch_unified.csv`) | 67,459 → 74,394 |
| 03 | EV bar | 1.81%, n=9,827 (`r1_ev_exact_summary.csv`); HVAC 12.70% and batch 3.97% taken from `r1_hvac_capacity.csv` / `r1_batch_unified.csv`, same values as before | 30,434 → 30,118 |
| 04b | batch curve k=2,3,4,6 = 3.50, 3.77, 3.92, 3.97%; ceiling 3.97%; annotation now "ceiling at k=6" (was k=4) | `r1_batch_unified.csv`. HVAC still from `capacity_sweep.csv`, asserted equal to `r1_hvac_capacity.csv` (within 1e-6) at k=2,3,4,6 and peak | 109,683 → 107,425 |
| 05 | grouped bars: no cap 12.70/11.24, M = 1.269 kW 8.29/7.58 | `r1_forecast_capped.csv`, arm milp | 21,674 → 26,606 |

Notes:
- **1-minute count.** The audit CSV naively gives 2,077 sessions under 1 minute, but 8 sessions have exactly 60 s of slack. The CSV writes 16 significant digits (0.0166666666666666), which reads back just below 1/60. The in-memory summary, and hence the caption, correctly counts 2,069 (21.05%). `fig_r1.py` applies a 1e-9 h guard and asserts agreement with `share_slack_lt_1min`.
- **Batch curve.** The old batch curve (227,529 jobs) also had k=0.5 and k=1 points. The R1 rerun sweeps only k=2,3,4,6, so the new curve starts at k=2.

Caption checks:
- 03, 04b and 05: the numbers match the figures.
- 02: the numbers match; the annotation prints 21.1% (1 decimal) against the caption's 21.05%. The caption's last sentence, "The heat-pump load is not shown…", contradicts the figure, which shows the HVAC (12 h) and batch (6 h) assigned constants. That mismatch predates R1 (the old figure also showed them) and is **not fixed**.

Text edits:
- Sec 4.4: after "and enforcing both gives 8.29\%" inserted `\BRev{, unchanged, because at $M = 3\times$ median the schedule's total draw already stays within 6.667~kW}`.
  - Confirmed: `milp_total_peak_kW` for "M=3x median" = 6.667, equal to the cap.
  - The two MILP values (8.2923 and 8.2896) differ by 0.003 pp. That is inside the solvers' relative-gap tolerance (about 0.008 pp on this scale); both round to 8.29.
- Deleted the `% TODO(Rajan): regenerate figures/05…` comment line.
- A word diff confirms these are the only changes in this pass.

Backup: `~/r1_backup_20260921_2245.tgz`, 55 entries. It uses the requested list plus `scripts/fig_r1.py`.
