# Brief: Presentation-layer statistics for the TGSIM AV-vs-HV comparison

## Project
`C:\Users\ptiu1\Documents\PhDStudy\INTR5100\lab1`
Python: `C:/Users/ptiu1/miniconda3/envs/intr5100/python.exe`

## Background
The TGSIM differentiable-IDM calibration is final (`results/tgsim_calibration.json`).
A presentation review (tutor) found the current analysis aggregates metrics only
as event-mean and pooled/sample-weighted, which hides two facts:
1. The HV–AV advantage in gap keeping / closest approach is driven by two
   **non-following sparse events** (HV–AV event_7/event_8, mean spacing 28–41 m,
   IDM interaction inactive). Excluding them, per-event median spacing is ~8.8 m
   (HV–AV) vs ~8.3 m (HV–HV), and 6 of the 8 HV–AV events have mean spacing <10 m.
2. For jerk/accel RMS the event-mean and event-median orderings of the two pairs
   reverse, so conclusions depend on aggregation.

Add a **presentation statistics** layer so the report can show raw physical
quantities with multiple aggregations and effect sizes. Do NOT change any
calibration result or existing JSON; add one new file.

## Inputs
- Per-event CSVs: `data/tgsim/HV-HV/event_*.csv`, `data/tgsim/HV-AV/event_*.csv`
  (columns include time, follower speed, leader speed, spacing/dhw — inspect and
  use exactly what preprocess_events.py wrote).
- Existing metrics conventions from `results/tgsim_analysis.json`
  (moving threshold, jerk smoothing window must match that script — read it).

## Sparse-event rule (hard-code, document)
An event is "sparse / non-interacting" if its **mean spacing > 20 m**
(expected to identify exactly HV–AV event_7 and event_8; verify and list which
events qualify per pair).

## Metrics (compute per event for both pairs)
- speed_mean (m/s)
- moving_fraction (same speed threshold the existing pipeline uses — state it)
- spacing_mean (m), spacing_min (m)
- accel_rms (m/s²) — same definition as existing analysis
- jerk_rms_smoothed (m/s³) — same smoothing as existing analysis

## Outputs — new file `results/tgsim_presentation_stats.json`
For every metric above and for each pair provide:
1. `event_mean` and `event_median` (median across events) and
   `pooled_sample_weighted`.
2. The same three aggregations on the **interaction subset** (sparse events
   excluded); also list excluded event ids and their mean spacing.
3. `event_values`: ordered per-event values for both pairs; for spacing_mean
   include `n_events_below_10m` and total event count.
4. Effect size for the per-event comparison of each metric: rank-biserial r
   (2U/(n_a n_b) − 1 from the Mann–Whitney U) plus the two-sided p-value
   (recompute; expect p > 0.14, low power with 10 vs 8 events).
5. A `method` section documenting: moving threshold, jerk smoothing,
   sparse rule (>20 m mean spacing), and a note that event-mean vs event-median
   orderings are both reported because they reverse for jerk/accel.

Use atomic write (`.tmp` + `os.replace`), deterministic key order, no
timestamps. Script under `src/tgsim/presentation_stats.py`, idempotent.

## Self-checks (print results)
- HV–AV interaction subset spacing mean ≈ 10.4 m and per-event median ≈ 8.8 m;
  HV–HV median ≈ 8.3 m (report exact computed values).
- 6/8 HV–AV events spacing_mean < 10 m (verify count).
- Demonstrate the mean-vs-median ordering reversal numerically for jerk
  and/or accel RMS.
- Recompute U/p independently against `tgsim_analysis.json` and confirm they
  agree with the existing p-values (rank_tests).
- Do not install packages, touch git, or modify any existing artifact.
