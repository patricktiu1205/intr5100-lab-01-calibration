# Brief: Five-group parameters comparison and validation analysis (Lab Session I)

## Project
`C:\Users\ptiu1\Documents\PhDStudy\INTR5100\lab1`
Python: `C:/Users/ptiu1/miniconda3/envs/intr5100/python.exe` (torch needs KMP_DUPLICATE_LIB_OK=TRUE)

## Data
`data/group_validation_raw/` contains five groups' validation runs (group_1 … group_5)
plus `idm_parameters_groups.json`. Every group ran the **same scenario** on the
same map: line1, leader_speed = 40, curve_speed = 35 (see each metadata.json).
Our submitted group is **group_4** (parameters: v0 41.12, T 2.09, s0 28.72,
a 13.97, b 185.16; note v0 = 123.36/3).

Per group:
- `commands.csv` columns: role, t_from_start, mode, reason, front_role, rear_role,
  gap, spacing, desired_velocity, v_cmd, a_idm, emergency, dt, time
  (spacing = gap − 250 platform convention; a_idm blank in curve/non-IDM modes).
- `follower_straight_trajectory.csv`: pose stream with t_from_start, x, y,
  velocity, s, lateral_error, route_distance, control_mode, etc.

## Required analyses

### A. Five parameter sets
1. Table of raw v0, T, s0, a, b, delta for all five groups (log scale for plots).
2. The raw parameters are not directly comparable because groups submitted in
   inconsistent scales. Therefore compute the **gauge-invariant dimensionless
   combinations** for each group (same formulae used elsewhere in the report):
   - ṽ0 = v0 / sqrt(b · s0)
   - ã = a / b
   - T̃ = T · sqrt(b / s0)
   These are invariant under the multiplicative unit rescaling and ARE
   comparable across groups. Report values and note spreads (min/max/CV).

### B. Five validation runs — parse follower rows of commands.csv
Use `t_from_start` (irregular dt; don't assume uniform). Compute per group:
1. Duration; follower rows count.
2. Spacing: min (report exact), mean, fraction of samples with spacing below
   thresholds 100 and 250 (platform-record units).
3. Safety layer: number and fraction of follower rows with emergency = 1
   (count distinct trigger episodes, i.e. consecutive emergency rows = one
   trigger), list trigger times and spacing at trigger.
4. Speed: mean and max of v_cmd (or desired_velocity; use v_cmd), moving
   fraction (v_cmd above the 0.3 threshold consistent with other analyses).
5. Dynamics where a_idm present: RMS and 95th percentile |a_idm|; jerk RMS
   from consecutive a_idm differences divided by dt (document the definition;
   skip across mode gaps larger than 1 s).
6. Lateral: RMS lateral_error from follower_straight_trajectory.csv.
7. Completion: max route_distance per group (all should complete line1; report).

### C. Interpretation
- Rank groups on safety (emergency episodes, spacing min) and efficiency
  (mean moving speed, duration) and relate to the dimensionless parameter
  combinations, not raw values.
- Be explicit about caveats: different physical cars/IPs per run (metadata),
  single run per group, no inferential claims; device/scale confounds.
- Report which groups stayed in pure IDM control vs needed the safety layer.

## Outputs
- `results/group_validation_analysis.json` (all tables/metrics + method notes;
  atomic write, deterministic keys, no timestamps)
- `results/figures/group_validation/fig_01..` at minimum:
  1. raw parameters comparison (log-scale grouped bars or parallel coords)
  2. dimensionless combinations grouped bars (five groups × 3 combos)
  3. spacing trajectories vs time for all five groups with emergency markers
  4. speed trajectories vs time for all five groups
  5. safety/efficiency metric comparison grouped bars
  Each plot: title, legend, tight layout, English labels (bilingual captions
  will be added to the report later — print-ready PNGs only).
- Script under `src/group_validation/`, idempotent, re-runnable.

## Rules
- Atomic writes (.tmp + os.replace), no timestamps, no package installs,
  no git, do not modify any existing files/results.
- Self-check: independent recomputation of key values (spacing min, emergency
  counts) via a second parsing path; print a per-group summary table.
