# Brief: Rebuild style radar with an absolute 0–1 scoring rubric

## Project
`C:\Users\ptiu1\Documents\PhDStudy\INTR5100\lab1`
Python: `C:/Users/ptiu1/miniconda3/envs/intr5100/python.exe`

## Problem
`src/tgsim/plot_events.py` (radar block, approx lines 170–205) normalises each
radar axis so the **larger of the two pair values becomes 1.0** ("each axis
normalised to the pair maximum"). This treats one dataset as the benchmark and
makes scores meaningless for any future third group. Replace with an **absolute,
a-priori-fixed 0–1 scoring rubric**, identical for both pairs.

## Required rubric (hard-code these anchors; do not derive them from the data)
Piecewise-linear clamp mapping, using **event-mean** raw values
(`{key}_event_mean`, as currently used):

1. Efficiency = `moving_fraction` — already 0–1: score = value.
2. Gap keeping = `spacing_mean_m` (m) — ideal window:
   - 5 m → 0 ; 10 m → 1 ; 25 m → 1 ; 40 m → 0 (linear between anchors).
     Too close is unsafe; too detached means no real car-following.
3. Closest approach = `spacing_min_m` (m) — 2 m → 0 ; 6 m → 1.
4. Smoothness = `jerk_rms_smoothed_mps3` (m/s³) — 3.0 → 0 ; 0.5 → 1
   (smaller jerk is better; i.e. score = clamp((3 − x)/(3 − 0.5), 0, 1)).
5. Comfort = `accel_rms_mps2` (m/s²) — 3.0 → 0 ; 0.5 → 1.

Implement a small documented helper `rubric_score(value, anchors)` (list of
(x, score) breakpoints, linear interpolation, clamp to [0,1]) so the anchors
are explicit in one place.

## Plot requirements
- Same five polar axes; axis titles should say "score (0–1)", not
  "inverse ..." since values are now scores, e.g. `efficiency\n(moving fraction)`,
  `gap keeping\n(mean spacing)`, `closest approach\n(min spacing)`,
  `smoothness\n(low jerk)`, `comfort\n(low accel)`.
- Radial ticks fixed 0–1 (0.2, 0.4, 0.6, 0.8, 1.0).
- Title must state the absolute rubric, e.g.:
  "Style profile scored against fixed physical anchors (0–1; higher = better)"
- Keep colours/labels/legend for HV–HV (human follower) and HV–AV (AV follower).
- Overwrite the same file `results/figures/tgsim/fig_09_style_radar.png`
  (atomically / via existing save helper); no other figures or files may change.
- Print a raw-value → score table for both pairs for verification.
- Do not install packages, do not touch git, do not modify JSON files.

## Self-check before finishing
- Both polygons are inside [0,1].
- Neither pair has any axis forced to exactly 1 merely because it is the max.
- Re-running the script is idempotent and only fig_09 changes on disk.
