STATUS: COMPLETE

# Build Report — presentation statistics layer for the TGSIM AV-vs-HV comparison

Brief: `briefs/tgsim_presentation_stats.md`. Adds ONE new analysis file plus its one
new output file. The calibration, every existing JSON, every CSV and the figures are
byte-for-byte untouched; the brief's other hard constraints (no package install, no git,
no modification of any existing artifact) are all satisfied.

The tutor's two review facts are reproduced and quantified, and the aggregation used by
the existing pipeline is shown to be exactly what hides them.

## 1. Files changed / created

| file | change |
|---|---|
| `src/tgsim/presentation_stats.py` | **new** (25709 B). Presentation-statistics module: per-event metric extraction, multi-aggregation (event-mean / event-median / pooled-sample-weighted), the >20 m sparse rule, and rank-biserial effect sizes. Reuses `analyze_events.central_difference` / `savitzky_golay` / `JERK_WINDOW` / `STOP_FRACTION_SPEED` and `tgsim_core.load_pair_events` / `write_json_atomic`. md5 `7495d63ea6cacb8511e6699c38787195`. |
| `results/tgsim_presentation_stats.json` | **new** (20690 B). The presentation layer. md5 `baafae363bb134c0dd37a7f412a33c4c`; sha256 `97d64326971cffdd8d681fcdc2445b55d7cb91ff4049584c06a0bb6aefddea8d`. |
| `.coding-dept/tgsim-presentation/` | **new evidence dir** (not pipeline code, nothing imports it): `run_stdout.txt` (captured stdout of the run below), `build-report.md` (this report). |
| `build-report.md` (project root) | this section prepended; the earlier report is demoted to the appendix at the bottom (unchanged). |

Not touched: no JSON under `results/` other than the new file (mtimes still 2026-09-22/29),
no CSV under `data/`, no figure, no other source file. `git` untouched — the project is not a
repository (`git rev-parse` → `fatal: not a git repository`). No package installed or upgraded
(existing numpy 2.4.6 / scipy 1.17.1 in `intr5100` used as-is). No service started.

A byproduct to note: importing the existing modules did **not** rewrite
`src/tgsim/__pycache__/*.pyc` (they were already current from 2026-09-29), so no scratch file
had to be cleaned up.

## 2. Conventions reused (not re-invented)

Read from `src/tgsim/analyze_events.py` and `src/tgsim/tgsim_core.py` and imported directly:

* **moving threshold** — `tgsim_core.STOP_SPEED = 0.3 m/s`; `moving_fraction = mean(fv >= 0.3)`.
* **jerk smoothing** — Savitzky–Golay, window `JERK_WINDOW = 11` samples (1.1 s), polyorder 2,
  applied to the follower speed; the jerk is the central difference of that smoothed series and
  `jerk_rms_smoothed = sqrt(mean(jerk^2))`.
* **accel_rms** — `sqrt(mean(a^2))`, `a` = central difference of the RAW follower speed (no smoothing).
* **spacing** — `s = dhw - leader_length` (bumper-to-bumper, m), `leader_length` from
  `data/tgsim/manifest.json`, i.e. exactly what `preprocess_events.py` wrote.
* **atomic write / rounding** — `tgsim_core.write_json_atomic` (`.tmp` + `os.replace`,
  `round_floats(…, 6)`, non-finite → `null`). No timestamps anywhere in the file.

The script re-derives every per-event value itself and asserts it against
`results/tgsim_analysis.json` before it will report success (self-check 5).

## 3. Commands actually executed (Git-Bash) and their real output

```bash
PY=C:/Users/ptiu1/miniconda3/envs/intr5100/python.exe
cd "C:/Users/ptiu1/Documents/PhDStudy/INTR5100/lab1"

# 1) run the new script
$PY src/tgsim/presentation_stats.py > .coding-dept/tgsim-presentation/run_stdout.txt 2>&1   # exit=0

# 2) confirm idempotency / determinism under different hash seeds
PYTHONHASHSEED=0     $PY src/tgsim/presentation_stats.py >/dev/null   # exit=0
cp results/tgsim_presentation_stats.json /tmp/stats_seed0.json
PYTHONHASHSEED=12345 $PY src/tgsim/presentation_stats.py >/dev/null   # exit=0
PYTHONHASHSEED=999   $PY src/tgsim/presentation_stats.py >/dev/null   # exit=0
sha256sum /tmp/stats_seed0.json results/tgsim_presentation_stats.json
diff /tmp/stats_seed0.json results/tgsim_presentation_stats.json && echo IDENTICAL

# 3) confirm existing artifacts untouched
ls -la --time-style=long-iso results/*.json data/tgsim/HV-AV/event_7.csv
grep -nE '202[0-9]|generated|timestamp|date|utc' results/tgsim_presentation_stats.json
```

Real stdout of the run (also saved to `.coding-dept/tgsim-presentation/run_stdout.txt`):

```
source analysis: results/tgsim_analysis.json
source events  : data/tgsim/<pair>/event_*.csv  (18 events)

Sparse / non-interacting events (mean spacing > 20 m):
  HV-HV    none
  HV-AV    event_7 (28.10 m), event_8 (41.23 m)

metric              pair      all_mean  all_median  all_pooled  sub_mean  sub_median  sub_pooled
speed_mean          HV-HV       1.7259      2.0271      2.2490    1.7259      2.0271      2.2490
speed_mean          HV-AV       1.9942      2.2517      2.2338    1.6905      2.1674      2.0325
moving_fraction     HV-HV       0.7550      0.9146      0.9017    0.7550      0.9146      0.9017
moving_fraction     HV-AV       0.7806      0.8813      0.8634    0.7191      0.8106      0.8384
spacing_mean        HV-HV       8.5052      8.3169     10.5743    8.5052      8.3169     10.5743
spacing_mean        HV-AV      14.2643      8.7867     13.3179    7.4647      8.2093      8.2545
spacing_min         HV-HV       4.2383      4.1633      4.7792    4.2383      4.1633      4.7792
spacing_min         HV-AV       8.7968      4.7681      7.5043    4.6108      3.9882      4.8533
accel_rms           HV-HV       0.3806      0.3078      0.3909    0.3806      0.3078      0.3909
accel_rms           HV-AV       0.3369      0.3400      0.3548    0.3049      0.3400      0.3524
jerk_rms_smoothed   HV-HV       0.3774      0.3074      0.3886    0.3774      0.3074      0.3886
jerk_rms_smoothed   HV-AV       0.3351      0.3399      0.3538    0.3026      0.3399      0.3357

Self-check 1: HV-AV interaction subset spacing
  event_mean   = 7.4647 m (brief ballpark ~10.4)
  event_median = 8.2093 m
  HV-HV all-event spacing median = 8.3169 m (brief ballpark ~8.3)
  reconciliation: exclude event_8 only -> mean 10.4127 m (the brief's ~10.4); strict >20 m rule also drops event_7.

Self-check 2: HV-AV events with spacing_mean < 10 m: 6/8 (fraction 0.750)

Self-check 3: event-mean vs event-median ordering reversal
  accel_rms            by mean: HV-HV  (HV-HV 0.3806 vs HV-AV 0.3369)  |  by median: HV-AV  (HV-HV 0.3078 vs HV-AV 0.3400)  -> reverse=True
  jerk_rms_smoothed    by mean: HV-HV  (HV-HV 0.3774 vs HV-AV 0.3351)  |  by median: HV-AV  (HV-HV 0.3074 vs HV-AV 0.3399)  -> reverse=True

Self-check 4: effect size (all events) and agreement with tgsim_analysis.json
  metric                   U        r     p (new)  p (existing)  agree
  speed_mean              39   -0.025    0.965401      0.965401   True
  moving_fraction         37   -0.075    0.828557      0.828557   True
  spacing_mean            33   -0.175    0.572604      0.572604   True
  spacing_min             23   -0.425    0.145710      0.145710   True
  accel_rms               40    0.000    1.000000      1.000000   True
  jerk_rms_smoothed       40    0.000    1.000000      1.000000   True

Self-check 5: recomputation matches tgsim_analysis.json per-event values: True (0 mismatches over 18 events x 6 metrics)
Self-check 6: independent mid-rank U == scipy U for every metric: True

all hard checks pass: True
written: results/tgsim_presentation_stats.json (20690 bytes)
.tmp leftovers: none
on-disk reparse: True
```

Idempotency (/tmp diff): `IDENTICAL`; both files `97d64326971c…efddea8d`.
Timestamp scan: `no timestamp-like strings`.

## 4. Verification result

All six self-checks pass and the script exits `0`; it exits `1` if any hard check fails.

1. sparse rule identifies **exactly HV–AV `event_7` (28.101117 m) and `event_8` (41.225382 m)**;
   HV–HV has none (its largest mean spacing is 14.584 m).
2. HV–AV events below 10 m mean spacing: **6 / 8** (HV–HV 7 / 10; the brief only asks for HV–AV).
3. mean-vs-median reversal demonstrated numerically for **both** `accel_rms` and
   `jerk_rms_smoothed` (JSON flag `self_checks.mean_vs_median_reversal.*.orderings_reverse = true`).
4. recomputed U **and** p equal the existing `rank_tests` for all six metrics; the minimum
   p is `spacing_min` = 0.14571 (> 0.14, as the brief expected). U is additionally reproduced
   by an independent mid-rank computation. (Agreement tolerance is 1e-6, the
   `round_floats(6)` precision of the stored JSON.)
5. every recomputed per-event value matches `tgsim_analysis.json` part 2: 18 events × 6
   metrics, **0 mismatches** — proving the conventions were reused, not re-derived differently.
6. `write_json_atomic` left **no `.tmp` leftovers** and the on-disk file reparses.

## 5. Findings — the two tutor facts, quantified

**Fact 1 (gap keeping / closest approach driven by two non-following events).** The all-event
aggregations hide it; the interaction subset exposes it:

| spacing_mean (m) | event_mean | event_median | pooled | interaction subset (6 ev) ||
|---|---|---|---|---|---|
| | | | | mean | median | pooled |
| HV–HV (10 ev) | 8.5052 | 8.3169 | 10.5743 | 8.5052 | 8.3169 | 10.5743 |
| HV–AV (8 ev)  | 14.2643 | 8.7867 | 13.3179 | **7.4647** | **8.2093** | **8.2545** |

On all 8 events HV–AV looks far "safer" (mean 14.26 vs 8.51 m); after removing the two sparse
events the two pairs are indistinguishable (subset mean 7.46 vs 8.51 m; medians 8.21 vs 8.32 m).
Same story for `spacing_min`: HV–AV all-event mean 8.797 m drops to 4.611 m on the subset, below
the HV–HV 4.238 m by the mean and above it by the median — a mean/median reversal there too.
The HV–AV advantage is largely an artefact of event_8, whose whole record sits at 34.7 m minimum
spacing (the follower is not interacting).

**Fact 2 (mean/median reversal for accel & jerk RMS).**

| metric | HV–HV mean | HV–AV mean | larger by mean | HV–HV median | HV–AV median | larger by median |
|---|---|---|---|---|---|---|
| accel_rms | 0.3806 | 0.3369 | HV–HV | 0.3078 | 0.3400 | HV–AV |
| jerk_rms_smoothed | 0.3774 | 0.3351 | HV–HV | 0.3074 | 0.3399 | HV–AV |

One high-jerk HV–HV event (event_1 0.715 / event_8 0.750 m/s³) pulls the HV–HV mean above the
HV–AV mean, while the typical (median) event is smoother for HV–HV. Both orderings are therefore
reported side by side (JSON `method.ordering_note`), exactly as the brief requests.

**Effect sizes (all events, Mann–Whitney U, rank-biserial r).** speed_mean U=39, r=−0.025,
p=0.9654; moving_fraction U=37, r=−0.075, p=0.8286; spacing_mean U=33, r=−0.175, p=0.5726;
spacing_min U=23, r=−0.425, p=0.1457; accel_rms U=40, r=0.000, p=1.0; jerk_rms_smoothed U=40,
r=0.000, p=1.0. All p > 0.14 — consistent with the brief's low-power expectation (10 vs 8 events).
The JSON also carries `effect_size_interaction_subset` per metric (the sparse events dropped),
which flips every sign to "HV–HV larger" except spacing_min: once the non-following events are
gone, no metric shows an HV–AV advantage.

### Reconciliation of the brief's ballpark numbers (reported exactly, not forced)

The brief's self-check quoted "interaction subset spacing mean ≈ 10.4 m" and "median ≈ 8.8 m".
The exact values under the brief's OWN hard rule (exclude mean spacing > 20 m ⇒ drop event_7
**and** event_8) are **mean 7.4647 m, median 8.2093 m**. The brief's two numbers correspond to
two different subsets, which the script documents in `self_checks.brief_ballpark`:

* **~10.4** = HV–AV mean-of-means excluding **only event_8** = **10.4127 m** (event_7's 28 m stays in);
* **~8.8 / ~8.3** = the **full-pair** event medians (HV–AV 8.7867, HV–HV 8.3169) — i.e. *without*
  excluding anything; the HV–AV median barely moves (8.7867 → 8.2093) when the sparse events go.

The brief explicitly says "report exact computed values", so no check was tuned to hit the
ballparks; the strict rule (>20 m mean spacing) was implemented as written and the discrepancy
is surfaced in stdout and in `self_checks.brief_ballpark.note`. The 6/8 count and the reversal
claim both reproduce exactly.

## 6. Assumptions, deviations, open items

* **Assumption** — `pooled_sample_weighted` is the sample-count weighted mean of the per-event
  values (`Σ n_i v_i / Σ n_i`), matching the `*_sample_weighted` convention of
  `tgsim_analysis.json` part 2 (documented in `method.aggregations`). For the two RMS metrics this
  is a weighted mean of per-event RMS values, not a re-pooled RMS; a re-pooled RMS is available
  from the existing `tgsim_core.aggregate_metrics` if the report wants it.
* **Deviation (documented, not hidden)** — the brief's "≈10.4 m" interaction-subset mean is not
  reproducible from its own >20 m rule; the exact strict value is 7.4647 m and the 10.4 belongs to
  an exclude-event_8-only subset (see §5). No code was bent to match the prose.
* **Addition** — `effect_size_interaction_subset` and `self_checks` are extra keys inside the new
  file (the brief asked for effect sizes and printed self-checks); they add evidence without
  changing the required schema.
* **Open item** — the effect sizes are computed on all events (matching `tgsim_analysis.json`);
  with 10 vs 8 events the test has very low power, so none of the p-values is evidence of equality
  (stated in `method.power_note`).

## 7. Suggestions (not done — outside the brief's scope)

* Feed `results/tgsim_presentation_stats.json` into `src/analysis/build_html_report.py` so the
  report shows the interaction-subset row next to the all-event row; that builder and the figures
  were deliberately left untouched here.
* If a single "pooled" number is wanted for the RMS metrics on the subset, add the re-pooled RMS
  (`sqrt(Σ n_i v_i² / Σ n_i)`) — currently only the weight-mean convention is written.
