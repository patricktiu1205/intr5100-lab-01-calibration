# Ask tutor: radar scoring anchors — tighten or present differently?

## Context (course extension, not part of the lab grade)
Working dir: `C:\Users\ptiu1\Documents\PhDStudy\INTR5100\lab1`
In the TGSIM I-90/I-94 extension, two car-following pairs were calibrated with
the differentiable IDM: HV–HV (human follower) and HV–AV (automated follower).
A five-axis **style radar** compares them. The student objected to the original
design (each axis normalised so the larger pair value = 1.0), rightly noting it
makes one dataset the benchmark. It was rebuilt with an **absolute, a-priori
0–1 scoring rubric**, one ruler for both pairs.

Current fixed anchors (piecewise-linear, using event-mean raw values):

| Axis | 0 anchor | 1 anchor / ideal |
|---|---|---|
| Efficiency | moving fraction, already 0–1 |  |
| Gap keeping (mean spacing) | 5 m→0; 10–25 m ideal; 40 m→0 |  |
| Closest approach (min spacing) | 2 m | 6 m |
| Smoothness (jerk RMS, m/s³) | 3.0 | 0.5 |
| Comfort (accel RMS, m/s²) | 3.0 | 0.5 |

Observed event-mean values:

| | HV–HV | HV–AV |
|---|---|---|
| moving fraction | 0.755 | 0.781 |
| spacing mean (m) | 8.51 | 14.26 |
| spacing min (m) | 4.24 | 8.80 |
| jerk RMS smoothed (m/s³) | 0.377 | 0.335 |
| accel RMS (m/s²) | 0.381 | 0.337 |

Scores now: HV–HV = 0.755 / 0.701 / 0.560 / 1.00 / 1.00;
HV–AV = 0.781 / 1.00 / 1.00 / 1.00 / 1.00.
Both pairs saturate (1.0) on smoothness and comfort; the radar separates them
only on three axes.

## Questions for the tutor
1. Is an absolute fixed-anchor 0–1 rubric a defensible presentation for a
   technical report (versus normalising to the data)? Any framing requirements
   so a reader does not over-interpret the scores?
2. The comfort/smoothness anchors were chosen from rough engineering judgement.
   Are there literature/standard-supported thresholds for acceleration and jerk
   (e.g. comfort accelerations in car-following, jerk limits, ISO/comfort
   bands) we could cite instead of inventing anchors? If we tighten them, what
   ranges are principled for *urban/stop-and-go* TGSIM following specifically?
3. Better alternatives: should we (a) keep the radar with tightened, cited
   anchors, (b) drop the radar and just report raw physical metrics with
   effect sizes, or (c) use a different normalisation (e.g. z-scores against a
   published reference distribution, or log-scale)? What is standard in the
   car-following / AV-HV comparison literature?
4. Any caution about using event-mean vs pooled (sample-weighted) values —
   pooled values are moving fraction 0.902/0.863, spacing mean 10.57/13.32,
   spacing min 4.78/7.50, jerk 0.389/0.354, accel 0.391/0.355.

Keep the answer concise and citable (chapter/page or named references where
possible). This concerns presentation only; the calibration is final.
