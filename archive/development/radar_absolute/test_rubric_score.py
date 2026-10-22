#!/usr/bin/env python3
"""Independent check of rubric_score against the brief's anchors (scratch file)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "tgsim"))
from plot_events import RADAR_RUBRIC, rubric_score  # noqa: E402

OK = True


def check(name, got, expected, tol=1e-12):
    global OK
    good = abs(got - expected) <= tol
    OK = OK and good
    print(f"  {'PASS' if good else 'FAIL'}  {name:46s} got {got!r:22s} expected {expected!r}")


print("rubric_score unit checks (values taken from the brief, not from the data)")
# 1. efficiency = identity on an already-0..1 fraction
eff = RADAR_RUBRIC["moving_fraction"][1]
check("efficiency 0.0", rubric_score(0.0, eff), 0.0)
check("efficiency 0.754953", rubric_score(0.754953, eff), 0.754953)
check("efficiency 1.0", rubric_score(1.0, eff), 1.0)
# 2. gap keeping: 5->0, 10->1, 25->1, 40->0
gap = RADAR_RUBRIC["spacing_mean_m"][1]
check("gap 5 m -> 0", rubric_score(5.0, gap), 0.0)
check("gap 7.5 m -> 0.5 (ramp up)", rubric_score(7.5, gap), 0.5)
check("gap 10 m -> 1", rubric_score(10.0, gap), 1.0)
check("gap 17.5 m -> 1 (plateau)", rubric_score(17.5, gap), 1.0)
check("gap 25 m -> 1", rubric_score(25.0, gap), 1.0)
check("gap 32.5 m -> 0.5 (ramp down)", rubric_score(32.5, gap), 0.5)
check("gap 40 m -> 0", rubric_score(40.0, gap), 0.0)
check("gap 4 m -> 0 (clamp below)", rubric_score(4.0, gap), 0.0)
check("gap 60 m -> 0 (clamp above)", rubric_score(60.0, gap), 0.0)
check("gap 8.50522 m", rubric_score(8.50522, gap), (8.50522 - 5.0) / 5.0)
# 3. closest approach: 2->0, 6->1
close = RADAR_RUBRIC["spacing_min_m"][1]
check("min spacing 2 m -> 0", rubric_score(2.0, close), 0.0)
check("min spacing 4.238255 m", rubric_score(4.238255, close), (4.238255 - 2.0) / 4.0)
check("min spacing 6 m -> 1", rubric_score(6.0, close), 1.0)
check("min spacing 9 m -> 1 (clamp)", rubric_score(9.0, close), 1.0)
# 4./5. jerk and accel: 3.0 -> 0, 0.5 -> 1  i.e. clamp((3 - x)/(3 - 0.5))
for key in ("jerk_rms_smoothed_mps3", "accel_rms_mps2"):
    anchors = RADAR_RUBRIC[key][1]
    check(f"{key} x=3.0 -> 0", rubric_score(3.0, anchors), 0.0)
    check(f"{key} x=1.75 -> 0.5", rubric_score(1.75, anchors), 0.5)
    check(f"{key} x=0.5 -> 1", rubric_score(0.5, anchors), 1.0)
    check(f"{key} x=0.2 -> 1 (clamp)", rubric_score(0.2, anchors), 1.0)
    check(f"{key} x=4.0 -> 0 (clamp)", rubric_score(4.0, anchors), 0.0)
    for x in (0.0, 0.377389, 1.0, 2.4, 5.0):
        check(f"{key} x={x} == clamp((3-x)/2.5)",
              rubric_score(x, anchors), min(max((3.0 - x) / 2.5, 0.0), 1.0))
# missing / non-finite -> 0.0, never an error
check("None -> 0.0", rubric_score(None, gap), 0.0)
check("NaN -> 0.0", rubric_score(float("nan"), gap), 0.0)
check("+inf -> 0.0", rubric_score(float("inf"), gap), 0.0)
# data-independence: the same raw value always gives the same score
check("data independence (0.9 twice)", rubric_score(0.9, eff), rubric_score(0.9, eff))
print(f"\nunit checks: {'ALL PASS' if OK else 'FAILURES PRESENT'}")
raise SystemExit(0 if OK else 1)
