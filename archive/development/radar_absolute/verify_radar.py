#!/usr/bin/env python3
"""Independent verification of the fig_09 style radar rubric (scratch file).

Re-derives the radar values from ``results/tgsim_analysis.json`` and checks the
brief's self-check list:
  * both polygons inside [0, 1];
  * no axis forced to exactly 1 merely because it is the larger of the two
    (the old pair-max normalisation did force that on every axis);
  * the score of a pair does not depend on the other pair's value.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "tgsim"))
from plot_events import RADAR_KEYS, RADAR_RUBRIC, rubric_score  # noqa: E402

PAIRS = ("HV-HV", "HV-AV")
OK = True


def check(name, condition, detail=""):
    global OK
    OK = OK and bool(condition)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")


part2 = json.loads((ROOT / "results" / "tgsim_analysis.json").read_text("utf-8"))[
    "part2_driving_style"]["per_pair"]
raw = {pair: {key: part2[pair][f"{key}_event_mean"] for key in RADAR_KEYS} for pair in PAIRS}
score = {pair: {key: rubric_score(raw[pair][key], RADAR_RUBRIC[key][1]) for key in RADAR_KEYS}
         for pair in PAIRS}


def ceiling_region(anchors, tol=1e-12):
    """Set of raw values whose piecewise-linear rubric score is 1.0.

    Returns a list of ``(lo, hi)`` half-open intervals (``lo``/``hi`` may be
    ``-inf``/``+inf``). Outside the outermost anchors the score is clamped to the
    outermost anchor score, so a ``1.0``-valued endpoint extends to infinity.
    """
    points = sorted((float(x), float(y)) for x, y in anchors)
    regions: list[tuple[float, float]] = []
    if abs(points[0][1] - 1.0) <= tol:
        regions.append((float("-inf"), points[0][0]))
    if abs(points[-1][1] - 1.0) <= tol:
        regions.append((points[-1][0], float("inf")))
    for index in range(len(points) - 1):
        (x0, y0), (x1, y1) = points[index], points[index + 1]
        if abs(y0 - 1.0) <= tol and abs(y1 - 1.0) <= tol:
            regions.append((x0, x1))
        elif abs(y0 - 1.0) <= tol:
            regions.append((x0, x0))
        elif abs(y1 - 1.0) <= tol:
            regions.append((x1, x1))
    return regions


def in_regions(value, regions, tol=1e-9):
    return any(lo - tol <= value <= hi + tol for lo, hi in regions)


print("radar self-checks (values re-derived from results/tgsim_analysis.json)")
print(f"\n  raw event means:")
for pair in PAIRS:
    print(f"    {pair}: " + ", ".join(f"{key}={raw[pair][key]:.6f}" for key in RADAR_KEYS))
print(f"\n  scores:")
for pair in PAIRS:
    print(f"    {pair}: " + ", ".join(f"{key}={score[pair][key]:.6f}" for key in RADAR_KEYS))

print()
check("all 10 scores inside [0, 1]",
      all(0.0 <= score[pair][key] <= 1.0 for pair in PAIRS for key in RADAR_KEYS),
      f"min={min(score[p][k] for p in PAIRS for k in RADAR_KEYS):.6f} "
      f"max={max(score[p][k] for p in PAIRS for k in RADAR_KEYS):.6f}")
check("polygon closure: 5 axes + repeat of first point",
      len(RADAR_KEYS) == 5)
check("radar uses exactly the five rubric axes of the brief",
      set(RADAR_KEYS) == {"moving_fraction", "spacing_mean_m", "spacing_min_m",
                          "jerk_rms_smoothed_mps3", "accel_rms_mps2"})

# 1) independence: a pair's score is a function of its own raw value only.
#    Delete the other pair from the input and recompute: nothing may move.
solo = {}
for pair in PAIRS:
    subset = {pair: part2[pair]}                       # other pair removed entirely
    solo[pair] = {key: rubric_score(subset[pair][f"{key}_event_mean"], RADAR_RUBRIC[key][1])
                  for key in RADAR_KEYS}
check("removing the other pair leaves every score bit-identical",
      all(solo[pair][key] == score[pair][key] for pair in PAIRS for key in RADAR_KEYS))
third = {"x": (0.5, 12.0, 5.0, 1.0, 1.5), "y": (0.99, 45.0, 1.0, 4.0, 5.0)}
third_scores = {name: [rubric_score(v, RADAR_RUBRIC[k][1]) for k, v in zip(RADAR_KEYS, vals)]
                for name, vals in third.items()}
check("a hypothetical third dataset scores on the same fixed scale",
      all(0.0 <= s <= 1.0 for row in third_scores.values() for s in row),
      "third A=" + ", ".join(f"{s:.3f}" for s in third_scores["x"])
      + " | third B=" + ", ".join(f"{s:.3f}" for s in third_scores["y"]))

# 2) every score of exactly 1.0 is justified by the anchors alone, not by being the max
for pair in PAIRS:
    for key in RADAR_KEYS:
        if abs(score[pair][key] - 1.0) <= 1e-12:
            regions = ceiling_region(RADAR_RUBRIC[key][1])
            value = raw[pair][key]
            check(f"{pair} {key}: 1.0 is the anchors' own plateau "
                  + ", ".join(f"[{lo:g},{hi:g}]" for lo, hi in regions),
                  in_regions(value, regions), f"raw={value:.4f}")

# 3) contrast with the old pair-max normalisation that this replaced
print("\n  old vs new (old: larger of the pair forced to 1.0 on every axis)")
forced_axes_old = 0
for key in RADAR_KEYS:
    pairs_at_one = [pair for pair in PAIRS if abs(score[pair][key] - 1.0) <= 1e-12]
    values = [raw[p][key] for p in PAIRS]
    if key in ("jerk_rms_smoothed_mps3", "accel_rms_mps2"):   # old code inverted these
        values = [1.0 / v for v in values]
    old_normalised = [v / max(values) for v in values]
    if abs(max(old_normalised) - 1.0) <= 1e-12:
        forced_axes_old += 1
    print(f"    {key:26s} new: " + ", ".join(f"{p}={score[p][key]:.4f}" for p in PAIRS)
          + f"   axes at 1.0: {len(pairs_at_one)}/2"
          + f"   old normalised: " + ", ".join(f"{v:.4f}" for v in old_normalised))
check("old scheme would put the larger pair at exactly 1.0 on all 5 axes "
      "(the problem this change removes)", forced_axes_old == 5,
      f"{forced_axes_old}/5 (larger value normalised, before any rubric)")
saturated = sum(1 for key in RADAR_KEYS if all(abs(score[p][key] - 1.0) <= 1e-12 for p in PAIRS))
check("new scheme: at least one axis separates the two pairs (not all saturated)",
      any(abs(score["HV-HV"][key] - score["HV-AV"][key]) > 1e-6 for key in RADAR_KEYS),
      f"axes where BOTH pairs are at 1.0: {saturated}/5")
check("HV-HV (human follower) polygon is not all-ones",
      any(score["HV-HV"][key] < 1.0 for key in RADAR_KEYS))
print(f"\nverification: {'ALL PASS' if OK else 'FAILURES PRESENT'}")
raise SystemExit(0 if OK else 1)
