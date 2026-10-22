#!/usr/bin/env python3
"""INTR5100 Lab Session I - validation and driving-style metrics.

Two metric families are produced for every event, human or validation:

A. prediction metrics - the run's identified IDM against the measurement.
   The simulation is ``idm_core.simulate_event`` (first measured ``(s, v)`` as the
   initial state, measured ``lv`` as the leader input, real time deltas) and the
   error metrics are ``cross_validate.regression_metrics`` - the cross-validation
   convention, imported and reused, never re-implemented. Both series are reported
   with MSE / RMSE / MAE / MAPE / R2:
   spacing is ``s = dhw - L_VEH`` (``L_VEH = 250``) and speed is the follower
   speed ``fv`` of the stage 4 table. Units: coordinate units (cu) and cu/s.

B. driving-style metrics - computed on the MEASURED trajectory only, so a human
   run and a validation run are described by the same yardstick. Three groups:

   efficiency  mean speed, speed achievement ``mean(v)/v0`` (v0 = the run's
               calibrated free speed), total and mean event duration, stop
               fraction (share of samples with ``v <= STOP_SPEED_THRESHOLD_CU_S``);
   comfort     mean / std / RMS of the longitudinal acceleration and of its
               first difference (jerk) - central difference with one-sided ends,
               the same ``idm_core.observed_acceleration`` convention stage 4
               uses for its speeds - plus the share of samples above the absolute
               acceleration limits and above ``2`` / ``3`` event sigma_a;
   safety      minimum spacing, minimum time-to-collision (TTC = s/dv while
               ``dv = lv - fv > 0``, ``inf`` when the leader is not closing),
               the share of samples below the TTC thresholds, and the number of
               samples with ``s <= 0`` (the violation convention of idm_core).

Every threshold is a module constant and is published through
:func:`thresholds` / :data:`THRESHOLD_DERIVATION`, so the HTML report can list
them with their basis instead of hiding them in the code.

Units: everything stays in the coordinate units of the stage 4 table (cu for a
distance, cu/s for a speed, cu/s^2 for an acceleration, cu/s^3 for a jerk), as
the earlier stages and the calibration do. ``L_VEH = 250`` cu is one vehicle
length by the lab brief, and the metadata of a collection defines its own
``max_accel_per_sec`` / ``max_decel_per_sec`` / ``min_spacing`` in the same
parameter system; the acceleration thresholds below use both anchors and the
report states why the sigma-relative one is the unit-free one.

Usage:
    python src/analysis/metrics.py --run run_1
    python src/analysis/metrics.py --run run_1 --validation-dir data/preprocess/validation_run_1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- import bootstrap (the project is a set of scripts, not a package) -------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/analysis/metrics.py -> lab1
_SRC = PROJECT_ROOT / "src"
_CALIBRATE = _SRC / "calibration"
for _entry in (str(_SRC), str(_CALIBRATE)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

import idm_core as core  # noqa: E402  (import after the path fix)

# --- threshold configuration (the single place every threshold lives) --------

# A sample counts as "stopped" at or below this speed (cu/s).
STOP_SPEED_THRESHOLD_CU_S = 1.0

# Absolute longitudinal-acceleration limits (cu/s^2) whose exceedance share is
# reported. 200 cu/s^2 is the upper end of the calibration box of the `a`
# parameter (idm_core.UPPER[3]), i.e. the strongest acceleration the identified
# model can produce in exactly these units - the unit-consistent anchor. 50 and
# 100 cu/s^2 are the tape's own acceleration ceilings from the collection
# metadata (idm_params.max_accel_per_sec / max_decel_per_sec) and the box
# midpoint; see THRESHOLD_DERIVATION.
ACCEL_ABS_LIMITS_CU_S2 = (50.0, 100.0, 200.0)

# Relative acceleration limits, in units of that event's own sigma_a.
ACCEL_SIGMA_LEVELS = (2.0, 3.0)

# TTC below these values counts as a critical (1.5 s) / warning (3.0 s) sample.
TTC_THRESHOLDS_S = (1.5, 3.0)

# TTC is only defined while the leader closes (dv = lv - fv > this).
TTC_MIN_DV_CU_S = 1e-9

# s <= 0 is the spacing-violation convention of idm_core (one vehicle length is
# already subtracted from dhw, so s is the bumper-to-bumper gap).
SPACING_VIOLATION_CU = 0.0

# Reported (not scored) reference gap: the collection metadata's min_spacing.
MIN_SPACING_WARN_CU = 80.0

# Distance bands whose sample count the safety comparison reports: 100 cu is the
# band quoted for the validation runs, 80 cu is the metadata min_spacing = the
# anti-collision spacing the vehicle itself used.
SPACING_BAND_CU = (100.0, 80.0)

# Scales of the normalised 0-1 style scores (see style_scores): each is the
# value at which the corresponding indicator saturates.
SCORE_ACCEL_SCALE_CU_S2 = 200.0
SCORE_JERK_SCALE_CU_S3 = 2000.0
SCORE_SPACING_SCALE_CU = 250.0
SCORE_TTC_SCALE_S = 3.0

# How each threshold above is justified. Printed by --thresholds and listed in
# the HTML report, so no threshold is applied without its basis being stated.
THRESHOLD_DERIVATION = {
    "stop_speed_cu_s": (
        "1 cu/s, about 0.7 percent of the calibrated free speed (-140 cu/s); a "
        "sample at or below it is treated as a standstill"
    ),
    "accel_abs_limits_cu_s2": (
        "50 / 100 / 200 cu/s^2: 50 is metadata.json idm_params.max_accel_per_sec, "
        "100 the midpoint of the calibration box of `a`, 200 its upper end "
        "(idm_core.UPPER[3]) and therefore the strongest acceleration the "
        "identified model can produce in these units"
    ),
    "accel_sigma_levels": (
        "2 and 3 event sigma_a: unit-free, so it needs no physical conversion and "
        "is the criterion used in the style scores"
    ),
    "ttc_thresholds_s": (
        "1.5 s critical and 3.0 s warning, the two thresholds of the usual TTC "
        "surrogate-safety literature; TTC is in seconds and therefore unit-free"
    ),
    "spacing_violation_cu": (
        "s <= 0, the same violation convention idm_core counts during a simulation"
    ),
    "min_spacing_warn_cu": (
        "80 cu, metadata.json idm_params.min_spacing (reported, not scored)"
    ),
}

# Aggregation rules of the run-level style summary (sample-weighted, so a
# short event cannot outvote a long one).
WEIGHTED_MEAN_FIELDS = (
    "mean_speed_cu_s",
    "speed_achievement",
    "speed_std_cu_s",
    "stop_fraction",
    "accel_mean_cu_s2",
    "accel_std_cu_s2",
    "accel_rms_cu_s2",
    "jerk_rms_cu_s3",
    "ttc_defined_share",
    "ttc_1.5_share",
    "ttc_3.0_share",
    "frac_accel_gt_2sigma",
    "frac_accel_gt_3sigma",
    "frac_accel_gt_50_cu_s2",
    "frac_accel_gt_100_cu_s2",
    "frac_accel_gt_200_cu_s2",
    "spacing_below_100_cu_share",
    "spacing_below_80_cu_share",
)
MIN_FIELDS = ("spacing_min_cu", "ttc_min_s")
MAX_FIELDS = ("accel_abs_max_cu_s2", "jerk_abs_max_cu_s3")
SUM_FIELDS = (
    "spacing_violation_count",
    "spacing_below_100_cu_count",
    "spacing_below_80_cu_count",
    "duration_s",
    "n_samples",
    "ttc_closing_samples",
)

STAGE4_SUBDIR = Path("data") / "preprocess" / "stage_4"
FINAL_PARAMS_FILENAME = "final_params.json"
# Name prefix of the isolated validation tree (see src/validate_historical_model.py).
VALIDATION_PREFIX = "validation_"
EVENT_FILE_RE = core.EVENT_FILE_RE

# Raw-collection file names (a validation collection has the same layout as a
# human collection; ``commands.csv`` only exists for the automated IDM runs).
RAW_FILENAME = "straight_trajectory.csv"
METADATA_FILENAME = "metadata.json"
COMMANDS_FILENAME = "commands.csv"

# A sample counts as "moving" for the speed-unit estimate above this speed.
UNIT_RATIO_MIN_SPEED_CU_S = 5.0
# Speeds below this (vehicle-reported units) are excluded from the estimate.
UNIT_RATIO_MIN_REPORTED_SPEED = 1.0


# --- thresholds ---------------------------------------------------------------


def thresholds() -> dict[str, object]:
    """The threshold configuration plus its derivation, for the report."""
    return {
        "stop_speed_cu_s": STOP_SPEED_THRESHOLD_CU_S,
        "accel_abs_limits_cu_s2": list(ACCEL_ABS_LIMITS_CU_S2),
        "accel_sigma_levels": list(ACCEL_SIGMA_LEVELS),
        "ttc_thresholds_s": list(TTC_THRESHOLDS_S),
        "ttc_min_dv_cu_s": TTC_MIN_DV_CU_S,
        "spacing_violation_cu": SPACING_VIOLATION_CU,
        "min_spacing_warn_cu": MIN_SPACING_WARN_CU,
        "score_scales": {
            "accel_cu_s2": SCORE_ACCEL_SCALE_CU_S2,
            "jerk_cu_s3": SCORE_JERK_SCALE_CU_S3,
            "spacing_cu": SCORE_SPACING_SCALE_CU,
            "ttc_s": SCORE_TTC_SCALE_S,
        },
        "derivation": dict(THRESHOLD_DERIVATION),
        "units": {
            "spacing": "coordinate units (cu), s = dhw - L_VEH",
            "speed": "cu/s",
            "acceleration": "cu/s^2",
            "jerk": "cu/s^3",
            "time": "s",
        },
    }


# --- loading ------------------------------------------------------------------


def load_final_params(root: str | Path | None = None) -> dict:
    """Read ``results/final_params.json`` (the per-run identified IDM)."""
    path = core.resolve_root(root) / core.RESULTS_SUBDIR / FINAL_PARAMS_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"final parameters not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def runs_with_final_params(root: str | Path | None = None) -> list[str]:
    """Run labels that have a final parameter vector, sorted."""
    payload = load_final_params(root)
    return sorted(
        run
        for run, entry in (payload.get("runs") or {}).items()
        if entry.get("theta")
    )


def theta_for_run(run: str, root: str | Path | None = None) -> tuple[np.ndarray, dict]:
    """``(theta, final_params_entry)`` of one run, validated against the box."""
    payload = load_final_params(root)
    entry = (payload.get("runs") or {}).get(run)
    if entry is None:
        raise KeyError(
            f"{run} has no entry in {FINAL_PARAMS_FILENAME} "
            f"(available: {sorted((payload.get('runs') or {}))})"
        )
    theta_values = entry.get("theta")
    if not theta_values:
        raise ValueError(f"{run} has no usable parameter vector in {FINAL_PARAMS_FILENAME}")
    theta = core.theta_from_dict({name: float(theta_values[name]) for name in core.PARAM_NAMES})
    if not core.inside_bounds(theta, tol=1e-6):
        raise ValueError(f"{run}: the stored parameters leave the calibration box")
    return theta, entry


def load_stage4_events(
    root: str | Path | None = None,
    subdir: Path = STAGE4_SUBDIR,
) -> dict[str, list[tuple[str, pd.DataFrame]]]:
    """Read a stage 4 directory as ``run -> [(event_name, DataFrame), ...]``.

    Thin alias of ``idm_core.load_events`` so the analysis side has one loader for
    both the human tables (``data/preprocess/stage_4``) and the validation tables
    (``data/preprocess/validation_<run>/stage_4``).
    """
    return core.load_events(core.resolve_root(root), subdir)


# --- raw collection: unit scale and on-board controller log -------------------


def collection_metadata(raw_dir: str | Path) -> dict:
    """Read ``metadata.json`` of a raw/validation collection."""
    path = Path(raw_dir) / METADATA_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"collection metadata not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def recorded_theta(raw_dir: str | Path) -> tuple[np.ndarray, dict]:
    """The IDM parameter vector the recording's metadata says was on board.

    A validation collection records the controller parameters it was driven
    with in ``metadata.json: idm_params``; this is that vector, as recorded and
    in the units of the controller (NOT necessarily the units of the recorded
    trajectory - see :func:`speed_unit_ratio`).
    """
    params = collection_metadata(raw_dir).get("idm_params") or {}
    missing = [name for name in core.PARAM_NAMES if name not in params]
    if missing:
        raise ValueError(f"{raw_dir}: metadata idm_params missing {missing}")
    theta = core.theta_from_dict({name: float(params[name]) for name in core.PARAM_NAMES})
    return theta, params


def speed_unit_ratio(raw_file: str | Path) -> dict[str, object]:
    """Measured ratio (coordinate speed) / (vehicle-reported speed) of a collection.

    The recorded trajectory table carries two independent speed measurements:
    ``x``, ``y`` (positions, whose difference gives a speed in the coordinate
    units of the trajectory) and ``velocity`` (the value the vehicle itself
    reports). Stage 4 keeps the first and discards the second, because they do
    not agree - this function measures by how much, per role, so the report can
    quote a number instead of an impression.

    Returns per-role sample counts, medians and means of the ratio plus a single
    ``ratio_median`` over both roles (the conversion factor used to express a
    controller speed in the trajectory's coordinate units).
    """
    frame = pd.read_csv(raw_file, low_memory=False)
    for column in ("role", "t_from_start", "x", "y", "velocity"):
        if column not in frame.columns:
            raise ValueError(f"{raw_file}: missing column {column!r}")
    per_role: dict[str, object] = {}
    ratios: list[np.ndarray] = []
    for role in ("follower", "leader"):
        block = frame[frame["role"] == role].sort_values("t_from_start")
        if len(block) < 3:
            continue
        time = block["t_from_start"].to_numpy(dtype=float)
        step = np.diff(time)
        distance = np.hypot(np.diff(block["x"].to_numpy(dtype=float)),
                           np.diff(block["y"].to_numpy(dtype=float)))
        speed_cu = np.concatenate([[np.nan], distance / np.where(step > 0.0, step, np.nan)])
        reported = block["velocity"].to_numpy(dtype=float)
        usable = (
            np.isfinite(speed_cu)
            & (speed_cu > UNIT_RATIO_MIN_SPEED_CU_S)
            & (reported > UNIT_RATIO_MIN_REPORTED_SPEED)
        )
        role_ratio = speed_cu[usable] / reported[usable]
        if role_ratio.size:
            ratios.append(role_ratio)
        per_role[role] = {
            "samples": int(len(block)),
            "usable_samples": int(role_ratio.size),
            "speed_cu_s_mean": float(np.nanmean(speed_cu)) if np.isfinite(speed_cu).any() else None,
            "speed_cu_s_max": float(np.nanmax(speed_cu)) if np.isfinite(speed_cu).any() else None,
            "reported_speed_mean": float(np.mean(reported)),
            "reported_speed_max": float(np.max(reported)),
            "ratio_median": float(np.median(role_ratio)) if role_ratio.size else None,
            "ratio_mean": float(np.mean(role_ratio)) if role_ratio.size else None,
        }
    pooled = np.concatenate(ratios) if ratios else np.array([])
    return {
        "file": str(raw_file).replace("\\", "/"),
        "per_role": per_role,
        "usable_samples": int(pooled.size),
        "ratio_median": float(np.median(pooled)) if pooled.size else None,
        "ratio_mean": float(np.mean(pooled)) if pooled.size else None,
        "minimum_speed_cu_s": UNIT_RATIO_MIN_SPEED_CU_S,
        "note": (
            "median over both roles of (position-difference speed in coordinate units per "
            "second) / (the vehicle-reported velocity column of the same sample), restricted "
            "to samples moving faster than "
            f"{UNIT_RATIO_MIN_SPEED_CU_S:g} cu/s and reporting more than "
            f"{UNIT_RATIO_MIN_REPORTED_SPEED:g}"
        ),
    }


def trajectory_frame_theta(
    theta_recorded,
    speed_ratio: float,
    scale_acceleration: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Express an on-board parameter vector in the trajectory's coordinate frame.

    The calibration of this lab is done in the coordinate units (cu) of the
    recorded trajectory: spacing in cu, speed in cu/s. A controller parameter
    vector that was *executed on the vehicle* is only comparable with a
    trajectory if its speed-like quantities are converted with the measured
    unit ratio :func:`speed_unit_ratio`: ``v0_cu = ratio * v0_recorded``.

    ``scale_acceleration`` additionally multiplies ``a`` and ``b`` by the same
    ratio; that is the other self-consistent reading of the on-board controller
    (its acceleration had to act on a speed that was itself in the smaller
    unit), and the report quotes both readings rather than hiding the choice.
    The vector is NOT required to stay inside the calibration search box: the
    box is the region the identification searched, not a physical limit.
    """
    recorded = core.as_theta(theta_recorded)
    frame = recorded.copy()
    frame[0] = recorded[0] * float(speed_ratio)
    if scale_acceleration:
        frame[3] = recorded[3] * float(speed_ratio)
        frame[4] = recorded[4] * float(speed_ratio)
    note = (
        f"v0_cu = {frame[0]:.4f} = {float(speed_ratio):.4f} * {recorded[0]:.4f}"
        + (
            f"; a, b also scaled: a = {frame[3]:.4f}, b = {frame[4]:.4f}"
            if scale_acceleration
            else "; T, s0, a, b taken as recorded"
        )
    )
    return frame, {
        "speed_ratio": float(speed_ratio),
        "scale_acceleration": bool(scale_acceleration),
        "theta_recorded": core.theta_to_dict(recorded),
        "theta_frame": core.theta_to_dict(frame),
        "within_calibration_box": bool(core.inside_bounds(frame)),
        "note": note,
    }


def controller_log(raw_dir: str | Path) -> dict[str, object] | None:
    """Summary of the on-board IDM controller log (``commands.csv``).

    The automated runs record every control step of the vehicle's own IDM
    controller: the measured gap and spacing, the speed it asked for, the
    acceleration it applied and its emergency flag. That log is direct evidence
    of what the controller did, independent of the trajectory reconstruction, so
    the safety comparison of the two validation runs quotes it. Returns ``None``
    when the collection has no (non-empty) ``commands.csv`` - a human run has
    none.
    """
    path = Path(raw_dir) / COMMANDS_FILENAME
    if not path.is_file() or path.stat().st_size == 0:
        return None
    frame = pd.read_csv(path)
    if frame.empty or "emergency" not in frame.columns:
        return None
    emergency = frame["emergency"].to_numpy(dtype=float) > 0.0
    # Contiguous emergency episodes = the number of anti-collision activations.
    episodes: list[list[float]] = []
    index = 0
    times = frame["t_from_start"].to_numpy(dtype=float)
    while index < len(emergency):
        if emergency[index]:
            start = index
            while index < len(emergency) and emergency[index]:
                index += 1
            episodes.append([float(times[start]), float(times[index - 1])])
        else:
            index += 1
    summary: dict[str, object] = {
        "file": str(path).replace("\\", "/"),
        "control_samples": int(len(frame)),
        "mode_counts": {str(key): int(value) for key, value in frame["mode"].value_counts().items()},
        "idm_samples": int((frame["mode"] == "idm_straight").sum()) if "mode" in frame else None,
        "emergency_samples": int(np.count_nonzero(emergency)),
        "emergency_episodes": len(episodes),
        "emergency_windows_s": episodes,
        "spacing_min_cu": float(frame["spacing"].min()),
        "spacing_mean_cu": float(frame["spacing"].mean()),
        "gap_min_cu": float(frame["gap"].min()),
        "dt_min_s": float(frame["dt"].min()),
        "dt_max_s": float(frame["dt"].max()),
    }
    for column in ("desired_velocity", "v_cmd", "a_idm"):
        if column in frame.columns:
            values = frame[column].to_numpy(dtype=float)
            summary[f"{column}_min"] = float(np.nanmin(values))
            summary[f"{column}_mean"] = float(np.nanmean(values))
            summary[f"{column}_max"] = float(np.nanmax(values))
    return summary


# --- metric A: prediction -----------------------------------------------------


def regression_metrics(simulated, observed) -> dict[str, object]:
    """MSE / RMSE / MAE / MAPE / R2, reusing the cross-validation implementation.

    Imported lazily from ``cross_validate`` (which pulls in the five calibration
    methods and torch) so a style-only run does not pay for it.
    """
    import cross_validate as cv  # noqa: PLC0415  (lazy: heavy import)

    return cv.regression_metrics(simulated, observed)


def event_prediction_metrics(
    frame: pd.DataFrame,
    theta,
    name: str = "",
    method: str = core.DEFAULT_METHOD,
) -> dict[str, object]:
    """Model-versus-measurement block of one event (cross-validation convention)."""
    data = core.to_event_data(frame, name)
    spacing, speed, status = core.simulate_event(data, theta, method, return_status=True)
    feasible = not status["diverged"] and bool(status["finite"])
    block: dict[str, object] = {
        "event": name,
        "n_samples": data.n,
        "duration_s": float(data.t[-1] - data.t[0]),
        "method": method,
        "feasible": feasible,
        "diverged": bool(status["diverged"]),
        "s_violations": int(status["s_violations"]),
        "sigma_s_cu": data.sigma_s,
        "sigma_v_cu_s": data.sigma_v,
        "spacing": regression_metrics(spacing, data.s_obs),
        "speed": regression_metrics(speed, data.v_obs),
    }
    block["simulated"] = {"spacing_cu": spacing, "speed_cu_s": speed}
    block["observed"] = {
        "time_s": data.t,
        "spacing_cu": data.s_obs,
        "speed_cu_s": data.v_obs,
        "leader_speed_cu_s": data.vL_obs,
    }
    return block


def strip_series(block: dict[str, object]) -> dict[str, object]:
    """Drop the raw arrays from a prediction block so it can be written to JSON."""
    return {key: value for key, value in block.items() if key not in ("simulated", "observed")}


# --- metric B: driving style --------------------------------------------------


def event_style_metrics(
    time,
    speed,
    spacing,
    leader_speed,
    v0: float,
    name: str = "",
) -> dict[str, object]:
    """Efficiency / comfort / safety metrics of one MEASURED trajectory.

    ``time``, ``speed`` and ``spacing`` are the follower's measured series;
    ``leader_speed`` is the measured ``lv``; ``v0`` is the run's calibrated
    free speed, used only by the speed-achievement ratio. All thresholds are the
    module constants; nothing is hard-coded here.
    """
    t = np.asarray(time, dtype=float)
    v = np.asarray(speed, dtype=float)
    s = np.asarray(spacing, dtype=float)
    vl = np.asarray(leader_speed, dtype=float)
    if not (t.size == v.size == s.size == vl.size):
        raise ValueError("time, speed, spacing and leader speed must have the same length")
    if t.size < 2:
        raise ValueError(f"{name}: at least 2 samples are needed for style metrics")
    n = int(t.size)

    data = core.EventData(name=name, t=t, s_obs=s, v_obs=v, vL_obs=vl)
    accel = core.observed_acceleration(data)
    accel_abs = np.abs(accel)
    sigma_a = float(np.std(accel))
    jerk = np.diff(accel) / np.diff(t)

    dv = vl - v
    closing = dv > TTC_MIN_DV_CU_S
    ttc = np.full(n, np.inf)
    ttc[closing] = s[closing] / dv[closing]
    finite_ttc = np.isfinite(ttc)

    mean_speed = float(np.mean(v))
    metrics: dict[str, object] = {
        "event": name,
        "n_samples": n,
        "duration_s": float(t[-1] - t[0]),
        # efficiency
        "mean_speed_cu_s": mean_speed,
        "speed_std_cu_s": float(np.std(v)),
        "speed_achievement": float(mean_speed / v0) if v0 > 0.0 else None,
        "v0_cu_s": float(v0),
        "stop_fraction": float(np.mean(v <= STOP_SPEED_THRESHOLD_CU_S)),
        "stop_samples": int(np.count_nonzero(v <= STOP_SPEED_THRESHOLD_CU_S)),
        # comfort
        "accel_mean_cu_s2": float(np.mean(accel)),
        "accel_std_cu_s2": sigma_a,
        "accel_rms_cu_s2": float(np.sqrt(np.mean(accel**2))),
        "accel_abs_max_cu_s2": float(np.max(accel_abs)),
        "sigma_a_cu_s2": sigma_a,
        "jerk_rms_cu_s3": float(np.sqrt(np.mean(jerk**2))),
        "jerk_abs_max_cu_s3": float(np.max(np.abs(jerk))),
        # safety
        "spacing_min_cu": float(np.min(s)),
        "spacing_mean_cu": float(np.mean(s)),
        "spacing_violation_count": int(np.count_nonzero(s <= SPACING_VIOLATION_CU)),
        "ttc_min_s": float(np.min(ttc[finite_ttc])) if finite_ttc.any() else None,
        "ttc_closing_samples": int(np.count_nonzero(finite_ttc)),
        "ttc_defined_share": float(np.mean(finite_ttc)),
    }
    for level in ACCEL_SIGMA_LEVELS:
        key = f"frac_accel_gt_{level:g}sigma"
        metrics[key] = float(np.mean(accel_abs > level * sigma_a)) if sigma_a > 0.0 else 0.0
    for limit in ACCEL_ABS_LIMITS_CU_S2:
        metrics[f"frac_accel_gt_{limit:g}_cu_s2"] = float(np.mean(accel_abs > limit))
    for threshold in TTC_THRESHOLDS_S:
        # One decimal is kept in the key name so it matches the aggregation list.
        metrics[f"ttc_{threshold:.1f}_share"] = float(np.mean(ttc < threshold))
    for band in SPACING_BAND_CU:
        below = int(np.count_nonzero(s < band))
        metrics[f"spacing_below_{band:g}_cu_count"] = below
        metrics[f"spacing_below_{band:g}_cu_share"] = float(below) / n
    return metrics


def aggregate_style(per_event: dict[str, dict[str, object]]) -> dict[str, object]:
    """Run-level style summary of one run, sample-weighted over its events.

    Means and fractions are weighted by the event's sample count (a 211-sample
    event must not be outvoted by a 27-sample one), minima are minima, maxima are
    maxima, counts and durations are summed. Events whose series are missing a
    field are skipped for that field only.
    """
    if not per_event:
        raise ValueError("no events to aggregate")
    weights = {name: float(metrics["n_samples"]) for name, metrics in per_event.items()}
    total_weight = sum(weights.values())
    summary: dict[str, object] = {"n_events": len(per_event), "n_samples": int(total_weight)}

    for field in WEIGHTED_MEAN_FIELDS:
        values = [
            (float(metrics[field]), weights[name])
            for name, metrics in per_event.items()
            if metrics.get(field) is not None
        ]
        if not values:
            summary[field] = None
            continue
        weight_sum = sum(weight for _value, weight in values)
        summary[field] = float(
            sum(value * weight for value, weight in values) / weight_sum
        )
    for field in MIN_FIELDS:
        values = [float(metrics[field]) for metrics in per_event.values() if metrics.get(field) is not None]
        summary[field] = min(values) if values else None
    for field in MAX_FIELDS:
        values = [float(metrics[field]) for metrics in per_event.values() if metrics.get(field) is not None]
        summary[field] = max(values) if values else None
    for field in SUM_FIELDS:
        if field == "n_samples":
            continue
        summary[field] = float(sum(float(metrics.get(field) or 0.0) for metrics in per_event.values()))
    summary["duration_mean_s"] = float(summary["duration_s"]) / len(per_event)
    summary["duration_total_s"] = float(summary["duration_s"])
    summary["aggregation"] = (
        "sample-weighted mean of the per-event means and fractions, min of minima, "
        "max of maxima, sum of counts and durations"
    )
    summary["events"] = sorted(per_event)
    return summary


# --- normalised style scores --------------------------------------------------


def _clamp01(value: float) -> float:
    """Clamp to [0, 1] (the radar/bar axes of the report)."""
    return float(min(1.0, max(0.0, value)))


def style_scores(style: dict[str, object]) -> dict[str, object]:
    """Normalised 0-1 efficiency / comfort / safety scores of one aggregated run.

    Every formula is explicit and reported, so the radar chart cannot be read as
    an opaque ranking:

        efficiency = clamp(mean(v) / v0)                    closer to the run's own
                                                           free speed is better
        comfort    = 1 - clamp(accel_rms / 200 cu/s^2)      less violent motion is
                                                           better (200 cu/s^2 = the
                                                           calibration box upper end)
        safety     = mean of clamp(s_min / 250 cu) and     larger margins are better
                     clamp(ttc_min / 3.0 s), and 0 when a
                     spacing violation occurred

    A run with no closing sample at all (TTC undefined everywhere) scores the TTC
    half as 1.0: no closing situation means no collision risk was observed.
    """
    achievement = style.get("speed_achievement")
    efficiency = _clamp01(float(achievement)) if achievement is not None else 0.0
    accel_rms = float(style.get("accel_rms_cu_s2") or 0.0)
    comfort = 1.0 - _clamp01(accel_rms / SCORE_ACCEL_SCALE_CU_S2)
    spacing_min = float(style.get("spacing_min_cu") or 0.0)
    ttc_min = style.get("ttc_min_s")
    spacing_score = _clamp01(spacing_min / SCORE_SPACING_SCALE_CU)
    ttc_score = _clamp01(float(ttc_min) / SCORE_TTC_SCALE_S) if ttc_min is not None else 1.0
    safety = 0.5 * spacing_score + 0.5 * ttc_score
    if float(style.get("spacing_violation_count") or 0.0) > 0.0:
        safety = 0.0
    jerk_rms = float(style.get("jerk_rms_cu_s3") or 0.0)
    return {
        "efficiency": efficiency,
        "comfort": comfort,
        "safety": safety,
        "sub_scores": {
            "speed achievement": efficiency,
            "non-stop share": 1.0 - _clamp01(float(style.get("stop_fraction") or 0.0)),
            "accel RMS margin": comfort,
            "jerk RMS margin": 1.0 - _clamp01(jerk_rms / SCORE_JERK_SCALE_CU_S3),
            "min spacing margin": spacing_score,
            "min TTC margin": ttc_score,
        },
        "formula": (
            "efficiency = clamp(mean(v)/v0); comfort = 1 - clamp(accel_rms/200); "
            "safety = 0.5*clamp(s_min/250) + 0.5*clamp(ttc_min/3.0), forced to 0 by any s<=0"
        ),
    }


# --- per-run evaluation ----------------------------------------------------


def aggregate_prediction(per_event: dict[str, dict[str, object]]) -> dict[str, object]:
    """Equal-weight mean over events of the prediction error metrics.

    Equal weight (not sample weight) keeps the number comparable with the
    cross-validation aggregation of the calibration step; R2 is left per event
    because a mean of R2 is not a coefficient of determination.
    """
    if not per_event:
        raise ValueError("no events to aggregate")
    summary: dict[str, object] = {"n_events": len(per_event)}
    for series in ("spacing", "speed"):
        for metric in ("mse", "rmse", "mae", "mape_percent"):
            values = [
                float(metrics[series][metric])
                for metrics in per_event.values()
                if metrics[series].get(metric) is not None
            ]
            summary[f"{series}_{metric}"] = float(np.mean(values)) if values else None
        summary[f"{series}_r2_per_event"] = {
            name: metrics[series].get("r2") for name, metrics in per_event.items()
        }
    summary["total_s_violations"] = int(
        sum(int(metrics["s_violations"]) for metrics in per_event.values())
    )
    summary["n_samples"] = int(sum(int(metrics["n_samples"]) for metrics in per_event.values()))
    summary["aggregation"] = "equal-weight mean over the events of that run"
    return summary


def evaluate_run(
    root: str | Path | None = None,
    run: str = "run_1",
    subdir: Path = STAGE4_SUBDIR,
    label: str = "human",
    method: str = core.DEFAULT_METHOD,
    keep_series: bool = False,
    theta=None,
    theta_source: object = None,
) -> dict[str, object]:
    """Full metric block of one run (all events of one run in one stage 4 dir).

    Returns the per-event prediction and style blocks, the two run-level
    aggregates, the normalised style scores and the provenance (which directory,
    which parameters). ``keep_series=True`` additionally keeps the simulated and
    measured arrays inside each event block (used for the plots and by the
    validation pipeline, never for JSON output).

    ``theta`` overrides the parameter vector read from ``final_params.json`` and
    ``theta_source`` describes where it came from instead; that is how a
    validation run is replayed with the parameter vector it was actually driven
    with (``metadata.json: idm_params``, possibly converted into the trajectory
    frame - see :func:`trajectory_frame_theta`).
    """
    resolved = core.resolve_root(root)
    if theta is None:
        theta, final_entry = theta_for_run(run, resolved)
        source: object = final_entry.get("method")
    else:
        theta = core.as_theta(theta)
        source = theta_source if theta_source is not None else "caller-supplied parameter vector"
    grouped = load_stage4_events(resolved, subdir)
    if run not in grouped:
        raise FileNotFoundError(
            f"{subdir}: no event table for {run} "
            f"(found: {sorted(grouped)})"
        )
    events = grouped[run]

    per_event: dict[str, dict[str, object]] = {}
    for name, frame in events:
        prediction = event_prediction_metrics(frame, theta, name, method)
        style = event_style_metrics(
            frame["time"].to_numpy(dtype=float),
            frame["fv"].to_numpy(dtype=float),
            frame["dhw"].to_numpy(dtype=float) - core.L_VEH,
            frame["lv"].to_numpy(dtype=float),
            v0=float(theta[0]),
            name=name,
        )
        block: dict[str, object] = {"prediction": prediction, "style": style}
        if not keep_series:
            block["prediction"] = strip_series(prediction)
        per_event[name] = block

    style_only = {name: block["style"] for name, block in per_event.items()}
    prediction_only = {name: block["prediction"] for name, block in per_event.items()}
    style_aggregate = aggregate_style(style_only)
    return {
        "label": label,
        "run": run,
        "stage4_dir": core.rel(resolved / subdir, resolved),
        "method": method,
        "L_VEH": core.L_VEH,
        "delta": core.DELTA,
        "theta": core.theta_to_dict(theta),
        "theta_source": source,
        "theta_within_calibration_box": bool(core.inside_bounds(theta)),
        "events": per_event,
        "n_events": len(per_event),
        "prediction_aggregate": aggregate_prediction(prediction_only),
        "style_aggregate": style_aggregate,
        "style_scores": style_scores(style_aggregate),
        "thresholds": thresholds(),
    }


# --- comparison and automatic conclusions ------------------------------------


def _pct(value: float) -> str:
    """Signed percentage with one decimal (a comparison sentence helper)."""
    return f"{value:+.1f}%"


def _relative_change(reference: float | None, other: float | None) -> float | None:
    """``100 * (other - reference) / |reference|`` or None when undefined."""
    if reference is None or other is None or reference == 0.0:
        return None
    return 100.0 * (float(other) - float(reference)) / abs(float(reference))


# Public alias: the report compares the same way the conclusions do.
relative_change = _relative_change


def compare_runs(human: dict[str, object], validation: dict[str, object]) -> dict[str, object]:
    """Numeric comparison of the human and validation style/prediction summaries."""
    hs = human["style_aggregate"]
    vs = validation["style_aggregate"]
    hp = human["prediction_aggregate"]
    vp = validation["prediction_aggregate"]
    fields = (
        "mean_speed_cu_s",
        "speed_achievement",
        "duration_total_s",
        "stop_fraction",
        "accel_mean_cu_s2",
        "accel_std_cu_s2",
        "accel_rms_cu_s2",
        "jerk_rms_cu_s3",
        "frac_accel_gt_200_cu_s2",
        "frac_accel_gt_2sigma",
        "spacing_min_cu",
        "ttc_min_s",
        "ttc_1.5_share",
        "ttc_3.0_share",
        "spacing_violation_count",
    )
    deltas: dict[str, object] = {}
    for field in fields:
        h_value = hs.get(field)
        v_value = vs.get(field)
        deltas[field] = {
            "human": h_value,
            "validation": v_value,
            "difference": (
                float(v_value) - float(h_value)
                if h_value is not None and v_value is not None
                else None
            ),
            "relative_percent": _relative_change(h_value, v_value),
        }
    prediction: dict[str, object] = {}
    for series in ("spacing", "speed"):
        for metric in ("mse", "rmse", "mae", "mape_percent"):
            h_value = hp.get(f"{series}_{metric}")
            v_value = vp.get(f"{series}_{metric}")
            prediction[f"{series}_{metric}"] = {
                "human": h_value,
                "validation": v_value,
                "relative_percent": _relative_change(h_value, v_value),
            }
    return {
        "style": deltas,
        "prediction": prediction,
        "human_scores": human["style_scores"],
        "validation_scores": validation["style_scores"],
    }


def text_conclusions(
    human: dict[str, object],
    validation: dict[str, object],
    material_percent: float = 5.0,
) -> list[str]:
    """Short English conclusions derived from the numbers, not from adjectives.

    ``material_percent`` is the relative change above which a difference is called
    material; below it the sentence says the two runs agree to within that band.
    """
    hs = human["style_aggregate"]
    vs = validation["style_aggregate"]
    hp = human["prediction_aggregate"]
    vp = validation["prediction_aggregate"]
    sentences: list[str] = []

    def sentence(
        label: str,
        human_value: float | None,
        validation_value: float | None,
        unit: str,
        direction: str = "higher",
        decimals: int = 2,
    ) -> None:
        """One comparison sentence; ``direction`` is higher / lower / neutral."""
        if human_value is None or validation_value is None:
            sentences.append(f"{label}: not available in both runs.")
            return
        human_value = float(human_value)
        validation_value = float(validation_value)
        suffix = f" {unit}" if unit else ""
        if direction == "neutral":
            verdict = "context, not a quality"
        else:
            verdict = f"{direction} is better"
        change = _relative_change(human_value, validation_value)
        if change is None:
            if human_value == 0.0 and validation_value == 0.0:
                sentences.append(f"{label}: 0 in both runs ({verdict}).")
                return
            # A zero reference makes the ratio undefined; report the absolute move
            # instead of silently dropping the sentence.
            sentences.append(
                f"{label}: human {human_value:.{decimals}f}{suffix} vs validation "
                f"{validation_value:.{decimals}f}{suffix} - the human value is zero, so only the "
                f"absolute difference ({validation_value - human_value:+.{decimals}f}{suffix}) "
                f"is defined ({verdict})."
            )
            return
        if abs(change) < material_percent:
            relation = f"agree to within {material_percent:g} percent"
        else:
            relation = f"{'higher' if change > 0 else 'lower'} by {abs(change):.1f} percent"
        if direction == "neutral":
            # A signed quantity without a better/worse direction: the ratio is not
            # informative, so only the absolute move is quoted.
            relation = f"absolute difference {validation_value - human_value:+.{decimals}f}{suffix}"
        sentences.append(
            f"{label}: human {human_value:.{decimals}f}{suffix} vs validation "
            f"{validation_value:.{decimals}f}{suffix} - {relation} ({verdict})."
        )

    sentence(
        "Mean follower speed",
        hs.get("mean_speed_cu_s"),
        vs.get("mean_speed_cu_s"),
        "cu/s",
        "higher",
        2,
    )
    sentence(
        "Speed achievement mean(v)/v0",
        hs.get("speed_achievement"),
        vs.get("speed_achievement"),
        "",
        "higher",
        3,
    )
    sentence(
        "Total driving time",
        hs.get("duration_total_s"),
        vs.get("duration_total_s"),
        "s",
        "neutral",
        1,
    )
    sentence(
        "Stop fraction",
        hs.get("stop_fraction"),
        vs.get("stop_fraction"),
        "",
        "lower",
        4,
    )
    sentence(
        "Mean acceleration (signed, near zero)",
        hs.get("accel_mean_cu_s2"),
        vs.get("accel_mean_cu_s2"),
        "cu/s^2",
        "neutral",
        2,
    )
    sentence(
        "Acceleration RMS",
        hs.get("accel_rms_cu_s2"),
        vs.get("accel_rms_cu_s2"),
        "cu/s^2",
        "lower",
        2,
    )
    sentence(
        "Jerk RMS",
        hs.get("jerk_rms_cu_s3"),
        vs.get("jerk_rms_cu_s3"),
        "cu/s^3",
        "lower",
        2,
    )
    sentence(
        "Minimum spacing",
        hs.get("spacing_min_cu"),
        vs.get("spacing_min_cu"),
        "cu",
        "higher",
        1,
    )
    sentence(
        "Minimum TTC",
        hs.get("ttc_min_s"),
        vs.get("ttc_min_s"),
        "s",
        "higher",
        3,
    )
    sentence(
        "Share of samples below TTC = 1.5 s",
        hs.get("ttc_1.5_share"),
        vs.get("ttc_1.5_share"),
        "",
        "lower",
        4,
    )
    sentence(
        "Spacing RMSE of the identified IDM",
        hp.get("spacing_rmse"),
        vp.get("spacing_rmse"),
        "cu",
        "lower",
        1,
    )
    sentence(
        "Speed RMSE of the identified IDM",
        hp.get("speed_rmse"),
        vp.get("speed_rmse"),
        "cu/s",
        "lower",
        2,
    )

    human_scores = human["style_scores"]
    validation_scores = validation["style_scores"]
    for axis in ("efficiency", "comfort", "safety"):
        h_value = float(human_scores[axis])
        v_value = float(validation_scores[axis])
        difference = v_value - h_value
        if abs(difference) < 0.02:
            relation = "practically the same"
        else:
            relation = "better on the validation run" if difference > 0 else "better on the human run"
        sentences.append(
            f"{axis.capitalize()} score: human {h_value:.3f} vs validation {v_value:.3f} "
            f"({difference:+.3f}) - {relation}."
        )

    if float(hp.get("total_s_violations") or 0) > 0 or float(vp.get("total_s_violations") or 0) > 0:
        sentences.append(
            "Spacing violations (s <= 0) during simulation: human "
            f"{int(hp.get('total_s_violations') or 0)}, validation "
            f"{int(vp.get('total_s_violations') or 0)} - a non-zero count means the "
            "reported errors were produced with the spacing clamped to the floor."
        )
    else:
        sentences.append(
            "No simulated spacing violation (s <= 0) occurred in either run, so every "
            "reported error was produced without clamping."
        )
    return sentences


# --- command line -------------------------------------------------------------


def _format(value, decimals: int = 3) -> str:
    """Fixed-point cell text; ``n/a`` when a metric does not exist."""
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, (int,)) or (isinstance(value, float) and float(value).is_integer() and decimals == 0):
        return f"{int(value)}"
    return f"{float(value):.{decimals}f}"


STYLE_COLUMNS = (
    ("mean v", "mean_speed_cu_s", 2),
    ("achv", "speed_achievement", 3),
    ("dur s", "duration_s", 1),
    ("stop", "stop_fraction", 4),
    ("a mean", "accel_mean_cu_s2", 2),
    ("a std", "accel_std_cu_s2", 2),
    ("a rms", "accel_rms_cu_s2", 2),
    ("|a|>200", "frac_accel_gt_200_cu_s2", 4),
    (">2sig", "frac_accel_gt_2sigma", 4),
    ("j rms", "jerk_rms_cu_s3", 1),
    ("s min", "spacing_min_cu", 1),
    ("ttc min", "ttc_min_s", 3),
    ("ttc<1.5", "ttc_1.5_share", 4),
    ("viol", "spacing_violation_count", 0),
)


def print_run(run: dict[str, object]) -> None:
    """Per-event style table of one run on stdout."""
    print(f"{run['label']} / {run['run']}  ({run['stage4_dir']}, {run['n_events']} event(s))")
    header = f"  {'event':<10}{'n':>5}" + "".join(f"{name:>10}" for name, _f, _d in STYLE_COLUMNS)
    print(header)
    for name, block in run["events"].items():
        style = block["style"]
        cells = "".join(
            f"{_format(style.get(field), decimals):>10}" for _label, field, decimals in STYLE_COLUMNS
        )
        print(f"  {name:<10}{int(style['n_samples']):>5}{cells}")
    aggregate = run["style_aggregate"]
    cells = "".join(
        f"{_format(aggregate.get(field), decimals):>10}" for _label, field, decimals in STYLE_COLUMNS
    )
    print(f"  {'AGG':<10}{int(aggregate['n_samples']):>5}{cells}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="project root (default: inferred)")
    parser.add_argument("--run", default="run_1", help="run label (run_1 / run_2)")
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=None,
        help="validation run directory (default: data/preprocess/validation_<run>)",
    )
    parser.add_argument("--method", default=core.DEFAULT_METHOD, choices=list(core.METHODS))
    parser.add_argument("--json-out", type=Path, default=None, help="write the metric blocks to JSON")
    parser.add_argument(
        "--thresholds",
        action="store_true",
        help="print the threshold configuration and exit",
    )
    args = parser.parse_args(argv)

    if args.thresholds:
        print(json.dumps(thresholds(), indent=2, ensure_ascii=False))
        return 0

    root = core.resolve_root(args.root)
    human = evaluate_run(root, args.run, STAGE4_SUBDIR, "human", args.method)
    print_run(human)
    print()

    payload: dict[str, object] = {"human": human}
    validation_subdir = (
        args.validation_dir / "stage_4"
        if args.validation_dir is not None
        else Path("data") / "preprocess" / f"validation_{args.run}" / "stage_4"
    )
    validation = None
    if (root / validation_subdir).is_dir():
        validation = evaluate_run(root, args.run, validation_subdir, "validation", args.method)
        print_run(validation)
        print()
        for line in text_conclusions(human, validation):
            print(f"  - {line}")
        payload["validation"] = validation
        payload["comparison"] = compare_runs(human, validation)
    else:
        print(f"no validation tables at {validation_subdir} - human metrics only")

    if args.json_out is not None:
        path = args.json_out if args.json_out.is_absolute() else root / args.json_out
        core.write_json_atomic(path, payload)
        print(f"written: {core.rel(path, root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
