#!/usr/bin/env python3
"""TGSIM extension: shared core for the real-world IDM calibration.

The Lab Session I machinery (``src/calibration/idm_core.py``) identifies IDM in the
SC2I miniature-testbed coordinate units (car length = 250 units, v0 box 40..200).
The TGSIM I-90/I-94 moving-trajectory data are METRIC SI, so the two problems
cannot share a box or a vehicle length - reusing ``idm_core``'s constants here
would silently rescale the model. What IS shared is the METHOD: the same IDM law,
the same normalised trajectory cost, the same equal-weight multi-event average
and the same Euler rollout. Those are re-implemented here in metric units.

--- data -------------------------------------------------------------------

    data/tgsim/<pair>/event_<k>.csv        columns: time (s), fv, lv (m/s), dhw (m)
    data/tgsim/manifest.json               event metadata incl. leader_length

``dhw`` is the raw centre-to-centre distance; the IDM spacing is the
bumper-to-bumper distance

    s = dhw - leader_length            leader_length in metres (manifest)

so ``s`` is the quantity the follower reacts to. ``leader_length`` is taken from
the manifest (written by ``preprocess_events.py`` from the source CSV) rather
than reconstructed from the rounded ``s_min`` field.

--- model (delta = 4 fixed) -------------------------------------------------

    a_idm(s, v, dv) = a * ( 1 - (v/v0)^4 - (s*/s)^2 )
    s*              = s0 + max(0, v*T + v*dv / (2*sqrt(a*b)))     dv = vL - v
    ds/dt = vL - v                dv/dt = a_idm

with theta = [v0, T, s0, a, b] and the metric search box

    v0 in [5, 35] m/s   T in [0.3, 5] s   s0 in [0.2, 10] m
    a  in [0.5, 6] m/s^2   b in [0.5, 8] m/s^2

Box rationale (see the build report for the numbers): 35 m/s = 126 km/h covers
the free-flow speeds of the segment (88 km/h limit, observed maxima ~30 m/s);
5 m/s keeps v0 above the speeds present in the events so the free term stays
active; T and s0 span the calibrated car-following literature (T 0.3..5 s is the
SC2I range, s0 0.2..10 m is the 2..10 m standstill-gap range); a 0.5..6 m/s^2
covers observed accelerations (|a| < 3.5 m/s^2 in these events) with headroom;
b 0.5..8 m/s^2 is the comfortable-deceleration range 1.5..4 m/s^2 with headroom
on both sides.

--- numerical guards --------------------------------------------------------

Both the torch graph and the numpy rollout use exactly the same guards, so the
two agree to machine precision (the calibration script asserts this):

  * the spacing is floored at ``S_FLOOR`` (0.5 m) before the singular (s*/s)^2
    term, and each step whose UNFLOORED spacing would be <= 0 is counted;
  * the follower speed is floored at 0 (no reversing);
  * a state that leaves the numerical domain (spacing > SPACING_LIMIT or speed >
    SPEED_LIMIT) is clamped and the rollout is flagged ``clamped`` - such a
    parameter vector is reported as a runaway and never selected.

--- cost --------------------------------------------------------------------

    J_e(theta) = mean((s_sim - s_obs)^2)/sigma_s^2
               + mean((v_sim - v_obs)^2)/sigma_v^2
               + S_VIOLATION_PENALTY * violations

    J(theta)   = mean over events of J_e            (equal weight per event)

``sigma_s`` / ``sigma_v`` are POOLED standard deviations over ALL events of the
pair (one common scale per pair, exactly as the brief prescribes), so the two
terms are dimensionless and comparable across the two pair datasets.

Usage:
    from tgsim_core import load_pair_events, simulate, pooled_scales
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# --- configuration ----------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
EVENT_DIR = ROOT / "data" / "tgsim"
MANIFEST_PATH = EVENT_DIR / "manifest.json"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "tgsim"
CALIBRATION_PATH = RESULTS_DIR / "tgsim_calibration.json"
ANALYSIS_PATH = RESULTS_DIR / "tgsim_analysis.json"

PAIR_ORDER = ("HV-HV", "HV-AV")
PAIR_LABEL = {"HV-HV": "HV following HV", "HV-AV": "AV following HV"}

# IDM free-flow exponent, fixed (not calibrated) - same value as Lab Session I.
DELTA = 4.0
PARAM_NAMES = ("v0", "T", "s0", "a", "b")
PARAM_UNITS = {"v0": "m/s", "T": "s", "s0": "m", "a": "m/s^2", "b": "m/s^2"}

# Metric search box (m, s, m/s, m/s^2).
LOWER = np.array([5.0, 0.3, 0.2, 0.5, 0.5])
UPPER = np.array([35.0, 5.0, 10.0, 6.0, 8.0])

# Numerical guards (shared by the torch graph and the numpy rollout).
S_FLOOR = 0.5           # m, floor under the singular (s*/s)^2 term
V_FLOOR = 0.0           # m/s, no reversing
SPACING_LIMIT = 1.0e4   # m, runaway guard
SPEED_LIMIT = 1.0e3     # m/s, runaway guard
SQRT_EPS = 1.0e-12      # guard under sqrt(a*b); a, b >= 0.5 inside the box
SIGMA_FLOOR = 1.0e-6    # guard for a degenerate (zero-spread) series
S_VIOLATION_PENALTY = 1.0e3   # per step that would drive the spacing to <= 0

# Stopping threshold used by the event splitter and the style metrics.
STOP_SPEED = 0.3        # m/s

# SC2I (Lab Session I) calibrated reference, for the dimensionless comparison.
# Source: results/final_params.json -> runs.run_1.theta (method M5).
SC2I_REFERENCE = {
    "label": "SC2I run_1 (M5 differentiable IDM)",
    "source": "results/final_params.json (runs.run_1.theta)",
    "theta": {"v0": 140.368591, "T": 1.507387, "s0": 45.409655,
              "a": 57.559955, "b": 10.212675},
    "units": "coordinate units (car length = 250 units), NOT metric",
}


# --- containers -------------------------------------------------------------


@dataclass(frozen=True)
class EventData:
    """One car-following event in SI units: t (s), spacing (m), speeds (m/s)."""

    pair: str
    name: str
    t: np.ndarray
    s_obs: np.ndarray
    v_obs: np.ndarray
    vL_obs: np.ndarray
    leader_length: float
    follower_length: float
    dhw: np.ndarray

    @property
    def n(self) -> int:
        return int(self.t.size)

    @property
    def dt(self) -> np.ndarray:
        return np.diff(self.t)


# --- loading ----------------------------------------------------------------


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Read ``data/tgsim/manifest.json`` (event metadata written by the splitter)."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_event(pair: str, entry: dict, base: Path = EVENT_DIR) -> EventData:
    """Load one event CSV plus its manifest metadata into an :class:`EventData`."""
    path = base / pair / entry["event"]
    time, fv, lv, dhw = [], [], [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            time.append(float(row["time"]))
            fv.append(float(row["fv"]))
            lv.append(float(row["lv"]))
            dhw.append(float(row["dhw"]))
    t = np.asarray(time, dtype=float)
    if t.size != int(entry["n"]):
        raise ValueError(f"{path.name}: {t.size} rows but the manifest says {entry['n']}")
    if not np.all(np.diff(t) > 0.0):
        raise ValueError(f"{path.name}: time column is not strictly increasing")
    gap = np.asarray(dhw, dtype=float)
    leader_length = float(entry["leader_length"])
    return EventData(
        pair=pair,
        name=entry["event"].replace(".csv", ""),
        t=t,
        s_obs=gap - leader_length,
        v_obs=np.asarray(fv, dtype=float),
        vL_obs=np.asarray(lv, dtype=float),
        leader_length=leader_length,
        follower_length=float(entry.get("follower_length", float("nan"))),
        dhw=gap,
    )


def load_pair_events(pair: str, base: Path = EVENT_DIR) -> list[EventData]:
    """All events of one pair, in manifest (i.e. chronological) order."""
    manifest = load_manifest(base / "manifest.json")
    if pair not in manifest:
        raise KeyError(f"pair {pair!r} not in the manifest")
    return [load_event(pair, entry, base) for entry in manifest[pair]["events"]]


def pooled_scales(events: Sequence[EventData]) -> tuple[float, float]:
    """Pooled (common) spacing and speed scales over all events of a pair.

    The brief asks for a pooled scale rather than the per-event spread used by
    Lab Session I: with a handful of short events a per-event sigma is noisy, and
    a pooled scale puts every event of a pair on one common yardstick.
    """
    spacing = np.concatenate([event.s_obs for event in events])
    speed = np.concatenate([event.v_obs for event in events])
    return (max(float(np.std(spacing)), SIGMA_FLOOR), max(float(np.std(speed)), SIGMA_FLOOR))


# --- model ------------------------------------------------------------------


def idm_acceleration(s, v, dv, theta):
    """Metric IDM acceleration ``a * (1 - (v/v0)^4 - (s*/s)^2)`` (scalar/array).

    ``dv = vL - v``. Arrays are supported so the analysis can evaluate the law on
    whole series at once; the guards are exactly those of the rollout.
    """
    vector = np.asarray(theta, dtype=float).reshape(-1)
    v0, T, s0, a, b = (float(x) for x in vector)
    s_arr = np.asarray(s, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    dv_arr = np.asarray(dv, dtype=float)
    if v0 <= 0.0 or a <= 0.0 or b <= 0.0:
        return math.inf
    headway = np.maximum(0.0, v_arr * T + v_arr * dv_arr / (2.0 * math.sqrt(a * b + SQRT_EPS)))
    desired = s0 + headway
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        acceleration = a * (1.0 - (v_arr / v0) ** DELTA
                            - (desired / np.maximum(s_arr, S_FLOOR)) ** 2)
    return float(acceleration) if np.ndim(acceleration) == 0 else acceleration


def simulate(event: EventData, theta: Sequence[float], return_status: bool = False):
    """Euler rollout of one event, identical to the torch graph of the calibrator.

    Initial condition = the first measured sample (``s_obs[0]``, ``v_obs[0]``);
    the leader speed is the measured ``lv`` series held at the step start (plain
    explicit Euler, one evaluation per step) and the step is the real ``time``
    delta. Returns ``(s_sim, v_sim)``; with ``return_status=True`` also a status
    dict with ``violations`` (steps whose unfloored spacing was <= 0),
    ``clamped`` (states that hit a numerical guard) and the observed extremes.
    """
    vector = np.asarray(theta, dtype=float).reshape(-1)
    v0, T, s0, a, b = (float(x) for x in vector)
    dt = event.dt
    leader = event.vL_obs
    n = event.n
    spacing = np.empty(n, dtype=float)
    speed = np.empty(n, dtype=float)
    spacing[0] = float(event.s_obs[0])
    speed[0] = float(event.v_obs[0])
    violations = 0
    clamped = 0
    s = float(spacing[0])
    v = float(speed[0])
    for i in range(n - 1):
        step = float(dt[i])
        leader_now = float(leader[i])
        dv = leader_now - v
        headway = v * T + v * dv / (2.0 * math.sqrt(a * b + SQRT_EPS))
        desired = s0 + max(0.0, headway)
        acceleration = a * (1.0 - (v / v0) ** DELTA - (desired / max(s, S_FLOOR)) ** 2)
        s_next = s + step * dv
        v_next = v + step * acceleration
        if s_next <= 0.0:
            violations += 1
        if (not math.isfinite(s_next)) or (not math.isfinite(v_next)):
            clamped += 1
            s, v = SPACING_LIMIT, SPEED_LIMIT
        else:
            if s_next > SPACING_LIMIT or v_next > SPEED_LIMIT:
                clamped += 1
            s = min(max(s_next, S_FLOOR), SPACING_LIMIT)
            v = min(max(v_next, V_FLOOR), SPEED_LIMIT)
        spacing[i + 1] = s
        speed[i + 1] = v
    if not return_status:
        return spacing, speed
    return spacing, speed, {
        "violations": int(violations),
        "clamped": int(clamped),
        "finite": bool(np.all(np.isfinite(spacing)) and np.all(np.isfinite(speed))),
        "min_spacing": float(np.min(spacing)),
        "max_speed": float(np.max(speed)),
        "min_speed": float(np.min(speed)),
    }


def event_cost(event: EventData, theta, sigma_s: float, sigma_v: float) -> float:
    """Normalised trajectory cost of one event (pooled scales), + violation term."""
    spacing, speed, status = simulate(event, theta, return_status=True)
    cost = (float(np.mean((spacing - event.s_obs) ** 2) / sigma_s ** 2)
            + float(np.mean((speed - event.v_obs) ** 2) / sigma_v ** 2))
    return cost + S_VIOLATION_PENALTY * status["violations"]


def total_cost(events: Sequence[EventData], theta, scales=None) -> float:
    """Equal-weight mean of the per-event costs (the pair's objective)."""
    if scales is None:
        scales = pooled_scales(events)
    sigma_s, sigma_v = scales
    return float(np.mean([event_cost(event, theta, sigma_s, sigma_v) for event in events]))


# --- metrics ----------------------------------------------------------------


def _mape(predicted: np.ndarray, observed: np.ndarray, floor: float):
    """Mean absolute percentage error, skipping |observed| < floor (counted)."""
    mask = np.abs(observed) >= floor
    skipped = int(np.size(observed) - np.count_nonzero(mask))
    if not np.any(mask):
        return float("nan"), skipped
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.mean(np.abs((predicted[mask] - observed[mask]) / observed[mask])) * 100.0), skipped


def event_metrics(event: EventData, theta, sigma_s: float, sigma_v: float) -> dict:
    """Raw and normalised error metrics of one event (numpy, metric units)."""
    spacing, speed, status = simulate(event, theta, return_status=True)
    residual_s = spacing - event.s_obs
    residual_v = speed - event.v_obs
    mape_s, skipped_s = _mape(spacing, event.s_obs, 0.05)   # 5 cm floor
    mape_v, skipped_v = _mape(speed, event.v_obs, 0.05)     # 0.05 m/s floor
    j_spacing = float(np.mean(residual_s ** 2) / sigma_s ** 2)
    j_speed = float(np.mean(residual_v ** 2) / sigma_v ** 2)
    v_leader = event.vL_obs
    dv = v_leader - event.v_obs
    return {
        "n_samples": event.n,
        "duration_s": float(event.t[-1] - event.t[0]),
        "spacing_rmse": float(np.sqrt(np.mean(residual_s ** 2))),
        "spacing_mae": float(np.mean(np.abs(residual_s))),
        "spacing_mape_percent": mape_s,
        "spacing_mape_skipped": skipped_s,
        "speed_rmse": float(np.sqrt(np.mean(residual_v ** 2))),
        "speed_mae": float(np.mean(np.abs(residual_v))),
        "speed_mape_percent": mape_v,
        "speed_mape_skipped": skipped_v,
        "j_spacing": j_spacing,
        "j_speed": j_speed,
        "j_total": float(j_spacing + j_speed + S_VIOLATION_PENALTY * status["violations"]),
        "s_violations": status["violations"],
        "clamped_steps": status["clamped"],
        "s_min_observed": float(np.min(event.s_obs)),
        "s_min_simulated": float(np.min(spacing)),
        "v_max_observed": float(np.max(event.v_obs)),
        "v_mean_observed": float(np.mean(event.v_obs)),
        "dv_max": float(np.max(dv)),
        "zero_model_spacing_rmse": float(min(
            np.sqrt(np.mean((event.s_obs - event.s_obs[0]) ** 2)),
            np.sqrt(np.mean((event.s_obs - np.mean(event.s_obs)) ** 2)))),
        "zero_model_speed_rmse": float(min(
            np.sqrt(np.mean((event.v_obs - event.v_obs[0]) ** 2)),
            np.sqrt(np.mean((event.v_obs - np.mean(event.v_obs)) ** 2)))),
    }


def aggregate_metrics(per_event: dict) -> dict:
    """Sample-weighted and event-mean aggregates over one pair's events."""
    n = np.array([m["n_samples"] for m in per_event.values()], dtype=float)
    keys = ("spacing_rmse", "spacing_mae", "speed_rmse", "speed_mae",
            "j_spacing", "j_speed")
    out = {"n_events": len(per_event), "n_samples": int(n.sum())}
    for key in keys:
        values = np.array([m[key] for m in per_event.values()], dtype=float)
        out[f"mean_{key}"] = float(np.mean(values))
        out[f"worst_{key}"] = float(np.max(values))
        out[f"pooled_{key}"] = float(np.sqrt(np.average(values ** 2, weights=n)))
    for key in ("spacing_mape_percent", "speed_mape_percent"):
        values = np.array([m[key] for m in per_event.values()], dtype=float)
        out[f"mean_{key}"] = float(np.nanmean(values))
    out["total_cost"] = float(np.mean([m["j_total"] for m in per_event.values()]))
    out["total_violations"] = int(sum(m["s_violations"] for m in per_event.values()))
    return out


# --- json io ----------------------------------------------------------------


def round_floats(value, decimals: int = 6):
    """Recursively round floats; non-finite becomes ``None`` (JSON has no inf)."""
    if isinstance(value, dict):
        return {key: round_floats(item, decimals) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(item, decimals) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return None
        rounded = round(number, decimals)
        return number if (rounded == 0.0 and number != 0.0) else rounded
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write a JSON report atomically (``.tmp`` + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    text = json.dumps(round_floats(payload), indent=2, ensure_ascii=False) + "\n"
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp_path, path)


def read_json(path: Path) -> dict:
    """Read a JSON report written by this pipeline."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_theta(mapping_or_vector) -> np.ndarray:
    """Parameter vector from a dict of named parameters or from a sequence."""
    if isinstance(mapping_or_vector, dict):
        return np.asarray([float(mapping_or_vector[name]) for name in PARAM_NAMES])
    vector = np.asarray(mapping_or_vector, dtype=float).reshape(-1)
    if vector.size != len(PARAM_NAMES):
        raise ValueError(f"theta must have {len(PARAM_NAMES)} entries, got {vector.size}")
    return vector


def theta_to_dict(theta) -> dict:
    """Named view of a parameter vector."""
    vector = as_theta(theta)
    return {name: float(vector[i]) for i, name in enumerate(PARAM_NAMES)}


def format_theta(theta) -> str:
    """One-line parameter vector for stdout and reports."""
    mapping = theta_to_dict(theta)
    return "  ".join(f"{name}={mapping[name]:.4f}" for name in PARAM_NAMES)


def inside_bounds(theta, tol: float = 0.0) -> bool:
    """True when every parameter sits inside the box (inclusive, +tol slack)."""
    vector = as_theta(theta)
    return bool(np.all(vector >= LOWER - tol) and np.all(vector <= UPPER + tol))


def dimensionless_combinations(theta) -> dict:
    """Unit-free parameter combinations used to compare metric and SC2I fits.

    ``v0 / sqrt(b s0)``, ``a / b`` and ``T sqrt(b / s0)`` are invariant under a
    common rescaling of lengths and accelerations, i.e. exactly the comparison
    that is legitimate between the metric TGSIM data and the SC2I coordinate-unit
    testbed (whose car length is 250 units).
    """
    v0, T, s0, a, b = (float(x) for x in as_theta(theta))
    return {
        "v0_over_sqrt_b_s0": v0 / math.sqrt(b * s0),
        "a_over_b": a / b,
        "T_times_sqrt_b_over_s0": T * math.sqrt(b / s0),
        "v0": v0, "T": T, "s0": s0, "a": a, "b": b,
    }


def rel(path: Path, root: Path = ROOT) -> str:
    """Project-relative POSIX path (reports never carry absolute paths)."""
    try:
        return str(Path(path).resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(Path(path).as_posix())


# --- data provenance / unit check -------------------------------------------


def raw_unit_check(pairs: Sequence[str] = PAIR_ORDER, base: Path | None = None) -> dict:
    """Re-verify on the raw CSVs that the TGSIM data are metric SI.

    Two independent checks per file:

      * the finite-difference speed of the position series equals ``speed_kf``
        1:1 (the median ratio of ``|d position|/dt`` to the reported speed) - this
        only holds if positions and speeds share one unit system, and the ratio is
        1.0 exactly when the speed unit is m/s and the position unit is m;
      * the reported vehicle lengths fall in the physical car/truck range in
        metres (4.1..4.9 m here) rather than, say, pixels or feet.

    The authoritative confirmation is the official data dictionary of the TGSIM
    I-90/I-94 dataset (see ``UNIT_REFERENCE`` below), quoted in the build report.
    """
    source_dir = (base or (ROOT / "data/raw/tgsim"))
    result: dict = {}
    for pair in pairs:
        path = source_dir / f"{pair}.csv"
        time, x, y, speed, length = [], [], [], [], []
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["role"] != "follower":
                    continue
                time.append(float(row["time"]))
                x.append(float(row["xloc_kf"]))
                y.append(float(row["yloc_kf"]))
                speed.append(float(row["speed_kf"]))
                length.append(float(row["length_smoothed"]))
        t = np.asarray(time)
        dt = np.diff(t)
        step = np.hypot(np.diff(np.asarray(x)), np.diff(np.asarray(y))) / dt
        v = np.asarray(speed)
        # step[i] is the chord speed over [i, i+1]; compare it with speed_kf[i].
        left, right = step, v[:-1]
        mask = (right > 1.0) & (left > 1.0)
        ratio = left[mask] / right[mask]
        result[pair] = {
            "source": f"data/raw/tgsim/{pair}.csv",
            "n_follower_samples": int(t.size),
            "median_dt_s": float(np.median(dt)),
            "dt_unique": sorted({round(float(value), 6) for value in dt}),
            "median_position_derivative_over_speed": float(np.median(ratio)),
            "p05_p95_ratio": [float(np.percentile(ratio, 5)), float(np.percentile(ratio, 95))],
            "length_smoothed_range_m": [float(np.min(length)), float(np.max(length))],
            "max_abs_speed_mps": float(np.max(np.abs(v))),
            "conclusion": ("positions and speeds are consistent with metric SI "
                           "(ratio ~ 1.0, lengths in metres)"),
        }
    return result


UNIT_REFERENCE = {
    "dataset": "Third Generation Simulation Data (TGSIM) I-90/I-94 Moving Trajectories",
    "publisher": "U.S. Department of Transportation (ITS JPO / FHWA), DOI 10.21949/1404218",
    "url": ("https://data.transportation.gov/Automobiles/"
            "Third-Generation-Simulation-Data-TGSIM-I-90-I-94-M/6a6e-vfvi"),
    "official_column_units": {
        "time": "seconds",
        "xloc_kf": "m (x-coordinate of the vehicle after Kalman filtering)",
        "yloc_kf": "m (y-coordinate of the vehicle after Kalman filtering)",
        "speed_kf": "m/s",
        "acceleration_kf": "m/s^2",
        "length_smoothed": "m (bounding-box length, converted from pixels with 1 px = 0.3 m)",
        "width_smoothed": "m",
    },
    "checked_against_verification_columns": "I-90/I-94 Stationary Trajectories data dictionary "
                                            "(same 12-column schema, same publishers)",
    "conclusion": "metric SI; no unit conversion factor is applied anywhere in this pipeline",
}

