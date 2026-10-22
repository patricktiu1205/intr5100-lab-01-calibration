#!/usr/bin/env python3
"""TGSIM extension: error, driving-style and supplementary analysis.

Reads ``results/tgsim_calibration.json`` (the fitted parameter sets) plus the
event tables and produces the three analysis parts of the brief:

Part 1 - calibration error, HV-HV vs HV-AV
    spacing / speed RMSE, MAE and MAPE, the normalised cost, per event and
    aggregated (event mean and sample-weighted pooled). All errors are reported in
    the ORIGINAL metric units (m, m/s), so the two pairs are directly comparable;
    the dimensionless cost is comparable WITHIN a pair only (each pair has its own
    pooled spread).

Part 2 - AV vs HV car-following style (model-free, from the measured series)
    parameter differences plus efficiency / comfort / safety indicators:
    mean and 85th percentile speed, stop fraction, acceleration and jerk RMS,
    minimum spacing, time headway, and the TTC distribution (TTC = s / dv with
    dv = v_follower - v_leader > 0). The AV-vs-HV difference is summarised with
    the group means and a rank test on the per-event values.

Part 3 - supplementary metrics
    * equilibrium spacing-speed relation of the fitted IDM against the measured
      spacing-speed cloud, plus a regression of the measured cloud;
    * leader / follower acceleration correlation and regression gain per event
      (a string-stability proxy: gain > 1 means the follower amplifies);
    * parameter correlation across the multi-start ensemble (identifiability);
    * dimensionless comparison with the SC2I testbed fit, which is the only
      legitimate way to compare the two datasets (different units).

Output: ``results/tgsim_analysis.json`` (atomic write, idempotent).

Usage:
    python src/tgsim/analyze_events.py [--calibration results/tgsim_calibration.json]
                                       [--out results/tgsim_analysis.json]
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tgsim_core as core  # noqa: E402

SCRIPT = "src/tgsim/analyze_events.py"
STOP_FRACTION_SPEED = core.STOP_SPEED        # 0.3 m/s
TTC_MIN_DV = 0.1                             # m/s, guard for the TTC denominator
TTC_CAP = 60.0                               # s, cap for the distribution
HARSH_ACCEL = 2.0                            # m/s^2, "harsh" event threshold
JERK_WINDOW = 11                             # samples (1.1 s) for the smoothed derivative
MARGIN_SPEED = 3.0                           # m/s, the interaction criterion of part 3


# --- measured-series features ------------------------------------------------


def central_difference(series: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Central-difference derivative, one-sided at both ends (dt from the table)."""
    out = np.empty_like(series)
    out[0] = (series[1] - series[0]) / (t[1] - t[0])
    out[-1] = (series[-1] - series[-2]) / (t[-1] - t[-2])
    out[1:-1] = (series[2:] - series[:-2]) / (t[2:] - t[:-2])
    return out


def savitzky_golay(series: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay smoothing (scipy if available, else a plain moving average).

    Used for the jerk estimate only: a central difference of a 10 Hz
    Kalman-filtered speed amplifies sensor noise, and the raw number is reported
    next to the smoothed one so the choice is visible. Both pairs are treated
    identically, so the comparison is unaffected.
    """
    if series.size < window:
        return series.copy()
    span = window if window % 2 == 1 else window + 1
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(series, span, polyorder)
    except Exception:  # pragma: no cover - scipy is present in this environment
        kernel = np.ones(span) / span
        padded = np.concatenate([np.full(span // 2, series[0]), series,
                                 np.full(span // 2, series[-1])])
        return np.convolve(padded, kernel, mode="valid")


def ttc_series(event: core.EventData, min_dv: float = TTC_MIN_DV) -> dict:
    """Time-to-collision distribution of one event (only closing samples)."""
    dv = event.v_obs - event.vL_obs           # > 0 means the follower is closing in
    mask = dv > min_dv
    if not np.any(mask):
        return {"n_closing_samples": 0, "available": False}
    ttc = np.minimum(event.s_obs[mask] / dv[mask], TTC_CAP)
    return {
        "available": True,
        "n_closing_samples": int(np.count_nonzero(mask)),
        "fraction_closing": float(np.mean(mask)),
        "min_s": float(np.min(ttc)),
        "p05_s": float(np.percentile(ttc, 5)),
        "p25_s": float(np.percentile(ttc, 25)),
        "median_s": float(np.median(ttc)),
        "mean_s": float(np.mean(ttc)),
        "fraction_below_1_5s": float(np.mean(ttc < 1.5)),
        "fraction_below_3s": float(np.mean(ttc < 3.0)),
        "fraction_below_5s": float(np.mean(ttc < 5.0)),
    }


def event_style(event: core.EventData) -> dict:
    """Model-free style indicators of one event (metric units)."""
    t = event.t
    v = event.v_obs
    vL = event.vL_obs
    a = central_difference(v, t)
    a_smooth = savitzky_golay(v, JERK_WINDOW, 2)
    jerk = central_difference(a, t)
    jerk_smooth = central_difference(a_smooth, t)
    dv = v - vL
    headway = np.divide(event.s_obs, v, out=np.full_like(v, np.nan), where=v > 0.5)
    return {
        "n_samples": event.n,
        "duration_s": float(t[-1] - t[0]),
        "speed_mean_mps": float(np.mean(v)),
        "speed_std_mps": float(np.std(v)),
        "speed_p85_mps": float(np.percentile(v, 85)),
        "speed_max_mps": float(np.max(v)),
        "stop_fraction": float(np.mean(v < STOP_FRACTION_SPEED)),
        "moving_fraction": float(np.mean(v >= STOP_FRACTION_SPEED)),
        "accel_mean_mps2": float(np.mean(a)),
        "accel_rms_mps2": float(np.sqrt(np.mean(a ** 2))),
        "accel_p95_abs_mps2": float(np.percentile(np.abs(a), 95)),
        "accel_harsh_fraction": float(np.mean(np.abs(a) > HARSH_ACCEL)),
        "accel_min_mps2": float(np.min(a)),
        "jerk_rms_mps3": float(np.sqrt(np.mean(jerk ** 2))),
        "jerk_rms_smoothed_mps3": float(np.sqrt(np.mean(jerk_smooth ** 2))),
        "spacing_mean_m": float(np.mean(event.s_obs)),
        "spacing_min_m": float(np.min(event.s_obs)),
        "spacing_std_m": float(np.std(event.s_obs)),
        "relative_speed_rms_mps": float(np.sqrt(np.mean(dv ** 2))),
        "relative_speed_max_closing_mps": float(np.max(dv)),
        "thw_median_s": float(np.nanmedian(headway)) if np.any(np.isfinite(headway)) else None,
        "ttc": ttc_series(event),
    }


def accel_correlation(event: core.EventData) -> dict:
    """Leader/follower acceleration correlation and gain (string-stability proxy).

    Pearson correlation and the slope of ``a_follower = gain * a_leader`` over the
    event's interior samples. ``gain > 1`` means the follower amplifies the
    leader's acceleration (string-unstable in the linear sense), ``gain`` near 0
    means the follower does not respond at all. The follower acceleration is
    estimated from its speed with the same central difference as the style
    metrics, and the leader acceleration identically.
    """
    t = event.t
    a_f = central_difference(savitzky_golay(event.v_obs, JERK_WINDOW, 2), t)
    a_l = central_difference(savitzky_golay(event.vL_obs, JERK_WINDOW, 2), t)
    if np.std(a_f) < 1e-9 or np.std(a_l) < 1e-9:
        return {"correlation": None, "gain": None, "n_samples": int(a_f.size),
                "leader_accel_rms_mps2": float(np.sqrt(np.mean(a_l ** 2))),
                "follower_accel_rms_mps2": float(np.sqrt(np.mean(a_f ** 2)))}
    correlation = float(np.corrcoef(a_f, a_l)[0, 1])
    gain = float(np.polyfit(a_l, a_f, 1)[0])
    return {
        "correlation": correlation,
        "gain": gain,
        "n_samples": int(a_f.size),
        "leader_accel_rms_mps2": float(np.sqrt(np.mean(a_l ** 2))),
        "follower_accel_rms_mps2": float(np.sqrt(np.mean(a_f ** 2))),
    }


# --- aggregation helpers ------------------------------------------------------


def weighted_mean(values, weights) -> float:
    """Weighted mean of finite values; ``nan`` when nothing is finite."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values)
    if not np.any(mask):
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def rank_test(group_a: list[float], group_b: list[float]) -> dict:
    """Two-sided Mann-Whitney U test on two small per-event samples (low power).

    Reported with an explicit warning: 10 vs 8 events give very little power, so a
    non-significant p-value is NOT evidence of no difference.
    """
    a = [value for value in group_a if value is not None and math.isfinite(value)]
    b = [value for value in group_b if value is not None and math.isfinite(value)]
    if len(a) < 3 or len(b) < 3:
        return {"available": False}
    try:
        from scipy.stats import mannwhitneyu
        statistic, p_value = mannwhitneyu(a, b, alternative="two-sided")
    except Exception:
        return {"available": False, "note": "scipy.stats.mannwhitneyu unavailable"}
    return {
        "available": True,
        "n_a": len(a), "n_b": len(b),
        "statistic": float(statistic), "p_value": float(p_value),
        "note": "two-sided Mann-Whitney U on per-event values; n = 10 vs 8, low power",
    }


# --- parts --------------------------------------------------------------------


def part1_errors(pairs: dict) -> dict:
    """Calibration error comparison (raw metric units + normalised cost)."""
    table = {}
    for pair, result in pairs.items():
        aggregate = result["aggregate"]
        table[pair] = {
            "label": result["label"],
            "n_events": aggregate["n_events"],
            "n_samples": aggregate["n_samples"],
            "pooled_sigma_s_m": result["pooled_scales"]["sigma_s_m"],
            "pooled_sigma_v_mps": result["pooled_scales"]["sigma_v_mps"],
            "cost": result["cost"],
            "spacing_rmse_mean_m": aggregate["mean_spacing_rmse"],
            "spacing_rmse_pooled_m": aggregate["pooled_spacing_rmse"],
            "spacing_rmse_worst_m": aggregate["worst_spacing_rmse"],
            "spacing_mae_mean_m": aggregate["mean_spacing_mae"],
            "spacing_mape_mean_percent": aggregate["mean_spacing_mape_percent"],
            "speed_rmse_mean_mps": aggregate["mean_speed_rmse"],
            "speed_rmse_pooled_mps": aggregate["pooled_speed_rmse"],
            "speed_rmse_worst_mps": aggregate["worst_speed_rmse"],
            "speed_mae_mean_mps": aggregate["mean_speed_mae"],
            "speed_mape_mean_percent": aggregate["mean_speed_mape_percent"],
            "events": {name: metrics for name, metrics in result["events"].items()},
        }
    comparison = {}
    if len(table) == 2:
        left, right = core.PAIR_ORDER
        for key in ("spacing_rmse_mean_m", "spacing_rmse_pooled_m", "spacing_mae_mean_m",
                    "speed_rmse_mean_mps", "speed_rmse_pooled_mps", "speed_mae_mean_mps",
                    "cost"):
            a = table[left][key]
            b = table[right][key]
            comparison[key] = {
                left: a, right: b,
                "ratio_HV-AV_over_HV-HV": float(b / a) if a else None,
                "better_pair": left if a < b else right,
            }
    return {
        "per_pair": table,
        "comparison": comparison,
        "note": ("Raw RMSE/MAE are in metric units (m, m/s) and are directly comparable "
                 "between the two pairs. The dimensionless cost is normalised by each "
                 "pair's OWN pooled spread (sigma_s = %.2f m for HV-HV vs %.2f m for "
                 "HV-AV), so it ranks parameter vectors within a pair and must not be "
                 "read as a cross-pair accuracy ranking."
                 % (table.get("HV-HV", {}).get("pooled_sigma_s_m", float("nan")),
                    table.get("HV-AV", {}).get("pooled_sigma_s_m", float("nan")))),
    }


def part2_style(pairs: dict, styles: dict) -> dict:
    """AV vs HV car-following style: parameters + efficiency/comfort/safety."""
    metrics = {
        "speed_mean_mps": "efficiency",
        "speed_p85_mps": "efficiency",
        "moving_fraction": "efficiency",
        "stop_fraction": "efficiency (congestion exposure)",
        "accel_rms_mps2": "comfort",
        "accel_p95_abs_mps2": "comfort",
        "accel_harsh_fraction": "comfort",
        "jerk_rms_mps3": "comfort",
        "jerk_rms_smoothed_mps3": "comfort",
        "spacing_min_m": "safety (closest approach)",
        "spacing_mean_m": "safety (gap keeping)",
        "relative_speed_rms_mps": "safety (speed mismatch)",
        "thw_median_s": "safety (time headway)",
    }
    table: dict = {}
    for pair in core.PAIR_ORDER:
        per_event = styles[pair]
        names = sorted(per_event, key=lambda key: int(key.split("_")[1]))
        n = np.array([per_event[name]["n_samples"] for name in names], dtype=float)
        entry = {"label": core.PAIR_LABEL[pair], "n_events": len(names),
                 "params": pairs[pair]["theta"], "per_event": {}}
        for key in metrics:
            values = [per_event[name][key] for name in names]
            finite = [value for value in values if value is not None and math.isfinite(value)]
            entry[f"{key}_event_mean"] = float(np.mean(finite)) if finite else None
            entry[f"{key}_sample_weighted"] = weighted_mean(values, n)
            entry[f"{key}_min"] = float(np.min(finite)) if finite else None
            entry[f"{key}_max"] = float(np.max(finite)) if finite else None
        # TTC: pooled over the closing samples of every event
        ttc_values = [per_event[name]["ttc"] for name in names
                      if per_event[name]["ttc"].get("available")]
        entry["ttc_events_with_closing"] = len(ttc_values)
        if ttc_values:
            entry["ttc_event_min_s"] = float(min(item["min_s"] for item in ttc_values))
            entry["ttc_event_median_min_s"] = float(np.median([item["min_s"] for item in ttc_values]))
            entry["ttc_pooled_fraction_below_1_5s"] = weighted_mean(
                [item["fraction_below_1_5s"] for item in ttc_values],
                [item["n_closing_samples"] for item in ttc_values])
            entry["ttc_pooled_fraction_below_3s"] = weighted_mean(
                [item["fraction_below_3s"] for item in ttc_values],
                [item["n_closing_samples"] for item in ttc_values])
        for name in names:
            entry["per_event"][name] = per_event[name]
        table[pair] = entry

    tests = {}
    for key in list(metrics) + ["ttc_event_min_s"]:
        a = [style[key] for style in
             [styles["HV-HV"][name] for name in sorted(styles["HV-HV"],
                                                       key=lambda k: int(k.split("_")[1]))]
             if style.get(key) is not None]
        b = [style[key] for style in
             [styles["HV-AV"][name] for name in sorted(styles["HV-AV"],
                                                       key=lambda k: int(k.split("_")[1]))]
             if style.get(key) is not None]
        tests[key] = rank_test(a, b)

    parameters = {}
    for key in core.PARAM_NAMES:
        a = pairs["HV-HV"]["theta"][key]
        b = pairs["HV-AV"]["theta"][key]
        parameters[key] = {
            "HV-HV": a, "HV-AV": b,
            "difference": float(b - a),
            "ratio": float(b / a) if a else None,
            "unit": core.PARAM_UNITS[key],
        }
    return {
        "metric_groups": metrics,
        "per_pair": table,
        "parameter_difference": parameters,
        "rank_tests": tests,
        "note": ("Style indicators come from the MEASURED series, so they are independent "
                 "of the IDM fit. The rank tests compare 10 (HV-HV) with 8 (HV-AV) event "
                 "values: low power, a non-significant p-value is not evidence of equality."),
    }


def part3_supplementary(pairs: dict, events_map: dict) -> dict:
    """Equilibrium spacing-speed, accel correlation, parameter correlation, scaling."""
    # --- equilibrium spacing-speed and the measured cloud ---------------------
    equilibrium = {}
    for pair in core.PAIR_ORDER:
        theta = core.as_theta(pairs[pair]["theta"])
        v0, T, s0, a, b = (float(x) for x in theta)
        observed_v_all = np.concatenate([event.v_obs for event in events_map[pair]])
        v_max = min(v0 * 0.98, float(np.max(observed_v_all)) * 1.4, 40.0)
        v_grid = np.linspace(0.0, v_max, 200)
        with np.errstate(invalid="ignore"):
            free = 1.0 - (v_grid / v0) ** 4
            s_eq = (s0 + v_grid * T) / np.sqrt(np.clip(free, 1e-6, None))
        observed_s = np.concatenate([event.s_obs for event in events_map[pair]])
        observed_v = np.concatenate([event.v_obs for event in events_map[pair]])
        slope, intercept = np.polyfit(observed_v, observed_s, 1)
        residual = observed_s - (slope * observed_v + intercept)
        equilibrium[pair] = {
            "theta": core.theta_to_dict(theta),
            "equilibrium_curve": {"speed_mps": v_grid.tolist(), "spacing_m": s_eq.tolist()},
            "equilibrium_spacing_at_observed_mean_speed_m": float(
                (s0 + float(np.mean(observed_v)) * T) / math.sqrt(
                    max(1.0 - (float(np.mean(observed_v)) / v0) ** 4, 1e-6))),
            "measured_cloud": {
                "speed_mean_mps": float(np.mean(observed_v)),
                "spacing_mean_m": float(np.mean(observed_s)),
                "linear_fit": {"slope_s": float(slope), "intercept_m": float(intercept),
                               "r_squared": float(1.0 - np.var(residual) / np.var(observed_s))},
                "correlation_spacing_speed": float(np.corrcoef(observed_s, observed_v)[0, 1]),
                "idm_equilibrium_slope_s": float(T / math.sqrt(
                    max(1.0 - (float(np.mean(observed_v)) / v0) ** 4, 1e-6))),
            },
        }

    # --- leader / follower acceleration correlation ---------------------------
    correlation = {}
    for pair in core.PAIR_ORDER:
        per_event = {}
        for event in events_map[pair]:
            per_event[event.name] = accel_correlation(event)
        values = [item["correlation"] for item in per_event.values() if item["correlation"] is not None]
        gains = [item["gain"] for item in per_event.values() if item["gain"] is not None]
        correlation[pair] = {
            "per_event": per_event,
            "mean_correlation": float(np.mean(values)) if values else None,
            "median_correlation": float(np.median(values)) if values else None,
            "mean_gain": float(np.mean(gains)) if gains else None,
            "fraction_gain_above_1": float(np.mean([gain > 1.0 for gain in gains])) if gains else None,
            "note": ("a_follower = gain * a_leader over the event (smoothed 1.1 s derivatives); "
                     "gain > 1 = amplification, a string-stability proxy"),
        }

    # --- parameter correlation across the multi-start ensemble ----------------
    parameter_correlation = {}
    for pair in core.PAIR_ORDER:
        starts = pairs[pair]["convergence"]["per_start"]
        feasible = [record for record in starts if record["feasible"]]
        matrix = np.array([[record["theta"][name] for name in core.PARAM_NAMES]
                           for record in feasible], dtype=float)
        if matrix.shape[0] >= 3:
            correlation_matrix = np.corrcoef(matrix, rowvar=False)
            parameter_correlation[pair] = {
                "n_starts": int(matrix.shape[0]),
                "names": list(core.PARAM_NAMES),
                "matrix": correlation_matrix.tolist(),
                "std_across_starts": {name: float(np.std(matrix[:, i]))
                                      for i, name in enumerate(core.PARAM_NAMES)},
                "range_across_starts": {name: [float(np.min(matrix[:, i])), float(np.max(matrix[:, i]))]
                                        for i, name in enumerate(core.PARAM_NAMES)},
                "note": ("correlation across the multi-start final vectors - evidence about "
                         "which parameters the data identify (a wide spread / strong "
                         "coupling means the parameter is weakly identified)"),
            }
        else:
            parameter_correlation[pair] = {"n_starts": int(matrix.shape[0]), "matrix": None}

    # --- dimensionless comparison with the SC2I testbed -----------------------
    reference = core.SC2I_REFERENCE
    dimensionless = {
        "definition": {
            "v0_over_sqrt_b_s0": "v0 / sqrt(b * s0)  (scale-free speed)",
            "a_over_b": "a / b  (scale-free acceleration aggressiveness)",
            "T_times_sqrt_b_over_s0": "T * sqrt(b / s0)  (scale-free time headway)",
        },
        "pairs": {pair: core.dimensionless_combinations(pairs[pair]["theta"])
                  for pair in core.PAIR_ORDER},
        "sc2i_reference": dict(core.dimensionless_combinations(reference["theta"]),
                               label=reference["label"], source=reference["source"],
                               units=reference["units"]),
        "why": ("TGSIM is metric SI while the SC2I miniature testbed is in coordinate units "
                "(car length 250 units). Raw parameters are not comparable; these three "
                "combinations are invariant under a common rescaling of lengths and "
                "accelerations and are therefore the legitimate comparison."),
    }
    return {
        "equilibrium_spacing_speed": equilibrium,
        "leader_follower_acceleration": correlation,
        "parameter_correlation": parameter_correlation,
        "dimensionless_comparison": dimensionless,
    }


def part1b_robustness(primary: dict, robustness: dict) -> dict:
    """Primary vs sparse-event-filtered re-calibration (same method, same seed).

    The HV-AV pair contains events in which the follower trails at 28..44 m: for
    most of such an event the IDM interaction term is inactive, so those events
    constrain the parameters only weakly (and inflate the pooled spacing scale).
    Re-fitting with them removed is the cheapest honest check that the reported
    HV-HV / HV-AV contrast is not produced by them.
    """
    out: dict = {
        "source": robustness.get("script", ""),
        "filter": robustness.get("event_selection", {}),
        "pairs": {},
        "comparison": {},
    }
    for pair in core.PAIR_ORDER:
        primary_pair = primary["pairs"][pair]
        robust_pair = robustness["pairs"][pair]
        selection = robustness.get("event_selection", {}).get(pair, {})
        out["pairs"][pair] = {
            "n_events_primary": primary_pair["aggregate"]["n_events"],
            "n_events_filtered": robust_pair["aggregate"]["n_events"],
            "excluded_events": selection.get("excluded", []),
            "rule": selection.get("rule", ""),
            "theta_filtered": robust_pair["theta"],
            "cost_filtered": robust_pair["cost"],
            "spacing_rmse_mean_m_filtered": robust_pair["aggregate"]["mean_spacing_rmse"],
            "speed_rmse_mean_mps_filtered": robust_pair["aggregate"]["mean_speed_rmse"],
        }
        out["comparison"][pair] = {
            "theta_shift": {name: float(robust_pair["theta"][name] - primary_pair["theta"][name])
                            for name in core.PARAM_NAMES},
            "theta_relative_change_percent": {
                name: float(100.0 * (robust_pair["theta"][name] - primary_pair["theta"][name])
                            / primary_pair["theta"][name]) for name in core.PARAM_NAMES},
            "spacing_rmse_change_m": float(robust_pair["aggregate"]["mean_spacing_rmse"]
                                           - primary_pair["aggregate"]["mean_spacing_rmse"]),
            "speed_rmse_change_mps": float(robust_pair["aggregate"]["mean_speed_rmse"]
                                           - primary_pair["aggregate"]["mean_speed_rmse"]),
        }
    return out


def spacing_convention_sensitivity(pairs: dict, events_map: dict) -> dict:
    """How much the choice of vehicle-length reference moves the fit.

    The event tables define ``s = centre distance - leader_length`` (the convention
    of ``preprocess_events.py`` and of the existing event tables). If the tracked
    point is the FRONT bumper, the spacing between the two bumpers is
    ``centre distance - follower_length`` instead, i.e. the two conventions differ
    by the constant ``offset = leader_length - follower_length`` (0.68 m for
    HV-HV, 0.22 m for HV-AV). A constant spacing offset is absorbed by the IDM's
    ``s0``, so this section quantifies the effect WITHOUT re-optimising: the
    calibrated vector is re-scored on the shifted spacing, once unchanged and once
    with ``s0 + offset``. A full re-fit under the alternative convention is listed
    as a follow-up in the build report.
    """
    out: dict = {}
    for pair in core.PAIR_ORDER:
        events = events_map[pair]
        offset = float(events[0].leader_length - events[0].follower_length)
        theta = core.as_theta(pairs[pair]["theta"])
        shifted = [replace(event, s_obs=event.s_obs + offset) for event in events]
        scales = (pairs[pair]["pooled_scales"]["sigma_s_m"],
                  pairs[pair]["pooled_scales"]["sigma_v_mps"])
        same = core.aggregate_metrics({event.name: core.event_metrics(event, theta, *scales)
                                       for event in shifted})
        adjusted = list(theta)
        adjusted[2] = theta[2] + offset
        with_s0 = core.aggregate_metrics({event.name: core.event_metrics(event, adjusted, *scales)
                                          for event in shifted})
        out[pair] = {
            "offset_m": offset,
            "leader_length_m": events[0].leader_length,
            "follower_length_m": events[0].follower_length,
            "primary_spacing_rmse_mean_m": pairs[pair]["aggregate"]["mean_spacing_rmse"],
            "alt_convention_same_theta_spacing_rmse_mean_m": same["mean_spacing_rmse"],
            "alt_convention_s0_shifted_spacing_rmse_mean_m": with_s0["mean_spacing_rmse"],
            "cost_alt_same_theta": same["total_cost"],
            "cost_alt_s0_shifted": with_s0["total_cost"],
            "cost_primary": pairs[pair]["cost"],
        }
    return {
        "per_pair": out,
        "note": ("the alternative convention is a constant offset of the spacing; the "
                 "calibrated vector re-scores with s0 shifted by the same offset, which is "
                 "how the IDM absorbs a standstill-gap offset"),
    }


# --- main ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calibration", type=Path, default=core.CALIBRATION_PATH)
    parser.add_argument("--robustness", type=Path,
                        default=core.RESULTS_DIR / "tgsim_calibration_robustness.json")
    parser.add_argument("--out", type=Path, default=core.ANALYSIS_PATH)
    args = parser.parse_args(argv)

    calibration = core.read_json(args.calibration)
    pairs = calibration["pairs"]
    events_map = {pair: core.load_pair_events(pair) for pair in core.PAIR_ORDER}
    styles = {pair: {event.name: event_style(event) for event in events_map[pair]}
              for pair in core.PAIR_ORDER}

    payload = {
        "script": SCRIPT,
        "source_calibration": core.rel(args.calibration),
        "method": calibration["method"],
        "part1_calibration_error": part1_errors(pairs),
        "part2_driving_style": part2_style(pairs, styles),
        "part3_supplementary": part3_supplementary(pairs, events_map),
        "part4s_spacing_convention": spacing_convention_sensitivity(pairs, events_map),
    }
    if args.robustness and Path(args.robustness).is_file():
        payload["part1b_robustness"] = part1b_robustness(calibration, core.read_json(args.robustness))
        payload["part1b_robustness"]["source"] = core.rel(args.robustness)
    else:
        payload["part1b_robustness"] = {"available": False,
                                        "reason": "no robustness calibration file found"}
    core.write_json_atomic(args.out, payload)

    # --- stdout summary -------------------------------------------------------
    print(f"source  : {core.rel(args.calibration)}")
    print()
    print("Part 1 - calibration error (raw metric units, directly comparable)")
    print(f"  {'pair':<8}{'ev':>4}{'n':>6}{'J':>9}{'sRMSE':>9}{'sMAE':>8}{'sMAPE':>8}"
          f"{'vRMSE':>9}{'vMAE':>8}{'vMAPE':>8}")
    for pair in core.PAIR_ORDER:
        entry = payload["part1_calibration_error"]["per_pair"][pair]
        print(f"  {pair:<8}{entry['n_events']:>4}{entry['n_samples']:>6}{entry['cost']:>9.4f}"
              f"{entry['spacing_rmse_mean_m']:>9.3f}{entry['spacing_mae_mean_m']:>8.3f}"
              f"{entry['spacing_mape_mean_percent']:>8.1f}"
              f"{entry['speed_rmse_mean_mps']:>9.3f}{entry['speed_mae_mean_mps']:>8.3f}"
              f"{entry['speed_mape_mean_percent']:>8.1f}")
    print()
    print("Part 1b - robustness: same fit with the sparse events removed")
    robust = payload["part1b_robustness"]
    if robust.get("available", True) and "comparison" in robust:
        for pair in core.PAIR_ORDER:
            item = robust["pairs"][pair]
            shift = robust["comparison"][pair]["theta_relative_change_percent"]
            print(f"  {pair:<8} events {item['n_events_primary']} -> {item['n_events_filtered']} "
                  f"(excluded {item['excluded_events']})  sRMSE "
                  f"{item['spacing_rmse_mean_m_filtered']:.3f} m  theta change "
                  + " ".join(f"{name} {shift[name]:+.1f}%" for name in core.PARAM_NAMES))
    else:
        print(f"  not available: {robust.get('reason')}")
    print()
    print("Part 2 - AV vs HV driving style (measured series, event means)")
    print(f"  {'metric':<32}{'HV-HV':>10}{'HV-AV':>10}{'p (MWU)':>10}")
    part2 = payload["part2_driving_style"]
    for key in part2["metric_groups"]:
        left = part2["per_pair"]["HV-HV"][f"{key}_event_mean"]
        right = part2["per_pair"]["HV-AV"][f"{key}_event_mean"]
        test = part2["rank_tests"][key]
        p_value = test.get("p_value") if test.get("available") else None
        print(f"  {key:<32}{left:>10.3f}{right:>10.3f}"
              f"{('%.3f' % p_value) if p_value is not None else 'n/a':>10}")
    print()
    print("  spacing-speed relation (measured cloud vs fitted IDM equilibrium)")
    for pair in core.PAIR_ORDER:
        item = payload["part3_supplementary"]["equilibrium_spacing_speed"][pair]
        cloud = item["measured_cloud"]
        print(f"  {pair:<8} measured slope {cloud['linear_fit']['slope_s']:.3f} s "
              f"(R^2 {cloud['linear_fit']['r_squared']:.3f}, corr "
              f"{cloud['correlation_spacing_speed']:.3f}); IDM equilibrium slope "
              f"{cloud['idm_equilibrium_slope_s']:.3f} s")
    print()
    print("  leader/follower acceleration correlation (string-stability proxy)")
    for pair in core.PAIR_ORDER:
        item = payload["part3_supplementary"]["leader_follower_acceleration"][pair]
        print(f"  {pair:<8} mean r = {item['mean_correlation']:.3f}, "
              f"mean gain = {item['mean_gain']:.3f}, "
              f"fraction gain > 1 = {item['fraction_gain_above_1']:.2f}")
    print()
    print("  spacing-convention sensitivity (alternative vehicle-length reference)")
    for pair in core.PAIR_ORDER:
        item = payload["part4s_spacing_convention"]["per_pair"][pair]
        print(f"  {pair:<8} offset {item['offset_m']:+.2f} m: sRMSE "
              f"{item['primary_spacing_rmse_mean_m']:.3f} (primary) -> "
              f"{item['alt_convention_same_theta_spacing_rmse_mean_m']:.3f} (same theta) -> "
              f"{item['alt_convention_s0_shifted_spacing_rmse_mean_m']:.3f} m (s0 shifted)")
    print()
    print("  dimensionless comparison (SC2I vs TGSIM)")
    print(f"  {'':<10}{'v0/sqrt(b s0)':>16}{'a/b':>10}{'T sqrt(b/s0)':>16}")
    for label, values in (("HV-HV", payload["part3_supplementary"]["dimensionless_comparison"]["pairs"]["HV-HV"]),
                          ("HV-AV", payload["part3_supplementary"]["dimensionless_comparison"]["pairs"]["HV-AV"]),
                          ("SC2I", payload["part3_supplementary"]["dimensionless_comparison"]["sc2i_reference"])):
        print(f"  {label:<10}{values['v0_over_sqrt_b_s0']:>16.3f}{values['a_over_b']:>10.3f}"
              f"{values['T_times_sqrt_b_over_s0']:>16.3f}")
    print()
    print(f"written: {core.rel(args.out)} ({args.out.stat().st_size} bytes)")
    leftovers = sorted(path.name for path in args.out.parent.glob("*.tmp"))
    print(f".tmp leftovers: {leftovers or 'none'}")
    reread = core.read_json(args.out)
    ok = (reread["part1_calibration_error"]["per_pair"]["HV-HV"]["n_events"]
          == payload["part1_calibration_error"]["per_pair"]["HV-HV"]["n_events"])
    print(f"on-disk reparse: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
