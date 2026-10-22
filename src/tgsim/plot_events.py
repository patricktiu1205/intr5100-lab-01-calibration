#!/usr/bin/env python3
"""TGSIM extension: figures for the calibration, the error and the style analysis.

Every figure is regenerated from ``results/tgsim_calibration.json`` and
``results/tgsim_analysis.json`` plus the event tables, so the whole figure set is
reproducible from the two scripts. Each PNG is written to a ``.tmp`` file and moved
into place with ``os.replace`` (same atomic convention as the JSON outputs).

Legend convention used throughout: "measured" is always the recorded trajectory
(plotted first, solid) and "IDM" the calibrated simulation (dashed). For the
style comparison the HV-HV pair is the human-following-human reference and the
HV-AV pair is the AV-following-human case, i.e. the AV-vs-human contrast the
brief asks for.

Usage:
    python src/tgsim/plot_events.py [--figures results/figures/tgsim]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # headless: never needs a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tgsim_core as core  # noqa: E402

SCRIPT = "src/tgsim/plot_events.py"
COLOR_HV = "#1f77b4"        # human follower (HV-HV)
COLOR_AV = "#d62728"        # AV follower (HV-AV)
COLOR_MEASURED = "#1f77b4"
COLOR_MODEL = "#ff7f0e"
COLOR_SC2I = "#2ca02c"


# --- absolute style rubric ----------------------------------------------------
#
# The style radar scores each axis against FIXED physical anchors hard-coded
# below - never derived from the data, and never from the other pair - so a
# score means the same thing for both pairs and a third dataset can be added
# without moving the scale. 0 = worst / physically unacceptable, 1 = ideal.
#
# Raw values are the per-event means of the measured series
# (``{key}_event_mean`` in ``part2_driving_style`` of ``tgsim_analysis.json``),
# exactly the quantities the style bars (fig_06) show.

RADAR_RUBRIC: dict[str, tuple[str, list[tuple[float, float]]]] = {
    # key -> (axis label, breakpoints [(raw value, score), ...])
    "moving_fraction": (
        "efficiency\n(moving fraction)",
        [(0.0, 0.0), (1.0, 1.0)],                  # already a 0..1 fraction
    ),
    "spacing_mean_m": (
        "gap keeping\n(mean spacing)",
        [(5.0, 0.0), (10.0, 1.0), (25.0, 1.0), (40.0, 0.0)],
        # ideal window 10..25 m: closer than 5 m is unsafe, beyond 40 m the pair
        # is not really car-following any more
    ),
    "spacing_min_m": (
        "closest approach\n(min spacing)",
        [(2.0, 0.0), (6.0, 1.0)],                  # 2 m or less is unsafe
    ),
    "jerk_rms_smoothed_mps3": (
        "smoothness\n(low jerk)",
        [(0.5, 1.0), (3.0, 0.0)],                  # = clamp((3 - x) / (3 - 0.5))
    ),
    "accel_rms_mps2": (
        "comfort\n(low accel)",
        [(0.5, 1.0), (3.0, 0.0)],                  # = clamp((3 - x) / (3 - 0.5))
    ),
}
RADAR_KEYS = tuple(RADAR_RUBRIC)


def clamp01(value: float) -> float:
    """Clamp a score to the rubric range [0, 1]."""
    return min(max(float(value), 0.0), 1.0)


def rubric_score(value, anchors) -> float:
    """Map a raw metric onto the absolute 0-1 rubric (piecewise-linear clamp).

    ``anchors`` is a sequence of ``(x, score)`` breakpoints; the score is
    interpolated linearly between the two breakpoints surrounding ``value`` and
    clamped to the outermost anchor score outside them, so the result always
    lies in [0, 1] and the breakpoints are not required to be monotone in
    ``score`` (a window merit like the spacing one rises and falls).

    ``value`` is the only data input: no other dataset, and no statistic of the
    data, enters the scale. A missing or non-finite ``value`` scores 0.0 (worst)
    instead of raising, so an absent metric can never look better than a
    measured one.
    """
    points = sorted((float(x), clamp01(score)) for x, score in anchors)
    if not points:
        raise ValueError("anchors must contain at least one (raw value, score) pair")
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(x):
        return 0.0
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        if x <= x1:
            fraction = 1.0 if x1 <= x0 else (x - x0) / (x1 - x0)
            return clamp01(y0 + fraction * (y1 - y0))
    return points[-1][1]


def save(fig, path: Path) -> str:
    """Save a figure atomically (``.tmp`` + ``os.replace``) and verify the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fig.savefig(tmp, dpi=130, bbox_inches="tight", format="png")
    plt.close(fig)
    os.replace(tmp, path)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"figure not written: {path}")
    return f"{core.rel(path)} ({path.stat().st_size} bytes)"


# --- per-pair trajectory overlays --------------------------------------------


def figure_trajectories(pairs: dict, events_map: dict, kind: str, figures: Path,
                        written: list[str]) -> None:
    """Grid of per-event measured-vs-IDM overlays (``kind`` = spacing or speed)."""
    for pair in core.PAIR_ORDER:
        theta = core.as_theta(pairs[pair]["theta"])
        events = events_map[pair]
        ncol, nrow = 2, int(np.ceil(len(events) / 2))
        fig, axes = plt.subplots(nrow, ncol, figsize=(11, 1.9 * nrow), sharex=False)
        axes = np.atleast_1d(axes).ravel()
        for axis, event in zip(axes, events):
            spacing, speed = core.simulate(event, theta)
            measured = event.s_obs if kind == "spacing" else event.v_obs
            modeled = spacing if kind == "spacing" else speed
            unit = "m" if kind == "spacing" else "m/s"
            axis.plot(event.t, measured, color=COLOR_MEASURED, lw=1.2, label="measured")
            axis.plot(event.t, modeled, color=COLOR_MODEL, lw=1.2, ls="--",
                      label="calibrated IDM")
            error = float(np.sqrt(np.mean((modeled - measured) ** 2)))
            axis.set_title(f"{pair} {event.name}  RMSE={error:.3f} {unit}", fontsize=8)
            axis.tick_params(labelsize=7)
            axis.grid(alpha=0.25, lw=0.5)
            if kind == "spacing":
                axis.set_ylabel("spacing [m]", fontsize=7)
            else:
                axis.set_ylabel("speed [m/s]", fontsize=7)
            axis.set_xlabel("time in event [s]", fontsize=7)
        for axis in axes[len(events):]:
            axis.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=8, frameon=False)
        follower = "AV follower" if pair == "HV-AV" else "human follower"
        fig.suptitle(f"{kind.capitalize()} fit, {pair} ({follower} behind a human leader)",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        index = {"spacing": 1, "speed": 2}.get(kind, 3)
        offset = 0 if pair == "HV-HV" else 2
        written.append(save(fig, figures / f"fig_{index + offset:02d}_{kind}_fit_{pair}.png"))


# --- error comparison ---------------------------------------------------------


def figure_error(analysis: dict, figures: Path, written: list[str]) -> None:
    """Per-event error bars plus the aggregate comparison of the two pairs."""
    part1 = analysis["part1_calibration_error"]["per_pair"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, metric, unit, suffix in ((axes[0], "spacing_rmse", "m", "m"),
                                       (axes[1], "speed_rmse", "m/s", "mps")):
        for pair, color in (("HV-HV", COLOR_HV), ("HV-AV", COLOR_AV)):
            events = part1[pair]["events"]
            names = sorted(events, key=lambda key: int(key.split("_")[1]))
            values = [events[name][metric] for name in names]
            axis.plot(range(1, len(names) + 1), values, "o-", color=color, lw=1.2,
                      ms=4, label=f"{pair} ({'human' if pair == 'HV-HV' else 'AV'} follower)")
            axis.axhline(part1[pair][f"{metric}_mean_{suffix}"], color=color,
                         lw=0.8, ls=":")
        axis.set_xlabel("event index")
        axis.set_ylabel(f"{metric.replace('_', ' ')} [{unit}]")
        axis.grid(alpha=0.25, lw=0.5)
        axis.legend(fontsize=8, frameon=False)
        axis.set_title(f"per-event {metric.replace('_', ' ')}")
    keys = ["spacing_rmse_mean_m", "spacing_rmse_pooled_m", "speed_rmse_mean_mps",
            "speed_rmse_pooled_mps"]
    labels = ["s RMSE\n(event mean)", "s RMSE\n(pooled)", "v RMSE\n(event mean)",
              "v RMSE\n(pooled)"]
    x = np.arange(len(keys))
    width = 0.38
    axes[2].bar(x - width / 2, [part1["HV-HV"][key] for key in keys], width,
                color=COLOR_HV, label="HV-HV (human follower)")
    axes[2].bar(x + width / 2, [part1["HV-AV"][key] for key in keys], width,
                color=COLOR_AV, label="HV-AV (AV follower)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=8)
    axes[2].set_ylabel("RMSE (metric units: m, m/s)")
    axes[2].set_title("aggregate error")
    axes[2].grid(alpha=0.25, lw=0.5, axis="y")
    axes[2].legend(fontsize=8, frameon=False)
    fig.suptitle("Calibration error: measured vs calibrated IDM (metric units)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    written.append(save(fig, figures / "fig_05_error_comparison.png"))


# --- style profile ------------------------------------------------------------


def figure_style(analysis: dict, figures: Path, written: list[str]) -> list[dict]:
    """Bars of the style indicators plus the radar scored on the absolute rubric.

    The bar panel keeps raw metric units (readability); the radar maps five of
    those indicators onto the fixed 0-1 rubric of :data:`RADAR_RUBRIC` and
    returns one ``{"pair", "key", "raw", "score"}`` row per axis pair so the
    caller can print the raw-value -> score table.
    """
    part2 = analysis["part2_driving_style"]["per_pair"]
    keys = ["speed_mean_mps", "moving_fraction", "stop_fraction", "accel_rms_mps2",
            "accel_harsh_fraction", "jerk_rms_smoothed_mps3", "spacing_min_m",
            "spacing_mean_m", "relative_speed_rms_mps", "thw_median_s"]
    labels = ["mean speed\n[m/s]", "moving\nfraction", "stop\nfraction",
              "accel RMS\n[m/s$^2$]", "harsh accel\nfraction", "jerk RMS\n[m/s$^3$]",
              "min spacing\n[m]", "mean spacing\n[m]", "rel. speed RMS\n[m/s]",
              "median THW\n[s]"]
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.ravel()
    for axis, key, label in zip(axes, keys, labels):
        values = [part2["HV-HV"][f"{key}_event_mean"], part2["HV-AV"][f"{key}_event_mean"]]
        axis.bar([0, 1], values, color=[COLOR_HV, COLOR_AV], width=0.6)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["HV-HV\n(human)", "HV-AV\n(AV)"], fontsize=8)
        axis.set_title(label, fontsize=9)
        axis.grid(alpha=0.25, lw=0.5, axis="y")
        for position, value in zip([0, 1], values):
            if value is not None:
                axis.text(position, value, f"{value:.3f}", ha="center", va="bottom",
                          fontsize=7)
    fig.suptitle("Driving-style indicators (measured series, per-event mean)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    written.append(save(fig, figures / "fig_06_style_bars.png"))

    # radar: five axes, each scored against the fixed absolute rubric
    # (see RADAR_RUBRIC above); 0 = worst, 1 = ideal, identical for both pairs
    angles = np.linspace(0, 2 * np.pi, len(RADAR_KEYS), endpoint=False).tolist()
    angles += angles[:1]
    fig, axis = plt.subplots(figsize=(6.4, 6.0), subplot_kw={"projection": "polar"})
    table: list[dict] = []
    for pair, color in (("HV-HV", COLOR_HV), ("HV-AV", COLOR_AV)):
        scores = []
        for key in RADAR_KEYS:
            raw = part2[pair][f"{key}_event_mean"]
            score = rubric_score(raw, RADAR_RUBRIC[key][1])
            scores.append(score)
            table.append({"pair": pair, "key": key, "raw": raw, "score": score})
        axis.plot(angles, scores + scores[:1], color=color, lw=1.6,
                  label=f"{pair} ({'human' if pair == 'HV-HV' else 'AV'} follower)")
        axis.fill(angles, scores + scores[:1], color=color, alpha=0.15)
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels([RADAR_RUBRIC[key][0] for key in RADAR_KEYS], fontsize=8)
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    axis.set_ylabel("score (0\u20131)", fontsize=8)
    axis.set_title("Style profile scored against fixed physical anchors\n"
                   "(0\u20131; higher = better)", fontsize=10, pad=18)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=8, frameon=False)
    written.append(save(fig, figures / "fig_09_style_radar.png"))
    return table


# --- TTC ----------------------------------------------------------------------


def figure_ttc(events_map: dict, analysis: dict, figures: Path, written: list[str]) -> None:
    """TTC distribution (closing samples only) and the per-event minimum."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    bins = np.linspace(0, 20, 41)
    for pair, color in (("HV-HV", COLOR_HV), ("HV-AV", COLOR_AV)):
        pooled = []
        for event in events_map[pair]:
            dv = event.v_obs - event.vL_obs
            mask = dv > 0.1
            if np.any(mask):
                pooled.append(np.minimum(event.s_obs[mask] / dv[mask], 60.0))
        values = np.concatenate(pooled) if pooled else np.array([])
        if values.size:
            axes[0].hist(values, bins=bins, density=True, histtype="step", lw=1.6,
                         color=color, label=f"{pair} (n={values.size})")
            ordered = np.sort(values)
            axes[1].plot(ordered, np.arange(1, ordered.size + 1) / ordered.size, lw=1.6,
                         color=color, label=pair)
        # per-event minimum TTC
        names = sorted(events_map[pair], key=lambda item: int(item.name.split("_")[1]))
        minima = []
        for event in names:
            dv = event.v_obs - event.vL_obs
            mask = dv > 0.1
            minima.append(float(np.min(event.s_obs[mask] / dv[mask])) if np.any(mask) else np.nan)
        axes[2].plot(range(1, len(minima) + 1), minima, "o-", color=color, ms=4, lw=1.2,
                     label=pair)
    for axis, title, xlabel in ((axes[0], "TTC distribution (closing samples)", "TTC [s]"),
                                (axes[1], "TTC CDF", "TTC [s]"),
                                (axes[2], "per-event minimum TTC", "event index")):
        axis.set_title(title, fontsize=10)
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.25, lw=0.5)
        axis.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("density")
    axes[2].set_ylabel("min TTC [s]")
    axes[2].axhline(1.5, color="grey", lw=0.8, ls="--")
    axes[2].axhline(3.0, color="grey", lw=0.8, ls=":")
    axes[2].text(1, 1.6, "1.5 s", fontsize=7, color="grey")
    axes[2].text(1, 3.1, "3 s", fontsize=7, color="grey")
    fig.suptitle("Time-to-collision (dv = v_follower - v_leader > 0 only)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    written.append(save(fig, figures / "fig_07_ttc.png"))


# --- spacing-speed ------------------------------------------------------------


def figure_spacing_speed(analysis: dict, events_map: dict, figures: Path,
                         written: list[str]) -> None:
    """Measured spacing-speed cloud vs the fitted IDM equilibrium curve."""
    item = analysis["part3_supplementary"]["equilibrium_spacing_speed"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for axis, pair in zip(axes[:2], core.PAIR_ORDER):
        color = COLOR_HV if pair == "HV-HV" else COLOR_AV
        for event in events_map[pair]:
            axis.scatter(event.v_obs, event.s_obs, s=3, alpha=0.25, color=color)
        cloud = item[pair]["measured_cloud"]
        fit = cloud["linear_fit"]
        grid = np.linspace(0, max(6.0, float(np.max(np.concatenate(
            [event.v_obs for event in events_map[pair]])))), 50)
        axis.plot(grid, fit["slope_s"] * grid + fit["intercept_m"], color="black", lw=1.2,
                  label=f"measured linear fit (slope {fit['slope_s']:.2f} s, "
                        f"R$^2$ {fit['r_squared']:.2f})")
        curve = item[pair]["equilibrium_curve"]
        axis.plot(curve["speed_mps"], curve["spacing_m"], color=COLOR_MODEL, lw=1.8, ls="--",
                  label="calibrated IDM equilibrium")
        axis.set_title(f"{pair} ({'human' if pair == 'HV-HV' else 'AV'} follower)", fontsize=10)
        axis.set_xlabel("speed [m/s]")
        axis.set_ylabel("spacing [m]")
        axis.grid(alpha=0.25, lw=0.5)
        axis.legend(fontsize=7.5, frameon=False, loc="upper left")
    # third panel: leader vs follower acceleration
    correlation = analysis["part3_supplementary"]["leader_follower_acceleration"]
    for pair, color in (("HV-HV", COLOR_HV), ("HV-AV", COLOR_AV)):
        followers, leaders = [], []
        for event in events_map[pair]:
            t = event.t
            from analyze_events import central_difference, savitzky_golay
            a_f = central_difference(savitzky_golay(event.v_obs, 11, 2), t)
            a_l = central_difference(savitzky_golay(event.vL_obs, 11, 2), t)
            leaders.append(a_l)
            followers.append(a_f)
        leaders = np.concatenate(leaders)
        followers = np.concatenate(followers)
        keep = np.abs(leaders) < 6.0
        axes[2].scatter(leaders[keep], followers[keep], s=2, alpha=0.15, color=color)
        grid = np.linspace(-6, 6, 20)
        gain = correlation[pair]["mean_gain"]
        axes[2].plot(grid, gain * grid, color=color, lw=1.6,
                     label=f"{pair}: r={correlation[pair]['mean_correlation']:.2f}, "
                           f"gain={gain:.2f}")
    axes[2].plot([-6, 6], [-6, 6], color="grey", lw=0.8, ls=":", label="gain = 1")
    axes[2].set_xlim(-6, 6)
    axes[2].set_ylim(-6, 6)
    axes[2].set_xlabel("leader acceleration [m/s$^2$]")
    axes[2].set_ylabel("follower acceleration [m/s$^2$]")
    axes[2].set_title("string-stability proxy", fontsize=10)
    axes[2].grid(alpha=0.25, lw=0.5)
    axes[2].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("Spacing-speed structure and leader/follower acceleration response", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    written.append(save(fig, figures / "fig_08_spacing_speed_and_accel.png"))


# --- dimensionless and convergence --------------------------------------------


def figure_dimensionless(analysis: dict, figures: Path, written: list[str]) -> None:
    """Dimensionless parameter combinations: HV-HV, HV-AV and the SC2I testbed fit."""
    item = analysis["part3_supplementary"]["dimensionless_comparison"]
    keys = ["v0_over_sqrt_b_s0", "a_over_b", "T_times_sqrt_b_over_s0"]
    labels = ["$v_0/\\sqrt{b\\,s_0}$\n(scale-free speed)", "$a/b$\n(acceleration ratio)",
              "$T\\sqrt{b/s_0}$\n(scale-free headway)"]
    values = [[item["pairs"]["HV-HV"][key], item["pairs"]["HV-AV"][key],
               item["sc2i_reference"][key]] for key in keys]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for axis, key, label, series in zip(axes, keys, labels, values):
        axis.bar([0, 1, 2], series, color=[COLOR_HV, COLOR_AV, COLOR_SC2I], width=0.6)
        axis.set_xticks([0, 1, 2])
        axis.set_xticklabels(["HV-HV\n(human)", "HV-AV\n(AV)", "SC2I testbed\n(coordinate units)"],
                             fontsize=8)
        axis.set_title(label, fontsize=10)
        axis.grid(alpha=0.25, lw=0.5, axis="y")
        for position, value in zip([0, 1, 2], series):
            axis.text(position, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Dimensionless parameter combinations (the only legitimate SC2I vs TGSIM "
                 "comparison)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    written.append(save(fig, figures / "fig_10_dimensionless.png"))


def figure_convergence(pairs: dict, figures: Path, written: list[str]) -> None:
    """Adam training curves of the batched multi-start, both pairs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for axis, pair in zip(axes, core.PAIR_ORDER):
        convergence = pairs[pair]["convergence"]
        axis.plot(convergence["best_curve"], color=COLOR_MODEL, lw=1.8,
                  label="best over all starts (running minimum)")
        axis.plot(convergence["step_min_curve"], color="grey", lw=0.9, alpha=0.8,
                  label="cheapest start at that step")
        for index, curve in enumerate(convergence["start_curves"]):
            axis.plot(np.arange(len(curve)) * 25, curve, lw=0.6, alpha=0.4)
        axis.set_yscale("log")
        axis.set_xlabel("Adam step")
        axis.set_ylabel("cost J (pooled-spread normalised)")
        axis.set_title(f"{pair} multi-start ({convergence['n_starts']} starts, "
                       f"best start {convergence['best_start_index']})", fontsize=10)
        axis.grid(alpha=0.25, lw=0.5)
        axis.legend(fontsize=8, frameon=False)
    fig.suptitle("Differentiable-IDM training convergence (torch Adam, float64)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    written.append(save(fig, figures / "fig_11_convergence.png"))


# --- main ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calibration", type=Path, default=core.CALIBRATION_PATH)
    parser.add_argument("--analysis", type=Path, default=core.ANALYSIS_PATH)
    parser.add_argument("--figures", type=Path, default=core.FIGURES_DIR)
    args = parser.parse_args(argv)

    calibration = core.read_json(args.calibration)
    analysis = core.read_json(args.analysis)
    pairs = calibration["pairs"]
    events_map = {pair: core.load_pair_events(pair) for pair in core.PAIR_ORDER}

    written: list[str] = []
    for kind in ("spacing", "speed"):
        figure_trajectories(pairs, events_map, kind, args.figures, written)
    figure_error(analysis, args.figures, written)
    style_table = figure_style(analysis, args.figures, written)
    figure_ttc(events_map, analysis, args.figures, written)
    figure_spacing_speed(analysis, events_map, args.figures, written)
    figure_dimensionless(analysis, args.figures, written)
    figure_convergence(pairs, args.figures, written)

    print(f"script  : {SCRIPT}")
    print(f"figures : {core.rel(args.figures)}")
    for line in written:
        print(f"  {line}")
    print("style radar: absolute 0-1 rubric (fixed anchors, identical for both pairs)")
    print("  axis (metric key)                        anchors: raw value -> score")
    for key in RADAR_KEYS:
        anchors = ", ".join(f"{x:g} -> {score:g}" for x, score in RADAR_RUBRIC[key][1])
        label = RADAR_RUBRIC[key][0].replace("\n", " ")
        print(f"  {label:38s} ({key})")
        print(f"  {'':38s} {anchors}")
    print("  pair    axis                                   raw value   score")
    for row in style_table:
        raw = row["raw"]
        label = RADAR_RUBRIC[row["key"]][0].replace("\n", " ")
        text = "     n/a" if raw is None else f"{raw:9.4f}"
        print(f"  {row['pair']:6s}  {label:38s} {text}   {row['score']:.4f}")
    leftovers = sorted(path.name for path in args.figures.glob("*.tmp"))
    print(f".tmp leftovers: {leftovers or 'none'}")
    return 0 if not leftovers else 1


if __name__ == "__main__":
    raise SystemExit(main())
