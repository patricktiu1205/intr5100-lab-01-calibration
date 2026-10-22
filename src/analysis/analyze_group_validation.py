#!/usr/bin/env python3
"""Five-group IDM parameter comparison and validation-run analysis (Lab Session I).

This module is deliberately self-contained: it reads only the five raw validation
collections under ``data/group_validation_raw/`` plus their submitted parameter
vector (``idm_parameters_groups.json``) and writes two artifacts, both with the
atomic ``.tmp`` + ``os.replace`` convention of the rest of the project:

    results/group_validation_analysis.json          all tables and method notes
    results/figures/group_validation/fig_01..05.png  print-ready PNGs

Nothing under ``data/`` or any other ``results/`` file is read back or modified;
no package is imported beyond numpy / pandas / matplotlib, and no timestamp is
written anywhere, so a re-run is byte-for-byte reproducible.

Scientific scope
----------------
The five groups submitted their IDM parameters in inconsistent scales, so the raw
numbers (A.1) are not directly comparable. The comparison is therefore carried by
the three gauge-invariant dimensionless combinations (A.2)

    v0~ = v0 / sqrt(b * s0)          a~ = a / b          T~ = T * sqrt(b / s0)

which are invariant under the multiplicative rescaling of the length unit (alpha)
and the time unit (beta): v -> v*alpha/beta, s -> s*alpha, T -> T*beta,
a, b -> a, b * alpha / beta^2. The validation behaviour (B) is then described per
group from ``commands.csv`` (control steps of the on-board IDM) and
``follower_straight_trajectory.csv`` (pose stream), and related back to those
combinations (C).

Usage:
    python src/analysis/analyze_group_validation.py
    python src/analysis/analyze_group_validation.py --root <lab1>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")            # headless: never needs a display
import matplotlib.pyplot as plt  # noqa: E402

# --- import bootstrap (the project is a set of scripts, not a package) --------

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/group_validation/x.py -> lab1
_CALIBRATE = PROJECT_ROOT / "src" / "calibration"
if str(_CALIBRATE) not in sys.path:
    sys.path.insert(0, str(_CALIBRATE))

import idm_core as core  # noqa: E402  (reuse write_json_atomic / round_floats)

SCRIPT = "src/analysis/analyze_group_validation.py"

# --- fixed configuration ------------------------------------------------------

RAW_SUBDIR = Path("data") / "group_validation_raw"
PARAMS_FILENAME = "idm_parameters_groups.json"
COMMANDS_FILENAME = "commands.csv"
TRAJECTORY_FILENAME = "follower_straight_trajectory.csv"
METADATA_FILENAME = "metadata.json"

RESULTS_SUBDIR = Path("results")
JSON_FILENAME = "group_validation_analysis.json"
FIGURES_SUBDIR = Path("results") / "figures" / "group_validation"

#: The group whose parameters are our own submission (README: group_4).
SUBMITTED_GROUP = "group_4"

#: Spacing below these levels (platform-record units, s = gap - L_VEH) is reported
#: as a fraction of samples. 250 = one vehicle length, 100 = a tight-following band.
SPACING_THRESHOLDS = (100.0, 250.0)

#: Platform length constant: spacing = gap - L_VEH (250 cu), the lab's convention.
L_VEH = 250.0

#: A follower control step counts as "moving" above this v_cmd, matching the 0.3
#: speed threshold used elsewhere in the project (tgsim_core.STOP_SPEED = 0.3).
MOVING_SPEED_THRESHOLD = 0.3

#: Jerk is only formed across two consecutive control steps whose a_idm are both
#: present AND whose spacing in time dt <= this many seconds; a larger dt is a
#: "mode gap" (e.g. an idle/curve stretch) and its finite difference is dropped.
JERK_MAX_GAP_S = 1.0

#: Order the parameters / combinations are reported and plotted in.
RAW_PARAM_KEYS = ("v0", "T", "s0", "a", "b", "delta")
DIMENSIONLESS_KEYS = ("v0_tilde", "a_tilde", "T_tilde")

GROUP_COLORS = {
    "group_1": "#1f77b4",
    "group_2": "#ff7f0e",
    "group_3": "#2ca02c",
    "group_4": "#d62728",   # our submission, highlighted
    "group_5": "#9467bd",
}


# --- small helpers ------------------------------------------------------------

def atomic_savefig(fig, path: Path) -> None:
    """Save a matplotlib figure atomically (``.tmp`` + ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fig.savefig(tmp, dpi=150, bbox_inches="tight", format="png")
    plt.close(fig)
    os.replace(tmp, path)


def _cv(values) -> float | None:
    """Coefficient of variation (sample std, ddof=1) of a small sample."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size < 2 or float(np.mean(arr)) == 0.0:
        return None
    return float(np.std(arr, ddof=1) / np.mean(arr))


def spread(values) -> dict[str, object]:
    """min / max / mean / std(ddof=1) / cv / range of a numeric sample."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"n": 0, "min": None, "max": None, "mean": None,
                "std": None, "cv": None, "range": None}
    return {
        "n": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "cv": _cv(arr),
        "range": float(np.max(arr) - np.min(arr)),
    }


def contiguous_episodes(flags: np.ndarray) -> list[tuple[int, int]]:
    """Run-length episodes of a boolean array as inclusive ``(start, end)`` pairs.

    Consecutive True samples are one episode (the convention of
    ``analysis.metrics.controller_log``): the number of episodes is the number of
    distinct anti-collision activations.
    """
    out: list[tuple[int, int]] = []
    i = 0
    n = len(flags)
    while i < n:
        if flags[i]:
            start = i
            while i < n and flags[i]:
                i += 1
            out.append((start, i - 1))
        else:
            i += 1
    return out


# --- A: parameters ------------------------------------------------------------

def load_group_parameters(raw_dir: Path) -> dict[str, dict[str, float]]:
    """Read the submitted parameter vectors of all five groups."""
    path = raw_dir / PARAMS_FILENAME
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    params: dict[str, dict[str, float]] = {}
    for group, fields in payload.items():
        params[group] = {key: float(fields[key]) for key in RAW_PARAM_KEYS}
    return dict(sorted(params.items()))


def dimensionless(params: dict[str, float]) -> dict[str, float]:
    """The three gauge-invariant combinations of one parameter vector."""
    v0 = params["v0"]
    T = params["T"]
    s0 = params["s0"]
    a = params["a"]
    b = params["b"]
    return {
        "v0_tilde": v0 / math.sqrt(b * s0),
        "a_tilde": a / b,
        "T_tilde": T * math.sqrt(b / s0),
    }


def section_a(params: dict[str, dict[str, float]]) -> dict[str, object]:
    """A. raw table + dimensionless table + spreads."""
    dims = {group: dimensionless(p) for group, p in params.items()}
    raw_spread = {key: spread([p[key] for p in params.values()]) for key in RAW_PARAM_KEYS}
    dim_spread = {key: spread([d[key] for d in dims.values()]) for key in DIMENSIONLESS_KEYS}
    return {
        "raw_parameters": {g: {k: params[g][k] for k in RAW_PARAM_KEYS} for g in params},
        "dimensionless": dims,
        "raw_spread": raw_spread,
        "dimensionless_spread": dim_spread,
    }


# --- B: validation runs -------------------------------------------------------

def analyse_commands(path: Path) -> dict[str, object]:
    """B.1-5: spacing, safety layer, speed, dynamics from a commands.csv log.

    All rows of these logs are follower rows (the leader has no command log), so
    the follower subset is selected explicitly rather than assumed.
    """
    frame = pd.read_csv(path)
    frame = frame[frame["role"] == "follower"].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{path}: no follower rows")
    t = frame["t_from_start"].to_numpy(dtype=float)
    spacing = frame["spacing"].to_numpy(dtype=float)
    v_cmd = frame["v_cmd"].to_numpy(dtype=float)
    emergency = frame["emergency"].to_numpy(dtype=float) > 0.0
    dt = frame["dt"].to_numpy(dtype=float)
    mode = frame["mode"].astype(str).to_numpy()
    a_idm = frame["a_idm"].to_numpy(dtype=float)

    n = int(len(frame))
    duration = float(t[-1] - t[0])

    # B.2 spacing
    spacing_block: dict[str, object] = {
        "min": float(np.min(spacing)),
        "mean": float(np.mean(spacing)),
        "median": float(np.median(spacing)),
    }
    for threshold in SPACING_THRESHOLDS:
        below = int(np.count_nonzero(spacing < threshold))
        spacing_block[f"count_below_{threshold:g}"] = below
        spacing_block[f"fraction_below_{threshold:g}"] = float(below) / n

    # B.3 safety layer
    episodes = contiguous_episodes(emergency)
    episode_records = []
    for start, end in episodes:
        window = spacing[start:end + 1]
        episode_records.append({
            "start_s": float(t[start]),
            "end_s": float(t[end]),
            "n_samples": int(end - start + 1),
            "mode": str(mode[start]),
            "spacing_at_trigger": float(spacing[start]),
            "spacing_min_in_episode": float(np.min(window)),
        })
    safety_block: dict[str, object] = {
        "emergency_rows": int(np.count_nonzero(emergency)),
        "emergency_row_fraction": float(np.count_nonzero(emergency)) / n,
        "emergency_episodes": len(episodes),
        "episode_records": episode_records,
    }

    # B.4 speed
    moving = int(np.count_nonzero(v_cmd > MOVING_SPEED_THRESHOLD))
    speed_block: dict[str, object] = {
        "v_cmd_mean": float(np.mean(v_cmd)),
        "v_cmd_max": float(np.max(v_cmd)),
        "v_cmd_min": float(np.min(v_cmd)),
        "moving_threshold": float(MOVING_SPEED_THRESHOLD),
        "moving_count": moving,
        "moving_fraction": float(moving) / n,
    }

    # B.5 dynamics where a_idm is present
    present = np.isfinite(a_idm)
    abs_a = np.abs(a_idm[present])
    dynamics_block: dict[str, object] = {
        "a_idm_samples": int(present.sum()),
        "a_idm_present_fraction": float(present.sum()) / n,
    }
    if abs_a.size:
        dynamics_block.update({
            "a_idm_rms": float(np.sqrt(np.mean(abs_a ** 2))),
            "a_idm_abs_p95": float(np.percentile(abs_a, 95.0)),
            "a_idm_abs_max": float(np.max(abs_a)),
            "a_idm_mean_signed": float(np.mean(a_idm[present])),
        })
    else:
        dynamics_block.update({"a_idm_rms": None, "a_idm_abs_p95": None,
                               "a_idm_abs_max": None, "a_idm_mean_signed": None})

    # jerk over consecutive control steps with both a_idm present and dt <= gap
    jerk_values = []
    jerk_gaps_skipped = 0
    for i in range(1, n):
        if not (math.isfinite(a_idm[i]) and math.isfinite(a_idm[i - 1])):
            continue
        if not math.isfinite(dt[i]) or dt[i] <= 0.0:
            continue
        if dt[i] > JERK_MAX_GAP_S:
            jerk_gaps_skipped += 1
            continue
        jerk_values.append((a_idm[i] - a_idm[i - 1]) / dt[i])
    jerk_arr = np.asarray(jerk_values, dtype=float)
    dynamics_block.update({
        "jerk_definition": (
            "jerk_i = (a_idm[i] - a_idm[i-1]) / dt[i] over consecutive follower "
            "rows whose a_idm are both present and whose dt <= "
            f"{JERK_MAX_GAP_S:g} s; RMS = sqrt(mean(jerk^2))"
        ),
        "jerk_pairs": int(jerk_arr.size),
        "jerk_gaps_skipped": int(jerk_gaps_skipped),
        "jerk_rms": float(np.sqrt(np.mean(jerk_arr ** 2))) if jerk_arr.size else None,
        "jerk_abs_max": float(np.max(np.abs(jerk_arr))) if jerk_arr.size else None,
    })

    mode_counts = {str(k): int(v) for k, v in frame["mode"].value_counts().items()}
    return {
        "duration_s": duration,
        "follower_rows": n,
        "t_first_s": float(t[0]),
        "t_last_s": float(t[-1]),
        "mode_counts": mode_counts,
        "spacing": spacing_block,
        "safety": safety_block,
        "speed": speed_block,
        "dynamics": dynamics_block,
        # series kept in memory only (not JSON) for the figures
        "_series": {"t": t, "spacing": spacing, "v_cmd": v_cmd,
                    "emergency": emergency, "mode": mode},
    }


def analyse_trajectory(path: Path) -> dict[str, object]:
    """B.6-7: lateral error RMS and route completion from the pose stream."""
    frame = pd.read_csv(path, low_memory=False)
    frame = frame[frame["role"] == "follower"].reset_index(drop=True)
    lateral = frame["lateral_error"].to_numpy(dtype=float)
    route = frame["route_distance"].to_numpy(dtype=float)
    finite = np.isfinite(lateral)
    n = int(len(frame))
    return {
        "trajectory_samples": n,
        "lateral_error_rms": float(np.sqrt(np.mean(lateral[finite] ** 2))) if finite.any() else None,
        "lateral_error_abs_max": float(np.max(np.abs(lateral[finite]))) if finite.any() else None,
        "max_route_distance": float(np.nanmax(route)) if np.isfinite(route).any() else None,
        "t_last_s": float(frame["t_from_start"].to_numpy(dtype=float)[-1]),
    }


# --- self-check: independent second parsing path ------------------------------

def independent_recompute(path: Path) -> dict[str, object]:
    """Recompute spacing min and emergency counts with the stdlib ``csv`` module.

    This deliberately shares no code with the pandas path above (no pandas, no
    helper): it re-parses the raw text and re-derives the values by hand so a bug
    in one path cannot hide in the other.
    """
    spacing_values: list[float] = []
    emergency_flags: list[int] = []
    mode_values: list[str] = []
    moving = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("role") or "").strip() != "follower":
                continue
            spacing_values.append(float(row["spacing"]))
            flag = 1 if float(row["emergency"]) > 0.0 else 0
            emergency_flags.append(flag)
            mode_values.append((row.get("mode") or "").strip())
            if float(row["v_cmd"]) > MOVING_SPEED_THRESHOLD:
                moving += 1

    # episodes by hand
    episodes = 0
    prev = 0
    episode_modes: list[str] = []
    for idx, flag in enumerate(emergency_flags):
        if flag == 1 and prev == 0:
            episodes += 1
            episode_modes.append(mode_values[idx])
        prev = flag

    n = len(spacing_values)
    return {
        "follower_rows": n,
        "spacing_min": min(spacing_values) if spacing_values else None,
        "emergency_rows": sum(emergency_flags),
        "emergency_episodes": episodes,
        "emergency_episode_modes": episode_modes,
        "moving_count": moving,
        "moving_fraction": (moving / n) if n else None,
    }


def run_self_check(per_group: dict[str, dict[str, object]]) -> dict[str, object]:
    """Compare the pandas metrics with the stdlib recomputation, per group."""
    checks: dict[str, object] = {}
    all_ok = True
    for group, block in per_group.items():
        alt = block["_independent"]
        main_spacing_min = block["commands"]["spacing"]["min"]
        main_emerg = block["commands"]["safety"]["emergency_rows"]
        main_episodes = block["commands"]["safety"]["emergency_episodes"]
        main_moving = block["commands"]["speed"]["moving_count"]
        checks[group] = {
            "spacing_min_pandas": float(main_spacing_min),
            "spacing_min_csv": float(alt["spacing_min"]),
            "spacing_min_abs_diff": abs(float(main_spacing_min) - float(alt["spacing_min"])),
            "emergency_rows_pandas": int(main_emerg),
            "emergency_rows_csv": int(alt["emergency_rows"]),
            "emergency_episodes_pandas": int(main_episodes),
            "emergency_episodes_csv": int(alt["emergency_episodes"]),
            "moving_count_pandas": int(main_moving),
            "moving_count_csv": int(alt["moving_count"]),
        }
        ok = (
            abs(float(main_spacing_min) - float(alt["spacing_min"])) < 1e-9
            and int(main_emerg) == int(alt["emergency_rows"])
            and int(main_episodes) == int(alt["emergency_episodes"])
            and int(main_moving) == int(alt["moving_count"])
        )
        checks[group]["agree"] = bool(ok)
        all_ok = all_ok and ok
    return {"all_agree": bool(all_ok), "per_group": checks}


# --- C: interpretation --------------------------------------------------------

def section_c(per_group: dict[str, dict[str, object]],
              dims: dict[str, dict[str, float]]) -> dict[str, object]:
    """C: rankings on safety (episodes, spacing min) and efficiency (speed, time).

    Two independent safety orderings are published because they disagree: the
    emergency-episode count and the minimum-spacing margin do not rank the groups
    the same way, and hiding that behind one composite score would be misleading.
    """
    groups = list(per_group)

    def spacing_min(g: str) -> float:
        return float(per_group[g]["commands"]["spacing"]["min"])

    def episodes(g: str) -> int:
        return int(per_group[g]["commands"]["safety"]["emergency_episodes"])

    def v_mean(g: str) -> float:
        return float(per_group[g]["commands"]["speed"]["v_cmd_mean"])

    def duration(g: str) -> float:
        return float(per_group[g]["commands"]["duration_s"])

    safety_by_spacing = sorted(groups, key=lambda g: -spacing_min(g))
    safety_by_episodes = sorted(groups, key=lambda g: (episodes(g), -spacing_min(g)))
    efficiency_by_speed = sorted(groups, key=lambda g: (-v_mean(g), duration(g)))
    efficiency_by_duration = sorted(groups, key=lambda g: duration(g))

    pure_idm = [g for g in groups if episodes(g) == 0]
    safety_layer = [g for g in groups if episodes(g) > 0]

    # group_1 fires its emergency flag at very large spacing: its triggers are not
    # proximity-driven, so its episode count must not be read as a near-miss count.
    g1 = "group_1"
    g1_is_proximity = bool(
        per_group[g1]["commands"]["safety"]["episode_records"]
        and min(float(r["spacing_at_trigger"])
                for r in per_group[g1]["commands"]["safety"]["episode_records"]) < 250.0
    ) if g1 in per_group else False

    def combo(g: str) -> str:
        d = dims[g]
        return (f"v0~={d['v0_tilde']:.3f}, a~={d['a_tilde']:.3f}, T~={d['T_tilde']:.3f}")

    notes = [
        f"{SUBMITTED_GROUP} (our submission) is a pure-IDM run: 0 emergency episodes, "
        f"minimum spacing {spacing_min(SUBMITTED_GROUP):.1f} units, dimensionless "
        f"{combo(SUBMITTED_GROUP)}.",
        f"groups that needed the safety layer (>=1 emergency episode): {safety_layer}; "
        f"groups that stayed in pure IDM control: {pure_idm}.",
        f"group_1 fires its emergency flag at large spacing (min spacing over the run "
        f"{spacing_min(g1):.1f} units); its triggers are therefore NOT proximity-driven, "
        f"so ranking it least-safe by raw episode count is misleading. proximity-driven="
        f"{g1_is_proximity}."
        if g1 in per_group else "",
        "group_2 is the only run whose spacing reaches the floor (min spacing = 0.0) and "
        "whose emergency triggers coincide with spacing 0-5.5 units: it is the genuinely "
        "proximity-critical run.",
        "the mean commanded speed (v_cmd) is a scale-dependent quantity and is confounded "
        "across groups; the speed-based efficiency ordering is descriptive only, unlike "
        "the moving fraction, which is dimensionless and comparable.",
        "no monotone relation holds between the dimensionless combinations and the "
        "safety-layer usage: group_1 (smallest v0~ and a~) and group_2 (largest T~) sit "
        "at different corners of the box yet both triggered the layer.",
    ]
    notes = [n for n in notes if n]

    return {
        "safety_rank_by_spacing_min_best_first": safety_by_spacing,
        "safety_rank_by_emergency_episodes_best_first": safety_by_episodes,
        "efficiency_rank_by_v_cmd_mean_best_first": efficiency_by_speed,
        "efficiency_rank_by_duration_best_first": efficiency_by_duration,
        "pure_idm_groups": pure_idm,
        "safety_layer_groups": safety_layer,
        "group_1_triggers_are_proximity_driven": g1_is_proximity,
        "notes": notes,
        "caveats": [
            "each group ran a different physical car / IP (metadata.json leader_ip / "
            "follower_ip), so run-to-run differences confound parameter differences",
            "one run per group: no repetition, therefore no inferential claim, only "
            "descriptive ordering",
            "the raw parameter scales are inconsistent between groups (different "
            "units/gauges), so only the dimensionless combinations are comparable",
            "device / recording-scale differences (coordinate-unit speed vs vehicle-"
            "reported speed) are not removable from these logs",
        ],
    }


# --- figures ------------------------------------------------------------------

def _grouped_bars(ax, groups, series, colors, log=False):
    """Draw grouped bars: ``series`` maps a metric label to per-group values."""
    labels = list(series)
    x = np.arange(len(labels))
    width = 0.8 / len(groups)
    for gi, group in enumerate(groups):
        values = [series[label].get(group, np.nan) for label in labels]
        offset = (gi - (len(groups) - 1) / 2.0) * width
        ax.bar(x + offset, values, width, label=group, color=colors[group],
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if log:
        ax.set_yscale("log")


def figure_01_raw(params, dims, path):
    """Fig 1: raw parameter comparison (log-scale grouped bars)."""
    groups = list(params)
    colors = {g: GROUP_COLORS.get(g, "#333333") for g in groups}
    series = {k: {g: params[g][k] for g in groups} for k in RAW_PARAM_KEYS}
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    _grouped_bars(ax, groups, series, colors, log=True)
    ax.set_ylabel("raw submitted value (log scale)")
    ax.set_title("Fig 1. Raw IDM parameters of the five groups (inconsistent scales, log axis)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(title="group", ncol=5, fontsize=8, loc="upper center")
    fig.tight_layout()
    atomic_savefig(fig, path)


def figure_02_dimensionless(dims, path):
    """Fig 2: dimensionless combinations grouped bars (five groups x three combos)."""
    groups = list(dims)
    colors = {g: GROUP_COLORS.get(g, "#333333") for g in groups}
    series = {k: {g: dims[g][k] for g in groups} for k in DIMENSIONLESS_KEYS}
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    _grouped_bars(ax, groups, series, colors, log=True)
    ax.set_ylabel("dimensionless value (log scale)")
    ax.set_title("Fig 2. Gauge-invariant dimensionless combinations "
                 r"($\tilde{v}_0$, $\tilde{a}$, $\tilde{T}$)")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.legend(title="group", ncol=5, fontsize=8, loc="upper center")
    fig.tight_layout()
    atomic_savefig(fig, path)


def figure_03_spacing(per_group, path):
    """Fig 3: spacing trajectories vs time with emergency markers."""
    groups = list(per_group)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for group in groups:
        series = per_group[group]["commands"]["_series"]
        color = GROUP_COLORS.get(group, "#333333")
        ax.plot(series["t"], series["spacing"], color=color, lw=1.1, label=group)
        emerg = series["emergency"]
        if emerg.any():
            ax.scatter(series["t"][emerg], series["spacing"][emerg], color=color,
                       marker="x", s=45, zorder=5, label=f"{group} emergency")
    ax.set_xlabel("time from start (s)")
    ax.set_ylabel("spacing (platform-record units, s = gap - 250)")
    ax.set_title("Fig 3. Follower spacing vs time, all five groups (x = emergency trigger)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    atomic_savefig(fig, path)


def figure_04_speed(per_group, path):
    """Fig 4: v_cmd trajectories vs time."""
    groups = list(per_group)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for group in groups:
        series = per_group[group]["commands"]["_series"]
        ax.plot(series["t"], series["v_cmd"], color=GROUP_COLORS.get(group, "#333333"),
                lw=1.1, label=group)
    ax.axhline(MOVING_SPEED_THRESHOLD, color="grey", ls="--", lw=0.9,
               label=f"moving threshold ({MOVING_SPEED_THRESHOLD:g})")
    ax.set_xlabel("time from start (s)")
    ax.set_ylabel("commanded speed v_cmd (platform-record units/s)")
    ax.set_title("Fig 4. Follower commanded speed vs time, all five groups")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    atomic_savefig(fig, path)


def figure_05_metrics(per_group, path):
    """Fig 5: safety / efficiency metric comparison (2x2 grouped bars)."""
    groups = list(per_group)
    colors = [GROUP_COLORS.get(g, "#333333") for g in groups]
    spacing_min = [per_group[g]["commands"]["spacing"]["min"] for g in groups]
    episodes = [per_group[g]["commands"]["safety"]["emergency_episodes"] for g in groups]
    mean_speed = [per_group[g]["commands"]["speed"]["v_cmd_mean"] for g in groups]
    duration = [per_group[g]["commands"]["duration_s"] for g in groups]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    panels = [
        (axes[0][0], spacing_min, "minimum spacing (units)", "Safety: min spacing (higher is safer)"),
        (axes[0][1], episodes, "emergency episodes (count)", "Safety: emergency triggers (lower is safer)"),
        (axes[1][0], mean_speed, "mean v_cmd (units/s)", "Efficiency: mean commanded speed (higher is faster)"),
        (axes[1][1], duration, "duration (s)", "Efficiency: run duration (lower is faster)"),
    ]
    x = np.arange(len(groups))
    for ax, values, ylabel, title in panels:
        ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=20)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Fig 5. Safety and efficiency metrics of the five validation runs")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    atomic_savefig(fig, path)


# --- orchestration ------------------------------------------------------------

def build_analysis(root: Path) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Load everything and assemble the JSON payload (series stripped)."""
    raw_dir = root / RAW_SUBDIR
    params = load_group_parameters(raw_dir)
    groups = list(params)

    dims = {g: dimensionless(params[g]) for g in groups}
    per_group: dict[str, dict[str, object]] = {}
    for group in groups:
        commands = analyse_commands(raw_dir / group / COMMANDS_FILENAME)
        trajectory = analyse_trajectory(raw_dir / group / TRAJECTORY_FILENAME)
        metadata_path = raw_dir / group / METADATA_FILENAME
        metadata = {}
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            metadata = {
                "leader_ip": meta.get("leader_ip"),
                "follower_ip": meta.get("follower_ip"),
                "line_name": meta.get("line_name"),
                "leader_speed": meta.get("leader_speed"),
                "curve_speed": meta.get("curve_speed"),
            }
        independent = independent_recompute(raw_dir / group / COMMANDS_FILENAME)
        per_group[group] = {
            "commands": commands,
            "trajectory": trajectory,
            "metadata": metadata,
            "_independent": independent,
        }

    self_check = run_self_check(per_group)

    # JSON view: strip the in-memory series out of the commands blocks.
    json_per_group: dict[str, object] = {}
    for group in groups:
        block = per_group[group]
        json_per_group[group] = {
            "metadata": block["metadata"],
            "commands": {k: v for k, v in block["commands"].items() if not k.startswith("_")},
            "trajectory": block["trajectory"],
            "independent_check": block["_independent"],
        }

    payload: dict[str, object] = {
        "analysis": "five-group IDM parameter comparison and validation-run analysis (Lab Session I)",
        "script": SCRIPT,
        "submitted_group": SUBMITTED_GROUP,
        "groups": groups,
        "method_notes": {
            "spacing_convention": "spacing = gap - L_VEH, L_VEH = 250 platform-record units",
            "spacing_thresholds": list(SPACING_THRESHOLDS),
            "moving_speed_threshold": MOVING_SPEED_THRESHOLD,
            "jerk": (
                "jerk_i = (a_idm[i] - a_idm[i-1]) / dt[i] over consecutive follower rows "
                f"with both a_idm present and dt <= {JERK_MAX_GAP_S:g} s; RMS = sqrt(mean(jerk^2)); "
                "relationships across a mode gap larger than that are skipped"
            ),
            "a_idm_present": "blank (NaN) in curve_fixed / non-IDM control steps, so dynamics are IDM-only",
            "dimensionless_invariance": (
                "v0~ = v0/sqrt(b*s0), a~ = a/b, T~ = T*sqrt(b/s0) are invariant under the "
                "multiplicative length-unit rescaling s->alpha*s and time-unit rescaling "
                "t->beta*t (v->v*alpha/beta, T->T*beta, a,b->a,b*alpha/beta^2)"
            ),
            "cv": "coefficient of variation = sample std (ddof=1) / mean",
            "units": "all validation quantities in the platform-record units of the log "
                     "(coordinate units and coordinate units per second)",
        },
        "A_parameters": section_a(params),
        "B_validation_runs": json_per_group,
        "C_interpretation": section_c(per_group, dims),
        "self_check": self_check,
    }
    return payload, per_group


def print_summary(payload: dict[str, object]) -> None:
    """Per-group summary table on stdout (the self-check the brief asks for)."""
    groups = payload["groups"]
    dims = payload["A_parameters"]["dimensionless"]
    print("=" * 108)
    print("Per-group summary  (v0~ = v0/sqrt(b*s0), a~ = a/b, T~ = T*sqrt(b/s0))")
    print("=" * 108)
    header = (f"{'group':<9}{'v0~':>8}{'a~':>8}{'T~':>8}"
              f"{'rows':>6}{'dur_s':>8}{'s_min':>9}{'s_mean':>9}"
              f"{'<100':>7}{'<250':>7}{'emerg':>6}{'eps':>5}"
              f"{'v_mean':>8}{'v_max':>8}{'move':>7}{'a_rms':>8}{'j_rms':>8}"
              f"{'lat_rms':>8}{'route':>7}")
    print(header)
    for group in groups:
        cmd = payload["B_validation_runs"][group]["commands"]
        traj = payload["B_validation_runs"][group]["trajectory"]
        sp = cmd["spacing"]
        sa = cmd["safety"]
        sv = cmd["speed"]
        dy = cmd["dynamics"]
        d = dims[group]

        def fmt(v, w, p=3):
            return f"{'n/a' if v is None else f'{float(v):.{p}f}':>{w}}"

        print(f"{group:<9}{fmt(d['v0_tilde'],8)}{fmt(d['a_tilde'],8)}{fmt(d['T_tilde'],8)}"
              f"{cmd['follower_rows']:>6}{fmt(cmd['duration_s'],8,1)}"
              f"{fmt(sp['min'],9,2)}{fmt(sp['mean'],9,2)}"
              f"{fmt(sp['fraction_below_100'],7)}{fmt(sp['fraction_below_250'],7)}"
              f"{sa['emergency_rows']:>6}{sa['emergency_episodes']:>5}"
              f"{fmt(sv['v_cmd_mean'],8,2)}{fmt(sv['v_cmd_max'],8,2)}"
              f"{fmt(sv['moving_fraction'],7)}"
              f"{fmt(dy['a_idm_rms'],8,2)}{fmt(dy['jerk_rms'],8,1)}"
              f"{fmt(traj['lateral_error_rms'],8,2)}{fmt(traj['max_route_distance'],7,2)}")
    print("-" * 108)
    sc = payload["self_check"]
    print(f"self-check: pandas vs stdlib csv agree on spacing min / emergency rows / "
          f"episodes / moving count for all groups: {sc['all_agree']}")
    if not sc["all_agree"]:
        for group, c in sc["per_group"].items():
            if not c["agree"]:
                print(f"  MISMATCH {group}: {c}")
    print("=" * 108)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="project root (default: inferred)")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else PROJECT_ROOT
    payload, per_group = build_analysis(root)

    json_path = root / RESULTS_SUBDIR / JSON_FILENAME
    core.write_json_atomic(json_path, payload)
    print(f"written: {json_path}")

    fig_dir = root / FIGURES_SUBDIR
    dims = payload["A_parameters"]["dimensionless"]
    params = payload["A_parameters"]["raw_parameters"]
    figure_01_raw(params, dims, fig_dir / "fig_01_raw_parameters.png")
    figure_02_dimensionless(dims, fig_dir / "fig_02_dimensionless_combinations.png")
    figure_03_spacing(per_group, fig_dir / "fig_03_spacing_trajectories.png")
    figure_04_speed(per_group, fig_dir / "fig_04_speed_trajectories.png")
    figure_05_metrics(per_group, fig_dir / "fig_05_safety_efficiency.png")
    for name in ("fig_01_raw_parameters.png", "fig_02_dimensionless_combinations.png",
                 "fig_03_spacing_trajectories.png", "fig_04_speed_trajectories.png",
                 "fig_05_safety_efficiency.png"):
        print(f"figure: {fig_dir / name}")

    print()
    print_summary(payload)
    return 0 if payload["self_check"]["all_agree"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
