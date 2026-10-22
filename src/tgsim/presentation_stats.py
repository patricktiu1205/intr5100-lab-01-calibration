#!/usr/bin/env python3
"""TGSIM presentation layer: multi-aggregation statistics with effect sizes.

The differentiable-IDM calibration (``results/tgsim_calibration.json``) and the
main analysis (``results/tgsim_analysis.json``) aggregate the measured per-event
style metrics as an event MEAN plus a sample-weighted pooled value. That
aggregation hides two facts a presentation review flagged:

1. the HV-AV advantage in gap keeping / closest approach is carried by two
   sparse, NON-FOLLOWING events (HV-AV event_7/event_8, mean spacing 28..41 m,
   the IDM interaction term inactive most of the time). Without them the two
   pairs are comparable - and 6 of the 8 HV-AV events are below 10 m;
2. for the jerk / acceleration RMS the event-mean and the event-median orderings
   of the two pairs REVERSE, so the conclusion depends on the aggregation used.

This script adds a presentation statistics layer. It does NOT touch any
calibration result or existing JSON; it writes ONE new file,
``results/tgsim_presentation_stats.json`` (atomic, deterministic, no timestamps).

For every metric and pair it reports

  * ``event_mean``, ``event_median`` and ``pooled_sample_weighted`` on all
    events, and the same three on the INTERACTION SUBSET (sparse events
    removed, with the excluded ids and their mean spacing);
  * the ordered per-event values (plus a <10 m count for spacing_mean);
  * the effect size of the HV-HV vs HV-AV per-event comparison: the
    rank-biserial correlation ``r = 2U/(n_a n_b) - 1`` from the two-sided
    Mann-Whitney U together with its two-sided p-value (recomputed and checked
    against the existing ``rank_tests`` in ``results/tgsim_analysis.json``);
  * a ``method`` block documenting every convention.

The metric definitions and the moving / jerk-smoothing conventions are imported
from ``src/tgsim/analyze_events.py`` so they are EXACTLY the existing ones; the
script asserts that its recomputed per-event values reproduce
``results/tgsim_analysis.json`` before it declares success.

Usage:
    python src/tgsim/presentation_stats.py [--analysis results/tgsim_analysis.json]
                                           [--out results/tgsim_presentation_stats.json]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tgsim_core as core  # noqa: E402
import analyze_events as analysis  # noqa: E402  (reuse the existing conventions)

SCRIPT = "src/tgsim/presentation_stats.py"
SPARSE_MEAN_SPACING_M = 20.0          # m, the sparse / non-interacting rule
SPACING_REFERENCE_M = 10.0            # m, count threshold for spacing_mean
OUT_PATH = core.RESULTS_DIR / "tgsim_presentation_stats.json"

# metric key -> (unit, analysis-JSON key, human description)
METRICS: tuple[tuple[str, str, str, str], ...] = (
    ("speed_mean", "m/s", "speed_mean_mps", "mean follower speed"),
    ("moving_fraction", "fraction", "moving_fraction",
     "fraction of samples with follower speed >= 0.3 m/s"),
    ("spacing_mean", "m", "spacing_mean_m",
     "mean bumper-to-bumper spacing s = dhw - leader_length"),
    ("spacing_min", "m", "spacing_min_m",
     "minimum spacing of the event (closest approach)"),
    ("accel_rms", "m/s^2", "accel_rms_mps2",
     "RMS of the central-difference follower acceleration (raw speed)"),
    ("jerk_rms_smoothed", "m/s^3", "jerk_rms_smoothed_mps3",
     "RMS of the central-difference jerk of the Savitzky-Golay smoothed speed"),
)
METRIC_UNIT = {key: unit for key, unit, _, _ in METRICS}
METRIC_ANALYSIS_KEY = {key: ak for key, _, ak, _ in METRICS}


# --- per-event measured statistics -------------------------------------------


def event_statistics(event: core.EventData) -> dict:
    """The six requested metrics of one event, using the EXISTING conventions.

    Reuses ``analyze_events.central_difference`` / ``savitzky_golay`` and the
    module constants (``STOP_FRACTION_SPEED`` = 0.3 m/s, ``JERK_WINDOW`` = 11
    samples with polyorder 2) so the numbers are identical to part 2 of
    ``results/tgsim_analysis.json``.
    """
    t = event.t
    v = event.v_obs
    accel = analysis.central_difference(v, t)
    speed_smooth = analysis.savitzky_golay(v, analysis.JERK_WINDOW, 2)
    jerk_smooth = analysis.central_difference(speed_smooth, t)
    return {
        "speed_mean": float(np.mean(v)),
        "moving_fraction": float(np.mean(v >= analysis.STOP_FRACTION_SPEED)),
        "spacing_mean": float(np.mean(event.s_obs)),
        "spacing_min": float(np.min(event.s_obs)),
        "accel_rms": float(np.sqrt(np.mean(accel ** 2))),
        "jerk_rms_smoothed": float(np.sqrt(np.mean(jerk_smooth ** 2))),
    }


# --- aggregation -------------------------------------------------------------


def aggregates(values: list[float], weights: list[float]) -> dict:
    """event_mean / event_median (across events) and sample-weighted pooled value.

    ``pooled_sample_weighted`` is the sample-count weighted mean of the per-event
    values (``sum(n_i v_i) / sum(n_i)``), the same convention as the
    ``*_sample_weighted`` fields of ``results/tgsim_analysis.json`` part 2.
    """
    array = np.asarray(values, dtype=float)
    weight = np.asarray(weights, dtype=float)
    mask = np.isfinite(array)
    if not np.any(mask):
        return {"n_events": int(array.size), "n_samples": 0,
                "event_mean": None, "event_median": None,
                "pooled_sample_weighted": None}
    kept_values = array[mask]
    kept_weights = weight[mask]
    return {
        "n_events": int(kept_values.size),
        "n_samples": int(kept_weights.sum()),
        "event_mean": float(np.mean(kept_values)),
        "event_median": float(np.median(kept_values)),
        "pooled_sample_weighted": float(np.average(kept_values, weights=kept_weights)),
    }


# --- effect size -------------------------------------------------------------


def _u_from_ranks(a: np.ndarray, b: np.ndarray) -> float:
    """Independent Mann-Whitney U of group ``a`` via mid-ranks (ties = 0.5)."""
    combined = np.concatenate([a, b])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(combined.size, dtype=float)
    sorted_values = combined[order]
    i = 0
    while i < sorted_values.size:
        j = i
        while j + 1 < sorted_values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0   # average rank of the tie block
        i = j + 1
    n_a = a.size
    return float(ranks[:n_a].sum() - n_a * (n_a + 1) / 2.0)


def rank_effect_size(group_a: list[float], group_b: list[float]) -> dict:
    """Two-sided Mann-Whitney U plus the rank-biserial correlation.

    ``r = 2U/(n_a n_b) - 1`` where U is the scipy statistic for group ``a``
    (HV-HV). ``r > 0`` means the HV-HV values tend to be larger, ``r < 0`` means
    the HV-AV values tend to be larger. The p-value is the two-sided
    ``scipy.stats.mannwhitneyu`` value (the same call as the existing analysis),
    and U is cross-checked against an independent mid-rank computation.
    """
    a = [float(x) for x in group_a if x is not None and math.isfinite(float(x))]
    b = [float(x) for x in group_b if x is not None and math.isfinite(float(x))]
    if len(a) < 3 or len(b) < 3:
        return {"available": False, "n_a": len(a), "n_b": len(b)}
    from scipy.stats import mannwhitneyu
    statistic, p_value = mannwhitneyu(a, b, alternative="two-sided")
    n_a, n_b = len(a), len(b)
    rank_biserial = 2.0 * float(statistic) / (n_a * n_b) - 1.0
    independent_u = _u_from_ranks(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return {
        "available": True,
        "n_a": n_a,
        "n_b": n_b,
        "u_statistic": float(statistic),
        "u_independent_midrank": independent_u,
        "u_agrees": bool(abs(float(statistic) - independent_u) < 1e-9),
        "rank_biserial_r": float(rank_biserial),
        "p_value_two_sided": float(p_value),
        "direction": ("HV-HV larger" if rank_biserial > 0
                      else ("HV-AV larger" if rank_biserial < 0 else "tie")),
    }


# --- sparse-event rule -------------------------------------------------------


def sparse_events(events: list[core.EventData]) -> tuple[list[core.EventData], list[dict]]:
    """Split one pair's events into (interacting, sparse) by the >20 m rule."""
    interacting, sparse = [], []
    for event in events:
        mean_spacing = float(np.mean(event.s_obs))
        if mean_spacing > SPARSE_MEAN_SPACING_M:
            sparse.append({"event": event.name, "spacing_mean_m": mean_spacing})
        else:
            interacting.append(event)
    return interacting, sparse


# --- assembly ----------------------------------------------------------------


def build_metric_block(metric: str, events_map: dict, statistics: dict,
                       sparse_map: dict) -> dict:
    """One metric entry: aggregations per pair (all + interaction subset)."""
    block: dict = {
        "unit": METRIC_UNIT[metric],
        "pairs": {},
        "event_values": {},
    }
    for pair in core.PAIR_ORDER:
        events = events_map[pair]
        values = [statistics[pair][event.name][metric] for event in events]
        weights = [float(event.n) for event in events]
        interacting, sparse = sparse_map[pair]
        keep = {event.name for event in interacting}
        # iterate the (ordered) event list, not the set, so the pooled weights
        # stay aligned with the values and the output is deterministic
        sub_values = [statistics[pair][event.name][metric] for event in events
                      if event.name in keep]
        sub_weights = [float(event.n) for event in events if event.name in keep]
        block["pairs"][pair] = {
            "all_events": aggregates(values, weights),
            "interaction_subset": dict(
                aggregates(sub_values, sub_weights),
                excluded_events=sparse,
            ),
        }
        block["event_values"][pair] = values
        if metric == "spacing_mean":
            below = int(sum(1 for value in values if value < SPACING_REFERENCE_M))
            block.setdefault("counts", {})[pair] = {
                "n_events_below_10m": below,
                "n_events": len(values),
                "fraction_below_10m": below / len(values) if values else None,
            }
    # effect size on all events and on the interaction subset
    test_all = rank_effect_size(
        block["event_values"]["HV-HV"], block["event_values"]["HV-AV"])
    sub_a = metrics_of(statistics, sparse_map, "HV-HV", metric)
    sub_b = metrics_of(statistics, sparse_map, "HV-AV", metric)
    block["effect_size"] = test_all
    block["effect_size_interaction_subset"] = rank_effect_size(sub_a, sub_b)
    return block


def metrics_of(statistics: dict, sparse_map: dict, pair: str, metric: str) -> list[float]:
    """Interaction-subset per-event values of one metric."""
    interacting, _ = sparse_map[pair]
    return [statistics[pair][event.name][metric] for event in interacting]


def method_block() -> dict:
    """Documentation of every convention used by this layer."""
    return {
        "moving_threshold_mps": analysis.STOP_FRACTION_SPEED,
        "moving_threshold_source": ("tgsim_core.STOP_SPEED = 0.3 m/s, reused by "
                                    "src/tgsim/analyze_events.py; moving_fraction = "
                                    "mean(follower speed >= 0.3 m/s)"),
        "speed_mean_definition": "mean of the follower speed series fv (m/s)",
        "spacing_definition": ("s = dhw - leader_length (bumper-to-bumper, m); dhw is the "
                               "raw centre distance and leader_length comes from "
                               "data/tgsim/manifest.json, the convention of "
                               "src/tgsim/preprocess_events.py and tgsim_core"),
        "accel_rms_definition": ("sqrt(mean(a^2)) with a = central difference of the raw "
                                 "follower speed over time (no smoothing), identical to "
                                 "analyze_events.event_style"),
        "jerk_rms_smoothed_definition": (
            "sqrt(mean(j^2)) with j = central difference of the Savitzky-Golay smoothed "
            "follower speed; SG window = 11 samples (1.1 s), polyorder 2 "
            "(analyze_events.JERK_WINDOW), identical to analyze_events.event_style"),
        "sparse_rule": ("an event is SPARSE / non-interacting when its mean spacing > "
                        "%.0f m; for such an event the IDM interaction term is inactive "
                        "for most of the record, so it is excluded from the interaction "
                        "subset" % SPARSE_MEAN_SPACING_M),
        "aggregations": {
            "event_mean": "arithmetic mean of the per-event values (each event equal weight)",
            "event_median": "median of the per-event values (across events)",
            "pooled_sample_weighted": ("sample-count weighted mean of the per-event values, "
                                       "sum(n_i v_i) / sum(n_i), the *_sample_weighted "
                                       "convention of results/tgsim_analysis.json part 2"),
        },
        "effect_size": ("rank-biserial r = 2U/(n_a n_b) - 1, U = scipy.stats.mannwhitneyu "
                        "(group HV-HV vs group HV-AV, two-sided); r > 0 => HV-HV values "
                        "tend to be larger, r < 0 => HV-AV values tend to be larger; the "
                        "p-value is the two-sided Mann-Whitney p"),
        "power_note": ("n = 10 HV-HV vs 8 HV-AV events gives very low power; a "
                       "non-significant p-value is NOT evidence of no difference"),
        "ordering_note": ("event-mean and event-median orderings of the two pairs REVERSE "
                          "for jerk_rms_smoothed (and accel_rms), so BOTH aggregations are "
                          "reported and must be read together: the mean is pulled by one "
                          "high-jerk HV-HV event while the median reflects the typical "
                          "event"),
        "source_analysis": core.rel(core.ANALYSIS_PATH),
        "source_events": "data/tgsim/{HV-HV,HV-AV}/event_*.csv via data/tgsim/manifest.json",
    }


def build_self_checks(metrics_block: dict, events_map: dict, sparse_map: dict,
                      statistics: dict, existing: dict) -> dict:
    """Machine-checkable summary of the brief's self-check list."""
    checks: dict = {}

    # 1. sparse events per pair
    checks["sparse_events"] = {
        pair: [entry["event"] for entry in sparse_map[pair][1]]
        for pair in core.PAIR_ORDER
    }
    checks["sparse_events_are_HV-AV_event_7_event_8"] = (
        checks["sparse_events"] == {"HV-HV": [], "HV-AV": ["event_7", "event_8"]}
    )

    # 2. interaction-subset spacing values
    hv_av_spacing = metrics_block["spacing_mean"]["pairs"]["HV-AV"]["interaction_subset"]
    hv_hv_spacing = metrics_block["spacing_mean"]["pairs"]["HV-HV"]
    checks["HV-AV_interaction_spacing_mean_m"] = hv_av_spacing["event_mean"]
    checks["HV-AV_interaction_spacing_median_m"] = hv_av_spacing["event_median"]
    checks["HV-HV_spacing_median_m"] = hv_hv_spacing["all_events"]["event_median"]
    # the brief's ballparks, reconciled explicitly
    checks["brief_ballpark"] = {
        "expected_HV-AV_interaction_mean_m": 10.4,
        "computed_HV-AV_interaction_mean_m": hv_av_spacing["event_mean"],
        "expected_HV-AV_median_m": 8.8,
        "computed_HV-AV_interaction_median_m": hv_av_spacing["event_median"],
        "expected_HV-HV_median_m": 8.3,
        "computed_HV-HV_median_m": hv_hv_spacing["all_events"]["event_median"],
        "note": ("the brief's ~8.8 / ~8.3 are the FULL-pair event medians (the HV-AV "
                 "median barely moves when the two sparse events are dropped); the "
                 "brief's ~10.4 is the HV-AV mean-of-means EXCLUDING ONLY event_8 - the "
                 "strict >20 m rule drops both event_7 and event_8, which lowers the "
                 "subset mean"),
        "exclude_event_8_only_mean_m": float(np.mean(
            [statistics["HV-AV"][event.name]["spacing_mean"]
             for event in events_map["HV-AV"] if event.name != "event_8"])),
    }

    # 3. <10 m count
    checks["HV-AV_events_below_10m"] = (
        metrics_block["spacing_mean"]["counts"]["HV-AV"])

    # 4. ordering reversal for accel / jerk
    reversal = {}
    for metric in ("accel_rms", "jerk_rms_smoothed"):
        pair_block = metrics_block[metric]["pairs"]
        hv_hv = pair_block["HV-HV"]["all_events"]
        hv_av = pair_block["HV-AV"]["all_events"]
        by_mean = ("HV-HV" if hv_hv["event_mean"] > hv_av["event_mean"] else "HV-AV")
        by_median = ("HV-HV" if hv_hv["event_median"] > hv_av["event_median"] else "HV-AV")
        reversal[metric] = {
            "HV-HV_event_mean": hv_hv["event_mean"],
            "HV-AV_event_mean": hv_av["event_mean"],
            "HV-HV_event_median": hv_hv["event_median"],
            "HV-AV_event_median": hv_av["event_median"],
            "larger_by_event_mean": by_mean,
            "larger_by_event_median": by_median,
            "orderings_reverse": bool(by_mean != by_median),
        }
    checks["mean_vs_median_reversal"] = reversal

    # 5. recomputation matches the existing analysis per-event values
    existing_styles = existing["part2_driving_style"]["per_pair"]
    mismatches = []
    for pair in core.PAIR_ORDER:
        per_event = existing_styles[pair]["per_event"]
        for name, stats in statistics[pair].items():
            for metric, _, analysis_key, _ in METRICS:
                if analysis_key == "moving_fraction":
                    ref = per_event[name].get("moving_fraction")
                else:
                    ref = per_event[name].get(analysis_key)
                if ref is None:
                    continue
                if abs(float(ref) - stats[metric]) > 5e-6:
                    mismatches.append({"pair": pair, "event": name, "metric": metric,
                                       "existing": float(ref), "recomputed": stats[metric]})
    checks["recomputation_matches_analysis_json"] = len(mismatches) == 0
    checks["recomputation_mismatches"] = mismatches

    # 6. recomputed U/p agree with the existing rank_tests
    existing_tests = existing["part2_driving_style"]["rank_tests"]
    agreement = {}
    for metric, _, analysis_key, _ in METRICS:
        mine = metrics_block[metric]["effect_size"]
        theirs = existing_tests.get(analysis_key, {})
        agreement[metric] = {
            "recomputed_u": mine.get("u_statistic"),
            "recomputed_p": mine.get("p_value_two_sided"),
            "existing_u": theirs.get("statistic"),
            "existing_p": theirs.get("p_value"),
            # the existing JSON is written with core.round_floats(6), so the
            # comparison tolerance is the stored 6-decimal precision
            "u_agrees": (mine.get("u_statistic") is not None
                         and abs(mine["u_statistic"] - float(theirs.get("statistic", np.nan))) < 1e-6),
            "p_agrees": (mine.get("p_value_two_sided") is not None
                         and abs(mine["p_value_two_sided"] - float(theirs.get("p_value", np.nan))) < 1e-6),
        }
    checks["existing_rank_tests_agree"] = all(
        item["u_agrees"] and item["p_agrees"] for item in agreement.values())
    checks["rank_test_comparison"] = agreement
    checks["all_hard_checks_pass"] = bool(
        checks["sparse_events_are_HV-AV_event_7_event_8"]
        and checks["HV-AV_events_below_10m"]["n_events_below_10m"] == 6
        and checks["recomputation_matches_analysis_json"]
        and checks["existing_rank_tests_agree"]
        and all(item["orderings_reverse"] for item in reversal.values()))
    return checks


# --- main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--analysis", type=Path, default=core.ANALYSIS_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)

    events_map = {pair: core.load_pair_events(pair) for pair in core.PAIR_ORDER}
    statistics = {pair: {event.name: event_statistics(event) for event in events_map[pair]}
                  for pair in core.PAIR_ORDER}
    sparse_map = {pair: sparse_events(events_map[pair]) for pair in core.PAIR_ORDER}

    metrics_block = {metric: build_metric_block(metric, events_map, statistics, sparse_map)
                     for metric, _, _, _ in METRICS}

    existing = core.read_json(args.analysis)
    checks = build_self_checks(metrics_block, events_map, sparse_map,
                               statistics, existing)

    payload = {
        "script": SCRIPT,
        "source_analysis": core.rel(args.analysis),
        "source_events": "data/tgsim/{HV-HV,HV-AV}/event_*.csv via data/tgsim/manifest.json",
        "pairs": list(core.PAIR_ORDER),
        "pair_labels": {pair: core.PAIR_LABEL[pair] for pair in core.PAIR_ORDER},
        "sparse_events": {pair: sparse_map[pair][1] for pair in core.PAIR_ORDER},
        "metrics": metrics_block,
        "method": method_block(),
        "self_checks": checks,
    }
    core.write_json_atomic(args.out, payload)

    # --- stdout self-check report --------------------------------------------
    print(f"source analysis: {core.rel(args.analysis)}")
    print(f"source events  : data/tgsim/<pair>/event_*.csv  ("
          f"{sum(len(events_map[p]) for p in core.PAIR_ORDER)} events)")
    print()
    print("Sparse / non-interacting events (mean spacing > %.0f m):" % SPARSE_MEAN_SPACING_M)
    for pair in core.PAIR_ORDER:
        entries = sparse_map[pair][1]
        if entries:
            text = ", ".join(f"{e['event']} ({e['spacing_mean_m']:.2f} m)" for e in entries)
        else:
            text = "none"
        print(f"  {pair:<8} {text}")
    print()
    print(f"{'metric':<20}{'pair':<8}{'all_mean':>10}{'all_median':>12}{'all_pooled':>12}"
          f"{'sub_mean':>10}{'sub_median':>12}{'sub_pooled':>12}")
    for metric, _, _, _ in METRICS:
        for pair in core.PAIR_ORDER:
            all_ev = metrics_block[metric]["pairs"][pair]["all_events"]
            sub = metrics_block[metric]["pairs"][pair]["interaction_subset"]
            print(f"{metric:<20}{pair:<8}{all_ev['event_mean']:>10.4f}"
                  f"{all_ev['event_median']:>12.4f}{all_ev['pooled_sample_weighted']:>12.4f}"
                  f"{sub['event_mean']:>10.4f}{sub['event_median']:>12.4f}"
                  f"{sub['pooled_sample_weighted']:>12.4f}")
    print()
    print("Self-check 1: HV-AV interaction subset spacing")
    print(f"  event_mean   = {checks['HV-AV_interaction_spacing_mean_m']:.4f} m "
          f"(brief ballpark ~{checks['brief_ballpark']['expected_HV-AV_interaction_mean_m']})")
    print(f"  event_median = {checks['HV-AV_interaction_spacing_median_m']:.4f} m")
    print(f"  HV-HV all-event spacing median = {checks['HV-HV_spacing_median_m']:.4f} m "
          f"(brief ballpark ~{checks['brief_ballpark']['expected_HV-HV_median_m']})")
    print(f"  reconciliation: exclude event_8 only -> mean "
          f"{checks['brief_ballpark']['exclude_event_8_only_mean_m']:.4f} m "
          f"(the brief's ~10.4); strict >20 m rule also drops event_7.")
    print()
    count = checks["HV-AV_events_below_10m"]
    print(f"Self-check 2: HV-AV events with spacing_mean < 10 m: "
          f"{count['n_events_below_10m']}/{count['n_events']} "
          f"(fraction {count['fraction_below_10m']:.3f})")
    print()
    print("Self-check 3: event-mean vs event-median ordering reversal")
    for metric, item in checks["mean_vs_median_reversal"].items():
        print(f"  {metric:<20} by mean: {item['larger_by_event_mean']:<6} "
              f"(HV-HV {item['HV-HV_event_mean']:.4f} vs HV-AV {item['HV-AV_event_mean']:.4f})"
              f"  |  by median: {item['larger_by_event_median']:<6} "
              f"(HV-HV {item['HV-HV_event_median']:.4f} vs HV-AV {item['HV-AV_event_median']:.4f})"
              f"  -> reverse={item['orderings_reverse']}")
    print()
    print("Self-check 4: effect size (all events) and agreement with tgsim_analysis.json")
    print(f"  {'metric':<20}{'U':>6}{'r':>9}{'p (new)':>12}{'p (existing)':>14}{'agree':>7}")
    for metric, _, _, _ in METRICS:
        item = checks["rank_test_comparison"][metric]
        effect = metrics_block[metric]["effect_size"]
        print(f"  {metric:<20}{effect['u_statistic']:>6.0f}{effect['rank_biserial_r']:>9.3f}"
              f"{item['recomputed_p']:>12.6f}{item['existing_p']:>14.6f}"
              f"{str(item['u_agrees'] and item['p_agrees']):>7}")
    print()
    print(f"Self-check 5: recomputation matches tgsim_analysis.json per-event values: "
          f"{checks['recomputation_matches_analysis_json']} "
          f"({len(checks['recomputation_mismatches'])} mismatches over "
          f"{sum(len(statistics[p]) for p in core.PAIR_ORDER)} events x {len(METRICS)} metrics)")
    print(f"Self-check 6: independent mid-rank U == scipy U for every metric: "
          f"{all(metrics_block[m]['effect_size']['u_agrees'] for m, _, _, _ in METRICS)}")
    print()
    print(f"all hard checks pass: {checks['all_hard_checks_pass']}")
    print(f"written: {core.rel(args.out)} ({args.out.stat().st_size} bytes)")
    leftovers = sorted(path.name for path in args.out.parent.glob("*.tmp"))
    print(f".tmp leftovers: {leftovers or 'none'}")
    reread = core.read_json(args.out)
    ok = reread["self_checks"]["all_hard_checks_pass"] == checks["all_hard_checks_pass"]
    print(f"on-disk reparse: {ok}")
    return 0 if (ok and checks["all_hard_checks_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
