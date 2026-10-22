#!/usr/bin/env python3
"""INTR5100 Lab Session I - Step C: leave-one-event-out CV and final parameters.

Step C of the study closes the calibration loop: instead of comparing the five
methods (M1..M5) on the data they were fitted to, it compares them on data they
have never seen, and picks the per-run winner with that evidence.

--- cross-validation scheme --------------------------------------------------

Leave-one-event-out, per run, on the stage 4 tables (the folds are the events the
preprocessing left in that run):

    run_1  events {event_2, event_3}  ->  2 folds (hold out e2 / e3)

A run_2 recording, once it is dropped into ``data/raw`` and the pipeline is
re-run, adds its own independent set of folds - the runs are never pooled.

Every fold:

  * the TRAINING events are handed to the method's OWN routine - the scripts
    ``m1_instant_ls`` .. ``m5_differentiable`` are imported and their
    ``calibrate_run`` is called as is, so the model, the cost, the bounds and
    the search of each method are bit-for-bit the ones of Step A/B. No algorithm
    is re-implemented here and nothing in the five scripts is modified;
  * the parameters that come back are used for a PURE PREDICTION of the held-out
    event: the simulation starts from the held-out event's first measured
    (s, v), is fed the measured ``lv`` series and is integrated forward over the
    whole event (``idm_core.event_metrics``). The held-out event influences
    nothing else - it is never used to fit, to rank or to scale.

A fold whose calibration or whose held-out prediction gives no usable trajectory
(a diverged rollout) is recorded as INFEASIBLE and does not interrupt the run
(a method that diverges on the full training set can also fail on a smaller
one).

--- metrics of one fold (held-out event, raw units) --------------------------

    spacing_rmse   sqrt(mean((s_sim - s_obs)^2))       original units
    speed_rmse     sqrt(mean((v_sim - v_obs)^2))       original units
    s_violations   integration steps that produced s <= 0
    objective_j    the event's normalised cost J_e (idm_core convention)

plus the ordinary regression metrics of each of the two series (spacing, with
s = dhw - L_VEH, and follower speed), all in the raw units of the measurement:

    mse  rmse  mae            coordinate units (units^2 for the mse)
    mape_percent              mean |residual / measurement| in percent; the
                              samples whose measurement is exactly 0 are skipped
                              (the ratio is undefined there) and the number of
                              skipped samples is reported alongside, so a MAPE
                              computed on a thinned series can never be mistaken
                              for one computed on the whole event
    r2                        1 - SS_res / SS_tot, SS_tot taken about the mean of
                              the MEASURED series of that event

Per run the folds are averaged with equal weight (spacing_rmse, speed_rmse
and every regression metric) and the violation counts are summed. R2 is the one
exception: a mean of per-fold R2 is not a coefficient of determination, so it is
RECOMPUTED on the pooled residuals - the residual sum of squares over every
held-out point of every fold, divided by the total sum of squares of those same
points about the mean of the fold each of them belongs to.

--- ranking ------------------------------------------------------------------

    score = mean_folds(spacing_rmse) / scale_s
          + mean_folds(speed_rmse)   / scale_v
          + VIOLATION_SCORE_PENALTY * sum_folds(s_violations)

``scale_s`` / ``scale_v`` are the POOLED standard deviations of the measured
spacing and speed over ALL of that run's events (every sample of every event
concatenated), so the two error kinds become comparable and the score is
dimensionless. An infeasible method does not enter the ranking at all (a
missing fold cannot be averaged into a mean without inventing a number).

The winner is retrained on ALL of that run's events with the method's FULL
budget (the cross-validation may run the global methods at half budget - see
below - but the final numbers are always the full-budget ones).

--- cost budget --------------------------------------------------------------

The brief allows the global methods to run at half budget inside the
cross-validation, because CV multiplies the number of optimisations by the
number of folds. That is what this script does, explicitly and only there:

    M1  single bounded least-squares            (no budget parameter)
    M2  13 multi-starts, maxiter 4000           FULL  (the module requires >= 11)
    M3  population 30 x generations 40          HALF of 60 x 80
    M4  20 particles x 40 iterations            HALF of 40 x 80
    M5  5 starts x 150 Adam steps               HALF of 10 x 300

The final retrain of the winner always uses the full budget (M3 60x80, M4 40x80,
M5 10x300, M2 13x4000). Both tables are written into the JSON, so which budget
produced which number is never ambiguous.

--- outputs ------------------------------------------------------------------

    results/cv_results.json     every fold / method metric, the per-run
                                averages, the scores and the ranking, and the
                                infeasible records
    results/final_params.json   per run: the winning method, the final
                                theta (v0, T, s0, a, b - delta = 4), the
                                full-set total_error and the per-event RMSEs,
                                plus ``selected_for_validation``: the run whose
                                parameters are handed to the validation replay

--- run_1 / run_2 selection ---------------------------------------------------

The two runs are recordings of the same platform on the same map, so the runs are
compared on the same *relative* scale: the winning method of each run is scored
with its own pooled spread as the unit

    normalized_score = mean_folds(spacing_rmse) / scale_s
                     + mean_folds(speed_rmse) / scale_v

(the error part of the stored ``score``, which adds ``1.0 * s_violations``). The
smaller score wins and its theta is stored as ``selected_for_validation`` together
with the comparison table and the reason, so the validation replay reads its
parameters from one place. The fold counts differ between the runs (run_1: 2
folds, run_2: its surviving events), which is exactly why the raw RMSEs cannot be
compared directly and are only listed as evidence.

plus the ranking table, the cross-run comparison table and the final parameters on
stdout. Timing is the only non-reproducible content: it is collected under a
single ``timing`` key so two runs can be compared field by field after dropping it.

Usage:
    python src/calibration/cross_validate.py [--root DIR] [--seed 20260921]
                                           [--method euler|rk4]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# This script drives the FIVE methods in ONE process, so scipy (conda/MKL, Intel
# OpenMP) and torch (its own bundled libiomp5md) are loaded together. Whichever of
# the two OpenMP runtimes initialises second aborts the whole process with
# "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already
# initialized" as soon as MKL enters a parallel region - measured: importing this
# module and then calling M1's first scipy.optimize.least_squares exits with code
# 3. KMP_DUPLICATE_LIB_OK=TRUE is Intel's documented switch for that mixed case,
# it is process-local, and it is set here - before numpy/scipy/torch are imported,
# and with setdefault so a caller can still override it. That the flag changes no
# number is verified, not assumed: in this runtime M1 reproduces the published
# Step A theta to the published 6-decimal rounding, and an M5 run in this mixed
# process prints byte-identical results to the same run in a torch-only process
# (see build-report.md, the runtime section).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402  (import after the OpenMP flag above)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import idm_core as core  # noqa: E402  (import after the path fix, same dir)
import m1_instant_ls as m1  # noqa: E402
import m2_neldermead as m2  # noqa: E402
import m3_ga as m3  # noqa: E402
import m4_pso as m4  # noqa: E402
import m5_differentiable as m5  # noqa: E402

SCRIPT = "src/calibration/cross_validate.py"
CV_OUTPUT_FILENAME = "cv_results.json"
FINAL_OUTPUT_FILENAME = "final_params.json"

METHOD_ORDER = ("M1", "M2", "M3", "M4", "M5")
METHOD_DESCRIPTIONS = {
    "M1": "instantaneous-acceleration least squares (bounded trf)",
    "M2": "Nelder-Mead trajectory fit, multi-start",
    "M3": "real-coded genetic algorithm",
    "M4": "gbest particle swarm",
    "M5": "differentiable IDM (torch) + Adam, multi-start",
}

DEFAULT_SEED = 20260921
VIOLATION_SCORE_PENALTY = 1.0
# ``idm_core.round_floats`` rounds the published JSON to 6 decimals, so a
# full-budget retrain that reproduces a Step A/B run exactly can still differ from
# the published number by up to half a unit in the 6th decimal.
PUBLISHED_RUN_TOLERANCE = 5.0e-7
SCORE_FORMULA = (
    "mean_folds(spacing_rmse)/scale_s + mean_folds(speed_rmse)/scale_v "
    "+ 1.0 * sum_folds(s_violations)"
)

# Regression metrics reported for both held-out series of every fold. The five
# entries are the ones the brief names; ``mape_skipped_points`` rides along for
# MAPE and the two sums are kept because the pooled R2 of a run is rebuilt
# from them (a mean of per-fold R2 would not be a coefficient of determination).
REGRESSION_METRIC_NAMES = ("mse", "rmse", "mae", "mape_percent", "r2")
REGRESSION_POOLED_R2_NOTE = (
    "pooled: 1 - sum(fold SS_res) / sum(fold SS_tot), each fold's SS_tot about "
    "that fold's own measured mean"
)

# Cross-validation budgets: the global methods run at HALF the Step A/B budget
# (their own module constants halved), M1/M2 keep theirs.
CV_BUDGETS: dict[str, dict[str, object]] = {
    "M1": {"mode": "full", "note": "one bounded least-squares fit (no budget parameter)"},
    "M2": {
        "mode": "full",
        "starts": m2.DEFAULT_STARTS,
        "maxiter": m2.DEFAULT_MAXITER,
        "note": "multi-start Nelder-Mead; the module requires at least 11 starts",
    },
    "M3": {
        "mode": "halved",
        "population": m3.POPULATION // 2,
        "generations": m3.GENERATIONS // 2,
        "full_population": m3.POPULATION,
        "full_generations": m3.GENERATIONS,
    },
    "M4": {
        "mode": "halved",
        "particles": m4.N_PARTICLES // 2,
        "iterations": m4.ITERATIONS // 2,
        "full_particles": m4.N_PARTICLES,
        "full_iterations": m4.ITERATIONS,
    },
    "M5": {
        "mode": "halved",
        "starts": max(2, m5.DEFAULT_STARTS // 2),
        "steps": m5.DEFAULT_STEPS // 2,
        "lr": m5.DEFAULT_LR,
        "full_starts": m5.DEFAULT_STARTS,
        "full_steps": m5.DEFAULT_STEPS,
    },
}

# Full budget, used for the final retrain of the winning method of each run.
FULL_BUDGETS: dict[str, dict[str, object]] = {
    "M1": {"mode": "full", "note": "one bounded least-squares fit (no budget parameter)"},
    "M2": {"mode": "full", "starts": m2.DEFAULT_STARTS, "maxiter": m2.DEFAULT_MAXITER},
    "M3": {"mode": "full", "population": m3.POPULATION, "generations": m3.GENERATIONS},
    "M4": {"mode": "full", "particles": m4.N_PARTICLES, "iterations": m4.ITERATIONS},
    "M5": {
        "mode": "full",
        "starts": m5.DEFAULT_STARTS,
        "steps": m5.DEFAULT_STEPS,
        "lr": m5.DEFAULT_LR,
    },
}

# The methods that were already run on the full data in Step A/B; their JSON is
# read (never recomputed) for the "calibration set" side of the comparison.
REFERENCE_RESULTS = {
    "M1": ("m1_params.json", m1.METHOD_NAME),
    "M2": ("m2_params.json", m2.METHOD_NAME),
    "M3": ("m3_params.json", m3.METHOD_NAME),
    "M4": ("m4_params.json", m4.METHOD_NAME),
    "M5": ("m5_params.json", m5.METHOD_NAME),
}


class FoldInfeasible(RuntimeError):
    """The calibration of one fold produced no usable parameter vector."""


# --- regression metrics -------------------------------------------------------


def regression_metrics(simulated, observed) -> dict[str, object]:
    """MSE / RMSE / MAE / MAPE / R2 of one simulated series against its measurement.

    All errors are in the original units of the series (the MSE in their square).
    ``mape_percent`` is ``100 * mean(|residual / measurement|)`` over the samples
    whose measurement is non-zero; the zero-denominator samples are skipped and
    counted in ``mape_skipped_points`` (skipping is the convention the brief asks
    for - a single zero would otherwise turn the whole MAPE into an infinity).
    ``r2`` is ``1 - SS_res / SS_tot`` with ``SS_tot`` about the mean of the
    MEASURED series, and the two sums are returned as well so a run can pool
    them across folds instead of averaging the per-fold R2.

    A non-finite input (which only a diverged rollout produces) returns ``None``
    metrics rather than a fabricated number.
    """
    predicted = np.asarray(simulated, dtype=float)
    actual = np.asarray(observed, dtype=float)
    if predicted.shape != actual.shape:
        raise ValueError("simulated and observed series must have the same shape")
    n_points = int(actual.size)
    if n_points == 0:
        raise ValueError("cannot compute regression metrics of an empty series")
    residual = predicted - actual
    blank: dict[str, object] = {name: None for name in REGRESSION_METRIC_NAMES}
    blank.update(n_points=n_points, mape_skipped_points=n_points, ss_res=None, ss_tot=None)
    if not np.all(np.isfinite(residual)):
        return blank

    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((actual - float(np.mean(actual))) ** 2))
    non_zero = actual != 0.0
    n_skipped = int(n_points - int(np.count_nonzero(non_zero)))
    mape = (
        float(np.mean(np.abs(residual[non_zero] / actual[non_zero])) * 100.0)
        if n_skipped < n_points
        else None
    )
    return {
        "mse": ss_res / n_points,
        "rmse": float(np.sqrt(ss_res / n_points)),
        "mae": float(np.mean(np.abs(residual))),
        "mape_percent": mape,
        "mape_skipped_points": n_skipped,
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else None,
        "n_points": n_points,
        "ss_res": ss_res,
        "ss_tot": ss_tot,
    }


def prediction_metrics(event, theta, integration: str) -> dict[str, object]:
    """The regression block of ONE predicted series pair (spacing and speed).

    The simulation is the shared ``idm_core.simulate_event`` - first measured
    ``(s, v)`` as the initial state, measured ``lv`` as the leader input, real
    time deltas - i.e. exactly the pure prediction the brief specifies. Only the
    two residual series are new here; the model, the integration and the cost are
    still the ones of :mod:`idm_core`.
    """
    data = core.as_event_data(event)
    spacing, speed, status = core.simulate_event(data, theta, integration, return_status=True)
    feasible = not status["diverged"] and bool(status["finite"])
    block: dict[str, object] = {
        "feasible": feasible,
        "diverged": bool(status["diverged"]),
        "s_violations": int(status["s_violations"]),
        "n_points": data.n,
        "spacing": regression_metrics(spacing, data.s_obs),
        "speed": regression_metrics(speed, data.v_obs),
    }
    return block


def pooled_r2(blocks: list[dict[str, object]], series: str) -> float | None:
    """R2 rebuilt from pooled residuals: ``1 - sum(SS_res) / sum(SS_tot)``.

    ``blocks`` are per-fold (or per-event) regression blocks; each one carries the
    sums of its own series. Pooling them this way measures the fit on every
    held-out point at once, with each fold's total sum of squares taken about the
    mean of the fold that owns the point - the "合并口径" of the brief. Averaging
    the per-fold R2 instead would let a short fold outvote a long one.
    """
    ss_res = 0.0
    ss_tot = 0.0
    for block in blocks:
        values = block.get(series)
        if not isinstance(values, dict):
            return None
        if values.get("ss_res") is None or values.get("ss_tot") is None:
            return None
        ss_res += float(values["ss_res"])
        ss_tot += float(values["ss_tot"])
    if ss_tot <= 0.0:
        return None
    return 1.0 - ss_res / ss_tot


def mean_metric(blocks: list[dict[str, object]], series: str, metric: str) -> float | None:
    """Equal-weight mean of one regression metric over the given blocks."""
    values: list[float] = []
    for block in blocks:
        series_block = block.get(series)
        if not isinstance(series_block, dict):
            return None
        value = series_block.get(metric)
        if value is None:
            return None
        values.append(float(value))
    if not values:
        return None
    return float(np.mean(values))


def aggregate_regression(blocks: list[dict[str, object]], unit_note: str) -> dict[str, object]:
    """Per-method regression summary over the (feasible) folds of one run.

    Error metrics are averaged with equal weight over the folds; R2 is pooled from
    the fold sums; the MAPE skip counts are summed so a reader can see that no
    MAPE was silently computed on a thinned series.
    """
    summary: dict[str, object] = {"unit": unit_note, "r2_aggregation": REGRESSION_POOLED_R2_NOTE}
    for series in ("spacing", "speed"):
        entry: dict[str, object] = {
            f"mean_{metric}": mean_metric(blocks, series, metric)
            for metric in REGRESSION_METRIC_NAMES
        }
        entry["pooled_r2"] = pooled_r2(blocks, series)
        entry[f"sum_mape_skipped_points"] = int(
            sum(
                int(block[series]["mape_skipped_points"])  # type: ignore[index]
                for block in blocks
                if isinstance(block.get(series), dict)
                and block[series].get("mape_skipped_points") is not None  # type: ignore[union-attr]
            )
        )
        entry["n_folds"] = len(blocks)
        entry["fold_mse"] = [block[series]["mse"] for block in blocks]  # type: ignore[index]
        entry["fold_rmse"] = [block[series]["rmse"] for block in blocks]  # type: ignore[index]
        entry["fold_mae"] = [block[series]["mae"] for block in blocks]  # type: ignore[index]
        entry["fold_mape_percent"] = [block[series]["mape_percent"] for block in blocks]  # type: ignore[index]
        entry["fold_r2"] = [block[series]["r2"] for block in blocks]  # type: ignore[index]
        entry["fold_ss_res"] = [block[series]["ss_res"] for block in blocks]  # type: ignore[index]
        entry["fold_ss_tot"] = [block[series]["ss_tot"] for block in blocks]  # type: ignore[index]
        summary[series] = entry
    summary["mean_rmse_matches_spacing_rmse"] = None
    return summary


# --- method adapters (import only, no algorithm is copied) -------------------


def _run_method(method: str, train_events, budget: dict, seed: int, integration: str):
    """Call the method's OWN ``calibrate_run`` on the training events.

    Every adapter forwards the module's own configuration constants (except the
    halved budget, where the brief allows it), so the CV uses exactly the search
    of Step A/B. Nothing is re-implemented.
    """
    tag = "cv"
    if method == "M1":
        return m1.calibrate_run(tag, train_events, integration, core.DEFAULT_THETA)
    if method == "M2":
        return m2.calibrate_run(
            tag, train_events, integration, int(budget["starts"]), seed,
            int(budget["maxiter"]), verbose=False,
        )
    if method == "M3":
        return m3.calibrate_run(
            tag, train_events, integration, int(budget["population"]),
            int(budget["generations"]), seed, verbose=False,
        )
    if method == "M4":
        return m4.calibrate_run(
            tag, train_events, integration, int(budget["particles"]),
            int(budget["iterations"]), seed, verbose=False,
        )
    if method == "M5":
        # The torch RNG is seeded per run, exactly as the M5 entry point does.
        m5.configure_torch(seed)
        return m5.calibrate_run(
            tag, train_events, int(budget["starts"]), int(budget["steps"]),
            float(budget["lr"]), seed, verbose=False,
        )
    raise ValueError(f"unknown method {method!r}")


def calibrate(train_events, method: str, budget: dict, seed: int, integration: str):
    """Train one fold: returns ``(theta, training_total_error)`` or raises.

    A method that raises (M2/M3/M4/M5 raise ``RuntimeError`` when no feasible
    start exists), that returns a non-finite vector, or whose parameters give an
    infeasible training trajectory is reported as an infeasible fold - the label
    the brief asks for - instead of being fitted with a fabricated number.
    """
    try:
        result = _run_method(method, train_events, budget, seed, integration)
    except Exception as exc:  # noqa: BLE001 - any failure is an infeasible fold
        raise FoldInfeasible(f"{type(exc).__name__}: {exc}") from exc

    theta = np.asarray(result["theta_vector"], dtype=float).reshape(-1)
    if theta.size != core.N_PARAMS or not np.all(np.isfinite(theta)):
        raise FoldInfeasible("the calibration returned a non-finite parameter vector")
    if not core.inside_bounds(theta):
        raise FoldInfeasible("the calibration returned a vector outside the parameter box")
    datasets = [data for _, data in train_events]
    train_cost = core.total_error(datasets, theta, integration)
    if not math.isfinite(train_cost):
        raise FoldInfeasible(
            "the calibrated parameters give a diverged trajectory on the training events"
        )
    return theta, float(train_cost)


# --- fold runner -------------------------------------------------------------


def run_fold(
    run: str,
    fold_index: int,
    held_name: str,
    events,
    method: str,
    budget: dict,
    seed: int,
    integration: str,
) -> dict[str, object]:
    """One (method, run, fold) record: train on the rest, predict the rest."""
    train_events = [item for index, item in enumerate(events) if index != fold_index]
    held_data = events[fold_index][1]

    record: dict[str, object] = {
        "run": run,
        "fold": f"leave_{held_name}_out",
        "held_out_event": held_name,
        "method": method,
        "method_description": METHOD_DESCRIPTIONS[method],
        "training_events": [name for name, _ in train_events],
        "n_training_events": len(train_events),
        "budget_mode": budget["mode"],
        "infeasible": False,
        "infeasible_reason": None,
        "theta": None,
        "train_total_error": None,
        "metrics": None,
        "regression": None,
        "elapsed_seconds": 0.0,
    }

    started = time.perf_counter()
    try:
        theta, train_cost = calibrate(train_events, method, budget, seed, integration)
    except FoldInfeasible as exc:
        record["infeasible"] = True
        record["infeasible_reason"] = f"calibration: {exc}"
        record["elapsed_seconds"] = float(time.perf_counter() - started)
        return record

    record["theta"] = core.theta_to_dict(theta)
    record["theta_vector"] = [float(value) for value in theta]
    record["train_total_error"] = train_cost

    # Pure prediction of the held-out event: first measured (s, v), measured lv,
    # forward integration over the whole event. The held-out event is used for
    # nothing else.
    metrics = core.event_metrics(held_data, theta, integration)
    if metrics["diverged"] or not metrics["finite"] or not math.isfinite(float(metrics["j_total"])):
        record["infeasible"] = True
        record["infeasible_reason"] = (
            "held-out prediction: the simulation diverged on the held-out event"
        )
        record["elapsed_seconds"] = float(time.perf_counter() - started)
        return record

    record["metrics"] = {
        "spacing_rmse": float(metrics["spacing_rmse"]),
        "speed_rmse": float(metrics["speed_rmse"]),
        "s_violations": int(metrics["s_violations"]),
        "objective_j": float(metrics["j_total"]),
        "j_spacing": float(metrics["j_spacing"]),
        "j_speed": float(metrics["j_speed"]),
        "spacing_rmse_normalized": float(metrics["spacing_rmse_normalized"]),
        "speed_rmse_normalized": float(metrics["speed_rmse_normalized"]),
        "n_samples": int(metrics["n_samples"]),
        "diverged": False,
        "zero_model_best_spacing_rmse": float(metrics["zero_model"]["best_spacing_rmse"]),
        "zero_model_best_speed_rmse": float(metrics["zero_model"]["best_speed_rmse"]),
    }

    # The regression metrics of the same prediction (MSE/RMSE/MAE/MAPE/R2 of the
    # spacing and of the speed, original units). The two RMSEs are recomputed from
    # the residual series of the shared simulator, so they must reproduce the ones
    # ``event_metrics`` reports - the flag below is that self-check, and ``main``
    # asserts it, i.e. the printed table can never drift from the JSON.
    regression = prediction_metrics(held_data, theta, integration)
    regression["rmse_matches_core"] = bool(
        regression["feasible"]
        and math.isclose(
            float(regression["spacing"]["rmse"]), float(metrics["spacing_rmse"]),
            rel_tol=1e-9, abs_tol=1e-9,
        )
        and math.isclose(
            float(regression["speed"]["rmse"]), float(metrics["speed_rmse"]),
            rel_tol=1e-9, abs_tol=1e-9,
        )
    )
    record["regression"] = regression
    record["elapsed_seconds"] = float(time.perf_counter() - started)
    return record


# --- per-run aggregation --------------------------------------------------


def pooled_scales(events) -> tuple[float, float]:
    """Pooled standard deviations of the measured s and v over ALL the run's events.

    Every sample of every event is concatenated before the standard deviation is
    taken, so the scale describes the whole run rather than one event (the
    per-event sigma of the cost function normalises inside an event; this one
    normalises the cross-validated errors between runs).
    """
    spacing = np.concatenate([data.s_obs for _, data in events])
    speed = np.concatenate([data.v_obs for _, data in events])
    return float(np.std(spacing)), float(np.std(speed))


def aggregate_run(run: str, events, fold_records: dict[str, dict[str, dict]]) -> dict[str, object]:
    """Per-run averages, scores and ranking over the five methods."""
    scale_s, scale_v = pooled_scales(events)
    per_method: dict[str, dict[str, object]] = {}
    for method in METHOD_ORDER:
        records = [fold_records[fold][method] for fold in fold_records]
        feasible = [record for record in records if not record["infeasible"]]
        infeasible = [record for record in records if record["infeasible"]]
        entry: dict[str, object] = {
            "method": method,
            "method_description": METHOD_DESCRIPTIONS[method],
            "budget_mode": CV_BUDGETS[method]["mode"],
            "n_folds": len(records),
            "n_feasible_folds": len(feasible),
            "n_infeasible_folds": len(infeasible),
            "infeasible_folds": [
                {
                    "fold": record["fold"],
                    "held_out_event": record["held_out_event"],
                    "reason": record["infeasible_reason"],
                }
                for record in infeasible
            ],
        }
        if feasible:
            spacing = [float(record["metrics"]["spacing_rmse"]) for record in feasible]
            speed = [float(record["metrics"]["speed_rmse"]) for record in feasible]
            violations = [int(record["metrics"]["s_violations"]) for record in feasible]
            mean_spacing = float(np.mean(spacing))
            mean_speed = float(np.mean(speed))
            violation_sum = int(np.sum(violations))
            score = (
                mean_spacing / scale_s
                + mean_speed / scale_v
                + VIOLATION_SCORE_PENALTY * violation_sum
            )
            regression = aggregate_regression(
                [record["regression"] for record in feasible],
                "coordinate units for the spacing, coordinate units/s for the speed",
            )
            # The regression table and the CV table must tell the same story: the
            # mean of the per-fold spacing RMSE recomputed from the residual series
            # has to be the mean_spacing_rmse the ranking uses.
            regression["mean_rmse_matches_spacing_rmse"] = bool(
                regression["spacing"]["mean_rmse"] is not None
                and math.isclose(
                    float(regression["spacing"]["mean_rmse"]), mean_spacing,
                    rel_tol=1e-9, abs_tol=1e-9,
                )
                and math.isclose(
                    float(regression["speed"]["mean_rmse"]), mean_speed,
                    rel_tol=1e-9, abs_tol=1e-9,
                )
            )
            entry.update(
                mean_spacing_rmse=mean_spacing,
                mean_speed_rmse=mean_speed,
                s_violations_sum=violation_sum,
                s_violations_penalty=float(VIOLATION_SCORE_PENALTY * violation_sum),
                score=float(score),
                mean_objective_j=float(np.mean([float(r["metrics"]["objective_j"]) for r in feasible])),
                fold_spacing_rmse=spacing,
                fold_speed_rmse=speed,
                fold_s_violations=violations,
                fold_held_out=[record["held_out_event"] for record in feasible],
                regression=regression,
            )
        else:
            entry.update(
                mean_spacing_rmse=None,
                mean_speed_rmse=None,
                s_violations_sum=None,
                s_violations_penalty=None,
                score=None,
                regression=None,
            )
        per_method[method] = entry

    ranked = sorted(
        (entry for entry in per_method.values() if entry["score"] is not None),
        key=lambda entry: float(entry["score"]),
    )
    ranking: list[dict[str, object]] = []
    for position, entry in enumerate(ranked, start=1):
        entry["rank"] = position
        ranking.append(
            {
                "rank": position,
                "method": entry["method"],
                "method_description": entry["method_description"],
                "score": entry["score"],
                "mean_spacing_rmse": entry["mean_spacing_rmse"],
                "mean_speed_rmse": entry["mean_speed_rmse"],
                "s_violations_sum": entry["s_violations_sum"],
                "feasible_folds": entry["n_feasible_folds"],
                "infeasible_folds": entry["n_infeasible_folds"],
            }
        )
    excluded = [
        {
            "method": entry["method"],
            "reason": "no feasible fold: the method does not participate in the ranking",
            "infeasible_folds": entry["infeasible_folds"],
        }
        for entry in per_method.values()
        if entry["score"] is None
    ]
    return {
        "scale_s": scale_s,
        "scale_v": scale_v,
        "score_formula": SCORE_FORMULA,
        "events": [
            {"event": name, "n_samples": data.n} for name, data in events
        ],
        "folds": list(fold_records),
        "per_method": per_method,
        "ranking": ranking,
        "best_method": ranking[0]["method"] if ranking else None,
        "excluded_methods": excluded,
    }


def calibration_set_reference(root: Path, run: str, events, integration: str) -> dict[str, object]:
    """The full-data Step A/B numbers, READ from ``results/m*_params.json``.

    These are the reference side of the comparison: the same five methods on the
    whole data set, where every event is training data. Reading them (instead of
    recomputing) keeps the two sides independent: the CV numbers cannot be
    contaminated by the fitted set.

    The stored parameter vector is replayed through the shared simulator to add
    the SAME regression metrics the cross-validation reports (each event then plays
    the role of a fold: every event is in the training set, and the per-event
    metrics of steps A/B are exactly the fit-set counterpart of a held-out event).
    Nothing is fitted here - the parameters are the published ones.
    """
    reference: dict[str, object] = {}
    for method, (filename, _name) in REFERENCE_RESULTS.items():
        path = core.results_dir(root) / filename
        entry: dict[str, object] = {"source": core.rel(path, root)}
        if not path.is_file():
            entry["available"] = False
            entry["note"] = "the file of this method is missing"
            reference[method] = entry
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        run_payload = (payload.get("runs") or {}).get(run)
        if run_payload is None:
            entry["available"] = False
            entry["note"] = "this method has no entry for the run"
            reference[method] = entry
            continue
        entry.update(
            available=True,
            theta=run_payload.get("theta"),
            total_error=run_payload.get("total_error"),
            mean_spacing_rmse=run_payload.get("mean_spacing_rmse"),
            mean_speed_rmse=run_payload.get("mean_speed_rmse"),
            per_event={
                name: {
                    "spacing_rmse": values.get("spacing_rmse"),
                    "speed_rmse": values.get("speed_rmse"),
                    "j_total": values.get("j_total"),
                    "s_violations": values.get("s_violations"),
                }
                for name, values in (run_payload.get("events") or {}).items()
            },
        )
        theta_values = run_payload.get("theta")
        if not theta_values:
            entry["regression"] = None
            entry["regression_note"] = (
                "the reference run of this method has no usable parameter vector for "
                "this run (see its total_error: null), so it cannot be replayed"
            )
            reference[method] = entry
            continue
        theta = core.theta_from_dict({name: float(value) for name, value in theta_values.items()})
        blocks = [prediction_metrics(data, theta, integration) for _name, data in events]
        entry["per_event_regression"] = {
            name: block for (name, _data), block in zip(events, blocks)
        }
        diverged = [name for (name, _data), block in zip(events, blocks) if not block["feasible"]]
        if diverged:
            entry["regression"] = None
            entry["regression_note"] = (
                f"the published parameters diverge on {diverged}: no regression metrics"
            )
        else:
            entry["regression"] = aggregate_regression(
                blocks,
                "coordinate units for the spacing, coordinate units/s for the speed",
            )
        reference[method] = entry
    return reference


# --- printing ----------------------------------------------------------------


# stdlib rank table: (header, width, decimals). ``decimals is None`` = text
# (left aligned), ``0`` = integer, ``>0`` = fixed point; a missing metric prints
# as ``n/a`` rather than as a zero that would look like a perfect fit.
RANKING_COLUMNS = (
    ("rank", 5, None),
    ("m", 4, None),
    ("s_RMSE", 10, 4),
    ("s_MAE", 10, 4),
    ("s_MAPE%", 9, 3),
    ("s_R2", 8, 4),
    ("s_J_e", 9, 3),
    ("v_RMSE", 10, 4),
    ("v_MAE", 10, 4),
    ("v_MAPE%", 9, 3),
    ("v_R2", 8, 4),
    ("viol", 6, 0),
    ("score", 10, 4),
    ("folds", 7, None),
)


def _format_cell(cell, width: int, decimals: int | None) -> str:
    """One fixed-width table cell (see :data:`RANKING_COLUMNS`)."""
    if decimals is None:
        return f"{str(cell):<{width}}"
    if cell is None:
        return f"{'n/a':>{width}}"
    if decimals == 0:
        return f"{int(cell):>{width}d}"
    return f"{float(cell):>{width}.{decimals}f}"


def ranking_table(aggregate: dict[str, object]) -> list[str]:
    """The ranking rows as formatted lines (header first, then one row per method)."""
    lines = ["  " + " ".join(_format_cell(name, width, None) for name, width, _ in RANKING_COLUMNS)]
    for row in aggregate["ranking"]:
        per_method = aggregate["per_method"][row["method"]]
        regression = per_method["regression"]
        cells = [
            row["rank"],
            row["method"],
            regression["spacing"]["mean_rmse"],
            regression["spacing"]["mean_mae"],
            regression["spacing"]["mean_mape_percent"],
            regression["spacing"]["pooled_r2"],
            per_method["mean_objective_j"],
            regression["speed"]["mean_rmse"],
            regression["speed"]["mean_mae"],
            regression["speed"]["mean_mape_percent"],
            regression["speed"]["pooled_r2"],
            row["s_violations_sum"],
            row["score"],
            f"{row['feasible_folds']}/{row['feasible_folds'] + row['infeasible_folds']}",
        ]
        lines.append(
            "  " + " ".join(
                _format_cell(cell, width, decimals)
                for cell, (_name, width, decimals) in zip(cells, RANKING_COLUMNS)
            )
        )
    return lines


def print_ranking(run: str, aggregate: dict[str, object]) -> None:
    """The per-run ranking table on stdout (brief: method + CV metrics + score).

    ``s_`` columns are the held-out spacing (``dhw - 250``, coordinate units),
    ``v_`` columns the held-out follower speed (coordinate units/s); ``s_J_e`` is
    the held-out event's normalised cost, kept next to the raw errors so the two
    scales can be read against each other. ``viol`` is the summed
    ``s_violations`` of the folds and ``score`` the value the winner is picked by.
    """
    print(f"  scale_s={aggregate['scale_s']:.4f}  scale_v={aggregate['scale_v']:.4f}"
          f"   (pooled over {len(aggregate['events'])} event(s))")
    print("  columns: s_* = held-out spacing (coordinate units), "
          "v_* = held-out speed (coordinate units/s),")
    print("           s_J_e = held-out normalised cost, viol = summed s_violations, "
          "folds = feasible/total")
    for line in ranking_table(aggregate):
        print(line)
    print()
    for entry in aggregate["excluded_methods"]:
        print(f"  excluded: {entry['method']} - {entry['reason']}")
        for fold in entry["infeasible_folds"]:
            print(f"      {fold['fold']:<22} {fold['reason']}")


def _num(value, decimals: int = 4, suffix: str = "") -> str:
    """Inline number for the stdout blocks (``n/a`` when a metric does not exist)."""
    if value is None:
        return "n/a"
    return f"{float(value):.{decimals}f}{suffix}"


def print_final(run: str, final: dict[str, object]) -> None:
    """The per-run final parameter block on stdout."""
    theta = core.as_theta(final["theta_vector"])
    print(f"  method             : {final['method']} - {final['method_description']}")
    print(f"  final theta        : {core.format_theta(theta)}   (delta = {core.DELTA:g} fixed)")
    print(f"  within bounds      : {final['parameters_within_bounds']}"
          f"   at a bound: {final['parameters_at_bound'] or 'none'}")
    print(f"  full-set J         : {final['total_error']:.6f}"
          f"  (equal weight mean of {final['n_events']} events, {final['n_samples']} samples)")
    print(f"  mean RMSE          : spacing {final['mean_spacing_rmse']:.4f}"
          f"   speed {final['mean_speed_rmse']:.4f}")
    regression = final.get("regression")
    if isinstance(regression, dict):
        print("  fit-set regression : "
              f"spacing RMSE {_num(regression['spacing']['mean_rmse'])}"
              f"  MAE {_num(regression['spacing']['mean_mae'])}"
              f"  MAPE {_num(regression['spacing']['mean_mape_percent'], 3, '%')}"
              f"  R2 {_num(regression['spacing']['pooled_r2'])}"
              f"   |   speed RMSE {_num(regression['speed']['mean_rmse'])}"
              f"  MAE {_num(regression['speed']['mean_mae'])}"
              f"  MAPE {_num(regression['speed']['mean_mape_percent'], 3, '%')}"
              f"  R2 {_num(regression['speed']['pooled_r2'])}")
    print(f"  {'event':<10}{'n':>5}{'s_RMSE':>11}{'s_MAE':>11}{'s_MAPE%':>9}{'s_R2':>9}"
          f"{'v_RMSE':>11}{'v_MAE':>11}{'v_MAPE%':>9}{'v_R2':>9}{'J_e':>11}{'viol':>6}")
    per_event_regression = final.get("per_event_regression") or {}
    for name, metrics in final["events"].items():
        regression_block = per_event_regression.get(name) or {}
        spacing = regression_block.get("spacing") or {}
        speed = regression_block.get("speed") or {}
        print(
            f"  {name:<10}{metrics['n_samples']:>5}{metrics['spacing_rmse']:>11.4f}"
            f"{_num(spacing.get('mae')):>11}{_num(spacing.get('mape_percent'), 3):>9}"
            f"{_num(spacing.get('r2')):>9}"
            f"{metrics['speed_rmse']:>11.4f}{_num(speed.get('mae')):>11}"
            f"{_num(speed.get('mape_percent'), 3):>9}{_num(speed.get('r2')):>9}"
            f"{metrics['j_total']:>11.4f}{metrics['s_violations']:>6d}"
        )


# --- main --------------------------------------------------------------------


def inside_and_at_bound(theta) -> tuple[bool, list[str]]:
    """Box membership plus the names of the coordinates sitting ON a bound."""
    vector = core.as_theta(theta)
    at_bound = [
        name
        for index, name in enumerate(core.PARAM_NAMES)
        if math.isclose(float(vector[index]), float(core.LOWER[index]), rel_tol=0.0, abs_tol=1e-6)
        or math.isclose(float(vector[index]), float(core.UPPER[index]), rel_tol=0.0, abs_tol=1e-6)
    ]
    return core.inside_bounds(vector), at_bound


def collect_self_checks(
    fold_records: dict[str, dict[str, dict[str, dict]]],
    aggregates: dict[str, dict[str, object]],
    finals: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Structural self-checks of the step C numbers (stored in the JSON, asserted).

    Three things could silently rot the comparison and are therefore checked
    mechanically rather than by eye:

      * every feasible fold recomputed its RMSEs from the residual series and got
        the same values as the shared ``idm_core`` metrics it reports beside them;
      * the regression table's mean RMSE equals the ``mean_spacing_rmse`` /
        ``mean_speed_rmse`` the ranking actually uses;
      * the final block has regression metrics for every event it reports, and the
        runs that hit a spacing violation are flagged (a violation means the
        printed error of that fold was produced with the spacing clamped to
        ``S_FLOOR``, which a reader must be told).
    """
    per_fold: dict[str, bool] = {}
    for run, folds in fold_records.items():
        for fold, methods in folds.items():
            for method, record in methods.items():
                if record["infeasible"]:
                    continue
                per_fold[f"{run}/{fold}/{method}"] = bool(record["regression"]["rmse_matches_core"])
    per_method: dict[str, bool] = {}
    mape_skipped: dict[str, object] = {}
    methods_with_violations: list[str] = []
    for run, aggregate in aggregates.items():
        for method, entry in aggregate["per_method"].items():
            if entry["regression"] is None:
                continue
            key = f"{run}/{method}"
            per_method[key] = bool(entry["regression"]["mean_rmse_matches_spacing_rmse"])
            mape_skipped[key] = {
                "spacing": entry["regression"]["spacing"]["sum_mape_skipped_points"],
                "speed": entry["regression"]["speed"]["sum_mape_skipped_points"],
            }
            if int(entry["s_violations_sum"] or 0) > 0:
                methods_with_violations.append(key)
    finals_complete = {
        run: all(
            block.get("spacing") and block.get("speed")
            for block in (entry.get("per_event_regression") or {}).values()
        )
        for run, entry in finals.items()
    }
    retrain_reproduced = {
        run: (entry.get("published_full_budget_run") or {}).get("reproduced")
        for run, entry in finals.items()
    }
    return {
        "per_fold_rmse_matches_idm_core": per_fold,
        "all_per_fold_rmse_match_idm_core": bool(per_fold) and all(per_fold.values()),
        "per_method_mean_rmse_matches_cv_table": per_method,
        "all_mean_rmse_match_cv_table": bool(per_method) and all(per_method.values()),
        "mape_skipped_points": mape_skipped,
        "mape_skipped_points_total": int(
            sum(
                int(item["spacing"]) + int(item["speed"])
                for item in mape_skipped.values()
            )
        ),
        "methods_with_spacing_violations": methods_with_violations,
        "final_per_event_regression_complete": finals_complete,
        "all_final_blocks_complete": bool(finals_complete) and all(finals_complete.values()),
        "full_budget_retrain_reproduces_published_run": retrain_reproduced,
        "all_full_budget_retrains_reproduce_published_run": all(
            value is not False for value in retrain_reproduced.values()
        ),
    }


def select_for_validation(
    aggregates: dict[str, dict[str, object]],
    finals: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Pick the run whose parameters are handed to the validation replay.

    The two runs are recordings of the same platform on the same map, so their
    parameters can only be compared on a *relative* scale: each run's errors are
    normalised by the pooled spread of that run's own measured signal

        normalized_score = mean_folds(spacing_rmse) / scale_s
                         + mean_folds(speed_rmse) / scale_v

    which is exactly the error part of the cross-validation score (the stored
    ``score`` adds ``1.0 * sum(s_violations)`` on top). The fold structure of the
    two runs differs (run_1: 2 folds, run_2: 8), so this normalisation is what
    makes the two numbers comparable at all.
    """
    rows: list[dict[str, object]] = []
    for run, entry in finals.items():
        aggregate = aggregates[run]
        method = str(entry["method"])
        per_method = aggregate["per_method"][method]  # type: ignore[index]
        scale_s = float(aggregate["scale_s"])
        scale_v = float(aggregate["scale_v"])
        spacing = float(per_method["mean_spacing_rmse"])
        speed = float(per_method["mean_speed_rmse"])
        violations = int(per_method["s_violations_sum"])
        regression = per_method["regression"] or {}
        rows.append(
            {
                "run": run,
                "method": method,
                "method_description": entry["method_description"],
                "cv_score": float(entry["cv_score"]),
                "normalized_score": spacing / scale_s + speed / scale_v,
                "score_with_violation_penalty": float(entry["cv_score"]),
                "mean_spacing_rmse": spacing,
                "mean_speed_rmse": speed,
                "s_violations_sum": violations,
                "spacing_mape_percent": (regression.get("spacing") or {}).get("mean_mape_percent"),
                "speed_mape_percent": (regression.get("speed") or {}).get("mean_mape_percent"),
                "scale_s": scale_s,
                "scale_v": scale_v,
                "n_folds": per_method["n_folds"],
                "n_feasible_folds": per_method["n_feasible_folds"],
                "n_events_trained_on_full_set": entry["n_events"],
                "n_samples_trained_on_full_set": entry["n_samples"],
                "mean_objective_j": per_method["mean_objective_j"],
                "theta": entry["theta"],
                "theta_vector": entry["theta_vector"],
            }
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["normalized_score"]),
            float(row["score_with_violation_penalty"]),
            str(row["run"]),
        ),
    )
    winner = ranked[0]
    scaled = ", ".join(
        f"{row['run']} {float(row['normalized_score']):.6f}" for row in ranked
    )
    reason = (
        f"{winner['run']} / {winner['method']} has the smallest normalised "
        f"cross-validation score ({scaled}): on its own pooled scale its held-out "
        f"folds average {float(winner['mean_spacing_rmse']):.4f} spacing units "
        f"({float(winner['mean_spacing_rmse']) / float(winner['scale_s']):.4f} sigma_s) "
        f"and {float(winner['mean_speed_rmse']):.4f} speed units/s "
        f"({float(winner['mean_speed_rmse']) / float(winner['scale_v']):.4f} sigma_v) "
        f"with {winner['s_violations_sum']} spacing violation(s) summed over its folds"
        + (
            "; every method of both runs is violation free, so the stored score "
            "orders the runs exactly like the normalised score does."
            if all(row["s_violations_sum"] == 0 for row in ranked)
            else "; the violation penalty of the stored score is listed beside the "
            "normalised score (a fold that hit a spacing violation was scored with "
            "the spacing clamped to S_FLOOR)."
        )
    )
    return {
        "rule": (
            "hand the parameters of the run with the smallest comparable "
            "cross-validation score to the validation replay"
        ),
        "criterion": (
            "normalized_score = mean_folds(spacing_rmse)/scale_s + "
            "mean_folds(speed_rmse)/scale_v, each run normalised by the pooled "
            "spread of its own measured signal (same platform, same map units); "
            "the stored score adds 1.0 * s_violations and is reported beside it"
        ),
        "selected_run": winner["run"],
        "method": winner["method"],
        "method_description": winner["method_description"],
        "theta": winner["theta"],
        "theta_vector": winner["theta_vector"],
        "normalized_score": winner["normalized_score"],
        "score_with_violation_penalty": winner["score_with_violation_penalty"],
        "comparison_table": ranked,
        "reason": reason,
        "self_check_selected_is_argmin": bool(
            math.isclose(
                float(winner["normalized_score"]),
                min(float(row["normalized_score"]) for row in rows),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ),
        "note": (
            "raw spacing/speed RMSE are in coordinate units and are not comparable "
            "between runs on their own (sampling and spread of the two recordings "
            "differ); they are listed as evidence next to the normalised score that "
            "drives the choice"
        ),
    }


def print_selection(selection: dict[str, object]) -> None:
    """The cross-run comparison table and the choice on stdout."""
    print("=" * 78)
    print("run_1 / run_2 selection for the validation replay")
    print("=" * 78)
    print(f"criterion: {selection['criterion']}")
    print()
    columns = (
        ("run", 7, None),
        ("method", 7, None),
        ("norm_score", 11, 6),
        ("score+viol", 11, 6),
        ("s_RMSE", 10, 4),
        ("s/sigma_s", 10, 4),
        ("v_RMSE", 10, 4),
        ("v/sigma_v", 10, 4),
        ("s_MAPE%", 9, 3),
        ("v_MAPE%", 9, 3),
        ("viol", 5, 0),
        ("folds", 8, None),
    )
    print("  " + " ".join(_format_cell(name, width, None) for name, width, _ in columns))
    for row in selection["comparison_table"]:  # type: ignore[union-attr]
        cells = (
            row["run"],
            row["method"],
            row["normalized_score"],
            row["score_with_violation_penalty"],
            row["mean_spacing_rmse"],
            float(row["mean_spacing_rmse"]) / float(row["scale_s"]),
            row["mean_speed_rmse"],
            float(row["mean_speed_rmse"]) / float(row["scale_v"]),
            row["spacing_mape_percent"],
            row["speed_mape_percent"],
            row["s_violations_sum"],
            f"{row['n_feasible_folds']}/{row['n_folds']}",
        )
        print(
            "  " + " ".join(
                _format_cell(cell, width, decimals)
                for cell, (_name, width, decimals) in zip(cells, columns)
            )
        )
    print()
    print(f"selected for validation : {selection['selected_run']} / {selection['method']}")
    theta = core.as_theta(selection["theta_vector"])
    print(f"  theta = {core.format_theta(theta)}   (delta = {core.DELTA:g} fixed)")
    print(f"  self-check (argmin) : {selection['self_check_selected_is_argmin']}")
    print(f"  reason: {selection['reason']}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="project root (default: inferred)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"seed shared by the multi-start methods (default {DEFAULT_SEED})")
    parser.add_argument("--method", default=core.DEFAULT_METHOD, choices=list(core.METHODS),
                        help="trajectory integration scheme used for training and prediction")
    args = parser.parse_args(argv)

    root = core.resolve_root(args.root)
    grouped = core.load_event_data(root)

    print(f"project root : {root}")
    print(f"input dir    : {core.rel(root / core.STAGE4_SUBDIR, root)}")
    print("method       : Step C - leave-one-event-out cross-validation + final parameters")
    print(f"seed         : {args.seed} (shared by the multi-start methods)")
    print(f"integration  : {args.method}, real time deltas, leader speed interpolated from lv")
    print(f"spacing      : s = dhw - {core.L_VEH:g}   (violation = simulated s <= 0)")
    print(f"score        : {SCORE_FORMULA}")
    print()
    print("cross-validation budgets (the brief allows HALF budget for the global methods here):")
    for method in METHOD_ORDER:
        budget = CV_BUDGETS[method]
        if method == "M1":
            detail = "1 least-squares fit"
        elif method == "M2":
            detail = f"{budget['starts']} starts x maxiter {budget['maxiter']}"
        elif method == "M3":
            detail = (f"population {budget['population']} x {budget['generations']} generations"
                      f"  (half of {budget['full_population']} x {budget['full_generations']})")
        elif method == "M4":
            detail = (f"{budget['particles']} particles x {budget['iterations']} iterations"
                      f"  (half of {budget['full_particles']} x {budget['full_iterations']})")
        else:
            detail = (f"{budget['starts']} starts x {budget['steps']} Adam steps"
                      f"  (half of {budget['full_starts']} x {budget['full_steps']})")
        print(f"  {method} [{budget['mode']:<6}] {detail}")
    print("final retrain budgets (always full, on all of the run's events):")
    for method in METHOD_ORDER:
        budget = FULL_BUDGETS[method]
        if method == "M1":
            detail = "1 least-squares fit"
        elif method == "M2":
            detail = f"{budget['starts']} starts x maxiter {budget['maxiter']}"
        elif method == "M3":
            detail = f"population {budget['population']} x {budget['generations']} generations"
        elif method == "M4":
            detail = f"{budget['particles']} particles x {budget['iterations']} iterations"
        else:
            detail = f"{budget['starts']} starts x {budget['steps']} Adam steps"
        print(f"  {method} {detail}")
    print()

    fold_records: dict[str, dict[str, dict[str, dict]]] = {}
    aggregates: dict[str, dict[str, object]] = {}
    timing: dict[str, object] = {}
    started_all = time.perf_counter()

    for run, events in grouped.items():
        names = [name for name, _ in events]
        print(f"{run}: {len(names)} event(s) {names} -> {len(names)} fold(s)")
        run_folds: dict[str, dict[str, dict]] = {}
        for index, held_name in enumerate(names):
            fold_name = f"leave_{held_name}_out"
            fold_label = f"{run}/{fold_name}"
            print(f"  {fold_label}: training on {[n for n, _ in events if n != held_name]}")
            fold_started = time.perf_counter()
            fold_folds: dict[str, dict] = {}
            for method in METHOD_ORDER:
                record = run_fold(
                    run, index, held_name, events, method,
                    CV_BUDGETS[method], args.seed, args.method,
                )
                fold_folds[method] = record
                if record["infeasible"]:
                    print(f"    {method:<4} INFEASIBLE  {record['infeasible_reason']}")
                else:
                    metrics = record["metrics"]
                    reg = record["regression"]
                    print(
                        f"    {method:<4} J_train={record['train_total_error']:10.4f}"
                        f"  held-out s_RMSE={metrics['spacing_rmse']:9.4f}"
                        f"  s_MAE={_num(reg['spacing']['mae']):>9}"
                        f"  s_MAPE%={_num(reg['spacing']['mape_percent'], 3):>8}"
                        f"  s_R2={_num(reg['spacing']['r2']):>8}"
                        f"  v_RMSE={metrics['speed_rmse']:9.4f}"
                        f"  v_MAE={_num(reg['speed']['mae']):>9}"
                        f"  v_MAPE%={_num(reg['speed']['mape_percent'], 3):>8}"
                        f"  v_R2={_num(reg['speed']['r2']):>8}"
                        f"  J_e={metrics['objective_j']:10.4f}"
                        f"  viol={metrics['s_violations']:2d}"
                        f"  [{record['elapsed_seconds']:6.2f} s]"
                    )
            run_folds[fold_name] = fold_folds
            timing[fold_label] = float(time.perf_counter() - fold_started)
            print()

        fold_records[run] = run_folds
        aggregate = aggregate_run(run, events, run_folds)
        aggregates[run] = aggregate
        print(f"  {run} cross-validation ranking:")
        print_ranking(run, aggregate)
        print()

    # The Step A/B numbers are read once per run. They are two things at the
    # same time: the fit-set side of the comparison, and the reference a
    # full-budget retrain has to reproduce - the retrain runs the SAME method on
    # the SAME events with the SAME seed, budget and integration scheme as the
    # published run, so any difference means the runtime changed the numbers.
    references: dict[str, dict[str, object]] = {
        run: calibration_set_reference(root, run, grouped[run], args.method)
        for run in grouped
    }

    # --- final retrain on all events of each run ---------------------------
    finals: dict[str, dict[str, object]] = {}
    final_timing: dict[str, float] = {}
    for run, events in grouped.items():
        aggregate = aggregates[run]
        datasets = [data for _, data in events]
        ranking = aggregate["ranking"]
        print(f"{run}: final retrain on all {len(events)} event(s)")
        winner: dict[str, object] | None = None
        final_theta = None
        fallbacks: list[dict[str, object]] = []
        for row in ranking:
            method = str(row["method"])
            budget = FULL_BUDGETS[method]
            retrain_started = time.perf_counter()
            try:
                theta, train_cost = calibrate(events, method, budget, args.seed, args.method)
            except FoldInfeasible as exc:
                print(f"  {method}: the full-budget retrain is INFEASIBLE ({exc}) - "
                      f"trying the next method of the ranking")
                fallbacks.append({"method": method, "reason": str(exc)})
                continue
            elapsed = float(time.perf_counter() - retrain_started)
            winner = {"method": method, "budget": budget, "train_total_error": train_cost}
            final_theta = theta
            final_timing[run] = elapsed
            print(f"  {method} (rank {row['rank']}, CV score {float(row['score']):.4f}) "
                  f"retrained on the full set in {elapsed:.2f} s, J_full={train_cost:.6f}")
            break

        if winner is None or final_theta is None:
            raise RuntimeError(f"{run}: no ranked method could be retrained on the full set")

        method = str(winner["method"])
        trajectory = core.summarize_events(datasets, final_theta, args.method)
        within_bounds, at_bound = inside_and_at_bound(final_theta)
        event_blocks = [prediction_metrics(data, final_theta, args.method) for data in datasets]
        final_regression: dict[str, object] | None = None
        if all(block["feasible"] for block in event_blocks):
            final_regression = aggregate_regression(
                event_blocks,
                "coordinate units for the spacing, coordinate units/s for the speed",
            )
        final_entry: dict[str, object] = {
            "run": run,
            "method": method,
            "method_description": METHOD_DESCRIPTIONS[method],
            "method_rank_in_cv": next(
                row["rank"] for row in ranking if row["method"] == method
            ),
            "cv_score": next(row["score"] for row in ranking if row["method"] == method),
            "theta_vector": [float(value) for value in final_theta],
            "theta": core.theta_to_dict(final_theta),
            "delta": core.DELTA,
            "L_VEH": core.L_VEH,
            "parameters_within_bounds": bool(within_bounds),
            "parameters_at_bound": at_bound,
            "total_error": float(trajectory["total_error"]),
            "total_error_method": trajectory["total_error_method"],
            "n_events": int(trajectory["n_events"]),
            "n_samples": int(trajectory["n_samples"]),
            "mean_spacing_rmse": float(trajectory["mean_spacing_rmse"]),
            "mean_speed_rmse": float(trajectory["mean_speed_rmse"]),
            "worst_spacing_rmse": float(trajectory["worst_spacing_rmse"]),
            "worst_speed_rmse": float(trajectory["worst_speed_rmse"]),
            "events": trajectory["events"],
            "regression": final_regression,
            "per_event_regression": {
                name: block for (name, _data), block in zip(events, event_blocks)
            },
            "training_events": [name for name, _ in events],
            "budget": FULL_BUDGETS[method],
            "method_reported_total_error": float(winner["train_total_error"]),
            "infeasible_ranked_methods": fallbacks,
            "input_dir": core.rel(root / core.STAGE4_SUBDIR, root),
            "integration_method": args.method,
            "seed": args.seed,
        }
        reference = references[run].get(method) or {}
        published_theta = reference.get("theta")
        published_error = reference.get("total_error")
        if published_theta:
            published_vector = core.as_theta(
                [float(published_theta[name]) for name in core.PARAM_NAMES]
            )
            theta_difference = float(np.max(np.abs(final_theta - published_vector)))
            error_difference = (
                abs(float(published_error) - float(trajectory["total_error"]))
                if published_error is not None
                else None
            )
            reproduced = bool(
                theta_difference <= PUBLISHED_RUN_TOLERANCE
                and error_difference is not None
                and error_difference <= PUBLISHED_RUN_TOLERANCE
            )
            published_note = (
                "the same method on the same events with the same seed, budget and "
                "integration scheme as its Step A/B run; that JSON rounds its floats to "
                "6 decimals, which is what the tolerance covers"
            )
        else:
            theta_difference = None
            error_difference = None
            reproduced = None
            published_note = reference.get("note") or reference.get("regression_note") or (
                "this method has no published parameter vector for this run"
            )
        final_entry["published_full_budget_run"] = {
            "source": reference.get("source"),
            "available": bool(published_theta),
            "note": published_note,
            "published_theta": published_theta,
            "published_total_error": published_error,
            "max_abs_theta_difference": theta_difference,
            "total_error_difference": error_difference,
            "tolerance": PUBLISHED_RUN_TOLERANCE,
            "reproduced": reproduced,
        }
        finals[run] = final_entry
        print_final(run, final_entry)
        published = final_entry["published_full_budget_run"]
        if published["available"]:
            print(f"  vs published Step A/B run (read from {published['source']}): "
                  f"max|theta difference| = {published['max_abs_theta_difference']:.3e}"
                  f"  (published JSON rounds to 6 decimals, tolerance "
                  f"{PUBLISHED_RUN_TOLERANCE:g}), total_error difference = "
                  f"{published['total_error_difference']:.3e}"
                  f"  -> reproduced: {published['reproduced']}")
        else:
            print(f"  vs published Step A/B run: not available "
                  f"({published.get('note') or 'no published parameter vector'})")
        print()

    # --- run_1 / run_2 selection for the validation replay ---------------------
    # The brief asks for the two runs to be compared on the same scale and for the
    # better one to be handed to the validation replay; the choice and the table it
    # rests on go into final_params.json as ``selected_for_validation``.
    selection = select_for_validation(aggregates, finals)
    print_selection(selection)
    if not selection["self_check_selected_is_argmin"]:
        print("WARNING: the selected run is not the argmin of the normalised score")

    total_seconds = float(time.perf_counter() - started_all)

    # --- self-checks ----------------------------------------------------------
    self_checks = collect_self_checks(fold_records, aggregates, finals)
    print("self-checks:")
    print("  per-fold RMSE recomputed from the residual series equals idm_core : "
          f"{self_checks['all_per_fold_rmse_match_idm_core']}")
    print("  mean RMSE of the regression table equals the CV table            : "
          f"{self_checks['all_mean_rmse_match_cv_table']}")
    print("  MAPE samples skipped for a zero measurement                       : "
          f"{self_checks['mape_skipped_points_total']}")
    print("  folds whose trajectory hit a spacing violation (clamped)         : "
          f"{self_checks['methods_with_spacing_violations'] or 'none'}")
    print("  final blocks carry regression metrics for every event            : "
          f"{self_checks['all_final_blocks_complete']}")
    print("  full-budget retrain reproduces the published Step A/B run        : "
          f"{self_checks['full_budget_retrain_reproduces_published_run']}")
    print()
    if not (
        self_checks["all_per_fold_rmse_match_idm_core"]
        and self_checks["all_mean_rmse_match_cv_table"]
        and self_checks["all_final_blocks_complete"]
    ):
        raise RuntimeError(f"the self-checks of the cross-validation failed: {self_checks}")

    # --- payloads -------------------------------------------------------------
    cv_payload: dict[str, object] = {
        "script": SCRIPT,
        "stage": "Step C - leave-one-event-out cross-validation",
        "input_dir": core.rel(root / core.STAGE4_SUBDIR, root),
        "seed": args.seed,
        "integration_method": args.method,
        "units": {
            "spacing_rmse": "coordinate units (raw, un-normalised)",
            "speed_rmse": "coordinate units/s (raw, un-normalised)",
            "objective_j": "dimensionless (event-normalised, idm_core convention)",
            "mse": "coordinate units^2 for the spacing, (coordinate units/s)^2 for the speed",
            "mae": "coordinate units for the spacing, coordinate units/s for the speed",
            "mape_percent": "percent (samples with a zero measurement skipped and counted)",
            "r2": "dimensionless, 1 - SS_res/SS_tot about the measured mean",
            "scale_s": "coordinate units (pooled std of all measured s of the run)",
            "scale_v": "coordinate units/s (pooled std of all measured v of the run)",
            "score": "dimensionless",
        },
        "cv_scheme": {
            "type": "leave-one-event-out",
            "folds": {run: list(folds) for run, folds in fold_records.items()},
            "training": "all remaining events of the same run",
            "prediction": (
                "pure forward simulation of the held-out event from its first measured "
                "(s, v) with the measured lv series; the held-out event is never used to fit"
            ),
            "per_fold_metrics": [
                "spacing_rmse", "speed_rmse", "s_violations", "objective_j",
                "spacing/speed mse, rmse, mae, mape_percent, r2",
            ],
            "aggregation": (
                "equal weight mean over the feasible folds for every error metric "
                "(mse/rmse/mae/mape); violation counts summed; r2 rebuilt on the pooled "
                "residuals (1 - sum SS_res / sum SS_tot, each fold about its own mean)"
            ),
        },
        "score": {
            "formula": SCORE_FORMULA,
            "violation_penalty_per_occurrence": VIOLATION_SCORE_PENALTY,
            "infeasible_methods": "excluded from the ranking",
        },
        "budgets": {
            "cross_validation": CV_BUDGETS,
            "final_retrain": FULL_BUDGETS,
            "note": (
                "the brief allows the global methods (GA/PSO/M5 multi-start) to run at half "
                "budget inside the cross-validation; the final retrain always uses the full budget"
            ),
        },
        "folds": fold_records,
        "runs": {
            run: {
                **aggregates[run],
                "calibration_set_reference": references[run],
            }
            for run in grouped
        },
        "finals": {run: entry["method"] for run, entry in finals.items()},
        "self_checks": self_checks,
        "timing": {
            "per_fold_seconds": timing,
            "final_retrain_seconds": final_timing,
            "total_seconds": total_seconds,
        },
    }
    cv_path = core.results_dir(root) / CV_OUTPUT_FILENAME
    core.write_json_atomic(cv_path, cv_payload)

    final_payload: dict[str, object] = {
        "script": SCRIPT,
        "stage": "Step C - final parameters (best cross-validated method, retrained on all events)",
        "input_dir": core.rel(root / core.STAGE4_SUBDIR, root),
        "seed": args.seed,
        "integration_method": args.method,
        "delta": core.DELTA,
        "L_VEH": core.L_VEH,
        "parameter_bounds": {
            name: [float(core.LOWER[i]), float(core.UPPER[i])]
            for i, name in enumerate(core.PARAM_NAMES)
        },
        "selection_rule": (
            "the method with the smallest cross-validation score "
            "(mean spacing RMSE / scale_s + mean speed RMSE / scale_v + 1.0 * violations)"
        ),
        "budget_note": (
            "cross-validation: global methods at half budget (M3 30x40, M4 20x40, M5 5x150 steps); "
            "final retrain: full budget (M2 13x4000, M3 60x80, M4 40x80, M5 10x300 steps)"
        ),
        "runs": finals,
        "selected_for_validation": selection,
        "self_checks": self_checks,
        "timing": {
            "final_retrain_seconds": final_timing,
            "total_seconds": total_seconds,
        },
    }
    final_path = core.results_dir(root) / FINAL_OUTPUT_FILENAME
    core.write_json_atomic(final_path, final_payload)

    print("=" * 78)
    print("final parameters")
    print("=" * 78)
    for run, entry in finals.items():
        theta = core.as_theta(entry["theta_vector"])
        print(f"{run}: best method {entry['method']} (CV rank {entry['method_rank_in_cv']},"
              f" score {float(entry['cv_score']):.4f})")
        print(f"  v0={theta[0]:.4f}  T={theta[1]:.4f}  s0={theta[2]:.4f}  "
              f"a={theta[3]:.4f}  b={theta[4]:.4f}   delta={core.DELTA:g} (fixed)")
        print(f"  full-set total_error = {entry['total_error']:.6f}"
              f"   at a bound: {entry['parameters_at_bound'] or 'none'}")
    print()
    print(f"total runtime: {total_seconds:.2f} s")
    print(f"written: {core.rel(cv_path, root)}")
    print(f"written: {core.rel(final_path, root)}")
    leftovers = sorted(
        path.name for path in cv_path.parent.glob("*.tmp")
    )
    print(f".tmp leftovers: {leftovers if leftovers else 'none'}")
    if not self_checks["all_full_budget_retrains_reproduce_published_run"]:
        # Not fatal - the artifacts are already on disk and the cause has to be read
        # from the log rather than swallowed. Exit code 2 says "look here"; the
        # structural checks above, which would mean a bug in this script, do raise.
        print("WARNING: a full-budget retrain did not reproduce its published Step A/B run:")
        for run, value in self_checks["full_budget_retrain_reproduces_published_run"].items():
            print(f"  {run}: reproduced={value}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
