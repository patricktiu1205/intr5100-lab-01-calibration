#!/usr/bin/env python3
"""INTR5100 Lab Session I - M1: instantaneous-acceleration least squares.

Method 1 of the IDM calibration study: fit the IDM acceleration law directly to
the measured acceleration instead of to the trajectory.

    theta_hat = argmin_theta  sum_events sum_i ((a_idm - a_obs) / sigma_a)^2

--- measured quantities (per event, from the stage 4 table) -------------------

    s   = dhw - L_VEH                    L_VEH = 250
    v   = fv                             follower speed
    dv  = fv - lv                        leader minus follower, SAME sample
    a_obs                                central difference of fv:
                                         interior  (fv[i+1] - fv[i-1]) / (t[i+1] - t[i-1])
                                         ends      one-sided over the adjacent interval

No derivative of the leader is needed and no integration is performed: every row
of the table is one independent observation of the acceleration.

--- normalisation ------------------------------------------------------------

The brief asks for the per-event residuals to be normalised by the standard
deviation of that event's measured acceleration before they are concatenated, so
``sigma_a`` above is ``std(a_obs)`` of the event the row belongs to. Without it a
single high-acceleration event (``run_1/event_3`` has ``std(a_obs) = 128.25``)
would dominate a calm one (``run_1/event_2`` has ``101.29``) purely through its
units. After normalisation every event contributes a comparable number of
dimensionless residuals and the sum is a plain, unweighted least-squares problem.

--- solver --------------------------------------------------------------------

Exactly one ``scipy.optimize.least_squares`` fit per run, started from the
brief's initial point (v0=120, T=1.2, s0=100, a=80, b=100) and with the brief's
box given to the solver itself (``bounds=(LOWER, UPPER)``, i.e. the trust-region
reflective method never evaluates a parameter outside the bounds - no penalty
term is needed here, unlike the trajectory methods). The reported ``cost`` is
scipy's ``0.5 * sum(residual^2)``; the optimality, the residual RMS and the
number of evaluations are printed as convergence evidence.

For the cross-method table the resulting parameters are ALSO evaluated as a
trajectory model (``trajectory_error`` / per-event spacing and speed RMSE, the
same quantities M2 reports), so M1 and M2 can be compared on the identical cost
scale even though they were fitted to different objectives.

Output: ``results/m1_params.json`` (per run: parameters, total_error, the
per-event raw RMSEs, convergence evidence and self-checks) plus the same numbers
on stdout.

Usage:
    python src/calibration/m1_instant_ls.py [--root DIR] [--method euler|rk4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))

import idm_core as core  # noqa: E402  (import after the path fix, same dir)

SCRIPT = "src/calibration/m1_instant_ls.py"
METHOD_NAME = "M1"
OUTPUT_FILENAME = "m1_params.json"


def build_design(events: list[tuple[str, core.EventData]]) -> list[dict[str, object]]:
    """Per-event design matrices of the instantaneous fit.

    One dict per event with the arrays the residual function needs: the observed
    spacing, speed, relative speed, the measured acceleration, the standard
    deviation that normalises this event's residuals and the sample count.
    """
    design = []
    for name, data in events:
        acceleration = core.observed_acceleration(data)
        sigma_a = float(np.std(acceleration))
        design.append(
            {
                "name": data.name or name,
                "data": data,
                "acceleration": acceleration,
                "sigma_a": sigma_a,
                "n": data.n,
                "sigma_a_used": sigma_a if sigma_a > core.MIN_SIGMA else core.MIN_SIGMA,
            }
        )
    return design


def residuals(theta: np.ndarray, design: list[dict[str, object]]) -> np.ndarray:
    """Concatenated normalised acceleration residuals of one run's events."""
    parts = []
    for item in design:
        data: core.EventData = item["data"]  # type: ignore[assignment]
        predicted = core.idm_acceleration(data.s_obs, data.v_obs, data.relative_gap, theta)
        parts.append((predicted - item["acceleration"]) / item["sigma_a_used"])  # type: ignore[operator]
    return np.concatenate(parts)


def acceleration_metrics(design: list[dict[str, object]], theta: np.ndarray) -> dict[str, dict[str, float]]:
    """Raw and normalised acceleration-fit errors per event (original units)."""
    metrics: dict[str, dict[str, float]] = {}
    for item in design:
        data: core.EventData = item["data"]  # type: ignore[assignment]
        predicted = core.idm_acceleration(data.s_obs, data.v_obs, data.relative_gap, theta)
        difference = predicted - item["acceleration"]
        sigma_a = float(item["sigma_a_used"])
        metrics[str(item["name"])] = {
            "acceleration_rmse": float(np.sqrt(np.mean(difference**2))),
            "acceleration_rmse_normalized": float(np.sqrt(np.mean((difference / sigma_a) ** 2))),
            "acceleration_bias": float(np.mean(difference)),
            "sigma_a": float(item["sigma_a"]),
        }
    return metrics


def calibrate_run(
    run: str,
    events: list[tuple[str, core.EventData]],
    method: str,
    initial: np.ndarray,
) -> dict[str, object]:
    """One bounded least-squares fit for one run, plus the trajectory metrics."""
    design = build_design(events)
    start_vector = core.as_theta(initial)
    solution = least_squares(
        residuals,
        start_vector,
        args=(design,),
        bounds=(core.LOWER, core.UPPER),
        method="trf",
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=20000,
    )
    theta = core.as_theta(solution.x)
    trajectory = core.summarize_events([data for _, data in events], theta, method)
    acceleration = acceleration_metrics(design, theta)
    for name, values in acceleration.items():
        trajectory["events"].setdefault(name, {}).update(values)  # type: ignore[union-attr]

    residual_vector = residuals(theta, design)
    raw_residuals = np.concatenate(
        [
            core.idm_acceleration(
                item["data"].s_obs, item["data"].v_obs, item["data"].relative_gap, theta
            )
            - item["acceleration"]
            for item in design
        ]
    )
    result: dict[str, object] = {
        "theta_vector": [float(value) for value in theta],
        "total_error": trajectory["total_error"],
        "total_error_method": trajectory["total_error_method"],
        "events": trajectory["events"],
        "n_events": trajectory["n_events"],
        "n_samples": trajectory["n_samples"],
        "mean_spacing_rmse": trajectory["mean_spacing_rmse"],
        "mean_speed_rmse": trajectory["mean_speed_rmse"],
        "worst_spacing_rmse": trajectory["worst_spacing_rmse"],
        "worst_speed_rmse": trajectory["worst_speed_rmse"],
        "acceleration_objective": {
            "value": float(0.5 * np.sum(residual_vector**2)),
            "definition": "0.5 * sum(residual^2) as returned by scipy.optimize.least_squares",
            "residual_count": int(residual_vector.size),
        },
        "acceleration_fit": {
            "residual_rms_normalized": float(np.sqrt(np.mean(residual_vector**2))),
            "residual_rms": float(np.sqrt(np.mean(raw_residuals**2))),
            "residual_sum_squares": float(np.sum(raw_residuals**2)),
        },
        "convergence": {
            "status": int(solution.status),
            "success": bool(solution.success),
            "message": str(solution.message),
            "optimality": float(solution.optimality),
            "nfev": int(solution.nfev),
            "njev": int(solution.njev) if solution.njev is not None else None,
            "cost": float(solution.cost),
            "initial_theta": [float(value) for value in start_vector],
            "solver": "scipy.optimize.least_squares (trf, bounds=LOWER/UPPER)",
        },
    }
    return result


def print_run(run: str, result: dict[str, object]) -> None:
    """Stdout block for one run."""
    theta = core.as_theta(result["theta_vector"])
    print(f"{run}:")
    print(f"  theta              : {core.format_theta(theta)}")
    print(f"  within bounds      : {core.inside_bounds(theta)}")
    convergence = result["convergence"]  # type: ignore[assignment]
    print(
        f"  least_squares      : success={convergence['success']} status={convergence['status']}"
        f" nfev={convergence['nfev']} optimality={convergence['optimality']:.3e}"
    )
    objective = result["acceleration_objective"]  # type: ignore[assignment]
    print(
        f"  acceleration LS    : cost={objective['value']:.6f}"
        f" residual RMS (normalised)={result['acceleration_fit']['residual_rms_normalized']:.6f}"  # type: ignore[index]
        f" over {objective['residual_count']} residuals"
    )
    print(
        f"  trajectory cost    : total_error={result['total_error']:.6f}"
        f"  (equal weight mean of {result['n_events']} events, {result['n_samples']} samples)"
    )
    print(f"  {'event':<10}{'n':>5}{'s_RMSE':>11}{'v_RMSE':>11}{'J_e':>11}"
          f"{'a_RMSE':>11}{'zero_s':>11}")
    for name, metrics in result["events"].items():  # type: ignore[union-attr]
        print(
            f"  {name:<10}{metrics['n_samples']:>5}{metrics['spacing_rmse']:>11.4f}"
            f"{metrics['speed_rmse']:>11.4f}{metrics['j_total']:>11.4f}"
            f"{metrics['acceleration_rmse']:>11.3f}"
            f"{metrics['zero_model']['best_spacing_rmse']:>11.4f}"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="project root (default: inferred)")
    parser.add_argument(
        "--method", default=core.DEFAULT_METHOD, choices=list(core.METHODS),
        help="trajectory integration scheme used for the reported metrics",
    )
    args = parser.parse_args(argv)

    root = core.resolve_root(args.root)
    grouped = core.load_event_data(root)

    print(f"project root : {root}")
    print(f"input dir    : {core.rel(root / core.STAGE4_SUBDIR, root)}")
    print(f"method       : {METHOD_NAME} - instantaneous-acceleration least squares")
    print("model        : a_idm = a*(1 - (v/v0)^4 - (s*/s)^2), "
          "s* = s0 + max(0, v*T + v*dv/(2*sqrt(a*b)))")
    print(f"observed a   : central difference of fv (ends one-sided), dt from the real time column")
    print(f"spacing      : s = dhw - {core.L_VEH:g}")
    print(f"normalisation: per-event residuals divided by std(a_obs) of that event")
    print(
        "solver       : scipy.optimize.least_squares (trf), bounds="
        f"v0[{core.LOWER[0]:g},{core.UPPER[0]:g}] T[{core.LOWER[1]:g},{core.UPPER[1]:g}] "
        f"s0[{core.LOWER[2]:g},{core.UPPER[2]:g}] a[{core.LOWER[3]:g},{core.UPPER[3]:g}] "
        f"b[{core.LOWER[4]:g},{core.UPPER[4]:g}]"
    )
    print(f"initial point: {core.format_theta(core.DEFAULT_THETA)}")
    print(f"trajectory   : integration method for the reported metrics = {args.method}")
    print()
    print("events:")
    for run, events in grouped.items():
        print(f"  {run:<10} " + ", ".join(f"{name} (n={data.n})" for name, data in events))
    print()

    per_run: dict[str, dict[str, object]] = {}
    for run, events in grouped.items():
        result = calibrate_run(run, events, args.method, core.DEFAULT_THETA)
        per_run[run] = result
        print_run(run, result)

    payload = core.build_results_payload(
        method=METHOD_NAME,
        script=SCRIPT,
        root=root,
        configuration={
            "objective": "sum_events sum_i ((a_idm - a_obs)/sigma_a)^2 (normalised, concatenated)",
            "observed_acceleration": "central difference of fv, one-sided at the ends",
            "solver": "scipy.optimize.least_squares, method=trf, bounds=LOWER/UPPER",
            "initial_theta": [float(value) for value in core.DEFAULT_THETA],
            "residual_normalisation": "per-event std(a_obs)",
            "trajectory_metrics_method": args.method,
            "L_VEH": core.L_VEH,
            "delta": core.DELTA,
            "parameter_bounds": {
                name: [float(core.LOWER[i]), float(core.UPPER[i])]
                for i, name in enumerate(core.PARAM_NAMES)
            },
            "spacing_formula": "s = dhw - L_VEH",
            "cost_normalisation": "per-event std(s_obs), std(v_obs) (trajectory metrics)",
        },
        per_run=per_run,
        focus_run="run_1",
        focus_event=None,
    )
    out_path = core.results_dir(root) / OUTPUT_FILENAME
    core.write_json_atomic(out_path, payload)

    # --- self-checks ---------------------------------------------------------
    checks = payload["self_checks"]  # type: ignore[index]
    print("self-checks:")
    for key in sorted(checks):
        if key in ("runs_found",):
            continue
        print(f"  {key:<46} {checks[key]}")
    print(f"  runs_found                                  {checks['runs_found']}")

    focus_ok = checks["focus_beats_zero_model"]
    if focus_ok is not True:
        print()
        print(f"WARNING: the focus event ({checks['focus_run']}/{checks['focus_event']}) spacing "
              "RMSE is NOT better than the constant zero model - see results for the numbers "
              "(warning only, not an error).")

    ok = (
        bool(checks["runs_found_nonempty"])
        and all(
            bool(checks[f"{run}_parameters_within_bounds"])
            and bool(checks[f"{run}_has_parameters"])
            for run in checks["runs_found"]
        )
    )

    print()
    print(f"written: {core.rel(out_path, root)}")
    tmp_leftovers = sorted(path.name for path in out_path.parent.glob("*.tmp"))
    print(f".tmp leftovers: {tmp_leftovers if tmp_leftovers else 'none'}")
    if not ok:
        print("ERROR: self-checks failed (see above).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
