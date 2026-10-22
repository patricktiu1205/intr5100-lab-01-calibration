#!/usr/bin/env python3
"""INTR5100 Lab Session I - M2: Nelder-Mead trajectory fit (multi-start).

Method 2 of the IDM calibration study: fit the IDM to the measured TRAJECTORY
instead of to the differentiated acceleration. The objective is the shared
multi-event cost of ``idm_core``,

    J(theta) = mean over events of
               [ mean((s_sim - s_obs)^2)/sigma_s^2
               + mean((v_sim - v_obs)^2)/sigma_v^2
               + S_VIOLATION_PENALTY * spacing_violations ]

with, per event, sigma_s / sigma_v the standard deviation of the MEASURED spacing
and follower speed of that event, initialised from the first measured sample and
integrated forward with the measured leader speed (see ``idm_core.simulate_event``).
Every event therefore contributes one dimensionless number and the events carry
equal weight.

--- optimiser -----------------------------------------------------------------

``scipy.optimize.minimise(method='Nelder-Mead')`` is derivative-free, which the
objective needs (the integration is sequential and the prototype model is not
written differentiably - M5 of the study covers the differentiable variant).
Being a local method it gets MULTI-START treatment, as the brief requires:

  * start 0 is the brief's reference point (v0=120, T=1.2, s0=100, a=80, b=100);
  * starts 1..N come from a Latin hypercube over the whole parameter box
    (``idm_core.latin_hypercube``, one jittered sample per stratum per dimension)
    with a FIXED seed, so the whole run - and therefore the reported optimum - is
    bit-for-bit reproducible;
  * every start is run to convergence and the cheapest feasible result wins.

The objective handed to the optimiser is ``idm_core.constrained_total_error``:
parameters outside the box return the large box penalty (the brief's out-of-bounds
penalty, so a start that walks out of the box cannot win) and an infeasible
trajectory returns the large finite ``INFEASIBLE_COST`` wall instead of a true
``inf``, which Nelder-Mead cannot handle. A true ``inf`` is what the shared cost
itself returns, and it is reported as such in the per-event evidence.

Output: ``results/m2_params.json`` (per run: parameters, total_error, per-event
raw RMSEs, the full multi-start table and the self-checks) plus the same numbers
on stdout.

Usage:
    python src/calibration/m2_neldermead.py [--root DIR] [--starts 13] [--seed 20260921]
                                          [--maxiter 2000] [--method euler|rk4]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))

import idm_core as core  # noqa: E402  (import after the path fix, same dir)

SCRIPT = "src/calibration/m2_neldermead.py"
METHOD_NAME = "M2"
OUTPUT_FILENAME = "m2_params.json"

# Multi-start configuration (the brief asks for at least 10 starts).
DEFAULT_STARTS = 13
DEFAULT_SEED = 20260921
DEFAULT_MAXITER = 4000
# Tolerances are tight on purpose: the box wall makes the objective flat outside
# the interesting region, so a loose tolerance would stop on a wall face.
NM_OPTIONS = {"adaptive": True, "xatol": 1e-8, "fatol": 1e-10}


def start_points(n_starts: int, seed: int) -> list[tuple[str, np.ndarray]]:
    """Reference point + Latin hypercube starts, all inside the box.

    Start 0 is the canonical initial point of the brief; the remaining
    ``n_starts - 1`` points come from a seeded Latin hypercube over the box.
    """
    if n_starts < 11:
        raise ValueError("the brief requires at least 10 multi-starts, plus the reference point")
    points: list[tuple[str, np.ndarray]] = [("reference theta0", core.as_theta(core.DEFAULT_THETA))]
    for index, point in enumerate(core.latin_hypercube(n_starts - 1, seed), start=1):
        points.append((f"lhs {index} (seed {seed})", core.as_theta(point)))
    return points


def calibrate_run(
    run: str,
    events: list[tuple[str, core.EventData]],
    method: str,
    n_starts: int,
    seed: int,
    maxiter: int,
    verbose: bool = True,
) -> dict[str, object]:
    """Multi-start Nelder-Mead on the shared trajectory cost of one run."""
    datasets = [data for _, data in events]
    objective = lambda theta: core.constrained_total_error(datasets, theta, method)  # noqa: E731

    start_cost = core.total_error(datasets, core.DEFAULT_THETA, method)
    runs: list[dict[str, object]] = []
    best_theta: np.ndarray | None = None
    best_cost = float("inf")
    best_index: int | None = None

    for index, (label, point) in enumerate(start_points(n_starts, seed)):
        started = time.perf_counter()
        solution = minimize(
            objective,
            point,
            method="Nelder-Mead",
            options={**NM_OPTIONS, "maxiter": maxiter, "disp": False},
        )
        # Nelder-Mead is not bound-aware: it converges onto a bound and may stop a
        # machine epsilon outside it (where the objective pays the box wall). The
        # vector it is REPORTED as having found is therefore its projection onto the
        # box, and the cost that ranks the starts is the honest trajectory cost of
        # that projected vector - never the wall value the optimiser stopped on.
        raw_theta = core.as_theta(solution.x)
        theta = np.clip(raw_theta, core.LOWER, core.UPPER)
        projected = not np.array_equal(theta, raw_theta)
        cost = core.total_error(datasets, theta, method)
        feasible = bool(np.all(np.isfinite(theta))) and math.isfinite(cost)
        runs.append(
            {
                "index": index,
                "label": label,
                "start_theta": [float(value) for value in point],
                "theta": [float(value) for value in theta],
                "raw_theta": [float(value) for value in raw_theta],
                "projected_onto_bounds": projected,
                "optimum_stopped_on_wall": bool(
                    math.isfinite(float(solution.fun)) and float(solution.fun) >= core.PARAM_PENALTY
                ),
                "optimiser_final_objective": float(solution.fun),
                "cost": cost,
                "inside_bounds": core.inside_bounds(theta),
                "feasible_candidate": feasible,
                "success": bool(solution.success),
                "status": int(solution.status),
                "message": str(solution.message),
                "nit": int(solution.nit),
                "nfev": int(solution.nfev),
            }
        )
        if verbose:
            print(
                f"    start {index:2d}/{n_starts - 1:<2d} {label:<26}"
                f" J={cost:12.6f} nit={solution.nit:5d} nfev={solution.nfev:5d}"
                f" {'feasible' if feasible else 'INFEASIBLE'}"
                f"{' (projected onto bounds)' if projected else ''}"
                f"  {time.perf_counter() - started:6.2f} s"
            )
        if feasible and cost < best_cost:
            best_cost = cost
            best_theta = theta
            best_index = index

    if best_theta is None:
        raise RuntimeError(f"{run}: no feasible multi-start result")

    trajectory = core.summarize_events(datasets, best_theta, method)
    feasible_costs = [float(run["cost"]) for run in runs if run["feasible_candidate"]]
    best_run = runs[best_index] if best_index is not None else None
    result: dict[str, object] = {
        "theta_vector": [float(value) for value in best_theta],
        "total_error": trajectory["total_error"],
        "total_error_method": trajectory["total_error_method"],
        "events": trajectory["events"],
        "n_events": trajectory["n_events"],
        "n_samples": trajectory["n_samples"],
        "mean_spacing_rmse": trajectory["mean_spacing_rmse"],
        "mean_speed_rmse": trajectory["mean_speed_rmse"],
        "worst_spacing_rmse": trajectory["worst_spacing_rmse"],
        "worst_speed_rmse": trajectory["worst_speed_rmse"],
        "multistart": {
            "n_starts": len(runs),
            "seed": seed,
            "sampling": "reference point + Latin hypercube over the parameter box",
            "feasible_starts": len(feasible_costs),
            "infeasible_starts": len(runs) - len(feasible_costs),
            "starts_stopped_on_wall": int(sum(1 for run in runs if run["optimum_stopped_on_wall"])),
            "starts_projected_onto_bounds": int(sum(1 for run in runs if run["projected_onto_bounds"])),
            "best_cost": best_cost,
            "best_start_index": best_index,
            "worst_feasible_cost": max(feasible_costs) if feasible_costs else None,
            "distinct_feasible_results": len({tuple(np.round(run["theta"], 6)) for run in runs if run["feasible_candidate"]}),
            "initial_point_cost": start_cost,
            "initial_point_improvement": (
                None if not np.isfinite(start_cost)
                else float(start_cost - best_cost)
            ),
            "runs": runs,
        },
        "convergence": {
            "solver": "scipy.optimize.minimize(method='Nelder-Mead', adaptive=True)",
            "options": {**NM_OPTIONS, "maxiter": maxiter},
            # The convergence evidence of the start that WON, not of the last start
            # evaluated: reporting the last one would describe a run that was thrown
            # away (the earlier draft of this script did exactly that).
            "best_start_index": best_index,
            "status": best_run["status"] if best_run else None,
            "success": best_run["success"] if best_run else None,
            "message": best_run["message"] if best_run else None,
            "nit": best_run["nit"] if best_run else None,
            "nfev": best_run["nfev"] if best_run else None,
            "stopped_on_wall": best_run["optimum_stopped_on_wall"] if best_run else None,
            "projected_onto_bounds": best_run["projected_onto_bounds"] if best_run else None,
            "objective": "idm_core.constrained_total_error (box penalty outside the bounds, INFEASIBLE_COST wall for infeasible trajectories)",
        },
    }
    return result


def print_run(run: str, result: dict[str, object]) -> None:
    """Stdout block for one run."""
    theta = core.as_theta(result["theta_vector"])
    multistart = result["multistart"]  # type: ignore[assignment]
    print(f"  best theta         : {core.format_theta(theta)}")
    print(f"  within bounds      : {core.inside_bounds(theta)}")
    worst = multistart["worst_feasible_cost"]
    print(
        f"  multi-start        : {multistart['n_starts']} starts,"
        f" {multistart['feasible_starts']} feasible,"
        f" best J={multistart['best_cost']:.6f},"
        f" worst feasible J={'n/a' if worst is None else format(float(worst), '.6f')}"
    )
    print(
        f"  distinct solutions : {multistart['distinct_feasible_results']}"
        f"   reference theta0 J={multistart['initial_point_cost']:.6f}"
        f"   improvement={multistart['initial_point_improvement']}"
    )
    print(f"  total_error        : {result['total_error']:.6f}  (equal weight mean of the per-event J_e)")
    print(f"  {'event':<10}{'n':>5}{'s_RMSE':>11}{'v_RMSE':>11}{'J_spacing':>11}{'J_speed':>10}"
          f"{'J_e':>11}{'zero_s':>11}")
    for name, metrics in result["events"].items():  # type: ignore[union-attr]
        print(
            f"  {name:<10}{metrics['n_samples']:>5}{metrics['spacing_rmse']:>11.4f}"
            f"{metrics['speed_rmse']:>11.4f}{metrics['j_spacing']:>11.4f}"
            f"{metrics['j_speed']:>10.4f}{metrics['j_total']:>11.4f}"
            f"{metrics['zero_model']['best_spacing_rmse']:>11.4f}"
        )
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="project root (default: inferred)")
    parser.add_argument("--starts", type=int, default=DEFAULT_STARTS,
                        help=f"number of multi-starts including the reference point (default {DEFAULT_STARTS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"seed of the Latin hypercube multi-start (default {DEFAULT_SEED})")
    parser.add_argument("--maxiter", type=int, default=DEFAULT_MAXITER,
                        help=f"Nelder-Mead iteration cap per start (default {DEFAULT_MAXITER})")
    parser.add_argument("--method", default=core.DEFAULT_METHOD, choices=list(core.METHODS),
                        help="trajectory integration scheme")
    args = parser.parse_args(argv)

    root = core.resolve_root(args.root)
    grouped = core.load_event_data(root)

    print(f"project root : {root}")
    print(f"input dir    : {core.rel(root / core.STAGE4_SUBDIR, root)}")
    print(f"method       : {METHOD_NAME} - Nelder-Mead trajectory fit, multi-start")
    print("objective    : constrained_total_error = equal weight mean of the per-event")
    print("               [ mean(ds^2)/sigma_s^2 + mean(dv^2)/sigma_v^2 + penalty*violations ]")
    print(f"spacing      : s = dhw - {core.L_VEH:g}   (spacing violation = simulated s <= 0)")
    print(f"references   : start 0 = {core.format_theta(core.DEFAULT_THETA)}")
    print(f"starts       : {args.starts} (1 reference + {args.starts - 1} Latin hypercube, seed {args.seed})")
    print(f"nelder-mead  : adaptive=True, xatol={NM_OPTIONS['xatol']:g}, fatol={NM_OPTIONS['fatol']:g}, "
          f"maxiter={args.maxiter}")
    print(f"bounds       : v0[{core.LOWER[0]:g},{core.UPPER[0]:g}] T[{core.LOWER[1]:g},{core.UPPER[1]:g}] "
          f"s0[{core.LOWER[2]:g},{core.UPPER[2]:g}] a[{core.LOWER[3]:g},{core.UPPER[3]:g}] "
          f"b[{core.LOWER[4]:g},{core.UPPER[4]:g}] (out-of-box penalty {core.PARAM_PENALTY:g})")
    print(f"integration  : {args.method} on the real time deltas, leader speed interpolated from lv")
    print()

    per_run: dict[str, dict[str, object]] = {}
    started = time.perf_counter()
    for run, events in grouped.items():
        print(f"{run}: Nelder-Mead multi-start")
        result = calibrate_run(
            run, events, args.method, args.starts, args.seed, args.maxiter
        )
        per_run[run] = result
        print_run(run, result)
    total_seconds = time.perf_counter() - started

    payload = core.build_results_payload(
        method=METHOD_NAME,
        script=SCRIPT,
        root=root,
        configuration={
            "objective": "equal weight mean over events of the normalised trajectory cost",
            "cost_definition": "mean((s_sim-s_obs)^2)/sigma_s^2 + mean((v_sim-v_obs)^2)/sigma_v^2 + S_VIOLATION_PENALTY*violations",
            "solver": "scipy.optimize.minimize(method='Nelder-Mead', adaptive=True)",
            "solver_options": {**NM_OPTIONS, "maxiter": args.maxiter},
            "n_starts": args.starts,
            "start_sampling": "reference theta0 + Latin hypercube over the parameter box",
            "seed": args.seed,
            "L_VEH": core.L_VEH,
            "delta": core.DELTA,
            "parameter_bounds": {
                name: [float(core.LOWER[i]), float(core.UPPER[i])]
                for i, name in enumerate(core.PARAM_NAMES)
            },
            "parameter_penalty": core.PARAM_PENALTY,
            "infeasible_cost": core.INFEASIBLE_COST,
            "spacing_violation_penalty": core.S_VIOLATION_PENALTY,
            "integration_method": args.method,
            "initial_conditions": "first measured sample of the event (s and v)",
            "leader_speed": "measured lv, linearly interpolated at the integration nodes",
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
    print(f"total runtime: {total_seconds:.2f} s")
    print(f"written: {core.rel(out_path, root)}")
    tmp_leftovers = sorted(path.name for path in out_path.parent.glob("*.tmp"))
    print(f".tmp leftovers: {tmp_leftovers if tmp_leftovers else 'none'}")
    if not ok:
        print("ERROR: self-checks failed (see above).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
