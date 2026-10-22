#!/usr/bin/env python3
"""INTR5100 Lab Session I - M4: particle swarm optimisation (gbest PSO).

Method 4 of the IDM calibration study. The objective is the SHARED cost of
``idm_core`` - unchanged between M1, M2, M3 and this method:

    J(theta) = mean over events of
               [ mean((s_sim - s_obs)^2)/sigma_s^2
               + mean((v_sim - v_obs)^2)/sigma_v^2
               + S_VIOLATION_PENALTY * spacing_violations ]

with, per event, sigma_s / sigma_v the standard deviation of the MEASURED spacing
and follower speed of that event, the state initialised from the first measured
sample and integrated forward with the measured leader speed
(``idm_core.simulate_event``, RK4 by default). Nothing about the model, the
bounds, the error definition or the data changes here: M4 only replaces the search.

--- algorithm -----------------------------------------------------------------

A textbook global-best (gbest) PSO on the real parameter box:

  * 40 particles, 80 iterations (the brief's numbers);
  * inertia weight ``w`` decaying LINEARLY from 0.9 to 0.4 across the iterations,
    w(t) = 0.9 + (0.4 - 0.9) * t / (iterations - 1), so the first iteration uses
    exactly 0.9 and the last exactly 0.4;
  * acceleration coefficients c1 = c2 = 1.5; every particle is pulled towards its
    own best position and towards the swarm's single best position (gbest);
  * velocity clamp: |v_d| <= 0.2 * (UPPER_d - LOWER_d), i.e. 20 % of that
    coordinate's box width per iteration (the brief's "0.2 x bound width"), applied
    BEFORE the position update;
  * bound handling - the ONE convention this script fixes and documents (the
    brief allows either): the position is clipped back into the box and the
    velocity component of every coordinate that had to be clipped is set to ZERO.
    Rationale: the objective is finite and meaningful on the boundary itself (the
    shared cost is evaluated at the projection of the vector onto the box), so
    parking a particle on the face is a legitimate outcome and must not be turned
    into a rejected step; zeroing that coordinate's velocity stops the particle
    from being dragged out again and again while leaving the other coordinates
    free to keep searching. The alternative (reflecting the velocity) would keep
    pushing the particle along the face and, with the IDM optimum known to sit ON
    several faces (s0, a and b hit their bounds in M2), it would spend iterations
    bouncing instead of consolidating. The count of clipped coordinates is
    recorded per run so the choice is visible in the evidence;
  * initial swarm: 40 positions from a seeded Latin hypercube over the box
    (``idm_core.latin_hypercube``) and 40 velocities from a standard normal scaled
    to 10 % of the box width, from a SEPARATE seeded generator (seed + 1) so the
    positions and the velocities do not share one stream; the whole run is
    therefore bit-for-bit reproducible;
  * fitness = the shared ``total_error``, MINIMISED, handed to the optimiser as
    ``idm_core.constrained_total_error`` (infeasible trajectory -> the large finite
    INFEASIBLE_COST wall, "worst particle"; a true ``inf`` would break the
    comparison ``cost < pbest_cost`` in a way that silently freezes the swarm).

--- convergence evidence ------------------------------------------------------

Recorded per run: the swarm's best cost after the initialisation and after
every iteration (the whole convergence curve), the first and last value, the
number of objective evaluations (particle-iterations, plus the number of distinct
points - the memo avoids re-evaluating a particle whose position did not change),
and the wall-clock time.

Output: ``results/m4_params.json`` (per run: parameters, total_error,
per-event raw RMSEs and normalised costs, the PSO convergence block and the
self-checks) plus the same numbers on stdout.

Usage:
    python src/calibration/m4_pso.py [--root DIR] [--particles 40] [--iterations 80]
                                   [--seed 20260921] [--method euler|rk4]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import idm_core as core  # noqa: E402  (import after the path fix, same dir)

SCRIPT = "src/calibration/m4_pso.py"
METHOD_NAME = "M4"
OUTPUT_FILENAME = "m4_params.json"

# --- PSO configuration (the brief's numbers) ---------------------------------

N_PARTICLES = 40
ITERATIONS = 80
DEFAULT_SEED = 20260921
W_START = 0.9
W_END = 0.4
C1 = 1.5
C2 = 1.5
V_MAX_FRACTION = 0.2   # velocity clamp = 0.2 * bound width, per coordinate
V_INIT_FRACTION = 0.1  # initial velocity scale = 0.1 * bound width
# The velocity stream is offset by this much from the position stream, so the two
# seeded draws are independent (documented, and it keeps the run reproducible).
VELOCITY_SEED_OFFSET = 1


class CostCache:
    """Memoised objective wrapping ``core.constrained_total_error`` (see M3).

    A particle whose position was clipped back to the point it already occupies
    costs nothing to re-evaluate, and a particle parked on a face does that every
    iteration. A hit returns the exact value of the first call; a non-finite
    value (a NaN that even the wall cannot fold in) is mapped to the large finite
    ``INFEASIBLE_COST`` so "worse than everything" is a real number the
    ``cost < pbest_cost`` test can order.
    """

    def __init__(self, datasets, method: str):
        self.datasets = datasets
        self.method = method
        self.cache: dict[tuple[float, ...], float] = {}
        self.calls = 0
        self.unique = 0

    def __call__(self, theta) -> float:
        vector = core.as_theta(theta)
        key = tuple(float(value) for value in np.round(vector, 12))
        self.calls += 1
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        cost = core.constrained_total_error(self.datasets, vector, self.method)
        if not math.isfinite(cost):
            cost = float(core.INFEASIBLE_COST)
        self.cache[key] = cost
        self.unique += 1
        return cost


def initial_swarm(particles: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded Latin hypercube positions and scaled-normal velocities in the box."""
    width = core.UPPER - core.LOWER
    positions = np.clip(core.latin_hypercube(particles, seed), core.LOWER, core.UPPER)
    velocity_rng = np.random.default_rng(seed + VELOCITY_SEED_OFFSET)
    velocities = velocity_rng.standard_normal(positions.shape) * (V_INIT_FRACTION * width)
    return positions, velocities


def calibrate_run(run: str, events, method: str, particles: int,
                     iterations: int, seed: int, verbose: bool = True) -> dict[str, object]:
    """One gbest PSO run for one run, on the shared trajectory cost."""
    datasets = [data for _, data in events]
    objective = CostCache(datasets, method)
    width = core.UPPER - core.LOWER
    v_max = V_MAX_FRACTION * width

    positions, velocities = initial_swarm(particles, seed)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()

    costs = np.array([objective(theta) for theta in positions], dtype=float)
    personal_best = positions.copy()
    personal_best_cost = costs.copy()
    best_index = int(np.argmin(costs))
    gbest = positions[best_index].copy()
    gbest_cost = float(costs[best_index])

    best_curve = [gbest_cost]
    mean_curve = [float(costs.mean())]
    clipped_coordinates = 0
    infeasible_reports = int(np.sum(costs >= core.INFEASIBLE_COST))
    outside_box_reports = int(sum(1 for theta in positions if not core.inside_bounds(theta)))

    for iteration in range(iterations):
        # Linear decay 0.9 -> 0.4: the first iteration uses W_START, the last W_END.
        w = W_START + (W_END - W_START) * iteration / max(iterations - 1, 1)
        r1 = rng.random(positions.shape)
        r2 = rng.random(positions.shape)
        velocities = (
            w * velocities
            + C1 * r1 * (personal_best - positions)
            + C2 * r2 * (gbest - positions)
        )
        velocities = np.clip(velocities, -v_max, v_max)
        positions = positions + velocities

        # Bound handling (the one convention this script fixes): clip the position
        # into the box and ZERO the velocity component of every clipped coordinate.
        outside = (positions < core.LOWER) | (positions > core.UPPER)
        clipped_coordinates += int(outside.sum())
        positions = np.clip(positions, core.LOWER, core.UPPER)
        velocities = np.where(outside, 0.0, velocities)

        costs = np.array([objective(theta) for theta in positions], dtype=float)
        # Infeasible particles (diverged trajectories, cost = the 1e12 wall) are a
        # normal part of a random initial swarm; a vector OUTSIDE THE BOX would be a
        # bug (positions are clipped), so the two are counted apart.
        infeasible_reports += int(np.sum(costs >= core.INFEASIBLE_COST))
        outside_box_reports += int(sum(1 for theta in positions if not core.inside_bounds(theta)))
        improved = costs < personal_best_cost
        personal_best[improved] = positions[improved]
        personal_best_cost[improved] = costs[improved]
        swarm_best = int(np.argmin(personal_best_cost))
        if float(personal_best_cost[swarm_best]) < gbest_cost:
            gbest_cost = float(personal_best_cost[swarm_best])
            gbest = personal_best[swarm_best].copy()

        best_curve.append(gbest_cost)
        mean_curve.append(float(personal_best_cost.mean()))
        if verbose and (iteration < 3 or (iteration + 1) % 10 == 0 or iteration == iterations - 1):
            print(f"    iter {iteration + 1:3d}/{iterations}  w={w:.4f}  best J={gbest_cost:12.6f}"
                  f"  mean pbest J={float(personal_best_cost.mean()):12.6f}  evals={objective.calls}")

    elapsed = time.perf_counter() - started
    best_theta = np.clip(gbest, core.LOWER, core.UPPER)
    if not core.inside_bounds(best_theta):
        raise RuntimeError(f"{run}: PSO returned a vector outside the box")
    reported_cost = core.total_error(datasets, best_theta, method)
    if not math.isfinite(reported_cost):
        raise RuntimeError(f"{run}: the PSO's gbest is infeasible")

    trajectory = core.summarize_events(datasets, best_theta, method)
    return {
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
        "pso": {
            "variant": "global best (gbest) particle swarm optimisation",
            "particles": particles,
            "iterations": iterations,
            "seed": seed,
            "inertia": f"w linearly {W_START:g} -> {W_END:g} over the iterations",
            "acceleration": f"c1={C1:g}, c2={C2:g} (cognitive / social)",
            "velocity_clamp": f"|v_d| <= {V_MAX_FRACTION:g} * bound width of coordinate d",
            "bound_handling": "position clipped to the box; velocity zeroed on clipped coordinates",
            "initial_swarm": (
                f"Latin hypercube positions (seed {seed}) + normal velocities scaled to "
                f"{V_INIT_FRACTION:g} * bound width (seed {seed + VELOCITY_SEED_OFFSET})"
            ),
            "fitness": "core.total_error (minimised); infeasible trajectory -> INFEASIBLE_COST wall",
            "objective": "idm_core.constrained_total_error",
            "clipped_coordinates": int(clipped_coordinates),
            "infeasible_particles": int(infeasible_reports),
            "infeasible_particles_note": "evaluations whose trajectory diverged (cost = the 1e12 wall); expected in a random swarm",
            "outside_box_particles": int(outside_box_reports),
            "outside_box_note": "must be 0: every position is clipped into the box",
            "elapsed_seconds": float(elapsed),
            "evaluations": int(objective.calls),
            "distinct_evaluations": int(objective.unique),
            "cache_hits": int(objective.calls - objective.unique),
        },
        "convergence": {
            "iterations": iterations,
            "best_curve": best_curve,
            "mean_pbest_curve": mean_curve,
            "initial_best": best_curve[0],
            "final_best": best_curve[-1],
            "improvement": float(best_curve[0] - best_curve[-1]),
            "monotone_nonincreasing": bool(all(
                best_curve[i + 1] <= best_curve[i] + 1e-12 for i in range(len(best_curve) - 1)
            )),
            "reported_best_cost": float(reported_cost),
            "elapsed_seconds": float(elapsed),
            "evaluations": int(objective.calls),
        },
    }


def print_run(run: str, result: dict[str, object]) -> None:
    """Stdout block for one run (parameters, J, RMSEs, convergence)."""
    theta = core.as_theta(result["theta_vector"])
    pso = result["pso"]            # type: ignore[assignment]
    conv = result["convergence"]   # type: ignore[assignment]
    print(f"  best theta         : {core.format_theta(theta)}")
    print(f"  within bounds      : {core.inside_bounds(theta)}")
    print(f"  total_error J      : {result['total_error']:.6f}  (equal weight mean of the per-event J_e)")
    print(f"  spacing RMSE (mean/worst): {result['mean_spacing_rmse']:.4f} / {result['worst_spacing_rmse']:.4f}")
    print(f"  speed   RMSE (mean/worst): {result['mean_speed_rmse']:.4f} / {result['worst_speed_rmse']:.4f}")
    print(f"  convergence        : iter 0 best J={conv['initial_best']:.6f} -> iter {conv['iterations']} "
          f"best J={conv['final_best']:.6f}  (improvement {conv['improvement']:.6f})")
    print(f"  curve monotone     : {conv['monotone_nonincreasing']}   evaluations={conv['evaluations']}"
          f" (distinct {pso['distinct_evaluations']}, cache hits {pso['cache_hits']})"
          f"   runtime={conv['elapsed_seconds']:.2f} s")
    print(f"  clipped coordinates: {pso['clipped_coordinates']} (velocity zeroed there)"
          f"   infeasible particles={pso['infeasible_particles']} (= the 1e12 wall)"
          f"   outside-box particles={pso['outside_box_particles']} (must be 0)")
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
    parser.add_argument("--particles", type=int, default=N_PARTICLES,
                        help=f"swarm size (default {N_PARTICLES})")
    parser.add_argument("--iterations", type=int, default=ITERATIONS,
                        help=f"number of iterations (default {ITERATIONS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"PSO seed, per run (default {DEFAULT_SEED})")
    parser.add_argument("--method", default=core.DEFAULT_METHOD, choices=list(core.METHODS),
                        help="trajectory integration scheme")
    args = parser.parse_args(argv)

    root = core.resolve_root(args.root)
    grouped = core.load_event_data(root)

    print(f"project root : {root}")
    print(f"input dir    : {core.rel(root / core.STAGE4_SUBDIR, root)}")
    print(f"method       : {METHOD_NAME} - gbest particle swarm optimisation")
    print(f"objective    : constrained_total_error = equal weight mean of the per-event")
    print(f"               [ mean(ds^2)/sigma_s^2 + mean(dv^2)/sigma_v^2 + penalty*violations ]")
    print(f"spacing      : s = dhw - {core.L_VEH:g}   (spacing violation = simulated s <= 0)")
    print(f"pso          : particles={args.particles}, iterations={args.iterations}, seed={args.seed} per run")
    print(f"coefficients : w {W_START:g} -> {W_END:g} linear, c1=c2={C1:g}, "
          f"|v| <= {V_MAX_FRACTION:g} * bound width, bounds: clip position + zero velocity")
    print(f"bounds       : v0[{core.LOWER[0]:g},{core.UPPER[0]:g}] T[{core.LOWER[1]:g},{core.UPPER[1]:g}] "
          f"s0[{core.LOWER[2]:g},{core.UPPER[2]:g}] a[{core.LOWER[3]:g},{core.UPPER[3]:g}] "
          f"b[{core.LOWER[4]:g},{core.UPPER[4]:g}]")
    print(f"integration  : {args.method} on the real time deltas, leader speed interpolated from lv")
    print()

    per_run: dict[str, dict[str, object]] = {}
    started = time.perf_counter()
    for run, events in grouped.items():
        print(f"{run}: particle swarm")
        result = calibrate_run(run, events, args.method, args.particles,
                                  args.iterations, args.seed)
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
            "algorithm": "global best (gbest) particle swarm optimisation",
            "particles": args.particles,
            "iterations": args.iterations,
            "seed": args.seed,
            "inertia": {"w_start": W_START, "w_end": W_END, "schedule": "linear decay"},
            "acceleration": {"c1": C1, "c2": C2},
            "velocity_clamp": {"fraction_of_bound_width": V_MAX_FRACTION},
            "bound_handling": "clip position into the box, zero the velocity on clipped coordinates",
            "infeasible_handling": "INFEASIBLE_COST wall (worst particle)",
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
    payload["total_runtime_seconds"] = float(total_seconds)  # type: ignore[index]
    out_path = core.results_dir(root) / OUTPUT_FILENAME
    core.write_json_atomic(out_path, payload)

    # --- self-checks ---------------------------------------------------------
    checks = payload["self_checks"]  # type: ignore[index]
    print("self-checks:")
    for key in sorted(checks):
        if key == "runs_found":
            continue
        print(f"  {key:<46} {checks[key]}")
    print(f"  runs_found                                  {checks['runs_found']}")

    focus_ok = checks["focus_beats_zero_model"]
    if focus_ok is not True:
        print()
        print(f"WARNING: the focus event ({checks['focus_run']}/{checks['focus_event']}) spacing "
              "RMSE is NOT better than the constant zero model - see results for the numbers "
              "(warning only, not an error).")

    curve_ok = all(
        bool(result["convergence"]["monotone_nonincreasing"])  # type: ignore[index]
        and float(result["convergence"]["improvement"]) >= 0.0  # type: ignore[index]
        for result in per_run.values()
    )
    if not curve_ok:
        print("WARNING: the PSO best curve is not non-increasing (gbest updates should prevent it).")

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
        print("ERROR: self-checks failed (failed above).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
