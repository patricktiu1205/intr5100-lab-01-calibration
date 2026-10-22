#!/usr/bin/env python3
"""INTR5100 Lab Session I - M5: differentiable IDM (torch) + Adam, multi-start.

Method 5 of the IDM calibration study. Instead of handing the trajectory error to
a derivative-free search, the IDM recursion itself is written as a differentiable
computation graph in **torch** and the SAME cost is minimised by gradient descent:

    J(theta) = mean over events of
               [ mean((s_sim - s_obs)^2)/sigma_s^2
               + mean((v_sim - v_obs)^2)/sigma_v^2
               + S_VIOLATION_PENALTY * spacing_violations ]

Nothing about the model, the bounds, the error definition or the data changes:
the model, the initial condition (first measured sample) and the leader input
(measured lv) are exactly those of ``idm_core``. No numpy simulation is used
inside the optimisation - the states are produced by the torch graph - and the
final parameters are then re-verified with ``idm_core`` itself (see below).

--- the differentiable model -------------------------------------------------

Euler rollout, one step per real time delta of the stage 4 table, all in double
precision, identical in form to the Euler branch of ``idm_core.simulate_event``:

    dv   = lv_i - v
    s*   = s0 + max(0, v*T + v*dv / (2*sqrt(a*b + eps)))
    a_i  = a * ( 1 - (v/v0)^4 - (max(s*,) / max(s, S_FLOOR))^2 )
    s'   = s + dt * (lv_i - v)
    v'   = v + dt * a_i

``eps = 1e-6`` is added under the square root exactly as the brief asks. It is a
pure numerical guard: the box enforces a, b >= 10, so the true value is never
near zero and the eps changes the interaction term by ~1e-8 relative - far below
the RK4-vs-Euler difference the consistency check below quantifies.

Numerical protection mirrors ``idm_core``'s three tiers so the graph stays finite
for ANY parameter vector (a NaN would poison the gradient):
  * a step that drives the spacing to s <= 0 is counted (``violations``), the
    state is floored at ``S_FLOOR`` and the excursion is paid for at
    ``S_VIOLATION_PENALTY`` per occurrence - exactly the shared cost's rule;
  * a step that would leave the numerical domain (spacing past
    ``SPACING_EXPLOSION_LIMIT``, |speed| past ``SPEED_EXPLOSION_LIMIT``) is
    clamped and counted (``clamped_steps``); such a start is a runaway and is
    flagged infeasible in the report rather than being silently fitted.

--- parameters ---------------------------------------------------------------

The optimiser works on unconstrained variables: ``z`` in R^5, mapped to the box by
the sigmoid,

    theta = LOWER + sigmoid(z) * (UPPER - LOWER)

so the box is respected exactly (no clipping, no penalty term, no boundary
stagnation in z-space) and the inverse ``z = logit((theta - LOWER)/(UPPER - LOWER))``
turns a starting point into its unconstrained image.

--- training -----------------------------------------------------------------

Adam, lr = 0.05, 300 steps, 10 multi-starts with the same FIXED seed strategy as
M2 (start 0 = the brief's reference theta0, starts 1..9 = a seeded Latin
hypercube, ``idm_core.latin_hypercube``, no scipy dependency).

The 10 starts are trained SIMULTANEOUSLY as one batch: ``z`` is a (10, 5) tensor
and the loss is the SUM over starts of each start's own cost. Adam is a
coordinate-wise rule and the loss is separable across rows, so a row's gradient
and its update are mathematically identical to training that start alone - the
batching changes only the wall-clock time, never the trajectory of the
optimisation. (The sum, not the mean, is deliberate: the mean would divide every
gradient by 10, which Adam tolerates but which makes the reported per-step cost
scale differ from the per-start cost.)

--- convergence evidence ------------------------------------------------------

Recorded per run: the cost of the cheapest start after every Adam step (the
training curve) together with its running minimum, which is the non-increasing
convergence curve (the cheapest start at a given step can tick upward, because a
start that is cheapest now may be walked away from its own best a step later);
the cost of every start every 25 steps; the initial and final cost of every start;
the number of starts that stayed feasible, how many distinct parameter vectors
they converged to, and the spread of their final costs (the brief's "reasonable
dispersion" of a multi-start run); and the wall-clock time. Because the 10 starts
are trained as one batch, that time is a single figure for the whole batch - it is
the time of the batched run, not 10 independent runs' times.

--- consistency check (the brief's requirement) ------------------------------

Every start's final parameters are RE-EVALUATED with numpy through
``idm_core.total_error(datasets, theta, method)``:
  * ``euler`` - the same scheme as the torch graph, so this number must agree
    with the torch cost to ~1e-9 and proves the graph implements the shared cost
    faithfully;
  * ``rk4`` - the scheme the other methods and the reported ``total_error`` use.
The relative difference between the torch cost and the RK4 re-computation is
printed per start, and a difference above 10 % raises a WARNING (the brief's
threshold). Starts are RANKED by the RK4 re-computation, i.e. by the same yardstick
as M1..M4, so the comparison table is apples to apples.

Output: ``results/m5_params.json`` (per run: parameters, total_error,
per-event raw RMSEs and normalised costs, the optimiser/convergence block and the
self-checks) plus the same numbers on stdout.

Usage:
    python src/calibration/m5_differentiable.py [--root DIR] [--starts 10]
                                              [--steps 300] [--lr 0.05]
                                              [--seed 20260921]
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import idm_core as core  # noqa: E402  (import after the path fix, same dir)

SCRIPT = "src/calibration/m5_differentiable.py"
METHOD_NAME = "M5"
OUTPUT_FILENAME = "m5_params.json"

# --- configuration -----------------------------------------------------------

DEFAULT_STARTS = 10       # the brief: 10 multi-starts
DEFAULT_STEPS = 300       # the brief: 300 Adam steps
DEFAULT_LR = 0.05         # the brief: Adam lr = 0.05
DEFAULT_SEED = 20260921   # same seed strategy as M2
SQRT_EPS = 1.0e-6         # the brief: eps under the sqrt
CONSISTENCY_TOLERANCE = 0.10   # the brief: warn above 10 % torch-vs-RK4 difference
START_LOGIT_CLIP = 6.0    # |z| cap when inverting a starting point into z-space
CURVE_SAMPLE_EVERY = 25   # per-start cost samples stored in the JSON


# --- torch setup -------------------------------------------------------------


def configure_torch(seed: int) -> None:
    """Double precision, single thread, seeded: reproducible on CPU.

    float64 matters (the trajectory is an integrated recurrence, and the other
    methods are float64 numpy), and one thread removes any thread-scheduling
    nondeterminism from the reduction order, so two runs produce the same numbers
    bit for bit. The tiny tensors here are faster on one thread anyway.
    """
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(seed)


# --- differentiable IDM ------------------------------------------------------


def decode_parameters(z: torch.Tensor) -> torch.Tensor:
    """Unconstrained z -> box theta through the sigmoid (exactly inside the box)."""
    span = torch.as_tensor(core.UPPER - core.LOWER, dtype=torch.float64)
    low = torch.as_tensor(core.LOWER, dtype=torch.float64)
    return low + torch.sigmoid(z) * span


def encode_parameters(theta) -> torch.Tensor:
    """Box theta -> unconstrained z (inverse of :func:`decode_parameters`)."""
    unit = np.clip(core.normalized_theta(theta), 1e-3, 1.0 - 1e-3)
    return torch.as_tensor(np.log(unit / (1.0 - unit)), dtype=torch.float64)


def build_event_tensor(event: core.EventData) -> dict[str, torch.Tensor]:
    """Pre-tensor one event: time deltas, measured series, leader and scales."""
    # ``torch.tensor`` (a copy) rather than ``as_tensor``: the numpy arrays are
    # read-only views of the frames, and torch warns about (and does not support)
    # non-writable storages.
    return {
        "name": event.name,
        "dt": torch.tensor(np.diff(event.t), dtype=torch.float64),
        "lv": torch.tensor(event.vL_obs, dtype=torch.float64),
        "s_obs": torch.tensor(event.s_obs, dtype=torch.float64),
        "v_obs": torch.tensor(event.v_obs, dtype=torch.float64),
        "s_first": float(event.s_obs[0]),
        "v_first": float(event.v_obs[0]),
        "sigma_s": max(float(event.sigma_s), core.MIN_SIGMA),
        "sigma_v": max(float(event.sigma_v), core.MIN_SIGMA),
    }


def rollout_event(ev: dict[str, torch.Tensor], theta: torch.Tensor):
    """Euler rollout of one event for a whole batch of parameter vectors.

    ``theta`` is (S, 5); the returned state series are (n, S). The leader speed is
    the measured ``lv`` series held as a constant tensor (the leader is an INPUT,
    never a state - same as ``idm_core``), and the step is the real time delta.
    """
    n_starts = theta.shape[0]
    # Plain 1-D slices, so every state tensor is (n_starts,) - a (n_starts, 1)
    # slice would broadcast against it and silently produce a (n_starts, n_starts)
    # matrix, i.e. cross terms between two different parameter vectors.
    v0 = theta[:, 0]
    T = theta[:, 1]
    s0 = theta[:, 2]
    a = theta[:, 3]
    b = theta[:, 4]

    dt = ev["dt"]
    lv = ev["lv"]

    s = torch.full((n_starts,), ev["s_first"], dtype=torch.float64)
    v = torch.full((n_starts,), ev["v_first"], dtype=torch.float64)
    spacing = [s]
    speed = [v]
    violations = torch.zeros(n_starts, dtype=torch.float64)
    clamped = torch.zeros(n_starts, dtype=torch.float64)

    for i in range(dt.shape[0]):
        leader_now = lv[i]
        dv = leader_now - v
        # eps under the sqrt, per the brief (pure guard: a, b >= 10 inside the box).
        headway = v * T + v * dv / (2.0 * torch.sqrt(a * b + SQRT_EPS))
        desired = s0 + torch.clamp(headway, min=0.0)
        safe_spacing = torch.clamp(s, min=core.S_FLOOR)
        acceleration = a * (1.0 - (v / v0) ** core.DELTA - (desired / safe_spacing) ** 2)
        s_next = s + dt[i] * dv
        v_next = v + dt[i] * acceleration

        violations = violations + (s_next <= 0.0).to(torch.float64)
        out_of_domain = (s_next > core.SPACING_EXPLOSION_LIMIT) | (
            torch.abs(v_next) > core.SPEED_EXPLOSION_LIMIT
        )
        clamped = clamped + out_of_domain.to(torch.float64)

        s = torch.clamp(s_next, min=core.S_FLOOR, max=core.SPACING_EXPLOSION_LIMIT)
        v = torch.clamp(v_next, min=-core.SPEED_EXPLOSION_LIMIT, max=core.SPEED_EXPLOSION_LIMIT)
        spacing.append(s)
        speed.append(v)

    return torch.stack(spacing), torch.stack(speed), violations, clamped


def torch_cost(events: list[dict[str, torch.Tensor]], z: torch.Tensor):
    """The shared multi-event cost in torch, per start of the batch.

    Returns ``(cost_per_start, violations_per_start, clamped_per_start)`` where
    ``cost_per_start`` has one entry per row of ``z``: the equal-weight mean over
    events of the per-event normalised cost, i.e. exactly ``idm_core.total_error``
    evaluated on the torch rollout.
    """
    theta = decode_parameters(z)
    n_starts = theta.shape[0]
    total = torch.zeros(n_starts, dtype=torch.float64)
    violations_total = torch.zeros(n_starts, dtype=torch.float64)
    clamped_total = torch.zeros(n_starts, dtype=torch.float64)
    for ev in events:
        spacing, speed, violations, clamped = rollout_event(ev, theta)
        spacing_error = (spacing - ev["s_obs"].unsqueeze(1)) / ev["sigma_s"]
        speed_error = (speed - ev["v_obs"].unsqueeze(1)) / ev["sigma_v"]
        event_cost = (
            (spacing_error**2).mean(dim=0)
            + (speed_error**2).mean(dim=0)
            + core.S_VIOLATION_PENALTY * violations
        )
        total = total + event_cost
        violations_total = violations_total + violations
        clamped_total = clamped_total + clamped
    return total / len(events), violations_total, clamped_total


# --- run run --------------------------------------------------------------


def start_points(n_starts: int, seed: int) -> list[tuple[str, np.ndarray]]:
    """Reference point + seeded Latin hypercube starts (same strategy as M2)."""
    if n_starts < 2:
        raise ValueError("at least 2 multi-starts are needed (reference point + Latin hypercube)")
    points: list[tuple[str, np.ndarray]] = [("reference theta0", core.as_theta(core.DEFAULT_THETA))]
    for index, point in enumerate(core.latin_hypercube(n_starts - 1, seed), start=1):
        points.append((f"lhs {index} (seed {seed})", core.as_theta(point)))
    return points


def calibrate_run(run: str, events, starts: int, steps: int, lr: float,
                     seed: int, verbose: bool = True) -> dict[str, object]:
    """Batched multi-start Adam on the differentiable IDM of one run."""
    datasets = [data for _, data in events]
    tensors = [build_event_tensor(data) for data in datasets]
    points = start_points(starts, seed)
    z = torch.stack([encode_parameters(point) for _, point in points]).clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([z], lr=lr)
    best_curve: list[float] = []       # best-so-far over the batch, one value per step
    step_best_curve: list[float] = []  # minimum over the 10 starts at that step
    best_so_far = float("inf")
    start_curves: list[list[float]] = [[] for _ in points]
    losses: list[float] = []
    started = time.perf_counter()
    initial_cost: np.ndarray | None = None
    final_cost: np.ndarray | None = None

    for step in range(steps):
        cost, _violations, _clamped = torch_cost(tensors, z)
        if not bool(torch.isfinite(cost).all()):
            # Should be unreachable thanks to the clamps, but a non-finite loss must
            # never be turned into a silent update: report it and stop this run.
            raise RuntimeError(f"{run}: non-finite torch cost at step {step}")
        loss = cost.sum()   # separable across starts -> per-row gradients unchanged
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        values = cost.detach().numpy()
        if initial_cost is None:
            initial_cost = values.copy()
        final_cost = values
        # Two different readings of "the best curve", both recorded:
        #  * step_min is the cheapest start at THAT step - it can tick upward, since
        #    a start that is cheapest now may be overtaken (and moved away from its
        #    own best) a step later;
        #  * best_so_far is the running minimum over starts and steps, which is the
        #    non-increasing convergence curve a training curve is expected to be.
        step_best = float(values.min())
        step_best_curve.append(step_best)
        best_so_far = min(best_so_far, step_best)
        best_curve.append(best_so_far)
        if step % CURVE_SAMPLE_EVERY == 0 or step == steps - 1:
            for k, value in enumerate(values):
                start_curves[k].append(float(value))
        losses.append(float(loss.detach()))
    elapsed = time.perf_counter() - started

    with torch.no_grad():
        final_cost_tensor, violations, clamped = torch_cost(tensors, z)
        final_theta = decode_parameters(z).numpy()

    start_records: list[dict[str, object]] = []
    for index, (label, point) in enumerate(points):
        theta = np.clip(core.as_theta(final_theta[index]), core.LOWER, core.UPPER)
        torch_value = float(final_cost_tensor[index])
        numpy_euler = core.total_error(datasets, theta, core.EULER)
        numpy_rk4 = core.total_error(datasets, theta, core.DEFAULT_METHOD)
        euler_finite = math.isfinite(numpy_euler)
        # A start whose torch rollout had to be clamped is a runaway: idm_core
        # reports that trajectory as diverged (an inf Euler cost), so the torch
        # number is a large finite artefact of the clamp and is NOT comparable to
        # it. Such a start is excluded from the consistency check (and from the
        # ranking) instead of being scored as a disagreement.
        comparable = bool(euler_finite and float(clamped[index]) == 0.0)
        relative = None
        if comparable and math.isfinite(numpy_rk4):
            relative = float(abs(numpy_rk4 - torch_value) / max(abs(numpy_rk4), 1e-12))
        record = {
            "index": index,
            "label": label,
            "start_theta": [float(value) for value in point],
            "theta": [float(value) for value in theta],
            "inside_bounds": core.inside_bounds(theta),
            "initial_cost_torch": float(initial_cost[index]) if initial_cost is not None else None,
            "final_cost_torch": torch_value,
            "final_cost_numpy_euler": float(numpy_euler),
            "final_cost_numpy_rk4": float(numpy_rk4),
            "torch_vs_rk4_relative_difference": relative,
            "torch_vs_euler_relative_difference": (
                None if not euler_finite else float(
                    abs(numpy_euler - torch_value) / max(abs(numpy_euler), 1e-12)
                )
            ),
            "spacing_violations": float(violations[index]),
            "clamped_steps": float(clamped[index]),
            "runaway": bool(clamped[index] > 0),
            "comparison_valid": comparable,
            "comparison_note": (
                "torch state clamped at the explosion limit; idm_core reports this "
                "trajectory as diverged (inf), so the torch cost is not comparable"
                if not comparable else "torch cost comparable with idm_core"
            ),
            "feasible_candidate": bool(
                core.inside_bounds(theta) and math.isfinite(numpy_rk4) and clamped[index] == 0
            ),
        }
        start_records.append(record)
        if verbose:
            print(
                f"    start {index:2d}/{len(points) - 1:<2d} {label:<22}"
                f" torch J={torch_value:12.6f}  numpy(euler)={numpy_euler:12.6f}"
                f"  numpy(rk4)={numpy_rk4:12.6f}"
                f"  rel={'-' if relative is None else f'{relative:.3e}'}"
                f"  {'RUNAWAY (clamped, infeasible)' if not comparable else ''}"
                f"{'  WARN >10%' if (relative or 0.0) > CONSISTENCY_TOLERANCE else ''}"
            )

    feasible = [record for record in start_records if record["feasible_candidate"]]
    if not feasible:
        raise RuntimeError(f"{run}: no feasible M5 multi-start result")
    best = min(feasible, key=lambda record: float(record["final_cost_numpy_rk4"]))
    best_theta = core.as_theta(best["theta"])
    trajectory = core.summarize_events(datasets, best_theta, core.DEFAULT_METHOD)
    if not math.isfinite(float(trajectory["total_error"])):
        raise RuntimeError(f"{run}: the best M5 vector is infeasible under idm_core")

    final_costs = np.array([float(record["final_cost_numpy_rk4"]) for record in feasible])
    distinct = len({tuple(np.round(core.as_theta(record["theta"]), 6)) for record in feasible})
    warnings = [
        f"{record['label']}: torch vs RK4 relative difference "
        f"{float(record['torch_vs_rk4_relative_difference']):.4%} > {CONSISTENCY_TOLERANCE:.0%}"
        for record in start_records
        if record["torch_vs_rk4_relative_difference"] is not None
        and float(record["torch_vs_rk4_relative_difference"]) > CONSISTENCY_TOLERANCE
    ]
    # Fidelity of the torch graph: torch (Euler) vs numpy (Euler) on the SAME
    # vectors, i.e. the scheme the graph implements. Only comparable (non-runaway)
    # starts can be probed, since idm_core returns inf for a diverged trajectory.
    euler_differences = [
        float(record["torch_vs_euler_relative_difference"])
        for record in start_records
        if record["comparison_valid"] and record["torch_vs_euler_relative_difference"] is not None
    ]
    rk4_differences = [
        float(record["torch_vs_rk4_relative_difference"])
        for record in start_records
        if record["torch_vs_rk4_relative_difference"] is not None
    ]
    runaway_starts = [str(record["label"]) for record in start_records if record["runaway"]]

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
        "differentiable": {
            "framework": f"torch {torch.__version__} (CPU, float64, 1 thread)",
            "rollout": "Euler, one step per real time delta, leader = measured lv (constant tensor)",
            "graph": "state (s, v) from the first measured sample, IDM written pointwise in torch",
            "sqrt_eps": SQRT_EPS,
            "parameterisation": "z in R^5 -> theta = LOWER + sigmoid(z) * (UPPER - LOWER)",
            "optimiser": "torch.optim.Adam",
            "learning_rate": lr,
            "steps": steps,
            "n_starts": len(points),
            "starts": "reference theta0 + seeded Latin hypercube (same strategy and seed as M2)",
            "seed": seed,
            "batching": "the 10 starts share one (n_starts, 5) tensor; loss = sum of per-start costs",
            "ranking": "starts ranked by idm_core.total_error with RK4 (same yardstick as M1..M4)",
            "objective": "idm_core.total_error as written in torch (equal weight mean of per-event costs)",
            "consistency_tolerance": CONSISTENCY_TOLERANCE,
            "consistency_warnings": warnings,
            "runaway_starts": runaway_starts,
            "torch_vs_numpy_euler_max_relative_difference": (
                max(euler_differences) if euler_differences else None
            ),
            "torch_vs_numpy_rk4_max_relative_difference": (
                max(rk4_differences) if rk4_differences else None
            ),
            "elapsed_seconds": float(elapsed),
            "steps_per_second": float(steps / elapsed) if elapsed > 0 else None,
        },
        "convergence": {
            "steps": steps,
            "best_curve": best_curve,
            "best_curve_sampling": "one value per Adam step (running minimum over the 10 starts and the steps)",
            "step_min_curve": step_best_curve,
            "step_min_curve_sampling": "one value per Adam step (cheapest start at that step; not necessarily monotone)",
            "start_curves": start_curves,
            "start_curves_sampling": f"one value every {CURVE_SAMPLE_EVERY} steps per start, rounded to 6 dp",
            "initial_best": best_curve[0],
            "final_best": best_curve[-1],
            "best_start_index": int(best["index"]),
            "best_start_label": str(best["label"]),
            "best_start_curve": start_curves[int(best["index"])],
            "feasible_starts": len(feasible),
            "infeasible_starts": len(start_records) - len(feasible),
            "final_cost_std_across_starts": float(np.std(final_costs)),
            "final_cost_min_across_starts": float(np.min(final_costs)),
            "final_cost_max_across_starts": float(np.max(final_costs)),
            "distinct_feasible_results": distinct,
            "per_start": start_records,
            "elapsed_seconds": float(elapsed),
        },
    }


def print_run(run: str, result: dict[str, object]) -> None:
    """Stdout block for one run (parameters, J, RMSEs, consistency check)."""
    theta = core.as_theta(result["theta_vector"])
    diff = result["differentiable"]  # type: ignore[assignment]
    conv = result["convergence"]     # type: ignore[assignment]
    print(f"  best theta         : {core.format_theta(theta)}")
    print(f"  within bounds      : {core.inside_bounds(theta)}")
    print(f"  total_error J      : {result['total_error']:.6f}  (equal weight mean of the per-event J_e, RK4)")
    print(f"  spacing RMSE (mean/worst): {result['mean_spacing_rmse']:.4f} / {result['worst_spacing_rmse']:.4f}")
    print(f"  speed   RMSE (mean/worst): {result['mean_speed_rmse']:.4f} / {result['worst_speed_rmse']:.4f}")
    print(f"  convergence        : step 0 best J={conv['initial_best']:.6f} -> step {conv['steps']} "
          f"best J={conv['final_best']:.6f}  (improvement {float(conv['initial_best']) - float(conv['final_best']):.6f})")
    print(f"  multi-start        : {conv['feasible_starts']}/{len(conv['per_start'])} feasible,"
          f" best = {conv['best_start_label']}, distinct solutions {conv['distinct_feasible_results']},"
          f" final cost across starts min/max/std = {conv['final_cost_min_across_starts']:.4f}/"
          f"{conv['final_cost_max_across_starts']:.4f}/{conv['final_cost_std_across_starts']:.4f}")
    print(f"  training           : {diff['steps']} Adam steps, lr={diff['learning_rate']:g},"
          f" runtime={diff['elapsed_seconds']:.2f} s ({diff['steps_per_second']:.1f} steps/s),"
          f" batched over {diff['n_starts']} starts")
    print(f"  {'event':<10}{'n':>5}{'s_RMSE':>11}{'v_RMSE':>11}{'J_spacing':>11}{'J_speed':>10}"
          f"{'J_e':>11}{'zero_s':>11}")
    for name, metrics in result["events"].items():  # type: ignore[union-attr]
        print(
            f"  {name:<10}{metrics['n_samples']:>5}{metrics['spacing_rmse']:>11.4f}"
            f"{metrics['speed_rmse']:>11.4f}{metrics['j_spacing']:>11.4f}"
            f"{metrics['j_speed']:>10.4f}{metrics['j_total']:>11.4f}"
            f"{metrics['zero_model']['best_spacing_rmse']:>11.4f}"
        )
    warnings = diff["consistency_warnings"]
    runaway = diff["runaway_starts"]
    euler_max = diff["torch_vs_numpy_euler_max_relative_difference"]
    rk4_max = diff["torch_vs_numpy_rk4_max_relative_difference"]
    print(f"  consistency        : torch(Euler) vs numpy(Euler) max rel. difference = "
          f"{'n/a' if euler_max is None else format(float(euler_max), '.3e')}"
          f"; torch(Euler) vs numpy(RK4) max rel. difference = "
          f"{'n/a' if rk4_max is None else format(float(rk4_max), '.4f')}")
    if runaway:
        print(f"  runaway starts     : {len(runaway)} - {runaway} (torch state clamped;"
              f" idm_core reports those trajectories as diverged, so no comparison is made)")
    if warnings:
        print(f"  WARNING            : {len(warnings)} start(s) disagree with RK4 by more than "
              f"{CONSISTENCY_TOLERANCE:.0%}:")
        for line in warnings:
            print(f"                       {line}")
    elif not runaway:
        print(f"  consistency check  : every start is within {CONSISTENCY_TOLERANCE:.0%} of the "
              f"numpy RK4 re-evaluation")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="project root (default: inferred)")
    parser.add_argument("--starts", type=int, default=DEFAULT_STARTS,
                        help=f"number of multi-starts (default {DEFAULT_STARTS})")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Adam steps (default {DEFAULT_STEPS})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Adam learning rate (default {DEFAULT_LR})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"seed of the Latin hypercube multi-start (default {DEFAULT_SEED})")
    args = parser.parse_args(argv)

    root = core.resolve_root(args.root)
    grouped = core.load_event_data(root)
    configure_torch(args.seed)

    print(f"project root : {root}")
    print(f"input dir    : {core.rel(root / core.STAGE4_SUBDIR, root)}")
    print(f"method       : {METHOD_NAME} - differentiable IDM (torch) + Adam, {args.starts} multi-starts")
    print(f"objective    : total_error as written in torch = equal weight mean of the per-event")
    print(f"               [ mean(ds^2)/sigma_s^2 + mean(dv^2)/sigma_v^2 + penalty*violations ]")
    print(f"spacing      : s = dhw - {core.L_VEH:g}   (spacing violation = simulated s <= 0)")
    print(f"torch        : {torch.__version__} CPU, float64, {torch.get_num_threads()} thread")
    print(f"rollout      : Euler, real dt, lv held constant (measured), sqrt eps={SQRT_EPS:g}, "
          f"state floored at S_FLOOR={core.S_FLOOR:g}")
    print(f"parameters   : theta = LOWER + sigmoid(z)*(UPPER-LOWER), z in R^5 (no clipping needed)")
    print(f"training     : Adam lr={args.lr:g}, {args.steps} steps, {args.starts} starts "
          f"(1 reference + {args.starts - 1} Latin hypercube, seed {args.seed}), batched")
    print(f"bounds       : v0[{core.LOWER[0]:g},{core.UPPER[0]:g}] T[{core.LOWER[1]:g},{core.UPPER[1]:g}] "
          f"s0[{core.LOWER[2]:g},{core.UPPER[2]:g}] a[{core.LOWER[3]:g},{core.UPPER[3]:g}] "
          f"b[{core.LOWER[4]:g},{core.UPPER[4]:g}]")
    print(f"reference    : starts = {core.format_theta(core.DEFAULT_THETA)}")
    print()

    per_run: dict[str, dict[str, object]] = {}
    started = time.perf_counter()
    for run, events in grouped.items():
        print(f"{run}: differentiable IDM + Adam, {args.starts} starts")
        result = calibrate_run(run, events, args.starts, args.steps, args.lr, args.seed)
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
            "algorithm": "differentiable IDM (torch Euler rollout) + Adam, multi-start",
            "framework": f"torch {torch.__version__} CPU float64",
            "learning_rate": args.lr,
            "steps": args.steps,
            "n_starts": args.starts,
            "start_sampling": "reference theta0 + Latin hypercube over the parameter box",
            "seed": args.seed,
            "sqrt_eps": SQRT_EPS,
            "consistency_tolerance": CONSISTENCY_TOLERANCE,
            "L_VEH": core.L_VEH,
            "delta": core.DELTA,
            "parameter_bounds": {
                name: [float(core.LOWER[i]), float(core.UPPER[i])]
                for i, name in enumerate(core.PARAM_NAMES)
            },
            "parameter_penalty": core.PARAM_PENALTY,
            "infeasible_cost": core.INFEASIBLE_COST,
            "spacing_violation_penalty": core.S_VIOLATION_PENALTY,
            "integration_method": core.DEFAULT_METHOD,
            "reported_metrics_method": core.DEFAULT_METHOD,
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

    convergence_warnings = 0
    for run, result in per_run.items():
        conv = result["convergence"]           # type: ignore[index]
        diff = result["differentiable"]        # type: ignore[index]
        if float(conv["final_best"]) > float(conv["initial_best"]):
            convergence_warnings += 1
            print(f"WARNING: {run}: the M5 best curve increased "
                  f"({conv['initial_best']} -> {conv['final_best']}).")
        if diff["consistency_warnings"]:
            convergence_warnings += len(diff["consistency_warnings"])
            for line in diff["consistency_warnings"]:
                print(f"WARNING: {run}: {line}")
        if int(conv["distinct_feasible_results"]) < 2:
            print(f"NOTE: {run}: the feasible M5 starts converged to "
                  f"{conv['distinct_feasible_results']} distinct parameter vector(s).")

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
    if convergence_warnings:
        # The brief asks for a WARNING on a >10 % torch-vs-RK4 difference, not for a
        # failed run; the numbers are in the JSON and in the block above.
        print(f"NOTE: {convergence_warnings} convergence/consistency warning(s) - see above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
