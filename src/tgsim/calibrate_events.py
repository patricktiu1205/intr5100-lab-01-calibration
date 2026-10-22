#!/usr/bin/env python3
"""TGSIM extension: differentiable IDM calibration of the real car-following events.

This is the "Comparison with real-world IDM" step of the Lab Session I report. It
applies the SAME identification method as M5 of the lab (``src/calibration/
m5_differentiable.py``) - a differentiable IDM written as a torch computation
graph, rolled out with explicit Euler, optimised by Adam from a seeded
multi-start with a sigmoid box mapping - to the metric TGSIM trajectories, once
for the HV-HV pair and once for the HV-AV pair.

The model, the cost and the guards live in ``tgsim_core``; ONLY the units and the
search box differ from the lab, because the TGSIM data are metric SI while the
SC2I testbed is in coordinate units (car length = 250 units). The two problems
must NOT share constants, and the comparison between them is made on unit-free
parameter combinations (``tgsim_core.dimensionless_combinations``).

--- optimisation ------------------------------------------------------------

    theta = LOWER + sigmoid(z) * (UPPER - LOWER)          z in R^5

so the box is respected exactly and no clipping is needed. All starts are trained
SIMULTANEOUSLY as one (n_starts, 5) tensor with ``loss = sum`` of the per-start
costs: the objective is separable across rows, so a row's gradient and its Adam
update are identical to training that start alone (the sum, not the mean, keeps
the per-step cost scale equal to the per-start cost).

Initial condition by construction: every event starts from its own first measured
(s, v); the leader speed is the measured ``lv`` series (an input, never a state).

--- cost (pooled scales, per the brief) -------------------------------------

    J_e = mean((s_sim - s_obs)^2)/sigma_s^2 + mean((v_sim - v_obs)^2)/sigma_v^2
          + S_VIOLATION_PENALTY * violations
    J   = mean over events of J_e

with sigma_s, sigma_v the POOLED standard deviations over all samples of the
pair. Same shape as the lab's ``idm_core.total_error`` (equal-weight mean of the
per-event normalised costs); only the scale is pooled instead of per-event.

--- verification ------------------------------------------------------------

  * every start is re-simulated with numpy (``tgsim_core.simulate``, the same
    Euler scheme and the same guards) and the torch cost is compared against it:
    the maximum relative difference over all starts must stay below 1e-8 (float64
    makes it ~1e-13). A disagreement means the graph does not implement the cost
    the report quotes, and it fails the run;
  * the winner is RANKED by the numpy cost, so selection and reporting use one
    and the same yardstick;
  * self-checks: parameters inside the box, finite costs, no runaway start, the
    output file re-parses to the same numbers, and the fitted spacing RMSE beats
    the constant zero model on every event (reported, not fatal).

Output: ``results/tgsim_calibration.json`` (atomic write, deterministic: same
seed and single-thread float64 give bit-identical results).

Usage:
    python src/tgsim/calibrate_events.py [--starts 12] [--steps 400] [--lr 0.05]
                                         [--seed 20260929]
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

import tgsim_core as core  # noqa: E402  (import after the path fix, same dir)

SCRIPT = "src/tgsim/calibrate_events.py"
METHOD_NAME = "M5-TGSIM"

DEFAULT_STARTS = 12
DEFAULT_STEPS = 400
DEFAULT_LR = 0.05
DEFAULT_SEED = 20260929
CONSISTENCY_TOLERANCE = 1.0e-8
CURVE_SAMPLE_EVERY = 25


# --- torch setup -------------------------------------------------------------


def configure_torch(seed: int) -> None:
    """float64 + one thread + seed: the run is reproducible bit for bit."""
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(seed)


def decode_parameters(z: torch.Tensor) -> torch.Tensor:
    """Unconstrained z -> box theta through the sigmoid (strictly inside the box)."""
    span = torch.as_tensor(core.UPPER - core.LOWER, dtype=torch.float64)
    low = torch.as_tensor(core.LOWER, dtype=torch.float64)
    return low + torch.sigmoid(z) * span


def encode_parameters(theta) -> torch.Tensor:
    """Box theta -> unconstrained z (inverse of :func:`decode_parameters`)."""
    unit = np.clip((core.as_theta(theta) - core.LOWER) / (core.UPPER - core.LOWER),
                   1e-3, 1.0 - 1e-3)
    return torch.as_tensor(np.log(unit / (1.0 - unit)), dtype=torch.float64)


def latin_hypercube(n_starts: int, seed: int) -> np.ndarray:
    """Seeded Latin hypercube over the metric box: (n_starts, 5), reproducible."""
    rng = np.random.default_rng(seed)
    strata = (np.arange(n_starts)[:, None] + rng.random((n_starts, 5))) / n_starts
    for column in range(5):
        strata[:, column] = strata[rng.permutation(n_starts), column]
    return core.LOWER + strata * (core.UPPER - core.LOWER)


# --- differentiable IDM ------------------------------------------------------


def build_event_tensor(event: core.EventData) -> dict:
    """Pre-tensor one event: real dt, measured series, first sample, leader speed."""
    return {
        "name": event.name,
        "dt": torch.tensor(event.dt, dtype=torch.float64),
        "lv": torch.tensor(event.vL_obs[:-1], dtype=torch.float64),
        "s_obs": torch.tensor(event.s_obs, dtype=torch.float64),
        "v_obs": torch.tensor(event.v_obs, dtype=torch.float64),
        "s_first": float(event.s_obs[0]),
        "v_first": float(event.v_obs[0]),
    }


def rollout_event(tensor: dict, theta: torch.Tensor):
    """Euler rollout of one event for a whole batch of parameter vectors.

    ``theta`` is (n_starts, 5); the state series are (n_starts, n_samples). The
    recursion is pointwise identical to ``tgsim_core.simulate`` (same guards), so
    the two agree to machine precision.
    """
    n_starts = theta.shape[0]
    # 1-D slices: a (n_starts, 1) slice would broadcast into a (n, n) matrix.
    v0 = theta[:, 0]
    T = theta[:, 1]
    s0 = theta[:, 2]
    a = theta[:, 3]
    b = theta[:, 4]

    dt = tensor["dt"]
    leader = tensor["lv"]
    s = torch.full((n_starts,), tensor["s_first"], dtype=torch.float64)
    v = torch.full((n_starts,), tensor["v_first"], dtype=torch.float64)
    spacing = [s]
    speed = [v]
    violations = torch.zeros(n_starts, dtype=torch.float64)
    clamped = torch.zeros(n_starts, dtype=torch.float64)

    for i in range(dt.shape[0]):
        dv = leader[i] - v
        headway = v * T + v * dv / (2.0 * torch.sqrt(a * b + core.SQRT_EPS))
        desired = s0 + torch.clamp(headway, min=0.0)
        acceleration = a * (1.0 - (v / v0) ** core.DELTA
                            - (desired / torch.clamp(s, min=core.S_FLOOR)) ** 2)
        s_next = s + dt[i] * dv
        v_next = v + dt[i] * acceleration
        violations = violations + (s_next <= 0.0).to(torch.float64)
        clamped = clamped + ((s_next > core.SPACING_LIMIT)
                             | (v_next > core.SPEED_LIMIT)).to(torch.float64)
        s = torch.clamp(s_next, min=core.S_FLOOR, max=core.SPACING_LIMIT)
        v = torch.clamp(v_next, min=core.V_FLOOR, max=core.SPEED_LIMIT)
        spacing.append(s)
        speed.append(v)
    return torch.stack(spacing, dim=1), torch.stack(speed, dim=1), violations, clamped


def torch_cost(tensors: list[dict], z: torch.Tensor, sigma_s: float, sigma_v: float):
    """Pooled-scale cost per start of the batch: (cost, violations, clamped)."""
    theta = decode_parameters(z)
    n_starts = theta.shape[0]
    total = torch.zeros(n_starts, dtype=torch.float64)
    viol_total = torch.zeros(n_starts, dtype=torch.float64)
    clamped_total = torch.zeros(n_starts, dtype=torch.float64)
    for tensor in tensors:
        spacing, speed, violations, clamped = rollout_event(tensor, theta)
        error_s = (spacing - tensor["s_obs"].unsqueeze(0)) / sigma_s
        error_v = (speed - tensor["v_obs"].unsqueeze(0)) / sigma_v
        event_cost = ((error_s ** 2).mean(dim=1) + (error_v ** 2).mean(dim=1)
                      + core.S_VIOLATION_PENALTY * violations)
        total = total + event_cost
        viol_total = viol_total + violations
        clamped_total = clamped_total + clamped
    return total / len(tensors), viol_total, clamped_total


# --- calibration of one pair -------------------------------------------------


def calibrate_pair(pair: str, events: list[core.EventData], starts: int, steps: int,
                   lr: float, seed: int, verbose: bool = True) -> dict:
    """Batched multi-start Adam on the differentiable IDM of one pair."""
    sigma_s, sigma_v = core.pooled_scales(events)
    tensors = [build_event_tensor(event) for event in events]
    start_points = latin_hypercube(starts, seed)
    start_points[0] = 0.5 * (core.LOWER + core.UPPER)   # start 0: documented box centre

    z = torch.stack([encode_parameters(point) for point in start_points])
    z = z.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=lr)

    best_curve: list[float] = []
    step_min_curve: list[float] = []
    start_curves: list[list[float]] = [[] for _ in range(starts)]
    best_so_far = math.inf
    initial_cost = None
    started = time.perf_counter()
    for step in range(steps):
        cost, _viol, _clamp = torch_cost(tensors, z, sigma_s, sigma_v)
        if not bool(torch.isfinite(cost).all()):
            raise RuntimeError(f"{pair}: non-finite torch cost at step {step}")
        loss = cost.sum()   # separable across starts -> per-row gradients unchanged
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        values = cost.detach().numpy()
        if initial_cost is None:
            initial_cost = values.copy()
        step_min = float(values.min())
        step_min_curve.append(step_min)
        best_so_far = min(best_so_far, step_min)
        best_curve.append(best_so_far)
        if step % CURVE_SAMPLE_EVERY == 0 or step == steps - 1:
            for k, value in enumerate(values):
                start_curves[k].append(float(value))
    elapsed = time.perf_counter() - started

    with torch.no_grad():
        final_torch, viol_tensor, clamp_tensor = torch_cost(tensors, z, sigma_s, sigma_v)
        final_theta = decode_parameters(z).numpy()

    records = []
    for index in range(starts):
        theta = np.clip(final_theta[index], core.LOWER, core.UPPER)
        numpy_total = core.total_cost(events, theta, (sigma_s, sigma_v))
        torch_total = float(final_torch[index])
        relative = abs(numpy_total - torch_total) / max(abs(numpy_total), 1e-12)
        records.append({
            "index": index,
            "start_theta": core.theta_to_dict(start_points[index]),
            "theta": core.theta_to_dict(theta),
            "initial_cost_torch": float(initial_cost[index]),
            "final_cost_torch": torch_total,
            "final_cost_numpy": numpy_total,
            "torch_vs_numpy_relative_difference": relative,
            "spacing_violations": float(viol_tensor[index]),
            "clamped_steps": float(clamp_tensor[index]),
            "runaway": bool(float(clamp_tensor[index]) > 0.0),
            "feasible": bool(core.inside_bounds(theta) and float(clamp_tensor[index]) == 0.0),
        })

    feasible = [record for record in records if record["feasible"]]
    if not feasible:
        raise RuntimeError(f"{pair}: no feasible multi-start result")
    best = min(feasible, key=lambda record: float(record["final_cost_numpy"]))
    theta = core.as_theta(best["theta"])

    per_event = {event.name: core.event_metrics(event, theta, sigma_s, sigma_v)
                 for event in events}
    aggregate = core.aggregate_metrics(per_event)

    numpy_costs = np.array([float(record["final_cost_numpy"]) for record in feasible])
    differences = [float(record["torch_vs_numpy_relative_difference"]) for record in records]
    check = {
        "torch_vs_numpy_max_relative_difference": max(differences),
        "tolerance": CONSISTENCY_TOLERANCE,
        "agrees_with_numpy": bool(max(differences) < CONSISTENCY_TOLERANCE),
        "ranking_by": "numpy total cost (the same function that produces the reported metrics)",
        "spacing_violations_total_best": float(best["spacing_violations"]),
        "runaway_starts": [record["index"] for record in records if record["runaway"]],
    }
    if verbose:
        print(f"{pair}  events={len(events)}  samples={aggregate['n_samples']}  "
              f"sigma_s={sigma_s:.4f} m  sigma_v={sigma_v:.4f} m/s")
        print(f"  theta              : {core.format_theta(theta)}")
        print(f"  cost J (numpy)     : {best['final_cost_numpy']:.6f}   "
              f"(torch {best['final_cost_torch']:.6f}, rel. diff "
              f"{best['torch_vs_numpy_relative_difference']:.2e})")
        print(f"  spacing RMSE       : mean {aggregate['mean_spacing_rmse']:.4f} m   "
              f"pooled {aggregate['pooled_spacing_rmse']:.4f} m   "
              f"worst {aggregate['worst_spacing_rmse']:.4f} m")
        print(f"  speed   RMSE       : mean {aggregate['mean_speed_rmse']:.4f} m/s   "
              f"pooled {aggregate['pooled_speed_rmse']:.4f} m/s   "
              f"worst {aggregate['worst_speed_rmse']:.4f} m/s")
        for name in sorted(per_event, key=lambda key: int(key.split("_")[1])):
            metrics = per_event[name]
            print(f"    {name:<9} n={metrics['n_samples']:>4} "
                  f"sRMSE={metrics['spacing_rmse']:6.3f} m "
                  f"vRMSE={metrics['speed_rmse']:5.3f} m/s "
                  f"J={metrics['j_total']:8.3f} "
                  f"s_min {metrics['s_min_observed']:5.2f}->{metrics['s_min_simulated']:5.2f}")
        print(f"  training           : {steps} Adam steps, lr={lr:g}, {starts} starts, "
              f"{elapsed:.1f} s; torch-vs-numpy max rel. diff "
              f"{check['torch_vs_numpy_max_relative_difference']:.2e}; "
              f"runaway starts {len(check['runaway_starts'])}")
        print()

    return {
        "label": core.PAIR_LABEL[pair],
        "theta": core.theta_to_dict(theta),
        "theta_vector": [float(value) for value in theta],
        "parameters_within_bounds": core.inside_bounds(theta),
        "cost": float(best["final_cost_numpy"]),
        "pooled_scales": {"sigma_s_m": sigma_s, "sigma_v_mps": sigma_v},
        "aggregate": aggregate,
        "events": per_event,
        "convergence": {
            "steps": steps,
            "learning_rate": lr,
            "n_starts": starts,
            "best_curve": best_curve,
            "best_curve_note": "running minimum over all starts and steps (non-increasing)",
            "step_min_curve": step_min_curve,
            "start_curves": start_curves,
            "start_curves_note": f"one value every {CURVE_SAMPLE_EVERY} steps per start",
            "initial_cost_torch": float(initial_cost.min()),
            "final_cost_torch": float(final_torch[best["index"]]),
            "best_start_index": int(best["index"]),
            "feasible_starts": len(feasible),
            "infeasible_or_runaway_starts": len(records) - len(feasible),
            "final_cost_min_across_feasible": float(np.min(numpy_costs)),
            "final_cost_max_across_feasible": float(np.max(numpy_costs)),
            "final_cost_std_across_feasible": float(np.std(numpy_costs)),
            "distinct_feasible_thetas": len({tuple(np.round(core.as_theta(record["theta"]), 4))
                                             for record in feasible}),
            "per_start": records,
            "elapsed_seconds": float(elapsed),
        },
        "torch_vs_numpy": check,
    }


# --- main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--starts", type=int, default=DEFAULT_STARTS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pairs", nargs="*", default=list(core.PAIR_ORDER))
    parser.add_argument("--exclude-sparse", type=float, default=None,
                        help="robustness mode: drop events whose mean observed spacing "
                             "exceeds this many metres (the leader then does not interact "
                             "with the follower for most of the event)")
    parser.add_argument("--out", type=Path, default=core.CALIBRATION_PATH)
    parser.add_argument("--skip-unit-check", action="store_true",
                        help="skip the raw-CSV unit re-verification (faster)")
    args = parser.parse_args(argv)

    configure_torch(args.seed)
    print(f"method       : {METHOD_NAME} - differentiable IDM (torch, Euler) + Adam, multi-start")
    print(f"script       : {SCRIPT}")
    print(f"input        : {core.rel(core.EVENT_DIR)}/<pair>/event_*.csv  (metric SI, dt = 0.1 s)")
    print(f"objective    : mean over events of [ mean(ds^2)/sigma_s^2 + mean(dv^2)/sigma_v^2 "
          f"+ {core.S_VIOLATION_PENALTY:g}*violations ], pooled sigma per pair")
    print("parameters   : theta = LOWER + sigmoid(z)*(UPPER-LOWER), z in R^5")
    print("bounds       : " + "  ".join(
        f"{name}[{core.LOWER[i]:g},{core.UPPER[i]:g}]" for i, name in enumerate(core.PARAM_NAMES)))
    print(f"guards       : s floor {core.S_FLOOR:g} m, v floor {core.V_FLOOR:g} m/s, "
          f"spacing limit {core.SPACING_LIMIT:g} m, speed limit {core.SPEED_LIMIT:g} m/s")
    print(f"training     : Adam lr={args.lr:g}, {args.steps} steps, {args.starts} starts "
          f"(start 0 = box centre, 1..{args.starts - 1} = Latin hypercube, seed {args.seed}), batched")
    print(f"torch        : {torch.__version__} CPU, float64, {torch.get_num_threads()} thread")
    print()

    if args.skip_unit_check:
        unit_check = {"skipped": True}
    else:
        unit_check = core.raw_unit_check(args.pairs)
        for pair, info in unit_check.items():
            print(f"unit check {pair}: median |d position|/dt over speed_kf = "
                  f"{info['median_position_derivative_over_speed']:.6f}, "
                  f"lengths {info['length_smoothed_range_m']} m, dt = {info['dt_unique']}")
        print()

    started = time.perf_counter()
    pairs: dict = {}
    event_selection: dict = {}
    for pair in args.pairs:
        events = core.load_pair_events(pair)
        event_selection[pair] = {
            "used": [event.name for event in events],
            "excluded": [],
            "rule": "all events of the pair (primary calibration)",
        }
        if args.exclude_sparse is not None:
            kept, dropped = [], []
            for event in events:
                (kept if float(np.mean(event.s_obs)) <= args.exclude_sparse else dropped).append(event)
            if not kept:
                raise RuntimeError(f"{pair}: the sparse-event filter removed every event")
            print(f"robustness filter: dropping {[event.name for event in dropped]} "
                  f"(mean spacing > {args.exclude_sparse:g} m) from {pair}")
            events = kept
            event_selection[pair] = {
                "used": [event.name for event in kept],
                "excluded": [event.name for event in dropped],
                "rule": (f"drop events whose mean observed spacing exceeds "
                         f"{args.exclude_sparse:g} m"),
            }
        pairs[pair] = calibrate_pair(pair, events, args.starts, args.steps, args.lr, args.seed)
    total_seconds = time.perf_counter() - started

    self_checks: dict = {"pairs": list(args.pairs)}
    for pair, result in pairs.items():
        theta = core.as_theta(result["theta"])
        self_checks[f"{pair}_parameters_within_bounds"] = result["parameters_within_bounds"]
        self_checks[f"{pair}_parameters_finite"] = bool(np.all(np.isfinite(theta)))
        self_checks[f"{pair}_cost_finite"] = bool(math.isfinite(result["cost"]))
        self_checks[f"{pair}_torch_matches_numpy"] = result["torch_vs_numpy"]["agrees_with_numpy"]
        self_checks[f"{pair}_no_runaway_starts"] = not result["torch_vs_numpy"]["runaway_starts"]
        beats = {name: bool(metrics["spacing_rmse"] < metrics["zero_model_spacing_rmse"])
                 for name, metrics in result["events"].items()}
        self_checks[f"{pair}_events_beating_zero_model"] = f"{sum(beats.values())}/{len(beats)}"
        self_checks[f"{pair}_all_events_beat_zero_model"] = all(beats.values())
        if not all(beats.values()):
            self_checks[f"{pair}_events_not_beating_zero_model"] = [
                name for name, value in beats.items() if not value]

    payload = {
        "method": METHOD_NAME,
        "script": SCRIPT,
        "input_dir": core.rel(core.EVENT_DIR),
        "units": {
            "system": "metric SI",
            "spacing": "m (bumper-to-bumper, s = centre distance - leader length)",
            "speed": "m/s",
            "acceleration": "m/s^2",
            "jerk": "m/s^3",
            "time": "s",
            "cost": "dimensionless (pooled-spread normalised, equal weight mean over events)",
        },
        "unit_verification": {"reference": core.UNIT_REFERENCE, "raw_csv_check": unit_check},
        "configuration": {
            "model": "IDM, delta = 4 fixed",
            "integrator": "explicit Euler, one step per real time delta (0.1 s), leader speed = "
                          "measured lv at the step start",
            "initial_condition": "first measured sample of each event (s, v)",
            "parameterisation": "theta = LOWER + sigmoid(z) * (UPPER - LOWER), z in R^5",
            "optimiser": "torch.optim.Adam",
            "learning_rate": args.lr,
            "steps": args.steps,
            "n_starts": args.starts,
            "start_sampling": "start 0 = box centre, starts 1..n-1 = seeded Latin hypercube",
            "seed": args.seed,
            "batching": "all starts in one (n_starts, 5) tensor, loss = sum of per-start costs",
            "cost_definition": "mean over events of [mean(ds^2)/sigma_s^2 + mean(dv^2)/sigma_v^2 "
                               "+ S_VIOLATION_PENALTY*violations], pooled sigma per pair",
            "spacing_violation_penalty": core.S_VIOLATION_PENALTY,
            "s_floor_m": core.S_FLOOR,
            "v_floor_mps": core.V_FLOOR,
            "spacing_limit_m": core.SPACING_LIMIT,
            "speed_limit_mps": core.SPEED_LIMIT,
            "delta": core.DELTA,
            "framework": f"torch {torch.__version__} CPU float64, {torch.get_num_threads()} thread",
            "parameter_bounds": {name: [float(core.LOWER[i]), float(core.UPPER[i])]
                                 for i, name in enumerate(core.PARAM_NAMES)},
            "parameter_units": core.PARAM_UNITS,
            "box_rationale": {
                "v0": "5..35 m/s (18..126 km/h): the 88 km/h segment limit plus free-flow "
                      "headroom; observed follower maxima stay below 30 m/s",
                "T": "0.3..5 s: the Lab Session I range, covering the 0.8..2.5 s literature band",
                "s0": "0.2..10 m: standstill-gap range; the observed minimum spacing is 2.09 m",
                "a": "0.5..6 m/s^2: covers the observed |a| < 3.5 m/s^2 with headroom",
                "b": "0.5..8 m/s^2: the 1.5..4 m/s^2 comfortable-deceleration band with headroom",
            },
            "spacing_definition": "s = Euclidean leader/follower distance - leader length (manifest)",
        },
        "pairs": pairs,
        "event_selection": event_selection,
        "self_checks": self_checks,
        "total_runtime_seconds": float(total_seconds),
    }
    core.write_json_atomic(args.out, payload)

    # on-disk self-check: the file must exist and re-parse to the same numbers
    reread = core.read_json(args.out)
    expected = core.round_floats(payload)
    on_disk_ok = all(
        reread["pairs"][pair]["theta_vector"] == expected["pairs"][pair]["theta_vector"]
        for pair in pairs)
    print("self-checks:")
    for key in sorted(self_checks):
        if key == "pairs":
            continue
        print(f"  {key:<44} {self_checks[key]}")
    print(f"  written_on_disk_reparses                      {on_disk_ok}")
    print(f"  output                                        {core.rel(args.out)} "
          f"({args.out.stat().st_size} bytes)")
    leftovers = sorted(path.name for path in args.out.parent.glob("*.tmp"))
    print(f"  .tmp leftovers                                {leftovers or 'none'}")
    print(f"total runtime: {total_seconds:.1f} s")

    ok = (on_disk_ok
          and all(bool(value) for key, value in self_checks.items()
                  if key.endswith(("_parameters_within_bounds", "_parameters_finite",
                                   "_cost_finite", "_torch_matches_numpy"))))
    if not ok:
        print("ERROR: self-checks failed (see above).", file=sys.stderr)
        return 1
    for pair in pairs:
        if not self_checks[f"{pair}_all_events_beat_zero_model"]:
            print(f"NOTE: {pair}: some events do not beat the constant zero model "
                  f"({self_checks[f'{pair}_events_beating_zero_model']}) - expected for very "
                  f"short or nearly free-flow events.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
