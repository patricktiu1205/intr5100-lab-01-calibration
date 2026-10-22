#!/usr/bin/env python3
"""INTR5100 Lab Session I - M3: genetic algorithm, real-coded (own implementation).

Method 3 of the IDM calibration study. The objective is the SHARED cost of
``idm_core`` - unchanged between M1, M2 and this method:

    J(theta) = mean over events of
               [ mean((s_sim - s_obs)^2)/sigma_s^2
               + mean((v_sim - v_obs)^2)/sigma_v^2
               + S_VIOLATION_PENALTY * spacing_violations ]

with, per event, sigma_s / sigma_v the standard deviation of the MEASURED spacing
and follower speed of that event, the state initialised from the first measured
sample and integrated forward with the measured leader speed
(``idm_core.simulate_event``, RK4 by default). Nothing about the model, the
bounds, the error definition or the data changes here: M3 only replaces the
search.

--- algorithm -----------------------------------------------------------------

A textbook real-coded generational GA, written from scratch (no scipy DE and no
external GA library):

  * real encoding in raw parameter units - theta = [v0, T, s0, a, b] directly,
    never a binary string (the brief asks for real coding);
  * initial population: the brief's reference point plus ``POPULATION - 1``
    Latin hypercube points over the box (``idm_core.latin_hypercube``, fixed
    seed), so the whole run is bit-for-bit reproducible;
  * selection: tournament, k = 3 drawn uniformly, the cheaper individual wins;
  * crossover: simulated binary crossover (SBX), distribution index eta = 15,
    applied per pair with probability ``SBX_PC`` = 0.9 (the brief fixes eta but
    not pc; 0.9 is the usual value and is recorded in the JSON);
  * mutation: bounded polynomial mutation, distribution index eta = 20, applied
    independently to each coordinate with probability 1/5 (the brief's value);
  * elitism: the 2 best individuals of a generation are copied unchanged into
    the next one (the brief's value);
  * every child - and every individual of the initial population - is clipped to
    the box after crossover and mutation, so the population can never leave it
    (the brief's "clip inside the bounds");
  * fitness = the shared ``total_error``, MINIMISED. The optimiser is handed
    ``idm_core.constrained_total_error`` so an infeasible trajectory arrives as
    the large finite ``INFEASIBLE_COST`` wall (1e12) rather than a true ``inf``:
    a true infinity breaks tournament ranking (``inf < inf`` is false, so a whole
    generation of infeasible individuals would rank at random) and it is the
    "worst individual" convention the brief asks for, made explicit.

The box ``bounds_penalty`` inside ``constrained_total_error`` is kept as a second
safety net: the clipping above already makes it unreachable, so a non-zero value
there would be a bug, and the script reports how often (never) it fired.

--- convergence evidence ------------------------------------------------------

Recorded per run: the best cost of every generation (one value per
generation, i.e. the whole convergence curve), the first-generation best the run
started from and the last-generation best it ended on, the number of objective
evaluations (calls, plus the number of distinct points - the GA re-evaluates
elites and duplicated children only once thanks to a memo), and the wall-clock
time.

Output: ``results/m3_params.json`` (per run: parameters, total_error,
per-event raw RMSEs and normalised costs, the GA convergence block and the
self-checks) plus the same numbers on stdout.

Usage:
    python src/calibration/m3_ga.py [--root DIR] [--population 60] [--generations 80]
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

SCRIPT = "src/calibration/m3_ga.py"
METHOD_NAME = "M3"
OUTPUT_FILENAME = "m3_params.json"

# --- GA configuration (the brief's numbers) ----------------------------------

POPULATION = 60
GENERATIONS = 80
DEFAULT_SEED = 20260921
TOURNAMENT_K = 3
SBX_ETA = 15.0
SBX_PC = 0.9          # per-pair crossover probability (not fixed by the brief)
MUT_ETA = 20.0
MUT_PM = 1.0 / 5.0    # per-coordinate mutation probability, the brief's "1/5"
N_ELITE = 2


# --- cache / evaluation bookkeeping ------------------------------------------


class CostCache:
    """Memoised objective wrapping ``core.constrained_total_error``.

    The GA re-encounters the same parameter vector constantly (the elites are
    copied every generation, and SBX without crossover reproduces its parents),
    so the memo is where most of the speed comes from. Nothing about the value is
    approximated: a hit returns the exact number the first call computed, and the
    key is the vector rounded to 12 decimals - far below the 1e-8 parameter
    resolution anything in this study cares about, and the JSON is rounded to 6
    decimals anyway.

    ``worst`` is the value an unusable vector reports: a true ``+inf`` (an
    infeasible trajectory that even the wall cannot fold in, or a NaN) is mapped
    to ``INFEASIBLE_COST`` so ranking stays total ("worst individual").
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


# --- GA operators ------------------------------------------------------------


def initial_population(population: int, seed: int) -> np.ndarray:
    """Reference point + seeded Latin hypercube over the box, all clipped in."""
    if population < N_ELITE + 2:
        raise ValueError("population is too small for elitism")
    lhs = core.latin_hypercube(population - 1, seed)
    sample = np.vstack([core.as_theta(core.DEFAULT_THETA), lhs])
    return np.clip(sample, core.LOWER, core.UPPER)


def tournament_select(population: np.ndarray, costs: np.ndarray, rng, k: int = TOURNAMENT_K) -> np.ndarray:
    """k-way tournament: the cheapest of k uniformly drawn individuals wins."""
    drawn = rng.integers(0, population.shape[0], size=k)
    winner = int(drawn[int(np.argmin(costs[drawn]))])
    return population[winner].copy()


def sbx_crossover(parent1: np.ndarray, parent2: np.ndarray, rng,
                  eta: float = SBX_ETA, pc: float = SBX_PC) -> tuple[np.ndarray, np.ndarray]:
    """Simulated binary crossover (Deb & Agrawal): two children from two parents.

    ``beta`` is drawn from the SBX distribution with spread factor ``eta``: the
    usual u <= 1/2 branch gives ``(2u)^(1/(eta+1))`` and the mirror branch
    ``(1/(2(1-u)))^(1/(eta+1))``. The children are NOT clipped here - the caller
    clips the whole population to the box (the brief's rule) - but ``u`` is kept
    strictly inside (0, 1) so neither branch can divide by zero.
    """
    child1 = parent1.copy()
    child2 = parent2.copy()
    if rng.random() > pc:
        return child1, child2
    exponent = 1.0 / (eta + 1.0)
    for j in range(core.N_PARAMS):
        u = float(min(max(rng.random(), 1e-12), 1.0 - 1e-12))
        if u <= 0.5:
            beta = (2.0 * u) ** exponent
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** exponent
        child1[j] = 0.5 * ((1.0 + beta) * parent1[j] + (1.0 - beta) * parent2[j])
        child2[j] = 0.5 * ((1.0 - beta) * parent1[j] + (1.0 + beta) * parent2[j])
    return child1, child2


def polynomial_mutation(individual: np.ndarray, rng, eta: float = MUT_ETA,
                        pm: float = MUT_PM) -> np.ndarray:
    """Bounded polynomial mutation (Deb): one Perturbation per coordinate at pm.

    The classic formulation, using the coordinate's position inside the box
    (``delta1`` below / ``delta2`` above) so the perturbation shrinks as the
    coordinate approaches a bound - this is what the "bounded" version buys over a
    plain power-law step. The result is clipped into the box by the caller; the
    clip is what the brief asks for and, since the formulation already respects
    the bounds, it normally changes nothing.
    """
    mutated = individual.copy()
    width = core.UPPER - core.LOWER
    exponent = 1.0 / (eta + 1.0)
    for j in range(core.N_PARAMS):
        if rng.random() > pm:
            continue
        u = float(min(max(rng.random(), 1e-12), 1.0 - 1e-12))
        delta1 = (mutated[j] - core.LOWER[j]) / width[j]
        delta2 = (core.UPPER[j] - mutated[j]) / width[j]
        if u <= 0.5:
            value = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta1) ** (eta + 1.0)
            delta_q = value**exponent - 1.0
        else:
            value = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta2) ** (eta + 1.0)
            delta_q = 1.0 - value**exponent
        mutated[j] = mutated[j] + delta_q * width[j]
    return np.clip(mutated, core.LOWER, core.UPPER)


# --- run run --------------------------------------------------------------


def calibrate_run(run: str, events, method: str, population: int,
                     generations: int, seed: int, verbose: bool = True) -> dict[str, object]:
    """One generational GA run for one run, on the shared trajectory cost."""
    datasets = [data for _, data in events]
    objective = CostCache(datasets, method)
    rng = np.random.default_rng(seed)

    started = time.perf_counter()
    pool = initial_population(population, seed)
    costs = np.array([objective(theta) for theta in pool], dtype=float)
    best_curve = [float(costs.min())]
    mean_curve = [float(costs.mean())]
    best_generation = 0
    infeasible_reports = 0
    outside_box_reports = 0

    for generation in range(1, generations + 1):
        order = np.argsort(costs, kind="stable")
        elite = pool[order[:N_ELITE]].copy()
        children: list[np.ndarray] = []
        while len(children) < population - N_ELITE:
            parent1 = tournament_select(pool, costs, rng)
            parent2 = tournament_select(pool, costs, rng)
            child1, child2 = sbx_crossover(parent1, parent2, rng)
            for child in (child1, child2):
                if len(children) >= population - N_ELITE:
                    break
                children.append(np.clip(polynomial_mutation(child, rng), core.LOWER, core.UPPER))
        pool = np.vstack([elite, np.asarray(children)])
        costs = np.array([objective(theta) for theta in pool], dtype=float)
        # Honest bookkeeping of two different things that must not be conflated:
        #  * an INFEASIBLE individual (a diverged trajectory, cost = the 1e12 wall)
        #    is normal - the Latin hypercube start draws plenty of them, and the GA
        #    is supposed to select away from them;
        #  * a vector OUTSIDE THE BOX would be a bug (children are clipped), so it is
        #    counted separately and must stay at zero.
        infeasible_reports += int(np.sum(costs >= core.INFEASIBLE_COST))
        outside_box_reports += int(sum(1 for theta in pool if not core.inside_bounds(theta)))
        generation_best = float(costs.min())
        best_curve.append(generation_best)
        mean_curve.append(float(costs.mean()))
        if generation_best < best_curve[best_generation]:
            best_generation = generation
        if verbose and (generation <= 3 or generation % 10 == 0 or generation == generations):
            print(f"    gen {generation:3d}/{generations}  best J={generation_best:12.6f}"
                  f"  mean J={float(costs.mean()):12.6f}  evals={objective.calls}")

    elapsed = time.perf_counter() - started
    best_theta = np.clip(pool[int(np.argmin(costs))].copy(), core.LOWER, core.UPPER)
    best_cost = core.total_error(datasets, best_theta, method)
    if not math.isfinite(best_cost):
        raise RuntimeError(f"{run}: the GA's best individual is infeasible")

    trajectory = core.summarize_events(datasets, best_theta, method)
    inside = core.inside_bounds(best_theta)
    if not inside:
        raise RuntimeError(f"{run}: GA returned a vector outside the box")

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
        "ga": {
            "encoding": "real coded, theta = [v0, T, s0, a, b] in raw units (no binary string)",
            "population": population,
            "generations": generations,
            "seed": seed,
            "initial_population": "reference theta0 + (population - 1) Latin hypercube points, fixed seed",
            "selection": f"tournament, k={TOURNAMENT_K}, cheapest of k wins",
            "crossover": f"simulated binary crossover (SBX), eta={SBX_ETA:g}, per-pair probability {SBX_PC:g}",
            "mutation": f"bounded polynomial mutation, eta={MUT_ETA:g}, per-coordinate probability {MUT_PM:g}",
            "elitism": f"{N_ELITE} best individuals copied unchanged into the next generation",
            "bounds": "every child clipped to the box after crossover and mutation",
            "fitness": "core.total_error (minimised); infeasible trajectory -> INFEASIBLE_COST wall, i.e. worst individual",
            "objective": "idm_core.constrained_total_error",
            "elapsed_seconds": float(elapsed),
            "evaluations": int(objective.calls),
            "distinct_evaluations": int(objective.unique),
            "cache_hits": int(objective.calls - objective.unique),
            "infeasible_individuals": int(infeasible_reports),
            "infeasible_individuals_note": "evaluations whose trajectory diverged (cost = the 1e12 wall); expected, the GA selects away from them",
            "outside_box_individuals": int(outside_box_reports),
            "outside_box_note": "must be 0: every child is clipped into the box",
        },
        "convergence": {
            "generations": generations,
            "best_curve": best_curve,
            "mean_curve": mean_curve,
            "initial_best": best_curve[0],
            "final_best": best_curve[-1],
            "improvement": float(best_curve[0] - best_curve[-1]),
            "best_generation": int(best_generation),
            "monotone_nonincreasing": bool(all(
                best_curve[i + 1] <= best_curve[i] + 1e-12 for i in range(len(best_curve) - 1)
            )),
            "reported_best_cost": float(best_cost),
            "elapsed_seconds": float(elapsed),
            "evaluations": int(objective.calls),
        },
    }


# --- stdout ------------------------------------------------------------------


def print_run(run: str, result: dict[str, object]) -> None:
    """Stdout block for one run (parameters, J, RMSEs, convergence)."""
    theta = core.as_theta(result["theta_vector"])
    ga = result["ga"]           # type: ignore[assignment]
    conv = result["convergence"]  # type: ignore[assignment]
    print(f"  best theta         : {core.format_theta(theta)}")
    print(f"  within bounds      : {core.inside_bounds(theta)}")
    print(f"  total_error J      : {result['total_error']:.6f}  (equal weight mean of the per-event J_e)")
    print(f"  spacing RMSE (mean/worst): {result['mean_spacing_rmse']:.4f} / {result['worst_spacing_rmse']:.4f}")
    print(f"  speed   RMSE (mean/worst): {result['mean_speed_rmse']:.4f} / {result['worst_speed_rmse']:.4f}")
    print(f"  convergence        : gen 0 best J={conv['initial_best']:.6f} -> gen {conv['generations']} "
          f"best J={conv['final_best']:.6f}  (improvement {conv['improvement']:.6f},"
          f" best at generation {conv['best_generation']})")
    print(f"  curve monotone     : {conv['monotone_nonincreasing']}   evaluations={conv['evaluations']}"
          f" (distinct {ga['distinct_evaluations']}, cache hits {ga['cache_hits']})"
          f"   runtime={conv['elapsed_seconds']:.2f} s")
    print(f"  infeasible ind.    : {ga['infeasible_individuals']} evaluations = the 1e12 wall"
          f"   outside-box ind.={ga['outside_box_individuals']} (must be 0)")
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
    parser.add_argument("--population", type=int, default=POPULATION,
                        help=f"GA population size (default {POPULATION})")
    parser.add_argument("--generations", type=int, default=GENERATIONS,
                        help=f"number of generations (default {GENERATIONS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"GA seed, per run (default {DEFAULT_SEED})")
    parser.add_argument("--method", default=core.DEFAULT_METHOD, choices=list(core.METHODS),
                        help="trajectory integration scheme")
    args = parser.parse_args(argv)

    root = core.resolve_root(args.root)
    grouped = core.load_event_data(root)

    print(f"project root : {root}")
    print(f"input dir    : {core.rel(root / core.STAGE4_SUBDIR, root)}")
    print(f"method       : {METHOD_NAME} - real-coded genetic algorithm (own implementation, no DE)")
    print(f"objective    : constrained_total_error = equal weight mean of the per-event")
    print(f"               [ mean(ds^2)/sigma_s^2 + mean(dv^2)/sigma_v^2 + penalty*violations ]")
    print(f"spacing      : s = dhw - {core.L_VEH:g}   (spacing violation = simulated s <= 0)")
    print(f"ga           : population={args.population}, generations={args.generations}, "
          f"seed={args.seed} per run, elite={N_ELITE}")
    print(f"operators    : tournament k={TOURNAMENT_K}, SBX eta={SBX_ETA:g} (pc={SBX_PC:g}), "
          f"polynomial mutation eta={MUT_ETA:g} (pm={MUT_PM:g} per coordinate), clip to box")
    print(f"bounds       : v0[{core.LOWER[0]:g},{core.UPPER[0]:g}] T[{core.LOWER[1]:g},{core.UPPER[1]:g}] "
          f"s0[{core.LOWER[2]:g},{core.UPPER[2]:g}] a[{core.LOWER[3]:g},{core.UPPER[3]:g}] "
          f"b[{core.LOWER[4]:g},{core.UPPER[4]:g}]")
    print(f"integration  : {args.method} on the real time deltas, leader speed interpolated from lv")
    print()

    per_run: dict[str, dict[str, object]] = {}
    started = time.perf_counter()
    for run, events in grouped.items():
        print(f"{run}: genetic algorithm")
        result = calibrate_run(run, events, args.method, args.population,
                                  args.generations, args.seed)
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
            "algorithm": "real-coded generational genetic algorithm",
            "population": args.population,
            "generations": args.generations,
            "seed": args.seed,
            "encoding": "real coded (theta in raw units)",
            "selection": f"tournament k={TOURNAMENT_K}",
            "crossover": f"SBX eta={SBX_ETA:g}, pc={SBX_PC:g}",
            "mutation": f"bounded polynomial mutation eta={MUT_ETA:g}, pm={MUT_PM:g}",
            "elitism": N_ELITE,
            "bounds_handling": "clip children into the box; constrained_total_error keeps the box wall as a safety net",
            "infeasible_handling": "INFEASIBLE_COST wall (worst individual)",
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
    payload["configuration"] = payload["configuration"]  # type: ignore[index]
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
        print("WARNING: a convergence curve is not non-increasing (elitism should prevent it).")

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
