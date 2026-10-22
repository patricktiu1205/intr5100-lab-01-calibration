#!/usr/bin/env python3
"""INTR5100 Lab Session I - IDM calibration core: model, simulation and cost.

Shared module for the calibration methods of the lab:

    M1  src/calibration/m1_instant_ls.py    instantaneous-acceleration least squares
    M2  src/calibration/m2_neldermead.py    Nelder-Mead trajectory fit (multi-start)
    M3  (planned) genetic algorithm       - reuses everything below
    M4  (planned) particle swarm          - reuses everything below
    M5  (planned) differentiable IDM      - reuses everything below

Everything the later methods need is exposed here: the parameter bounds, the
IDM acceleration law, the trajectory simulation, the two error levels, the
stage 4 loader, the parameter transforms and the optimiser-facing cost.

--- data ---------------------------------------------------------------------

The input is the stage 4 calibration table ``data/preprocess/stage_4/<run>_
event_<k>.csv`` with columns ``time,fv,lv,dhw``:

    time  0.1 s grid timestamp (the real dt is read from this column - stage 4
          guarantees it is strictly increasing and the raw grid may contain
          holes, so the step is NOT assumed to be constant)
    fv    follower speed   (coordinate units / s)
    lv    leader speed     (coordinate units / s)
    dhw   distance to the leader along the follower heading, car length NOT
          subtracted (coordinate units)

The spacing the IDM works on is the bumper-to-bumper distance

    s = dhw - L_VEH            L_VEH = 250 coordinate units

``L_VEH`` is a fixed constant of the identification problem, not a parameter:
it is given by the lab brief.

--- model (delta = 4 fixed) --------------------------------------------------

    a_idm(s, v, dv) = a * ( 1 - (v/v0)^delta - (s*/s)^2 )

    s* = s0 + max(0, v*T + v*dv / (2*sqrt(a*b)))          dv = v_leader - v

    ds/dt = v_leader - v
    dv/dt = a_idm(s, v, dv)

    theta = [v0, T, s0, a, b]

with the bounds

    v0 in [40, 200]   T in [0.3, 5.0]   s0 in [10, 400]
    a  in [10, 200]   b in [10, 200]

--- integration --------------------------------------------------------------

``simulate_event`` mirrors the measured ``lv`` series exactly (the leader is an
input, not a state): starting from the event's first measured spacing and
follower speed, it marches forward on the real ``time`` deltas and reads the
leader speed off the linear interpolant of ``lv`` at every integration node.
The leader therefore never drifts away from the measurement and no leader model
is needed. Two schemes are available - explicit Euler (one acceleration
evaluation per step, the scheme the brief names first) and classical RK4 (four
evaluations, midpoint leader speed = the mean of the two bracketing samples).
``DEFAULT_METHOD`` is RK4; the Euler-vs-RK4 difference on the real data is small
enough that both give the same optimum (measured, see the build report).

Numerical protection, per the brief:

  * a state with ``s <= 0`` is counted (``s_violations``), the spacing is
    clamped to ``S_FLOOR`` so the integration can continue and the violation is
    paid for as ``S_VIOLATION_PENALTY`` per occurrence in the event cost. The run
    is NOT aborted: the excursion stays visible in the cost instead of being
    replaced by an abstract "infeasible";
  * a non-finite state, a spacing past ``SPACING_EXPLOSION_LIMIT`` or a speed
    past ``SPEED_EXPLOSION_LIMIT`` marks the run ``diverged`` and the event cost
    becomes ``+inf`` (such a parameter vector is unusable whatever the fit);
  * ``+inf`` is converted to the large finite ``INFEASIBLE_COST`` by
    ``constrained_total_error``, the function the optimisers are given - Nelder-
    Mead, GA and PSO all behave badly when handed a true ``inf``, while a
    1e12 wall is numerically identical in effect (no feasible cost ever gets
    close to it).

--- error levels -------------------------------------------------------------

``event_cost`` / ``trajectory_error`` (per event, self-scaling):

    J_e = mean((s_sim - s_obs)^2)/sigma_s^2 + mean((v_sim - v_obs)^2)/sigma_v^2
          + S_VIOLATION_PENALTY * s_violations

``sigma_s`` / ``sigma_v`` are the standard deviations of THAT event's measured
spacing and speed, so every event is scored in its own units and a tight-following
event (small dynamic range) is not drowned out by a high-speed one.

``total_error`` (multi-event) averages the per-event costs with EQUAL weight
(mean of the J_e, never a pool of the raw residuals - pooling would weight an
event by its number of samples). One diverged event makes the whole vector
infeasible, as it should.

``constrained_total_error`` = ``total_error`` plus the box penalty (below), with
``+inf`` mapped to ``INFEASIBLE_COST``: the optimiser-facing objective.

--- bounds --------------------------------------------------------------------

``bounds_penalty`` is exactly zero inside the box and at least ``PARAM_PENALTY``
(1e6) outside it, plus a term proportional to the relative overshoot (each
coordinate's overshoot over that coordinate's box width, so the term is
scale-free). Feasible costs are O(1e2), so a vector that leaves the box can never
win - the "large penalty outside the bounds" of the brief. The offset is what
makes it a wall rather than a slope: Nelder-Mead is not bound-aware and parks
exactly on (or a machine epsilon beyond) a bound, and a purely proportional
penalty would score that ghost as a near-perfect ~1e-10 cost. Since it cannot
exceed the box, the trajectory cost is evaluated at the PROJECTION of the vector
onto the box, so the boundary itself carries its true, finite cost.

--- results -------------------------------------------------------------------

``event_metrics`` reports the raw, un-normalised RMSEs (original units) plus the
normalised cost components of one event, and the "constant zero model"
baselines (predict the first / the mean spacing and speed and never move) so a
fit can be judged against the trivial predictor. ``build_results_payload`` and
``write_json_atomic`` assemble and land the ``results/*.json`` files of M1/M2
with one shared shape.

Unit conventions (as in stage 4): speeds in coordinate units/s, spacing in
coordinate units, time in seconds. Error values in the cost are dimensionless
(they are ratios to the event's own spread).

Usage:
    from idm_core import load_events, simulate_event, total_error, LOWER, UPPER
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# --- configuration ---------------------------------------------------------

# Fixed vehicle length of the identification problem: s = dhw - L_VEH.
L_VEH = 250.0

# IDM free-flow exponent, fixed by the brief (not calibrated).
DELTA = 4.0

# Parameter vector layout.
PARAM_NAMES = ("v0", "T", "s0", "a", "b")
PARAM_UNITS = {
    "v0": "coordinate units/s",
    "T": "s",
    "s0": "coordinate units",
    "a": "coordinate units/s^2",
    "b": "coordinate units/s^2",
}
N_PARAMS = 5

# Box of the identification problem (brief) and the initial point used by M1/M2.
LOWER = np.array([40.0, 0.3, 10.0, 10.0, 10.0])
UPPER = np.array([200.0, 5.0, 400.0, 200.0, 200.0])
DEFAULT_THETA = np.array([120.0, 1.2, 100.0, 80.0, 100.0])

# Integration schemes and the one used unless a caller says otherwise.
EULER = "euler"
RK4 = "rk4"
METHODS = (EULER, RK4)
DEFAULT_METHOD = RK4

# Cost protection constants.
S_VIOLATION_PENALTY = 1.0e6  # per integration step that produced s <= 0
PARAM_PENALTY = 1.0e6        # scale of the out-of-box penalty
INFEASIBLE_COST = 1.0e12     # finite stand-in for +inf, safe for NM/GA/PSO
# Numerical clamp used when a step drives the spacing to s <= 0. It must stay far
# from zero: the IDM interaction term (s*/s)^2 is singular at s = 0, and a floor of
# a machine epsilon makes the very next acceleration ~-1e14, i.e. an immediate
# explosion that turns every violation into "infeasible" and leaves the violation
# count of the brief dead code. At 1 coordinate unit (the smallest observed gap in
# this data is 124) the run can continue, the excursion is counted, and it is paid
# for at S_VIOLATION_PENALTY per occurrence.
S_FLOOR = 1.0
SPACING_EXPLOSION_LIMIT = 1.0e6
SPEED_EXPLOSION_LIMIT = 1.0e4
MIN_SIGMA = 1.0e-9           # guard for a degenerate (zero-spread) event

# Input location and columns.
STAGE4_SUBDIR = Path("data") / "preprocess" / "stage_4"
RESULTS_SUBDIR = Path("results")
RUN_REGISTRY_PATH = Path("data") / "run_registry.json"
INPUT_COLUMNS = ("time", "fv", "lv", "dhw")
EVENT_FILE_RE = re.compile(r"^(?P<run>.+)_event_(?P<ordinal>\d+)\.csv$")


# --- paths -----------------------------------------------------------------


def project_root() -> Path:
    """Project root = grandparent of the directory holding this file (src/calibration)."""
    return Path(__file__).resolve().parents[2]


def resolve_root(root: str | os.PathLike[str] | None = None) -> Path:
    """Absolute project root: the argument when given, else the inferred one."""
    return Path(root).resolve() if root is not None else project_root()


def rel(path: Path, root: Path) -> str:
    """Project-relative POSIX path (reports never carry absolute paths)."""
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# --- parameter helpers -----------------------------------------------------


def as_theta(theta: Sequence[float]) -> np.ndarray:
    """Parameter vector as a float array of length 5 (validated shape)."""
    vector = np.asarray(theta, dtype=float).reshape(-1)
    if vector.size != N_PARAMS:
        raise ValueError(f"theta must have {N_PARAMS} entries, got {vector.size}")
    return vector


def theta_to_dict(theta: Sequence[float]) -> dict[str, float]:
    """Named view of a parameter vector (for JSON reports and printing)."""
    vector = as_theta(theta)
    return {name: float(vector[i]) for i, name in enumerate(PARAM_NAMES)}


def theta_from_dict(mapping: dict[str, float]) -> np.ndarray:
    """Inverse of :func:`theta_to_dict`."""
    return as_theta([mapping[name] for name in PARAM_NAMES])


def format_theta(theta: Sequence[float]) -> str:
    """One-line parameter vector, one line per method output."""
    mapping = theta_to_dict(theta)
    return "  ".join(f"{name}={mapping[name]:.4f}" for name in PARAM_NAMES)


def inside_bounds(theta: Sequence[float], tol: float = 0.0) -> bool:
    """True when every parameter sits inside its box (inclusive, +tol slack)."""
    vector = as_theta(theta)
    return bool(np.all(vector >= LOWER - tol) and np.all(vector <= UPPER + tol))


def bounds_penalty(theta: Sequence[float]) -> float:
    """Large WALL penalty for parameters outside the box, exactly 0 inside it.

    Any out-of-box vector costs at least ``PARAM_PENALTY`` (1e6), plus a term
    proportional to how far outside it sits (each coordinate's overshoot divided
    by that coordinate's box width, so the term is scale-free and comparable
    across parameters). Feasible costs are O(10), so the box can never be left by
    a winner.

    The offset matters: a bare proportional penalty turns a coordinate that IEEE
    rounding leaves 1e-14 above a bound into a cost of ~1e-10, i.e. an
    artificially PERFECT objective at an infeasible point - Nelder-Mead is not
    bound-aware and parks exactly on (or a machine epsilon beyond) a bound, so it
    would report that ghost as its optimum. With the offset, every point outside
    the box is worse than every point inside it.
    """
    vector = as_theta(theta)
    width = UPPER - LOWER
    if not np.all(np.isfinite(vector)):
        # A non-finite vector cannot be clipped into usefulness: wall it off.
        return float(PARAM_PENALTY * (1.0 + N_PARAMS))
    below = np.clip(LOWER - vector, 0.0, None) / width
    above = np.clip(vector - UPPER, 0.0, None) / width
    overshoot = float((below + above).sum())
    if overshoot == 0.0:
        return 0.0
    return float(PARAM_PENALTY * (1.0 + overshoot))


def normalized_theta(theta: Sequence[float]) -> np.ndarray:
    """Map the box onto the unit cube (for optimisers that work in [0, 1]^n)."""
    return (as_theta(theta) - LOWER) / (UPPER - LOWER)


def denormalized_theta(unit: Sequence[float]) -> np.ndarray:
    """Inverse of :func:`normalized_theta` (values are NOT clipped to the box)."""
    return LOWER + np.asarray(unit, dtype=float).reshape(-1) * (UPPER - LOWER)


def latin_hypercube(n_samples: int, seed: int, lower=LOWER, upper=UPPER) -> np.ndarray:
    """Latin hypercube sample of the box: (n_samples, n_params), reproducible.

    One jittered sample per stratum and dimension, then an independent random
    permutation per dimension, i.e. a plain LHS that needs no scipy. With a fixed
    ``seed`` the sample is bit-for-bit reproducible, which the multi-start
    methods of this project rely on.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    lower_arr = np.asarray(lower, dtype=float).reshape(-1)
    upper_arr = np.asarray(upper, dtype=float).reshape(-1)
    if lower_arr.shape != upper_arr.shape:
        raise ValueError("lower and upper must have the same shape")
    dim = lower_arr.size
    rng = np.random.default_rng(seed)
    sample = np.empty((n_samples, dim), dtype=float)
    for j in range(dim):
        strata = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        sample[:, j] = strata[rng.permutation(n_samples)]
    return lower_arr + sample * (upper_arr - lower_arr)


# --- event containers ------------------------------------------------------


@dataclass(frozen=True)
class EventData:
    """One calibration event as plain arrays: t, measured s / v and leader speed.

    ``s_obs`` is already ``dhw - L_VEH``. Kept as a numpy-only container so the
    optimisers never pay pandas indexing costs inside their loops.
    """

    name: str
    t: np.ndarray
    s_obs: np.ndarray
    v_obs: np.ndarray
    vL_obs: np.ndarray

    @property
    def n(self) -> int:
        return int(self.t.size)

    @property
    def sigma_s(self) -> float:
        """Standard deviation of the measured spacing (this event's scale)."""
        return float(np.std(self.s_obs))

    @property
    def sigma_v(self) -> float:
        """Standard deviation of the measured follower speed."""
        return float(np.std(self.v_obs))

    @property
    def relative_gap(self) -> np.ndarray:
        """Measured ``dv = v_leader - v_follower`` at every sample."""
        return self.vL_obs - self.v_obs


def to_event_data(df: pd.DataFrame, name: str = "") -> EventData:
    """Build an :class:`EventData` from a stage 4 table (validated)."""
    missing = [column for column in INPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"event table is missing column(s): {missing}")
    if len(df) < 2:
        raise ValueError(f"event table has {len(df)} row(s), at least 2 are needed")
    time = df["time"].to_numpy(dtype=float)
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("time column is not strictly increasing")
    dhw = df["dhw"].to_numpy(dtype=float)
    return EventData(
        name=name,
        t=time,
        s_obs=dhw - L_VEH,
        v_obs=df["fv"].to_numpy(dtype=float),
        vL_obs=df["lv"].to_numpy(dtype=float),
    )


def as_event_data(event) -> EventData:
    """Accept either a DataFrame or an :class:`EventData` and return the latter."""
    if isinstance(event, EventData):
        return event
    if isinstance(event, tuple) and len(event) == 2:
        # (name, df) or (name, EventData) - the shape load_events returns.
        data = as_event_data(event[1])
        return data if data.name else replace(data, name=str(event[0]))
    if isinstance(event, pd.DataFrame):
        return to_event_data(event)
    raise TypeError(f"cannot interpret {type(event).__name__} as a calibration event")


# --- loader ----------------------------------------------------------------


def load_events(
    root: str | os.PathLike[str] | None = None,
    subdir: Path = STAGE4_SUBDIR,
) -> dict[str, list[tuple[str, pd.DataFrame]]]:
    """Read the stage 4 tables: ``run -> list of (event_name, DataFrame)``.

    Runs are ordered by name and events by their file ordinal, so every
    method sees the same event list. The stage 4 files are opened read-only; this
    module never writes anywhere near them.
    """
    base = resolve_root(root) / subdir
    if not base.is_dir():
        raise FileNotFoundError(f"stage 4 directory not found: {base}")
    found: list[tuple[str, int, str, pd.DataFrame]] = []
    for path in sorted(base.glob("*.csv")):
        match = EVENT_FILE_RE.match(path.name)
        if not match:
            raise ValueError(f"{path.name}: does not look like a calibration table")
        frame = pd.read_csv(path)
        missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{path.name}: missing column(s): {missing}")
        found.append(
            (match.group("run"), int(match.group("ordinal")), f"event_{int(match.group('ordinal'))}", frame)
        )
    if not found:
        raise FileNotFoundError(f"no calibration table found in {base}")
    found.sort(key=lambda item: (item[0], item[1]))
    grouped: dict[str, list[tuple[str, pd.DataFrame]]] = {}
    for run, _ordinal, name, frame in found:
        grouped.setdefault(run, []).append((name, frame))
    return grouped


def load_event_data(
    root: str | os.PathLike[str] | None = None,
    subdir: Path = STAGE4_SUBDIR,
) -> dict[str, list[tuple[str, EventData]]]:
    """Same as :func:`load_events` but with the numpy container already built.

    This is what the optimisation methods use: the conversion (and therefore the
    pandas cost) happens once per event, not once per cost evaluation.
    """
    grouped = load_events(root, subdir)
    return {
        run: [(name, to_event_data(frame, name)) for name, frame in events]
        for run, events in grouped.items()
    }


def observed_acceleration(event, one_sided_ends: bool = True) -> np.ndarray:
    """Measured follower acceleration from the ``fv`` series (central difference).

    Interior samples use the chord over the two neighbouring samples,
    ``(fv[i+1] - fv[i-1]) / (t[i+1] - t[i-1])``; the two end samples have no
    second neighbour and fall back to the one-sided difference over the single
    adjacent interval (the same convention stage 4 uses for its speeds). All dt
    come from the real ``time`` column, never from an assumed 0.1 s step.
    """
    data = as_event_data(event)
    if not one_sided_ends:
        raise ValueError("only one-sided end handling is implemented")
    time, speed = data.t, data.v_obs
    n = data.n
    acceleration = np.empty(n, dtype=float)
    acceleration[0] = (speed[1] - speed[0]) / (time[1] - time[0])
    acceleration[-1] = (speed[-1] - speed[-2]) / (time[-1] - time[-2])
    acceleration[1:-1] = (speed[2:] - speed[:-2]) / (time[2:] - time[:-2])
    return acceleration


# --- model -----------------------------------------------------------------


def idm_acceleration(s, v, dv, theta) -> float | np.ndarray:
    """IDM acceleration ``a * (1 - (v/v0)^delta - (s*/s)^2)``, delta = 4.

    Args:
        s:  bumper-to-bumper spacing (``dhw - L_VEH``), scalar or array.
        v:  follower speed, scalar or array.
        dv: relative speed ``v_leader - v``, scalar or array.
        theta: ``[v0, T, s0, a, b]``.

    Returns a Python float for scalar inputs and an array for array inputs.
    Non-physical parameters (``v0 <= 0`` or ``a <= 0`` or ``b <= 0``) return
    ``+inf`` instead of raising, so an optimiser that leaves the box simply sees
    an unusable cost rather than a crash. A non-positive spacing is floored at
    ``S_FLOOR`` for the same reason the simulator floors it - the interaction
    term is singular at ``s = 0`` - so the array path and the scalar path used by
    the integrator agree. The simulation counts and penalises each such step
    explicitly (see :func:`simulate_event`); this function never reports the
    violation itself.
    """
    vector = as_theta(theta)
    v0, T, s0, a, b = (float(x) for x in vector)
    s_arr = np.asarray(s, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    dv_arr = np.asarray(dv, dtype=float)
    if v0 <= 0.0 or a <= 0.0 or b <= 0.0:
        result = np.full(np.broadcast(s_arr, v_arr, dv_arr).shape, np.inf)
        return float(result) if result.ndim == 0 else result
    desired = s0 + np.maximum(0.0, v_arr * T + v_arr * dv_arr / (2.0 * math.sqrt(a * b)))
    free_term = 1.0 - (v_arr / v0) ** DELTA
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        interaction = (desired / np.maximum(s_arr, S_FLOOR)) ** 2
    acceleration = a * (free_term - interaction)
    return float(acceleration) if acceleration.ndim == 0 else acceleration


def leader_speed_at(data: EventData, t_eval) -> np.ndarray:
    """Measured leader speed linearly interpolated on the event's ``time`` grid.

    The leader is an input trajectory, so at any integration node the leader
    speed is read off the piecewise-linear interpolant of ``lv`` (outside the
    measured window the end value is held, matching ``np.interp``).
    """
    return np.interp(np.asarray(t_eval, dtype=float), data.t, data.vL_obs)


def _idm_scalar(s: float, v: float, dv: float, v0: float, T: float, s0: float,
                a: float, b: float) -> float:
    """Scalar IDM acceleration used inside the integration loop (plain floats).

    The spacing is floored at ``S_FLOOR`` rather than allowed to divide by zero:
    the interaction term is singular at ``s = 0``, and returning an infinity would
    poison the whole RK4 step (any intermediate stage below zero would turn the
    step into ``inf``), which would make the brief's violation counter unreachable
    for the RK4 scheme. With the floor the step always completes, the caller counts
    the excursion and clamps the state - the violation is therefore visible and
    paid for, in Euler and RK4 alike.

    ``nan`` is returned instead of raising when the arithmetic overflows: an
    intermediate RK4 stage can carry an enormous ``v`` and Python's float ``**``
    raises ``OverflowError`` where the mathematical result is simply a huge
    number. A non-finite acceleration propagates to a non-finite state, which
    :func:`simulate_event` reports as ``diverged`` (i.e. an infeasible
    trajectory) - the honest outcome, and never a fabricated value.
    """
    if v0 <= 0.0 or a <= 0.0 or b <= 0.0:
        return math.inf
    if s <= 0.0:
        s = S_FLOOR
    headway = v * T + v * dv / (2.0 * math.sqrt(a * b))
    if headway < 0.0:
        headway = 0.0
    desired = s0 + headway
    try:
        ratio = v / v0
        return a * (1.0 - ratio ** DELTA - (desired / s) ** 2)
    except OverflowError:
        return math.nan


def simulate_event(
    event,
    theta: Sequence[float],
    method: str = DEFAULT_METHOD,
    return_status: bool = False,
):
    """Forward-simulate one event's spacing and follower speed.

    Initialisation is the first measured sample: ``s(0) = s_obs[0]``,
    ``v(0) = v_obs[0]``, so no free initial condition is introduced. The leader
    speed is the measured ``lv`` series linearly interpolated at every
    integration node (RK4 evaluates the midpoint too, where the linear
    interpolant is the mean of the two bracketing samples). The step is the real
    ``time`` delta of the table, so holes in the grid are integrated exactly as
    far as the sampling goes.

    Returns ``(s_sim, v_sim)``, two arrays as long as the input. With
    ``return_status=True`` also returns a status dict with ``diverged`` (state
    left the numerical domain), ``s_violations`` (steps that produced ``s <= 0``),
    ``finite`` and the observed extremes. A diverged or non-finite run leaves
    ``nan`` in the tail of the returned arrays on purpose - no caller may fit
    against fabricated values.
    """
    data = as_event_data(event)
    if method not in METHODS:
        raise ValueError(f"unknown integration method {method!r}, expected one of {METHODS}")
    vector = as_theta(theta)
    v0, T, s0, a, b = (float(x) for x in vector)

    time = data.t
    leader = data.vL_obs
    n = data.n
    spacing = np.empty(n, dtype=float)
    speed = np.empty(n, dtype=float)

    s = float(data.s_obs[0])
    v = float(data.v_obs[0])
    spacing[0] = s
    speed[0] = v

    violations = 0
    diverged = False

    for i in range(n - 1):
        dt = float(time[i + 1] - time[i])
        if dt <= 0.0:
            raise ValueError(f"{data.name}: non-positive dt at index {i} ({dt})")
        leader_now = float(leader[i])
        leader_next = float(leader[i + 1])
        if method == EULER:
            # Explicit Euler on the real step, leader speed at the step start.
            acceleration = _idm_scalar(s, v, leader_now - v, v0, T, s0, a, b)
            s_new = s + dt * (leader_now - v)
            v_new = v + dt * acceleration
        else:
            # Classical RK4; the leader interpolant is linear, so its midpoint
            # value is exactly the mean of the two bracketing samples.
            leader_mid = 0.5 * (leader_now + leader_next)
            k1_s = leader_now - v
            k1_v = _idm_scalar(s, v, leader_now - v, v0, T, s0, a, b)
            s2 = s + 0.5 * dt * k1_s
            v2 = v + 0.5 * dt * k1_v
            k2_s = leader_mid - v2
            k2_v = _idm_scalar(s2, v2, leader_mid - v2, v0, T, s0, a, b)
            s3 = s + 0.5 * dt * k2_s
            v3 = v + 0.5 * dt * k2_v
            k3_s = leader_mid - v3
            k3_v = _idm_scalar(s3, v3, leader_mid - v3, v0, T, s0, a, b)
            s4 = s + dt * k3_s
            v4 = v + dt * k3_v
            k4_s = leader_next - v4
            k4_v = _idm_scalar(s4, v4, leader_next - v4, v0, T, s0, a, b)
            s_new = s + dt / 6.0 * (k1_s + 2.0 * k2_s + 2.0 * k3_s + k4_s)
            v_new = v + dt / 6.0 * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        if not (math.isfinite(s_new) and math.isfinite(v_new)):
            diverged = True
            spacing[i + 1:] = np.nan
            speed[i + 1:] = np.nan
            break
        if s_new <= 0.0:
            # Not fatal: counted, paid for, and clamped so the run can continue.
            violations += 1
            s_new = S_FLOOR
        if abs(v_new) > SPEED_EXPLOSION_LIMIT or s_new > SPACING_EXPLOSION_LIMIT:
            diverged = True
            spacing[i + 1:] = np.nan
            speed[i + 1:] = np.nan
            break
        s, v = s_new, v_new
        spacing[i + 1] = s
        speed[i + 1] = v

    if not return_status:
        return spacing, speed
    finite = bool(np.all(np.isfinite(spacing)) and np.all(np.isfinite(speed)))
    status = {
        "diverged": bool(diverged),
        "s_violations": int(violations),
        "finite": finite,
        "min_spacing": float(np.nanmin(spacing)) if finite else float("nan"),
        "max_abs_speed": float(np.nanmax(np.abs(speed))) if finite else float("nan"),
        "method": method,
    }
    return spacing, speed, status


# --- cost functions --------------------------------------------------------


def event_cost(event, theta: Sequence[float], method: str = DEFAULT_METHOD) -> float:
    """Normalised trajectory cost J_e of one event (``+inf`` when unusable).

    ``mean((s_sim - s_obs)^2)/sigma_s^2 + mean((v_sim - v_obs)^2)/sigma_v^2``
    with the per-event spreads as scales (see the module docstring), plus the
    spacing-violation penalty. Both terms are means, so the cost does not scale
    with the event's length.
    """
    data = as_event_data(event)
    spacing, speed, status = simulate_event(data, theta, method, return_status=True)
    if status["diverged"] or not status["finite"]:
        return float("inf")
    sigma_s = max(data.sigma_s, MIN_SIGMA)
    sigma_v = max(data.sigma_v, MIN_SIGMA)
    spacing_error = (spacing - data.s_obs) / sigma_s
    speed_error = (speed - data.v_obs) / sigma_v
    if not (np.all(np.isfinite(spacing_error)) and np.all(np.isfinite(speed_error))):
        return float("inf")
    cost = float(np.mean(spacing_error**2) + np.mean(speed_error**2))
    return cost + S_VIOLATION_PENALTY * status["s_violations"]


# The brief names this error ``trajectory_error``; keep that name as the public one.
trajectory_error = event_cost


def total_error(events: Iterable, theta: Sequence[float], method: str = DEFAULT_METHOD) -> float:
    """Multi-event cost: the EQUAL-WEIGHT mean of the per-event costs.

    ``events`` is any iterable of events (DataFrames, ``EventData`` or
    ``(name, df)`` pairs). Averaging the per-event costs - rather than pooling
    the residuals - is what gives every event the same say: a 211-sample event
    cannot outvote a 27-sample one, and each event is scored in its own units.
    ``+inf`` if any event is infeasible; the mean of an empty sequence is an
    error (silently scoring zero events as 0.0 would make any parameter optimal).
    """
    costs = [event_cost(event, theta, method) for event in events]
    if not costs:
        raise ValueError("no events given to total_error")
    if any(not math.isfinite(cost) for cost in costs):
        return float("inf")
    return float(np.mean(costs))


def constrained_total_error(events: Iterable, theta: Sequence[float],
                            method: str = DEFAULT_METHOD) -> float:
    """Optimiser-facing objective: the box as a wall plus the trajectory cost.

    This is the function M2 (and later M3/M4/M5) minimise, and it is designed so
    that a bound-aware-free optimiser such as Nelder-Mead cannot be fooled:

      * the trajectory cost is always evaluated at the PROJECTION of the vector
        onto the box (``np.clip``), so the objective is finite and meaningful on
        the boundary itself and a parameter that drifts a machine epsilon outside
        is scored as the boundary point it is - not as a spike;
      * a vector outside the box additionally pays the ``bounds_penalty`` wall
        (>= 1e6), so it can never beat an interior point (the brief's "large
        penalty outside the bounds");
      * an infeasible trajectory returns the large finite ``INFEASIBLE_COST``
        wall instead of a true ``inf`` - Nelder-Mead breaks down on an infinity,
        while no feasible cost ever approaches 1e12, so the optimum is unaffected.

    The true ``inf`` of the shared cost is what the reported evidence uses.
    """
    vector = as_theta(theta)
    clipped = np.clip(vector, LOWER, UPPER)
    cost = total_error(events, clipped, method)
    if not math.isfinite(cost):
        cost = INFEASIBLE_COST
    penalty = bounds_penalty(vector)
    return cost if penalty == 0.0 else penalty + cost


# --- metrics ---------------------------------------------------------------


def _rmse(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Root-mean-square error in the original units of the series."""
    difference = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    if not np.all(np.isfinite(difference)):
        return float("inf")
    return float(np.sqrt(np.mean(difference**2)))


def zero_model_metrics(event) -> dict[str, float]:
    """RMSE of the constant "zero model" baselines of one event.

    The trivial predictors a fit has to beat: predict a constant spacing and a
    constant follower speed and hold them for the whole event. Both the first
    sample and the event mean are offered, and ``best_*`` is the smaller (i.e.
    the harder baseline).
    """
    data = as_event_data(event)
    n = data.n
    first_s = np.full(n, data.s_obs[0])
    mean_s = np.full(n, float(np.mean(data.s_obs)))
    first_v = np.full(n, data.v_obs[0])
    mean_v = np.full(n, float(np.mean(data.v_obs)))
    spacing_first = _rmse(first_s, data.s_obs)
    spacing_mean = _rmse(mean_s, data.s_obs)
    speed_first = _rmse(first_v, data.v_obs)
    speed_mean = _rmse(mean_v, data.v_obs)
    return {
        "spacing_rmse_const_first": spacing_first,
        "spacing_rmse_const_mean": spacing_mean,
        "speed_rmse_const_first": speed_first,
        "speed_rmse_const_mean": speed_mean,
        "best_spacing_rmse": min(spacing_first, spacing_mean),
        "best_speed_rmse": min(speed_first, speed_mean),
    }


def event_metrics(event, theta: Sequence[float], method: str = DEFAULT_METHOD) -> dict[str, object]:
    """Full per-event report: raw RMSEs, normalised cost parts and status."""
    data = as_event_data(event)
    spacing, speed, status = simulate_event(data, theta, method, return_status=True)
    metrics: dict[str, object] = {
        "n_samples": data.n,
        "duration_s": float(data.t[-1] - data.t[0]),
        "method": method,
        "diverged": status["diverged"],
        "finite": status["finite"],
        "s_violations": status["s_violations"],
        "sigma_s": data.sigma_s,
        "sigma_v": data.sigma_v,
    }
    if status["diverged"] or not status["finite"]:
        metrics.update(
            spacing_rmse=float("inf"),
            speed_rmse=float("inf"),
            spacing_rmse_normalized=float("inf"),
            speed_rmse_normalized=float("inf"),
            j_spacing=float("inf"),
            j_speed=float("inf"),
            j_total=float("inf"),
        )
    else:
        sigma_s = max(data.sigma_s, MIN_SIGMA)
        sigma_v = max(data.sigma_v, MIN_SIGMA)
        spacing_difference = spacing - data.s_obs
        speed_difference = speed - data.v_obs
        j_spacing = float(np.mean(spacing_difference**2) / sigma_s**2)
        j_speed = float(np.mean(speed_difference**2) / sigma_v**2)
        metrics.update(
            spacing_rmse=_rmse(spacing, data.s_obs),
            speed_rmse=_rmse(speed, data.v_obs),
            spacing_rmse_normalized=float(np.sqrt(j_spacing)),
            speed_rmse_normalized=float(np.sqrt(j_speed)),
            j_spacing=j_spacing,
            j_speed=j_speed,
            j_total=float(j_spacing + j_speed + S_VIOLATION_PENALTY * status["s_violations"]),
        )
    metrics["zero_model"] = zero_model_metrics(data)
    return metrics


def summarize_events(events: Iterable, theta: Sequence[float],
                     method: str = DEFAULT_METHOD) -> dict[str, object]:
    """Per-event metrics of a whole run plus the equal-weight aggregates."""
    per_event: dict[str, dict[str, object]] = {}
    costs: list[float] = []
    for event in events:
        data = as_event_data(event)
        metrics = event_metrics(data, theta, method)
        per_event[data.name or f"event_{len(per_event) + 1}"] = metrics
        costs.append(float(metrics["j_total"]))
    if not per_event:
        raise ValueError("no events to summarise")
    spacing_rmse = [float(m["spacing_rmse"]) for m in per_event.values()]
    speed_rmse = [float(m["speed_rmse"]) for m in per_event.values()]
    return {
        "events": per_event,
        "n_events": len(per_event),
        "n_samples": int(sum(int(m["n_samples"]) for m in per_event.values())),
        "total_error": float(np.mean(costs)),
        "total_error_method": "equal weight mean of the per-event normalised costs",
        "mean_spacing_rmse": float(np.mean(spacing_rmse)),
        "mean_speed_rmse": float(np.mean(speed_rmse)),
        "worst_spacing_rmse": float(np.max(spacing_rmse)),
        "worst_speed_rmse": float(np.max(speed_rmse)),
    }


def self_checks(per_run: dict[str, dict[str, object]],
                focus_run: str = "run_1", focus_event: str | None = None) -> dict[str, object]:
    """Numerical sanity self-checks shared by M1/M2 (never raise, only report).

    Checks: every run produced a parameter vector, every vector sits inside
    the box, the parameters are finite, and - per the brief - the fit of the
    focus event beats the constant zero model on spacing. The last one is a
    warning flag, not an error: it is reported, printed, and left to the report.

    The focus event defaults to the FIRST event of the focus run (``None``), so
    the check follows the data: a run whose first event was dropped by the
    preprocessing stage (run_1 loses ``event_1``) is checked on the event it
    really has instead of on a name that does not exist.
    """
    checks: dict[str, object] = {}
    for run, result in per_run.items():
        theta = as_theta(result["theta_vector"])
        checks[f"{run}_has_parameters"] = bool(np.all(np.isfinite(theta)))
        checks[f"{run}_parameters_within_bounds"] = inside_bounds(theta)
    checks["runs_found"] = sorted(per_run)
    checks["runs_found_nonempty"] = bool(per_run)

    events_of_focus_run = list((per_run.get(focus_run, {}).get("events") or {}).keys())
    if focus_event is None:
        focus_event = events_of_focus_run[0] if events_of_focus_run else None
    checks["focus_run"] = focus_run
    checks["focus_event"] = focus_event
    checks["focus_beats_zero_model"] = None
    checks["focus_improvement_factor"] = None
    if focus_event is None:
        checks["focus_note"] = "the focus run has no event to check"
        return checks
    focus = per_run.get(focus_run, {}).get("events", {}).get(focus_event)
    if focus is None:
        checks[f"{focus_run}_{focus_event}_beats_zero_model"] = None
        checks[f"{focus_run}_{focus_event}_note"] = "focus event missing"
        return checks
    zero = float(focus["zero_model"]["best_spacing_rmse"])
    fitted = float(focus["spacing_rmse"])
    checks[f"{focus_run}_{focus_event}_spacing_rmse"] = fitted
    checks[f"{focus_run}_{focus_event}_zero_model_spacing_rmse"] = zero
    beats = bool(fitted < zero)
    checks[f"{focus_run}_{focus_event}_beats_zero_model"] = beats
    improvement = float(zero / fitted) if fitted > 0.0 and math.isfinite(fitted) else None
    checks[f"{focus_run}_{focus_event}_improvement_factor"] = improvement
    # Generic aliases: they carry the same verdict under a name that does not
    # change when the focus event changes.
    checks["focus_beats_zero_model"] = beats
    checks["focus_improvement_factor"] = improvement
    return checks


# --- results output --------------------------------------------------------


def results_dir(root: str | os.PathLike[str] | None = None,
                subdir: Path = RESULTS_SUBDIR) -> Path:
    """The project's ``results/`` directory, created if missing."""
    path = resolve_root(root) / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_run_registry(root: str | os.PathLike[str] | None = None) -> dict[str, object]:
    """Read ``data/run_registry.json`` (run label <-> collection directory).

    Written by ``src/preprocess_stage1.py``; read here so every results file
    carries the provenance of the run labels it reports. A missing registry is
    not an error (the results are still valid) - it is reported as
    ``available: false`` instead of being invented.
    """
    base = resolve_root(root)
    path = base / RUN_REGISTRY_PATH
    if not path.is_file():
        return {
            "available": False,
            "path": rel(path, base),
            "note": "no run registry found; run src/preprocess_stage1.py to write one",
        }
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {"available": True, "path": rel(path, base), "runs": payload.get("runs", {}),
            "collections": payload.get("collections", {})}


def round_floats(value, decimals: int = 6):
    """Recursively round floats so the JSON stays readable and reproducible.

    Two deliberate exceptions to plain rounding:

      * a non-finite value becomes ``None`` (JSON ``null``) - the report contains
        no ``Infinity`` token, which is not valid JSON, and a ``null`` cost means
        exactly "infeasible trajectory", never "missing";
      * a non-zero value that would round away to zero keeps full precision, so a
        solver tolerance such as ``fatol = 1e-10`` cannot be published as ``0.0``.
    """
    if isinstance(value, dict):
        return {key: round_floats(item, decimals) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(item, decimals) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return None
        rounded = round(number, decimals)
        return number if (rounded == 0.0 and number != 0.0) else rounded
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON report atomically (``.tmp`` + ``os.replace``), like the stages."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    text = json.dumps(round_floats(payload), indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp_path, path)


def build_results_payload(
    method: str,
    script: str,
    root: Path,
    configuration: dict[str, object],
    per_run: dict[str, dict[str, object]],
    focus_run: str = "run_1",
    focus_event: str | None = None,
) -> dict[str, object]:
    """Assemble the ``results/*.json`` payload shared by M1 and M2.

    ``per_run`` maps a run to ``{"theta": {...}, "theta_vector": [...],
    "events": {...}}`` (the shape :func:`summarize_events` returns plus the
    vector); the payload is filled in here the same way for every method so the
    later M3/M4/M5 outputs can be compared field by field.
    """
    runs: dict[str, object] = {}
    for run, result in per_run.items():
        theta = as_theta(result["theta_vector"])
        entry: dict[str, object] = {
            "theta": theta_to_dict(theta),
            "theta_vector": [float(value) for value in theta],
            "parameters_within_bounds": inside_bounds(theta),
            "total_error": result["total_error"],
            "total_error_method": result.get(
                "total_error_method", "equal weight mean of the per-event normalised costs"
            ),
            "n_events": result["n_events"],
            "n_samples": result["n_samples"],
            "mean_spacing_rmse": result["mean_spacing_rmse"],
            "mean_speed_rmse": result["mean_speed_rmse"],
            "worst_spacing_rmse": result["worst_spacing_rmse"],
            "worst_speed_rmse": result["worst_speed_rmse"],
            "events": result["events"],
        }
        for key, value in result.items():
            if key not in entry and key not in ("theta_vector", "events", "theta"):
                entry[key] = value
        runs[run] = entry
    payload: dict[str, object] = {
        "method": method,
        "script": script,
        "input_dir": rel(root / STAGE4_SUBDIR, root),
        "configuration": configuration,
        "units": {
            "spacing": "coordinate units",
            "speed": "coordinate units/s",
            "acceleration": "coordinate units/s^2",
            "spacing_rmse": "coordinate units (raw, un-normalised)",
            "speed_rmse": "coordinate units/s (raw, un-normalised)",
            "total_error": "dimensionless (per-event spread normalised, equal weight mean)",
        },
        "runs": runs,
        "run_registry": load_run_registry(root),
        "self_checks": self_checks(per_run, focus_run, focus_event),
    }
    return payload
