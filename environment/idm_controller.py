# -*- coding: utf-8 -*-
"""IDM controller used by the TA-side validation mode."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class IDMParams:
    # Units must match the vehicle telemetry/command units used in your sandbox.
    v0: float = 70.0
    T: float = 1.2
    s0: float = 250.0
    a: float = 20.0
    b: float = 30.0
    delta: float = 4.0
    vehicle_length: float = 250.0
    min_spacing: float = 80.0
    emergency_spacing: float = 80.0
    max_decel_per_sec: float = 80.0
    max_accel_per_sec: float = 50.0

    @classmethod
    def from_json(cls, path: str | Path) -> "IDMParams":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


class IDMController:
    def __init__(self, params: IDMParams):
        self.params = params
        self.last_cmd_v = 0.0

    def acceleration(self, v_ego: float, v_front: float, spacing: float) -> float:
        p = self.params
        spacing = max(float(spacing), p.min_spacing)
        dv = float(v_ego) - float(v_front)
        sqrt_ab = math.sqrt(max(p.a * p.b, 1e-6))
        s_star = p.s0 + max(0.0, v_ego * p.T + v_ego * dv / (2.0 * sqrt_ab))
        free_term = (max(v_ego, 0.0) / max(p.v0, 1e-6)) ** p.delta
        interact_term = (s_star / spacing) ** 2
        return p.a * (1.0 - free_term - interact_term)

    def next_velocity(self, v_ego: float, v_front: float, spacing: float, dt: float, v_max: float | None = None) -> float:
        p = self.params
        a_idm = self.acceleration(v_ego, v_front, spacing)
        desired = max(0.0, float(v_ego) + a_idm * max(dt, 1e-3))
        if v_max is not None:
            desired = min(desired, float(v_max))
        # Extra command-rate limiting to avoid sudden jumps on the small car.
        if self.last_cmd_v <= 0.0:
            self.last_cmd_v = min(desired, p.v0)
        max_up = p.max_accel_per_sec * max(dt, 1e-3)
        max_down = p.max_decel_per_sec * max(dt, 1e-3)
        desired = clip(desired, self.last_cmd_v - max_down, self.last_cmd_v + max_up)
        self.last_cmd_v = desired
        return desired
