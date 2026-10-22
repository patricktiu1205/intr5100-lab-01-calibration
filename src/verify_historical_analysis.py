"""Read-only checks of the existing Lab I implementation and saved results.

Run in the shared intr5100 Conda environment. Existing analysis outputs are not
changed or refitted; evidence is written under review/.
"""
from pathlib import Path
import sys
import argparse
import json
import math
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for folder in [ROOT / "src/calibration", ROOT / "src/analysis", ROOT / "src/tgsim",
               ROOT / "environment"]:
    sys.path.insert(0, str(folder))
import idm_core as core
import metrics
import m5_differentiable as m5
import tgsim_core as tgsim
from idm_controller import IDMController, IDMParams

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "review/historical_verification_results.json")
    args = parser.parse_args()
    theta = np.array([30., 1.5, 2., 1., 2.])
    v, leader_v, s = 10., 5., 20.
    ta = IDMController(IDMParams(v0=30, T=1.5, s0=2, a=1, b=2,
                                min_spacing=0.01, emergency_spacing=0.01))
    standard = ta.acceleration(v, leader_v, s)
    existing = core.idm_acceleration(s, v, leader_v - v, theta)
    existing_tgsim = tgsim.idm_acceleration(s, v, leader_v - v, theta)
    assert standard < 0 < existing and abs(existing - existing_tgsim) < 1e-8
    event = core.EventData("synthetic_closing", np.array([0., .1]),
                           np.full(2, s), np.full(2, v), np.full(2, leader_v))
    ts, tv, *_ = m5.rollout_event(m5.build_event_tensor(event),
                                  torch.tensor(theta[None, :], dtype=torch.float64))
    torch_accel = float((tv[1, 0] - tv[0, 0]) / .1)
    assert abs(torch_accel - existing) < 1e-8

    closing_ttc = metrics.event_style_metrics([0., 1.], [10., 10.], [20., 20.],
                                             [5., 5.], 30.)["ttc_min_s"]
    opening_ttc = metrics.event_style_metrics([0., 1.], [5., 5.], [20., 20.],
                                             [10., 10.], 30.)["ttc_min_s"]
    assert closing_ttc is None and abs(opening_ttc - 4.) < 1e-10

    saved = json.loads((ROOT / "results/m5_params.json").read_text())["runs"]["run_2"]
    selected = core.as_theta(saved["theta_vector"])
    events = core.load_event_data(ROOT)["run_2"]
    original_metrics = {}
    corrected_metrics = {}
    for name, ev in events:
        ps, pv = core.simulate_event(ev, selected)
        original_metrics[name] = {
            "spacing_rmse": float(np.sqrt(np.mean((ps - ev.s_obs) ** 2))),
            "speed_rmse": float(np.sqrt(np.mean((pv - ev.v_obs) ** 2))),
            "saved_spacing_rmse": saved["events"][name]["spacing_rmse"],
        }
    original_function = core._idm_scalar
    try:
        # Reverse only the acceleration-law argument. Gap evolution stays vL - v.
        core._idm_scalar = lambda s, v, gap_rate, *p: original_function(s, v, -gap_rate, *p)
        for name, ev in events:
            ps, pv, status = core.simulate_event(ev, selected, return_status=True)
            corrected_metrics[name] = {
                "spacing_rmse": float(np.sqrt(np.mean((ps - ev.s_obs) ** 2))),
                "speed_rmse": float(np.sqrt(np.mean((pv - ev.v_obs) ** 2))),
                "status": status,
            }
    finally:
        core._idm_scalar = original_function
    assert max(abs(x["spacing_rmse"] - x["saved_spacing_rmse"])
               for x in original_metrics.values()) < 1e-3

    validation = {}
    for suffix in ["152115", "155015"]:
        folder = ROOT / f"data/validation/idm_test_20260922_{suffix}"
        meta = json.loads((folder / "metadata.json").read_text())
        commands = pd.read_csv(folder / "commands.csv")
        commands = commands[commands.role == "follower"].sort_values("t_from_start")
        flags = commands.emergency.to_numpy() > 0
        starts = flags & ~np.r_[False, flags[:-1]]
        validation[suffix] = {
            "idm_params": meta["idm_params"],
            "command_rows": len(commands),
            "logged_spacing_min": float(commands.spacing.min()),
            "emergency_rows": int(flags.sum()),
            "emergency_episodes": int(starts.sum()),
            "gap_minus_spacing_range": [float((commands.gap - commands.spacing).min()),
                                         float((commands.gap - commands.spacing).max())],
            "flagged_rows_by_reason": {
                str(k): int(v) for k, v in commands.loc[flags, "reason"].value_counts().items()
            },
            "episodes": [
                {"start_s": float(row.t_from_start), "reason": str(row.reason)}
                for row in commands.loc[starts].itertuples()
            ],
            "spacing_min_by_mode": {
                str(k): float(v) for k, v in commands.groupby("mode").spacing.min().items()
            },
        }

    groups = json.loads((ROOT / "results/group_validation_analysis.json").read_text())
    group_means = {g: b["commands"]["spacing"]["mean"]
                   for g, b in groups["B_validation_runs"].items()}
    stage1 = pd.read_csv(ROOT / "data/preprocess/stage_1/run_2.csv")
    stage4_samples = sum(ev.n for _, ev in events)
    map_data = yaml.safe_load((ROOT / "environment/6e68_map.yaml").read_text())
    evidence = {
        "scope": "Existing outputs remain unchanged; no calibration was refitted.",
        "idm_sign_check": {
            "scenario": "gap=20, follower speed=10, leader speed=5; theta=[30,1.5,2,1,2]",
            "guideline_and_ta_acceleration": float(standard),
            "existing_calibration_acceleration": float(existing),
            "existing_torch_training_acceleration": torch_accel,
            "existing_tgsim_acceleration": float(existing_tgsim),
            "opposite_sign_confirmed": True,
        },
        "ttc_sign_check": {
            "closing_scenario_correct_ttc_s": 4.0,
            "closing_scenario_current_ttc_s": closing_ttc,
            "opening_scenario_correct_ttc_s": None,
            "opening_scenario_current_ttc_s": opening_ttc,
        },
        "run_2_existing_model_reproduction": original_metrics,
        "run_2_standard_sign_same_parameters_diagnostic_only": corrected_metrics,
        "calibration_event_mean_spacing_rmse_existing": float(np.mean([x["spacing_rmse"] for x in original_metrics.values()])),
        "validation_raw_logs": validation,
        "group_mean_logged_spacing": group_means,
        "stage_1_run_2_rows": len(stage1),
        "stage_4_run_2_events": len(events),
        "stage_4_run_2_samples": stage4_samples,
        "map_top_level_keys": list(map_data),
        "map_has_explicit_top_level_unit_or_scale": any(
            any(word in str(key).lower() for word in ["unit", "scale", "resolution"])
            for key in map_data),
    }
    destination = args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in evidence.items()
                      if "run_2_" not in k and k != "validation_raw_logs"}, indent=2))
    print("Saved read-only diagnostic evidence:", destination)


if __name__ == "__main__":
    main()
