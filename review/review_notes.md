# Lab I review

## Report revision status - October 22, 2026

Following this review, the user specified that the experimental results are fixed and calibration/testing cannot be repeated. The current [PDF report](../report/lab_01_report.pdf), [HTML report](../report/lab_01_report.html) and [LaTeX source](../report/source/main.tex) were therefore revised using existing records only. Historical fits are labeled as such; validation summaries and five-group comparisons are traceable to the logs. Unsupported safety, comfort and causal claims were removed. No new fit or validation-prediction error is presented.

The original review below describes the earlier report and retains prospective analysis recommendations for reference. Those recommendations were not performed during this report-only revision. [Revision checks](report_revision_checks.json) confirm that 132 protected data/result files were unchanged.

## Original review

Reviewed October 22, 2026 using the shared `intr5100` Conda environment. Scope: the lab guidelines, report template, current HTML/LaTeX reports, calibration and TGSIM code, saved fit results, and original validation logs. The report and scientific analysis outputs were not edited or refitted. Numerical evidence is in `historical_verification_results.json`; the main checks are reproducible with `../src/verify_historical_analysis.py`.

## Overall finding

The raw experiments remain useful, but the existing fit results do not establish calibration of the IDM printed in the report. Two confirmed sign errors require correction before relying on the model-fit and TTC conclusions. The next step is to fix and test the analysis, rerun affected fits, then update the report from traceable outputs. Original physical-run data and deployed parameter records must remain unchanged.

## 1. Critical IDM sign error

The report and guidelines define approaching speed as delta_v = v_follower - v_leader. The positive v*delta_v term in the desired gap must use that convention. Gap evolution separately remains ds/dt = v_leader - v_follower.

In `src/calibration/idm_core.py`, `EventData.relative_gap` returns leader-minus-follower speed, and both Euler and RK4 pass that same sign into the positive dynamic-gap term. `src/calibration/m5_differentiable.py` repeats it in the training graph. The real-world extension repeats it in both `src/tgsim/tgsim_core.py` and `src/tgsim/calibrate_events.py`. The supplied TA `IDMController.acceleration` uses follower-minus-leader, correctly.

Confirmed numerical example: gap 20, follower speed 10, leader speed 5, parameters [v0=30, T=1.5, s0=2, a=1, b=2]. The standard/TA IDM gives acceleration -2.01870; the current NumPy calibration, Torch training graph and TGSIM implementation give +0.97765. Thus agreement between the training and evaluation code is not an independent check of the physical equation.

This error is connected to the saved results: rerunning the existing run_2 model with the saved selected parameters reproduces the event-mean spacing RMSE 39.4544145, matching `m5_params.json` and the report's 39.45. A diagnostic with only the acceleration-law sign reversed changes the event predictions. It is not a refit and must not be published as a corrected optimum.

Required correction: separate `closing_speed = v - vL` from `gap_rate = vL - v`; update NumPy, scalar, Torch and TGSIM implementations consistently. Add acceleration-law comparisons against the independent TA implementation and closing/opening scenarios. Refit the affected calibration methods and TGSIM cases, regenerate fit comparisons, and reevaluate all fitted-parameter interpretations. Keep the original deployed theta as historical experimental configuration.

## 2. Critical TTC sign error

`src/analysis/metrics.py`, lines 552-555, treats v_leader - v_follower > 0 as closing. For a follower approaching its leader, the denominator must instead be v_follower - v_leader > 0.

At gap 20 with follower speed 10 and leader speed 5, correct TTC is 4 seconds; the current function reports no finite TTC. When the speeds are swapped, the current function reports 4 seconds even though the vehicles separate. Recompute all Lab I metrics and plots that use this function. The separate TGSIM `analyze_events.ttc_series` already uses the correct sign and should not be changed blindly.

## 3. Validation logs and report claims need reconciliation

Directly reading follower commands gives:

| Quantity | Run 152115 | Run 155015 |
| --- | --- | --- |
| Configured v0 | 123.36 | 41.12 |
| Configured emergency_spacing | 80 | 20 |
| Command rows | 737 | 749 |
| Minimum logged spacing, all modes | 52.2791 | 143.2861 |
| Minimum logged spacing, idm_straight | 52.2791 | 149.2610 |
| Emergency-flagged rows | 17 | 0 |
| Contiguous flagged episodes | 5 | 0 |
| Episode reasons | 3 spacing; 2 center-sensor | None |

The report quotes minimum spacing 60.8 and six distinct triggers for run 152115. The command log does not support those figures as whole-run command-log statistics. Its 17 flagged rows comprise 10 spacing-emergency rows and 7 center-sensor rows; spacing events also occur around 128 seconds, in addition to the two earlier windows described in the report.

The quoted 149.2 for run 155015 agrees with its IDM-only minimum, whereas the whole-run minimum is 143.2861 during bypass. Clearly state signal source, mode mask and episode definition. If 60.8 or six triggers came from another estimator/subset, identify and reproduce that estimator rather than mixing definitions.

The second run also changed `emergency_spacing` from 80 to 20, so the runs differ in more than v0. Its active minimum still exceeds 80, so this threshold change does not itself explain the larger observed gap. It must nevertheless be disclosed, and absence of native flags must not be treated as identical safety evidence under the two settings.

## 4. Required report content is missing

The supplied report template explicitly asks for five-group calibrated-parameter comparison and interpretation. The report currently says the parameter sets are not compared. Add the available parameter table, distinguish Group 4's calibrated v0=123.36 from its deployed v0=41.12, and relate observed behavior cautiously to parameter/settings differences. `results/group_validation_analysis.json` already contains useful parameter tables. Different calibration methods and datasets are limitations to discuss, not a reason to omit the requested comparison.

The template also requires numerical validation errors between model-generated and measured validation trajectories. Current validation sections mainly compare safety/style statistics and behavior across runs. Add explicitly defined prediction-error tables after resolving the sign and unit conventions; describe whether the simulation represents ideal IDM or the implemented command/vehicle-response chain. Reconstruction of calibration data is in-sample fit, not independent validation.

The lab calibration subsection should state its objective equation, normalization, initialization, optimizer budget and integration scheme. The code trains M5 with Euler and evaluates/ranks final candidates with RK4; explain and check that distinction. The real-world comparison is explicitly requested by the report template, despite project notes calling it an unrelated extension.

## 5. Units and causal claims are overstated

The approximate 2.8 position-derived/reported-speed ratio is evidence of a channel discrepancy. It does not identify all telemetry, command-response and localization units or establish the sole cause of failed following. The reversed calibration law is an additional concrete model/deployment mismatch. Replace claims that the IDM structure has been proved correct, units were the sole cause, all other deficiencies were ruled out, or the v0/3 run is generally safe.

Treat the proposed parameter conversion as conditional on its assumed speed/acceleration relationship. A consistent physical model must also account for ds/dt and distinguish telemetry conversion from actuator command gain. It has not been physically validated. The symbolic scaling argument does not determine a physical length scale from dimensionless data.

The route map `Labs/Shared Environment/6e68_map.yaml` is already supplied. Its top-level keys are all_point, cross_line, path and stop_point; there is no explicit top-level physical unit/scale. Replace the stale task to obtain the map with a request to establish its physical scale/reference using platform documentation or a known physical measurement. Do not assert that the map alone supplies metric units.

Center-sensor flags at large route gaps cannot be identified as phantom obstacles from the route gap alone. An obstacle sensor can detect something other than the tracked leader. Preserve the logged reason and mark its physical cause unverified.

## 6. Comfort and group comparisons need correct metric labels

The cross-group analysis computes jerk from differences of logged `a_idm`. This is an unconstrained model acceleration-demand signal, not measured vehicle acceleration or physical ride jerk after rate limits, actuator lag and safety overrides. Relabel it and avoid using it as an established ride-comfort ranking. Coordinate-derived acceleration/jerk needs its own timing/noise and unit assessment; large numerical derivatives are not proof of aggressive physical driving.

The sentence saying Group 4 has the largest spacing contradicts the report's own table and plot: Group 1 mean spacing is 1561.69, Group 4 is 589.74. Group 4 also does not have the largest minimum gap. Use precisely defined descriptive comparisons and separate starting gaps, bypass behavior and active control.

## 7. Numerical consistency and reporting details

- Stage 1 lists 727 records, which belongs to run_1. The adopted run_2 has 3435 Stage-1 records, 8 Stage-4 events and 1140 Stage-4 samples.
- The reconstruction table lists event-1 spacing RMSE 170.8, while the saved selected-theta RK4 output reproduces 180.6484. Other rows also differ. Identify which implementation and parameter precision produced each table; regenerate from one corrected source.
- The TGSIM error table combines pooled RMSE with event-mean MAE without labeling the different aggregation. Existing pooled spacing MAEs are 2.223369 and 7.986212, while the displayed 1.48 and 6.94 are event means. Use one aggregation or label every column explicitly when regenerating results.
- The TGSIM objective equation sums over events, while the saved cost definition is the mean over events. Include the missing event-count normalization if quoting the current numerical cost.
- TGSIM text alternates between center and front-bumper reference positions. Verify the source reference point and use the corresponding gap formula; do not silently assume these conventions are interchangeable.
- Events from one recorded pair are not independent vehicles or independent drivers. Keep the HV-HV/HV-AV comparison descriptive; event-level tests cannot establish general AV/HV population differences.

## Recommended update order

1. Correct and independently test the IDM/TTC signs; confirm spacing and unit definitions.
2. Refit Lab I and TGSIM models in separate reviewed outputs; preserve recorded physical data and deployed configurations.
3. Recompute validation prediction errors, flag episodes and clearly labeled comfort/safety diagnostics.
4. Regenerate tables/figures, add the required five-group parameter discussion, and shorten unsupported causal claims.
5. Update both the HTML report and LaTeX source consistently, compile and visually check the final report, then remove temporary build/render files.

This review does not justify changing raw experimental records or labeling a newly refitted parameter set as the one originally deployed.

## Subsequent repository organization

The later publication task grouped the current report under `report/`, moved original TGSIM inputs under `data/raw/tgsim/`, standardized folder names and made the supplied references local. Model equations and original scientific result bytes were preserved; code changes concern file loading, report builds and entry points. Historical snapshots keep their original paths. Use the organization manifest and current publication/build checks for the reorganized repository.
