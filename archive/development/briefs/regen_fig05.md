# Brief: Regenerate group validation metric figure (no duration; distinguish trigger types)

## Project
`C:\Users\ptiu1\Documents\PhDStudy\INTR5100\lab1`
Python: `C:/Users/ptiu1/miniconda3/envs/intr5100/python.exe`

## Task
Replace `results/figures/group_validation/fig_05_safety_efficiency.png`.
The old figure plotted safety-layer episodes AND run duration. Run duration must be
removed because the five drives are not of consistent length.

New figure (single grouped bar chart) — for each of group_1…group_5 plot the
follower safety-layer episodes split into two stacked/categorised counts, parsed
from that group's `commands.csv` follower rows in `data/group_validation_raw/`:
1. `proximity triggers`: emergency = 1 rows where spacing <= 250 (genuine
   close-range interventions).
2. `sensor transient flags`: emergency = 1 rows where spacing > 250
   (the `center_sensor_emergency` flags at spacing 942–1974).

Use two stacked segments per group (labels: "proximity trigger", "sensor transient
flag"), y-axis "number of episodes", x-axis group labels; annotate counts on bars;
same clean style as the other group_validation figures; English labels; tight layout.

Expected: group_1 = 0 proximity + 4 sensor; group_2 = 2 proximity + 0 sensor;
groups 3/4/5 = 0 + 0 — verify by parsing.

Overwrite only fig_05 atomically; print the count table; idempotent; do not touch
any other file, no installs/git.
