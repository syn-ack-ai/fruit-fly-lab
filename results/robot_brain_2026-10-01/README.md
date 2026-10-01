# A robot's brain: the fly's eyes in Habitat, then the trimmed brain, 2026-10-01

ROADMAP.md section 0. Habitat 3.0 on the box, brains on the Mac (split mode,
`sim/habitat_bridge/run_cortex_life.sh` with `HAB_HOST`), rover body, real
senses (`REAL=1`), lidar steering (`petsteer`, the rover's setup), merged
brain, held-out seeds 41-43, 10 days of 120 s each (30 days per condition).
The baseline (`petsteer`, whole brain) reproduces 2026-09-29's held-out
score exactly (80.1%). Scores: `sim/habitat_bridge/score_pets.py`;
significance: `compare_brains_stats.py` (Bonferroni, p < 0.0045 and all
seeds apart).

## 1. The flyvis eye in Habitat: worse

`petsteereye`: the Habitat server renders the pet camera (`--eye-camera`:
0.35 m up, pitched 20 deg, 150 deg wide, as the virtual camera the person
geometry uses); it and the lidar go through flyvis into the optic lobes
(`robot/eye.HabitatEye`, simulated time). Files: `score_heldout_eye.txt`,
`stats_noeye_vs_eye.txt`.

| | without the eye | with the eye |
|---|---|---|
| total | 80.1% | 76.9% |
| aliveness | 91.8% | 87.3% |
| walked m/day | 21.7 | 16.6 (significant) |
| speed variation while moving | 0.59 | 0.69 (significant) |
| turns toward the person | 78% | 59% (seeds apart) |
| pinned s/day | 10.5 | 18.5 (seeds apart) |
| escape / flight states, % of time | ~2% | ~46% |

Milo's own movement sweeps and expands walls and furniture across the eye;
the looming circuits (LC4 / LPLC2 -> DNp02 / DNp04 / DNp11) read it as an
attack. A robot should startle only at real danger, and a wheeled robot
cannot fly: hence the plan (ROADMAP.md section 0). The eye stays an option
(`--eye`), not the robot's default.

## 2. The trimmed brain: the same, at 60% less compute

`FLY_TRIM=robot` (`brain/neurons/registry.trim_connectome`): the optic lobes
(intrinsic neurons and photoreceptors) and the nerve cord (interneurons,
sensory, motor) left out. The visual projection neurons (LC4, LPLC2, LC10a,
the lobula plate's HS / VS ...) stay: AI vision will drive them. The
ascending neurons stay (inputs for a future body model).

| | whole | trimmed |
|---|---|---|
| neurons | 165,122 | 51,268 |
| connections / synapses | 6.73 M / 89 M | 2.78 M / 40.5 M |
| fly exam (calibrated, no robustness) | 31/32 | 31/32 (`exam_full.txt`, `exam_trimmed.txt`) |
| resting walking command DNg100 | 3.1 Hz | 3.54 Hz (`data/metadata/body_readout_merged_robot.json`) |
| proboscis under sugar | 8.56 Hz | 9.43 Hz |
| Orin, brain alone, ms per 100 ms | 48.2 | 26.8 |
| Orin, whole robot stack (simulated sensors, neocortex, learning, voice, dashboard) | 57.7 (p95 67.8) | 36.2 (p95 49.1) |

Habitat held-out (`score_heldout_trim.txt`, `stats_full_vs_trimmed.txt`): no
significant difference in any metric.

| | whole | trimmed |
|---|---|---|
| total | 80.1% | 80.5% |
| aliveness | 91.8% | 94.3% |
| walked m/day | 21.7 | 21.5 |
| turns toward the person | 78% | 87% |
| pinned s/day | 10.5 | 15.5 (p = 0.036, not significant) |
| meals / dock days | 5 / 12 | 5 / 12 |

The exam's one failure in both is the documented odour lateralization.

What had to change for the trimmed brain:

- Saved whole-brain indices (the sensory modality map) are translated
  (`registry.own_indices`).
- Receptive fields of the visual projection neurons are anatomy, estimated
  through the optic lobes: a trimmed brain takes them from the whole brain,
  by root id (`retinotopy.Retinotopy._full`); identical (LC10a: 274 of 274).
- The body's readouts are fitted per brain:
  `body_readout_<dataset>[_<trim>].json` (`config.BRAIN_KEY`).

The rover (`brain_client --rover`) now runs the trimmed brain by default
(`FLY_TRIM=` keeps the whole brain; `--eye` needs it). Habitat, the exam and
the lab keep the whole brain unless `FLY_TRIM=robot` is set.
