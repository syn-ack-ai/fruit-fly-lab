# Lidar obstacle avoidance for the pet, 2026-09-26/27

How the battery pet (cortex `pet`, `sim/habitat_bridge/run_cortex_life.sh`)
keeps off the furniture with a simulated 2D lidar (90 beams at 20 cm,
`robot/lidar.py`). Both runs: Habitat small house, near-person speed limit on
(`SAFE_SPEED=1`), battery 0.7 at the start, 3 seeds x 10 days (30 pet-days per
condition), FlyWire FAFB brain (the default brain until 2026-09-27), CUDA engine.
Summaries: `sim/habitat_bridge/compare_lidar.py <output dir>`.

## 1. Spot body (Habitat's own 1.1 m robot), `lidar_v2`

| | pet-caused person bumps | scene contact frames/day | walked m/day | days reached dock | meals |
|---|---|---|---|---|---|
| pet (no lidar) | 42 | 3210 | 33 | 20/30 | 11 |
| petlidar (lidar -> looming + touch senses, and a forward-cone speed limit) | **15** | **1160** | 23 | 20/30 | 7 |
| petobst (speed limit only) | 18 | 1803 | 19 | 17/30 | 8 |
| petsense (senses only) | 62 | 2686 | 30 | 15/30 | 6 |

"Scene contact frames" is Habitat's contact count: every contact point at every
1/120 s step, a measure of time spent touching, not a number of bumps. The
speed limit does most of the work; the fly's own looming/touch senses alone made
bumps into the person worse. A footprint time-to-collision filter (`petttc`,
Nav2 Collision Monitor style) trapped the 1.1 m Spot against walls (it walked
2 m/day) -- the reason for the rover body below.

## 2. Rover body, `organic_v1`

`BODY=rover` (`sim/habitat_bridge/bodies.py`): the Waveshare UGV Rover's 253 x
231 mm footprint on its own navmesh (it fits under tables); bumps are the
rover's rectangle touching the scene (rays at 5, 15 and 25 cm; counted once per
contact), "pressed" is time the navmesh held the base against something.

| | furniture bumps/day | pressed s/day | safety brake s/day | pivot s/day | walked m/day | days reached dock | person bumps (pet-caused) |
|---|---|---|---|---|---|---|---|
| pet (no lidar) | **18.1** | 51 | - | - | 29 | 13/30 | 25 (6) |
| petlidar (senses + brake) | 4.7 | 5.2 | 55 | - | 22 | 18/30 | 28 (4) |
| **petsteer** (+ steering reflex) | **0.4** | 4.4 | 42 | 4.7 | 27 | 16/30 | 34 (6) |
| petroute (+ obstacle map and routes) | **0.3** | 5.5 | 43 | 6.1 | 26 | 16/30 | 51 (12) |

- **Organic avoidance works**: the steering reflex (`robot/avoid.py`) bends the
  fly brain's goal / pursuit steering toward open space before an obstacle, as
  animals steer rather than brake (insect centering, Srinivasan et al. 1991;
  cat obstacle planning, Drew & Marigold 2015). Furniture bumps fall 98% vs no
  lidar and >10x vs the brake alone, while the pet keeps walking (27 vs 22 m/day)
  and the safety brake has to intervene less. The pivot (turn in place when
  pinned) stands in for nerve-cord turning reflexes the brain-only FAFB lacks.
- The neocortex's obstacle map and routes (`cortex/obstacle_map.py`) add little
  in this small house and the pet bumped into its person more (12 pet-caused vs
  6) -- open question.
- Caveats: one house; the FAFB brain has a measured left/right steering bias
  (see `results/complete_brain_2026-09-27/`). Rerun on the complete brain with
  only the rover's real senses (`REAL=1`): `results/habitat_real_world_2026-09-27/`.
