# Neocortex prototype in the Habitat home, 2026-09-25

`sim/habitat_bridge/run_cortex_life.sh` (10 days x 120 s, small HSSD house, 150 deg
view, mushroom-body learning on) and `run_cortex_oracle.sh` (6 x 60 s).
`summary.txt` columns: day, episode, first_bowl_s, near_bowl_s, eating_s,
mean_bowl_dist, plant_s, pets, treats, near_person_s, bumps, wall_s.

| Condition | Run 1 (v2 dynamics, before the CUDA stream fix) | Run 2 (v3 dynamics, fixed engine) |
|---|---|---|
| fly brain alone | 2/10 days reached bowl, 1.2 s eating/day | 3/10, 2.8 s |
| cortex, place memory wiped nightly | 3/10, 2.5 s | 5/10, 5.2 s |
| cortex v0 | 8/10, 9.9 s | 6/10, 8.8 s |

Run 1 shared the GPU between processes while the CUDA engine could read stale
inputs (fixed in native/lif_cuda.cu); all conditions were affected alike.
In run 2 the cortex held the bowl as its goal on days 2 and 8, came within
0.6-0.7 m (eating radius 0.6 m) and overshot: the v3 brain walks almost
continuously (~0.45 m/s) and the cortex cannot slow the approach. Ten days
per condition cannot separate memory from no-memory.
