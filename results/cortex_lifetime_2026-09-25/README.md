# Neocortex prototype in the Habitat home, 2026-09-25

`sim/habitat_bridge/run_cortex_life.sh` (10 days x 120 s per lifetime, small HSSD
house, 150 deg view, mushroom-body learning on in every condition).

## Final run (`summary_final.txt`): all review fixes, calibrated v3, 2 seeds

| Condition | Days reaching bowl | Eating per day | Person bumps |
|---|---|---|---|
| fly brain alone | 8/20 | 5.6 s | 71 |
| cortex, place memory (and critic) wiped nightly | 8/20 | 5.5 s | 46 |
| cortex v0 | 17/20 | 11.8 s | 124 |

Cortex v0 vs nightly-wiped cortex: days reaching the bowl p = 0.008 (Fisher exact),
eating time p = 0.011 (Mann-Whitney); days 1-9 only (memory possible): 15/18 vs
7/18, p = 0.015. Wiped cortex vs fly alone: identical (8/20 vs 8/20, p = 1.0).
Caveats: days within a lifetime are not independent; two seeds.

## Earlier runs (superseded; kept for the record)

`summary.txt`: run 1 (v2 dynamics, before the CUDA stream fix) and run 2 (v3
before the code-review fixes: Habitat person motion read "always moving" after
day 0, the amnesic control kept its critic, compensation/consensus compounded).

| Condition | Run 1 | Run 2 |
|---|---|---|
| fly brain alone | 2/10, 1.2 s | 3/10, 2.8 s |
| cortex, memory wiped nightly | 3/10, 2.5 s | 5/10, 5.2 s |
| cortex v0 | 8/10, 9.9 s | 6/10, 8.8 s |

Oracle channel test (6 x 60 s, before the fixes): none 2/6, FC2 goal 3/6, LC10a
attend 5/6, both 6/6 reached the bowl.
