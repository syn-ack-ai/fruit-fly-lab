# Cat-like manners around the person, 2026-09-26 (RTX 3080 Ti, CUDA engine)

    OUT=... SEEDS="1 2" CONDS="none v0 manners" SAFE_SPEED=1 sim/habitat_bridge/run_cortex_life.sh

2 seeds x 10 days x 120 s, Habitat small house, mushroom-body learning on, v3
dynamics. All three conditions have the robot's near-person speed limit
(robot/safety.py). "manners" = cortex v0 with cat-like manners (cortex/v0.py):
contact welcome when it wants company, loses interest in chasing and steps back
when content, never cuts in front of a walking person.

| condition | days reached bowl | eating s/day | bumps | pet ran into person | person walked into pet | pet bumped when it did not want company | pets received |
|---|---|---|---|---|---|---|---|
| fly brain alone | 6/20 | 3.4 | 73 | 36 | 36 | - | 25 |
| cortex v0 | 16/20 | 12.2 | 104 | 50 | 50 | 19 | 16 |
| cortex v0 + manners | 16/20 | 10.5 | 59 | 8 | 49 | 0 | 15 |

(summary.txt totals by the client's contact counter; the who-bumped-whom split,
sim/habitat_bridge/bump_analysis.py, can differ by one where a contact spans
two steps.) Pet-caused bumps per day, manners vs v0: Mann-Whitney p = 0.036;
manners vs fly alone p = 0.42 (fly-alone bumps cluster in a few days). Eating,
manners vs v0: p = 0.51. Bowl, manners vs fly alone: Fisher p = 0.004.

Who bumped whom: in the 0.1 s before contact, the pet counts as the one that
bumped only if it was moving toward the person (> 0.02 m/s) at least as fast as
the person was moving toward it. Most remaining contacts are the Habitat
person (who never avoids the pet) walking into it, often from behind, outside
the pet's 150-degree view.

Limits: contact in Habitat happens at 0.6-0.8 m between centres (Spot-sized
robot body), so the speed limit (crawl inside 0.5 m) still lets contacts happen
at 0.1-0.15 m/s; the fly-alone bump count is the same as without the limit
(71 in the 2026-09-25 run). Two seeds.
