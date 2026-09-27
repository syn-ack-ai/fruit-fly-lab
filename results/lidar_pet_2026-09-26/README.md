# Lidar senses, first test, 2026-09-26: INVALID (bug), kept for the record

**These results are invalid.** An independent code review found that the
simulated lidar converted the robot's heading to radians twice, so the beams
stayed fixed to the world axes instead of turning with the robot
(sim/habitat_bridge/habitat_server.py, fixed with a regression test). The
looming "99.8%" figure, the clearances and the touch sides below were all
computed in the wrong frame, and the diagnosis in point 1 was built on that
bug. The table also mixes three changes (looming input, touch input and, in
the uncommitted follow-up, the obstacle speed limit) and adding encoders shifts
the random-number stream, so seeds are not paired. A corrected run with
ablations replaces this.

    OUT=... SEEDS="1 2 3" CONDS="pet petlidar" SAFE_SPEED=1 BATTERY=0.7 sim/habitat_bridge/run_cortex_life.sh

Battery pet (run D code) with and without a simulated 2D lidar (90 beams at
20 cm, switched on by brain_client --lidar via the server's "lidar" command) feeding looming (LC4/LPLC2) and antennal/
vibrissal touch by side (robot/lidar.py). 3 seeds x 10 days.

| | pet | pet + lidar |
|---|---|---|
| pet ran into person | 45 | 51 |
| person walked into pet | 70 | 103 |
| robot-scene contact count (Habitat) | 93,788 | 81,182 (-13%) |
| near person / pets | 449 s / 23 | 584 s / 27 |
| days reached dock / complete meals / rest | 19 / 12 / 83 s | 19 / 7 / 51 s |

(The "pet" column reproduces battery run D exactly: same code and seeds.)

Why it did not help:
1. The lidar looming signal was on in 99.8% of control steps: the scan is fixed
   to the robot, so its own turning and walking make obstacles "approach" in
   some beams. LC4/LPLC2 were driven almost continuously (likely the cause of
   fewer meals and naps). Fix: compensate for self-motion (the fly's efference
   copy) and loom only for things moving toward the pet.
2. Touch was felt before 32 of the 51 pet-caused bumps (median clearance 8 cm
   in the 0.8 s before), but in this connectome antennal touch slows walking
   and steers inconsistently (DNa02 toward the touched side, DNa01 away;
   experiments/antenna_touch_test.py), so it does not steer around obstacles.
   Fix: obstacle clearance in the direction of travel belongs to the robot's
   safety layer (slow / stop), like the near-person speed limit, with touch kept
   as a feeling (slowing) rather than the avoidance mechanism.
