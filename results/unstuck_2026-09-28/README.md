# Getting unstuck: rapidly adapting touch and the unstick reflex, 2026-09-28

## The problem

On the held-out runs of 2026-09-27, Milo sat within touch distance of a wall
or furniture, barely moving, for about a quarter of all simulated time.

| | stuck time |
|---|---|
| without the motor lag | 19% |
| with the motor lag | 26-28% |

Some days started pinned and never got free. None of the scores showed it: the
"pinned" metric counts only time the base is physically blocked. The new
reported metric `stuck` (`sim/habitat_bridge/score_pets.stuck_s`) counts runs
of 5 s or more in which an obstacle is within 15 cm and the body moved less
than 10 cm in 3 s.

## Why

The lidar's "touch" (an obstacle within 15 cm) drove the fly's head bristles
at a steady 80 Hz for as long as the wall was there. The brain stayed in a
touch response:

- **Backing:** MDN backward walking at ~18 Hz, alternating with walking back
  toward the goal. The motor lag averaged that into nothing.
- **Freezing:** sometimes proboscis extension, which freezes the body.
- **No pivot:** the pivot reflex never engaged, because the brain was not
  asking to go forward.

A first unstick reflex, which pointed the goal and attention at the most open
way, fired ~9 times a day and freed nothing. Over each 2.5 s pull the heading
changed ~1.5 deg: the touch response dominated.

## The fixes

1. **Rapidly adapting touch** (`robot/lidar.LidarTouch`, `FLY_TOUCH_ADAPT`).
   Bristle mechanoreceptors signal a deflection and fall mostly silent under
   steady contact (NompC; Walker, Willingham & Zuker 2000). The drive is now
   the contact level minus its adapted level (time constant 0.5 s) plus a 15%
   tonic part. A new contact or a harder press still bursts.
2. **The unstick reflex** (`robot/avoid.Unstick`, with lidar steering,
   `FLY_UNSTICK`).
   - **Trigger:** within 15 cm of something and less than 10 cm of movement
     in 3 s.
   - **Pull:** the goal and attention point at the most open corridor for
     2.5 s. The fly brain turns; the reflex biases, it never drives.
   - **Target:** the direction is fixed in the world when it triggers. A
     first version kept it relative to the body and overshot by up to 130 deg
     (review).
   - **Pivot:** the pivot reflex is told the same side.
   - **Not while standing still on purpose:** it does not trigger while
     docked or mid-meal, while yielding, or beside the person.

## Results

**Tuning seeds 11-13** (3 days, lidar steering; `score_tuning_touch.txt`):

| | steady touch | adapting touch | adapting + unstick |
|---|---|---|---|
| stuck s/day | 41.7 | 23.9 | 22.5 |
| furniture bumps/day | 1.56 | 0.11 | 0.00 |
| total | 71.6% | 79.6% | 81.1% |

**Held-out seeds 41-43** (10 days; `score_heldout_arms.txt`,
`stats_before_vs_final.txt`):

- "before" = the 2026-09-27 robot;
- "touch + voice" adds the adapting touch and the voice
  (`results/milo_voice_2026-09-28/`);
- "+ unstick" adds the unstick reflex;
- "final" = all of it, after two more review rounds: unstick gating, the
  excitement fade near the person, the step-mean song readout.

Pairs are lidar / lidar steering:

| | before | touch + voice | + unstick | **final** |
|---|---|---|---|---|
| stuck s/day | 43.6 / 25.1 | 16.3 / 2.4 | 16.3 / 1.7 | **15.2 / 2.3** |
| furniture bumps/day | 2.03 / 0.47 | 0.33 / 0.03 | 0.33 / 0.20 | **0.27 / 0.00** |
| pinned / pressing s/day | 9.8 / 2.7 | 0.7 / 1.6 | 0.7 / 1.1 | 1.3 / 0.6 |
| pet-caused person bumps/day | 0.50 / 0.03 | 0.20 / 0.40 | 0.20 / 0.20 | 0.33 / 0.20 |
| walked m/day | 5.9 / 7.4 | 7.4 / 8.1 | 7.4 / 8.2 | 7.4 / 8.7 |
| turns toward the person | 70% / 66% | 70% / 74% | 70% / 71% | **78% / 79%** |
| aliveness | 88% / 87% | 88% / 89% | 88% / 87% | **90% / 90%** |
| total (lidar conditions) | 75.3% | 78.3% | 80.1% | **79.0%** |

- **Unstick** only runs with lidar steering; with lidar alone the "touch +
  voice" and "+ unstick" arms are identical, as they must be.
- **Stuck time** falls by two thirds with lidar and by ~90% with lidar
  steering.
- **Furniture bumps** fall 7-fold and to zero, seeds apart (p = 0.04-0.05;
  not significant after the Bonferroni correction over 33 tests).

**Person bumps.** In the "touch + voice" arm all 12 pet-caused person bumps
with lidar steering happened in greeting mode, with excitement held at 0.3 and
the song on. P1 also drives courtship pursuit, so Milo chased into the
person's legs (gently, at the 0.08 m/s crawl the governor enforces near
people). The greeting term was removed, and excitement now fades to zero
within 0.3 m of the person.

In the final arm, pet-caused person bumps total **16, as before (16)** across
both conditions. They shifted from lidar alone (15 -> 10) to lidar steering
(1 -> 6); neither shift is significant (p = 0.16). All of them happen while
greeting or yielding, as before.
