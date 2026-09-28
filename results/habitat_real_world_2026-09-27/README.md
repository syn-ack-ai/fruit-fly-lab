# Milo in the real-world Habitat test: the complete brain vs FlyWire FAFB, 2026-09-27

The robot is **Milo**: a small home robot (in the spirit of Johnny 5 / WALL-E)
whose lower brain is the complete wiring diagram of a real fruit fly. It is a
robot and says so -- it never presents itself as a cat or any animal
(`cortex/personality.py`, `robot/face_page/face.html`: lens eyes, robot sounds;
honesty checks in `cortex/llm_bench.py`).

## Setup

Habitat 3.0 on the Linux box, brains on the Mac Studio (split mode,
`sim/habitat_bridge/run_cortex_life.sh` with `HAB_HOST`). Rover body, "real
senses" (`REAL=1`: no smell, no petting or treats, no plant; the dock's contact
is taste), speed governor near people, battery starting at 70% and carried over
between days. 3 seeds x 10 days x 3 conditions per brain:

- `pet`: no lidar (the brain alone);
- `petlidar`: lidar safety layer;
- `petsteer`: lidar safety layer + steering around obstacles (`robot/avoid.py`).
  This is the rover's intended setup.

Brains:

- **fafb**: FlyWire FAFB (female brain), calibrated dynamics v3;
- **merged**: the complete brain, default since 2026-09-27
  (`results/complete_brain_2026-09-27`).

Both use the same body, cortex and robot layers. Runs:
`simulation/outputs/habitat/real_v1_fafb`, `real_v3_merged`.

**Bug found and fixed before the final merged run.** On male-CNS brains the
body's proboscis readout was empty: its FlyWire label group names FlyWire neuron
IDs. The pet could never eat, and had 0 meals in every earlier merged run. It
now reads MN9, the proboscis-extension motor neuron that the exam calibrates
sugar against (`brain/motor/descending.py`, test in `tests/test_merge.py`).

Why MN9 alone: the 61 male neurons of the FlyWire group's types average
11.6 Hz at 120 Hz sugar, because MN10 and MNx01-04 are silent under sugar.
MN9 alone fires at 59 Hz (FAFB group 29 Hz).

## Scores (`sim/habitat_bridge/score_pets.py`)

These brain-vs-brain runs used the robot as it was then: no motor lag, the
single-cell long-mode escape readout, no orienting reflex, and the complete
brain before the autapse fix. The later sections change those; the comparison
has not been re-run with them.

Absolute anchors per metric (100% = ideal, 0% = a set limit), fixed before
the merged run finished. The weights are judgement calls; every metric's own
score is in `score_fafb_vs_merged.txt`.

| category | weight | metrics |
|---|---|---|
| safety | 28 | furniture bumps, pet-caused person bumps, lidar brake time, pinned/pressing time |
| self-care | 21 | time below 20% battery, days with a flat battery, eating when full |
| life & company | 21 | walked m/day, time near the person |
| aliveness | 30 | speed variation, heading reversals (twitching), move/pause bouts and their irregularity, its own turn toward the person when they appear |

The scores below were recomputed after an independent code review found two
biased aliveness measures:

- **"turns toward the person"** counted the bearing shrinking. Most
  appearances were made by the robot's own twitching, and a robot spinning in
  place scored 1.0. It now counts only the robot's own rotation of 20 degrees
  or more toward them, and skips appearances its own turn caused.
- **speed** was computed from a pose rounded to 1 cm at 10 Hz. That is a
  0.1 m/s step, which inflated the slower brain's speed variation. Speed and
  walked distance now use 0.5 s of displacement, and new logs keep 0.1 mm.

A first report ("complete brain 74.0% vs FAFB 72.7%") used the biased
measures.

Total per condition (mean over seeds, [range]):

| condition | FAFB | complete brain |
|---|---|---|
| pet (no lidar) | 61.4% [57-70] | 62.1% [54-69] |
| petlidar | 74.1% [71-76] | 74.2% [73-76] |
| petsteer (rover setup) | 71.1% [69-75] | 69.6% [67-73] |
| **overall, lidar conditions** | **72.6%** | **71.9%** |
| overall, all conditions | 68.9% | 68.6% |
| without aliveness, lidar | 75.4% | 76.4% |

By category, lidar conditions:

| category | FAFB | complete brain |
|---|---|---|
| safety | 50-59% | **75-83%** |
| self-care | 100% | 100% |
| life | **76-80%** | 49% |
| aliveness | **64-68%** | 58-65% |

Overall the two brains are **tied**: the difference is well inside the seed
ranges. The complete brain is clearly safer, and FAFB is clearly more active
and slightly more alive. Our brain therefore does not yet meet the goal of
being "as good or better" on the aliveness-weighted total. Section "Motor
dynamics" below is the work on that.

## What is really different (`stats_fafb_vs_merged.txt`)

A difference is called REAL only if two things hold. First, the Mann-Whitney
test over days survives a Bonferroni correction over all 33 tests
(p < 0.0015). Second, every seed of one brain lies beyond every seed of the
other. With 3 vs 3 seeds, separation alone happens by chance 10% of the time.

Complete brain better (REAL):

- **Lidar brake time: 4-5 s/day vs 39-54 s/day.** Nearly 10x less. The
  complete brain rarely drives into a situation where the safety layer has to
  step in.
- **Pinned / pressing on things: 1.4-2.2 vs 4.7-9.4 s/day** with lidar, and
  8 vs 46 s/day without.
- **Furniture bumps without lidar: 5.1 vs 13.6 per day.** With lidar both are
  about 1/day (no difference).

FAFB better (REAL):

- **Walks about 2.5x further:** 23-31 vs 10-11 m/day. The complete brain is
  calmer, which also explains part of its safety.
- **Smoother speed:** CV 0.45-0.57 vs 0.75-0.77.
- **Twitches a little less:** 135-154 vs 170-189 heading reversals per active
  minute. Both are far from smooth.

Different, neither better:

- **Bouts:** the complete brain moves in shorter, more frequent bouts
  (7-9 vs 3-4 per minute). This is REAL, but both are inside the ideal band.

Not significant:

- pet-caused person bumps (fewer for the complete brain with lidar, 0.10-0.20
  vs 0.30-0.53 per day, but few events);
- time near the person;
- turning toward the person;
- **eating time** (4-5 vs 6-9 s/day: seeds apart, but p = 0.15-0.83).

FAFB completed 5-9 full meals per 30 days and the complete brain 0-1. Neither
battery ran low (at most 0.15 s/day below 20%, no flat days).

## Feeding: eats when hungry vs snacks

The dock's taste signal scales with hunger, as starvation raises sugar-taste
sensitivity in real flies.

- FAFB's MN9 fires at 20 Hz on an 18 Hz sugar signal.
- The complete brain's MN9 is silent below about 40 Hz of sugar input, so it
  starts a meal only when moderately hungry.

This is not a pure brain-to-brain comparison, because the bodies read
different neurons. The FAFB body reads the whole FlyWire proboscis label group
(62 motor and premotor neurons: 6 Hz at 18 Hz sugar, 29 Hz at 120 Hz). The
male body reads MN9 alone. The exam constrains MN9 only at strong sugar.

Self-care scored 100% for both brains: the days (120 s) are too short for the
battery to run low. A longer-horizon test would separate them.

## Motor dynamics: making Milo move like something alive

Heading reversals are 135-190 per active minute with either brain: the heading
flips left/right about 3 times a second. The cause is in the body, not the
brain.

- The turn rate is read from steering-neuron spike counts in each 100 ms
  window, so it jumps (+101, +38, -34, +16, +93 deg/s).
- The simulated base follows that exactly.
- A body with mass and muscles would integrate it.

`robot/motion.py` (`MotorLag`) adds that integration: a first-order lag on
speed and turn rate, optionally in stages. It sits before the safety layers,
so braking stays immediate. Its state follows what the wheels actually do, so
it restarts smoothly after a brake (anti-windup). Startle escapes are not
smoothed. Set it with `FLY_MOTOR_TAU="tau_v,tau_w[,stages[,tau_w_fast]]"`; it is
on by default since the held-out validation below (0.3 s / 1.5 s; `FLY_MOTOR_TAU=0`
turns it off).

Tuning ran on seeds 11-13 (3 days, petsteer, complete brain; the pre-review
lag without anti-windup, scored with the pre-review metrics):

| turn lag | reversals/active min | aliveness | safety | total |
|---|---|---|---|---|
| none | 165 | 66% | 63% | 69% |
| 1.0 s | 11 | 79% | 97% | 84% |
| 1.5 s | 1.7 | 91% | 65% | 76% |
| 2 x 0.35 s | 28 | 59% | 53% | 63% |
| 2 x 0.5 s | 16 | 72% | 64% | 70% |

Safety differences in this table are noisy (9 days per arm).

**Held-out validation.** Held-out seeds 21-23, 10 days, petlidar and petsteer,
code after the review fixes, complete brain (`score_motor_heldout.txt`,
`stats_motor_heldout_none_vs_1p5.txt`). Lidar conditions:

| | no lag | 1.0 s | **1.5 s** |
|---|---|---|---|
| heading reversals / active min | 184 | 13 | **2** |
| aliveness | 56% | 68% | **82%** |
| safety | 73% | 69% | 75% |
| total | 68.5% | 70.2% | **77.0%** |
| total without aliveness | 74.1% | 71.0% | **75.0%** |

With 1.5 s the twitching disappears (REAL: p = 3e-11, all seeds apart). There
is no significant change in furniture bumps, person bumps, braking, pinned
time or turning toward the person (p = 0.07-0.8). **0.3 s speed / 1.5 s turn
is now the default** (`brain_client.MOTOR_TAU_DEFAULT`; `FLY_MOTOR_TAU=0`
turns it off). Aliveness was favoured when choosing, per the user; here it
cost nothing. The hard safety layers are not part of that trade. These runs
used the long-mode escape readout from before the fix below, equally in all
three arms.

## Stray startles

In the brain-comparison runs the complete brain started a long-mode takeoff
2.7 times a day with no looming object, no touch and no person nearby.
FAFB's escapes all coincided with a lidar loom.

**Diagnosis** (the escape DNs logged at every step, 3 seeds x 2 days):

- DNp11 was active at 14 of 21 escapes, almost always while the pet headed
  for its person.
- The body read each long-mode DN per cell over its 50 ms window, so two
  spikes of one DNp11 launched a takeoff.
- In the male CNS, DNp11 receives 107 synapses from LC10a, the pursuit
  neurons the neocortex drives toward the person. FAFB has 5. This comes from
  the raw male data (41 and 65 per side), not from the merge.

**Fix** (`brain/motor/descending.py`, `POOLED_ALWAYS`): the long-mode takeoff
is read from the pooled rate of its looming DNs (DNp02, DNp04, DNp11, both
sides). It is a population behaviour (von Reyn et al. 2014; Ache et al. 2019).
The Giant Fibre, one command cell, keeps the single-cell reading.

Real looms still trigger it: LC4/LPLC2 at 20-150 Hz, both sides or one, fire
it in 57-60 of 60 windows on both brains. Two spikes of one cell now read 0.10
(threshold 0.30).

**Habitat check** (the same 3 seeds x 2 days; `tests/test_escape_readout.py`):

| | before | after |
|---|---|---|
| no lidar | 21 escapes, 15 with no cause | **0 escapes** |
| lidar steering | - | 6 at real looms (1/day) and 1 other |

The other one was a Giant Fibre takeoff with an obstacle 20 cm away, which is
plausibly real even though the lidar's loom flag did not mark it.

**Giant Fibre stray spikes at rest.** These come from DNg33: the left and right
cells excite each other through about 750 synapses each way (FAFB about 140)
and latch at 130-140 Hz with no input. The stray spikes are single spikes,
below the body's takeoff threshold, and none of the stray (no-cause) Habitat
escapes came from the Giant Fibre.

A rate-dependent adaptation of descending neurons
(`vnc.descending_adapt_mV_per_spike`, off by default) lowers DNg33
(140 -> 24 Hz). But it only reduces the GF spikes 12 -> 5 per 36 s, and it
weakens pursuit steering by up to 40%, so it was not adopted.

**Side finding, fixed since:** the merge created 255 autapses (a neuron wired to
itself; the raw male data has 26). They are now moved to the next cell of the
same type after the build (`merge.move_autapses`); see
`results/complete_brain_2026-09-27/README.md` for the rebuild and re-calibration.

## Orienting reflex

With or without the motor lag, the brain turned toward a person who came into
view only ~50% of the time.

- A faster, adaptive turn lag did not help: an offline replay of the logged
  commands and a Habitat run agreed that it let twitching back.
- The limit was the brain's attention, not the motors.

The neocortex (`cortex/v0.py`, `orient=True` for the pet, `FLY_ORIENT=0`
turns it off) now has an orienting reflex:

- When the person comes into view after >= 1 s out of sight, it points the
  fly's pursuit neurons (LC10a, the "attend" channel) at them for 2 s, and the
  connectome turns the body.
- It stays quiet while resting, eating, during a meal, giving space or
  yielding.
- Obstacle avoidance checks the corridor toward the person separately from the
  goal's (`robot/avoid.adjust`). The first version checked only the goal's
  corridor or straight ahead, and bumps rose.

**Held-out validation** (seeds 31-33, 10 days, the final brain and calibration;
`score_orient_heldout.txt`, `stats_orient_heldout.txt`):

| lidar / lidar steering | reflex off | reflex on |
|---|---|---|
| turns toward the person on appearance | 57% / 51% | **69% / 67%** |
| aliveness | 85.3% / 82.5% | 87.6% / 86.0% |
| pet-caused person bumps / day | 0.30 / 0.37 | 0.20 / 0.13 |
| furniture bumps / day | 0.13 / 0.70 | 0.50 / 0.77 |
| total | 76.5% | **78.6%** |

The rise in turning toward the person is consistent in both conditions but
not significant alone (p = 0.2-0.3; about 1 appearance a day). No safety
difference is significant. The reflex is on by default.

