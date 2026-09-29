# The body's readouts, fitted to each brain, 2026-09-29

The ROADMAP question: why does the complete male brain walk about half as far
as FAFB? The answer turned out to be mostly in the body's readouts, not the
brain. They were tuned on FAFB, whose command neurons fire several times faster
than the male brain's, and the male brain paid for that twice: in walking and
in feeding (docking to charge).

## 1. What drives walking above rest (`experiments/walking_drive_test.py`)

Each neocortex channel was driven alone on top of the resting input (6 seeds,
open loop; `walking_drive_*.txt`). DNg100 is the forward-walking command, as a
multiple of its rate at rest:

| input | male brain | FAFB |
|---|---|---|
| compass only | 0.88x | 1.06x |
| goal ahead / 60 deg / behind | 1.30 / 1.46 / 1.09x | 1.49 / 1.31 / 1.23x |
| pursuit neurons (LC10a) ahead | 1.28x (MDN 6 -> 13 Hz) | 0.93x |
| goal + LC10a (FAFB's routing) | 1.16x | 1.68x |
| P1 excitement 0.3 / 1.0 | 0.78 / 0.26x | - |
| sleep drive (ER5) | 1.01x | 0.86x |

The goal channel raises walking about as much in both brains. P1 excitement
(the voice) slows the male brain.

## 2. FAFB's resting rate was stale

The body walks at a fly's pace (12 mm/s) when DNg100 fires at the brain's
resting rate. The male brain's rate (3.1 Hz) was measured with a 1 s warm-up;
FAFB's constant (14.6 Hz, 2026-09-25) was not. Measured the same way FAFB
rests at **18.5 +- 1.2 Hz** (`cognition/calibrate_body.py`, now also
`data/metadata/body_readout_fafb.json`). So FAFB had walked 27% faster than a
fly's pace, and the "1.6x vs 1.2x rest" gap in the closed loop was mostly that.
Relative to its true rest FAFB's DNg100 runs at 1.2-1.3x in the closed loop,
like the male brain's.

Held-out with only this fix, FAFB walked 18.0 / 27.6 m/day instead of
19.5 / 29.0 (`score_heldout_fafb_rest.txt`): speed is not distance, most of
the gap remained.

## 3. Feeding: the proboscis flickered

The Habitat dock is the robot's food: it charges while its proboscis is out.
In the held-out runs the male robot sat at its dock tasting sugar with its
proboscis out only **16%** of the time (FAFB 50%), and never finished a meal
in 30 days (FAFB 6-8) (`dock_proboscis_heldout_before.txt`).

- Under sustained sugar the male brain's two MN9 cells hold 8.6 Hz, FAFB's 62
  proboscis motor neurons 15.1 Hz. The extension threshold (10.6 Hz) was set on
  FAFB.
- Both fire in ~250 ms bursts; the body read them through a 50 ms window.

**Fix** (`brain/motor/descending.proboscis_gain`, `fly/body/foraging_body.py`):
each brain's proboscis rate is scaled to FAFB's under the same sugar (male
x1.76), and the body follows the rate over the last ~second, with hysteresis
(retract below 0.65 x the threshold). Replayed spike trains
(`experiments/feeding_hold_test.py`, `feeding_hold_*.txt`):

| male brain | before | now |
|---|---|---|
| sugar | 16% out | **98%** |
| resting input | 1.5% | 0% |
| P1 excitement 0.3 / max | 5.6% / 8.7% | 0% / 6.8% |
| retraction after the sugar (rest / max excitement) | - | 1.2 s / 2.9 s |

FAFB: 55% -> 99% under sugar. Tuning seeds starting hungry (30% battery,
`score_tuning_hungry_feeding.txt`): eating 6.5 / 5.0 -> 12.2 / 14.0 s/day,
meals 0 -> 5, battery at day end 0.29 -> 0.50.

## 4. Walking: shot noise and thresholds set on FAFB

The body logs (`behaviour_fractions_heldout_before.txt`) showed the male robot:

- walking **backward 7.5%** of the time (FAFB never): MDN groups are 2 cells
  per side at ~5 Hz, and 3 spikes in one 50 ms window read as 30 Hz, over the
  backward threshold;
- rarely in command-started walking bouts (4% vs FAFB 30%): its DNg100 never
  reached FAFB's absolute threshold except in chance bursts;
- slower while walking (0.15 vs 0.25 m/s) at the same DNg100 / rest (1.3x):
  read over FAFB's 0.3 s, two cells at 3 Hz are mostly shot noise (SD 1.6x
  rest vs FAFB 0.74x), and the speed limits cut the bursts off (mean 1.30x ->
  0.81x after the wheels' 0.5 m/s; FAFB 1.32x -> 1.15x).

**Fix** (`fly/body/foraging_body.py`; FAFB unchanged apart from the first
point):

- walking direction switches follow the forward / backward commands over
  0.2 s (`WALK_CMD_TAU_S`);
- a bout starts when DNg100 scaled to FAFB's rest reaches the forward
  threshold;
- the DNg100 readouts average the same expected number of spikes on every
  brain: smoothing x 18.5 / 3.1 on the male brain (`DNG100_TAU_SCALE`).

Tuning seeds (11-13, 3 days; `score_tuning_walk_*.txt`): smoothing alone (E)
removed the backing but also the male robot's only walking triggers (resting
4% -> 14%, walked less); adding the scaled trigger (F) gave it FAFB's
behaviour profile; adding the scaled smoothing (G) raised walking with lidar
steering (21 -> 25 m/day) but not with lidar alone (17 -> 14, more braking).
The differences between these variants were within tuning noise; G was taken
on mechanistic grounds and tested on held-out seeds.

`FLY_PROBOSCIS_HOLD=0`, `FLY_WALK_CMD_TAU=0` and `FLY_DNG100_TAU_SCALE=0`
restore the old readouts for comparisons.

## 5. Held-out seeds 41-43, 10 days, both brains with today's body

Files: `score_heldout.txt`, `stats_before_vs_final.txt`,
`stats_fafb_vs_final.txt`, `behaviour_fractions_heldout_final.txt`. "Before" =
`results/touch_subtypes_2026-09-28/` (the same male brain, the old body). Pairs
are lidar / lidar steering:

| | FAFB | before | **final** |
|---|---|---|---|
| walked m/day | 17.5 / 28.7 | 10.2 / 15.8 | **12.0 / 21.7** |
| speed while walking, m/s | 0.17 / 0.27 | 0.10 / 0.15 | 0.13 / 0.22 |
| walking backward, % of time | 0 / 0 | 7.5 / 7.6 | **0 / 0** |
| meals (30 days) | 6 / 7 | 0 / 0 | **4 / 5** |
| eating s/day | 5.8 / 6.8 | 2.6 / 3.2 | 5.0 / 6.1 |
| s/day below 20% battery | 0 / 0 | 17.8 / 15.3 | **2.7 / 0** |
| battery at day end | 0.58 / 0.63 | 0.37 / 0.37 | **0.56 / 0.57** |
| safety-layer brake s/day | 54 / 31 | 24 / 12 | 44 / 26 |
| pinned/pressing s/day | 6.5 / 9.7 | 9.3 / 6.6 | 11.7 / 10.5 |
| pet-caused person bumps/day | 0.23 / 0.63 | 0.37 / 0.20 | 0.03 / 0.70 |
| stuck s/day | 35 / 0.2 | 38 / 1.0 | 45 / 1.1 |
| turns toward the person | 88% / 67% | 78% / 69% | 86% / 78% |
| heading reversals / active min | 2.4 / 2.7 | 0.7 / 1.7 | 0.8 / 1.7 |
| self-care score | 100% / 100% | 85% / 87% | **98% / 100%** |
| aliveness score | 94% / 88% | 93% / 89% | **95% / 92%** |
| total | 81.0% / 78.5% | 75.4% / 81.0% | 79.4% / 80.1% |
| **total, both** | **79.8%** | **78.2%** | **79.8%** |

**What it means.**

- **Milo now eats.** The complete brain finishes meals and keeps its battery
  up like FAFB (the self-care gap is closed).
- **It walks further, and never backward by accident.** With lidar steering
  (the rover's setup) walking rose 15.8 -> 21.7 m/day (significant after
  Bonferroni). It still walks ~3/4 as far as FAFB (significant): its speed
  while walking is lower (0.22 vs 0.27 m/s) and it rests a little more.
- **Moving more costs some safety score**, as it did for FAFB: the brake
  doubles (significant), and with steering person bumps rose to FAFB's level
  (0.70 vs 0.63/day; not significant). Aliveness is the highest of the three.
- **Overall the complete brain now ties FAFB** (79.8% vs 79.8%), up from
  78.2%.
- **Open:** with lidar alone Milo is still stuck ~45 s/day (FAFB 35). The
  rover runs lidar steering, where it is ~1 s/day.
