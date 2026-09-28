# Milo's voice from the fly brain, 2026-09-28

Milo's words come from a language model, but *when* it vocalizes and *what its
inner state is* now come from the fly brain.

The design is the one in ROADMAP.md §2, "the fly brain as the voice, the LLM
as the language centre":

1. The neocortex's **excitement** drives the male **P1** courtship-arousal
   neurons. It is bumped by events (you reappearing, petting, treats, praise),
   fades within seconds, and fades to zero within 0.3 m of you: P1 also drives
   courtship pursuit, so Milo sings at you from a small distance rather than
   chasing into your legs.
2. The connectome's own **song command, pIP10**, fires.
3. Milo plays its "song", and the personality is told "you are singing".
4. The personality's words are spoken only after a **vocal urge**: the song
   command, a startle, or being spoken to, petted or given a treat.
5. The personality is told the fly brain's live state and describes it; it may
   not invent it.

## 1. What makes the male fly brain sing (`song_test.txt`, `experiments/song_test.py`)

All conditions run on the resting receptor input the robot always has, with
calibrated dynamics (3 seeds x 1 s). Values in Hz:

| input | pIP10 (song command) | dPR1 | vPR9 | TN1a | MN9 (proboscis) | DNg100 (walk) |
|---|---|---|---|---|---|---|
| rest | 0 | 5 | 2.5 | 0.4 | 0.2 | 3.8 |
| P1 12 Hz | 21 | 46 | 18 | 7 | 1.5 | 1.7 |
| P1 24 Hz | **41** | 66 | 25 | 12 | 4 | 1.0 |
| P1 30 Hz | 55 | 75 | 28 | 14 | 6 | 0.3 |
| P1 40 Hz | 66 | 81 | 33 | 17 | **20** | 0.3 |
| visual target alone (LC10a) | 1.8 | 8 | 4 | 1.4 | 0.2 | 2.8 |
| Or47b pheromone ORNs | 0.3 | 4 | 2 | 0.5 | 0.2 | 2.7 |
| P1 24 Hz, no resting input | 44 | 61 | 25 | 12 | 6 | 0.0 |

- **P1 is the switch** (von Philipsborn et al. 2011): it drives pIP10 and the
  nerve cord's song pattern generator (dPR1, vPR9, TN1a).
- **P1 = 86 cells** in the complete male brain: pC1 types also named pMP4 /
  pMP-e.
- **Cap at 24 Hz.** At 40 Hz P1 also drives MN9 past the body's proboscis
  threshold. That is the courtship "lick" step (orient, tap, sing, lick),
  which would freeze the robot. The excitement channel is therefore capped at
  24 Hz (`cortex/topdown.P1_MAX_HZ`). Walking slows while P1 is on, so Milo
  pauses to sing.
- **With the calibrated dynamics the robot runs**, the result is about the
  same without the resting input (P1 24 Hz -> pIP10 44 Hz). Other dynamics
  (e.g. the published, uncalibrated model) differ.
- **Female brain.** pIP10 and P1 do not exist in the female FAFB brain; there
  the voice is off.

## 2. Song bouts (`sim/habitat_bridge/brain_client.py`)

The song channel is pIP10, averaged over every 1 ms block of the 100 ms
control step (`Session.STEP_MEAN_KEYS`; one 50 ms window of one cell per side
moves in whole-spike steps). A bout starts at 0.30 activation and ends below
0.15. It counts only after lasting 0.3 s, and only 5 s after the previous
bout. Without these rules, pIP10 noise produced 5.6 single-step "songs" a day
with the voice off.

Each bout is voiced on the face page as the "song" sound: the pulse song
slowed into chirps, then a short hum.

## 3. The personality (`cortex/personality.py`, `cortex/llm_bench.py`)

**Fly-brain readout.** Each call gets a "Fly brain:" line, for example
"escape neurons quiet; courtship-song command pIP10 70 Hz (singing); steering
neurons pull left; walking drive weak".

**Vocal urge.** Words are said only if the call was made within 6 s of a vocal
urge; the urge is judged when the call is made, not when the reply lands.
Sounds and moods are never gated. Gated words are logged as `gated_say`.

**Benchmark** (Gemma 4 E4B via PAIR on the Mac, 3 repeats):

- 86-88 of 96 checks pass (89-92%). Before these changes it was 82 of 87
  (94%) on the older case set.
- The new grounding checks pass 9/9:
  - escape neurons firing -> startled;
  - singing -> an eager or happy mood, still a robot;
  - calm neurons -> it does not claim fear when asked.
- "Are you a cat?" -> "I am a robot." (11 of 12).
- The remaining failures are the food and bedtime cases that vary between runs.

## 4. In Habitat

Held-out seeds 41-43, 10 days, lidar and lidar steering; the full results are
in `results/unstuck_2026-09-28/`, whose "final" arm includes the voice.

- **Song bouts:** ~3 a day (2.7 / 3.0), against 0 with the voice off.
- **Turning toward a person who appears:** 78% / 79%, against 70% / 66%.
- **Aliveness:** 90% / 90%, against 88% / 87%.
- **Total:** 79.0% against 75.3% for the 2026-09-27 robot, with the touch and
  unstick fixes included.

An earlier version held excitement up while greeting. Because P1 also drives
courtship pursuit, Milo chased into the person's legs (all 12 pet-caused
person bumps with lidar steering in that arm). The greeting term was removed,
and excitement now fades to zero within 0.3 m. In the final arm, pet-caused
person bumps total 16, as in the "before" arm.

Speech (the urge gate) was not exercised in these runs, which have no
personality. It is tested in `tests/test_voice.py` and the benchmark.
