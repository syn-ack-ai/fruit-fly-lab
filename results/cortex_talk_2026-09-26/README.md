# Talking pet: LLM personality on the cortex, 2026-09-26 (RTX 3080 Ti, CUDA engine)

> **Note (2026-09-27):** this run used an earlier persona: a cat-like pet named "Mote" with cat
> sounds. Since 2026-09-27 the robot is **Milo**, a robot that never presents itself as an
> animal, with robot sounds (beep, boop, chirp, trill, whirr, buzz); see `cortex/personality.py`.


    OUT=... SEEDS="1 2" CONDS="manners talk" SAFE_SPEED=1 SPEECH=1 sim/habitat_bridge/run_cortex_life.sh

2 seeds x 10 days x 120 s, Habitat small house, v3 dynamics, mushroom-body
learning, speed limit. A scripted person calls ("come here, Mote!"), praises an
answered call and says "ouch, careful!" when the pet runs into them
(sim/habitat_bridge/speech.py). "manners" = cortex v0 with manners, nothing
listens; "talk" = the same + the LLM personality (cortex/personality.py) via
NVIDIA PAIR. Per lifetime (seed 1 / seed 2):

| | manners | talk, Qwen3.8-27B (run 1) | talk, Gemma 4 E4B + fixes |
|---|---|---|---|
| days reached bowl | 8 / 8 | 5 / 0 | 9 / 10 |
| eating s/day | 10.6 / 10.3 | 8.3 / 0.0 | 14.0 / 10.5 |
| calls answered | 2/9, 6/16 | 8/12, 9/16 | 7/11, 9/16 |
| pet ran into person | 3 / 6 | 8 / 25 | 17 / 24 |
| LLM reply, median (p90) | - | 4.6 (8.2) s | 0.77 (0.82) s |

Pooled: calls answered 16/27 (Gemma) vs 8/25 (manners), Fisher p = 0.058.

Run 1 (27B) settings: INTENT_GAIN 1.5, no food memory in the prompt; the model
chose "go to the person" 79 of 160 times and begged ("feed me please"), and the
pet stopped finding food. Gemma run: INTENT_GAIN 0.5, the prompt says where
food is remembered and that the person rarely feeds on request, reasoning off,
sleepiness not sent (no naps in this condition). Intents: eat 75, seek_person
51, explore 37, follow 31. Phrases mostly "Meow?", "Mrow?", "Mrrp?".

Findings: (1) the personality layer answers calls about twice as often, and with
the fixes no longer costs food; (2) it runs into the person 4-5x more than the
manners-only pet: seeking the person more overwhelms the hand-written manners,
and verbal scolding (read as -1 in 8 of 10 scolds) did not teach it to stop
within 10 days -- the case for learned manners (roadmap RL3).

Caveats: two lifetimes per condition; days within a lifetime are not
independent; call times differ between conditions (no call while the pet is
within 2 m); the manners column is from the run of the same day with the same
manners code (cortex_life_talk), not re-run. Bumps: sim/habitat_bridge/
bump_analysis.py (corrected rule).
