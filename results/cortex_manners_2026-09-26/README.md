# Cat-like manners around the person, 2026-09-26 (RTX 3080 Ti, CUDA engine)

    OUT=... SEEDS="1 2" CONDS="none v0 manners" SAFE_SPEED=1 sim/habitat_bridge/run_cortex_life.sh

2 seeds x 10 days x 120 s, Habitat small house, mushroom-body learning on, v3
dynamics. All three conditions have the robot's near-person speed limit
(robot/safety.py). "manners" = cortex v0 with cat-like manners (cortex/v0.py):
contact welcome when it wants company, loses interest in chasing and steps back
when content, never cuts in front of a walking person.

Per lifetime (seed 1 | seed 2):

| condition | days reached bowl | eating s/day | pet ran into person | pet bumped when it did not want company |
|---|---|---|---|---|
| fly brain alone | 4 \| 2 | 3.5 \| 3.2 | 7 \| 25 | - |
| cortex v0 | 9 \| 7 | 14.3 \| 10.2 | 21 \| 27 | 19 in total |
| cortex v0 + manners | 8 \| 8 | 10.6 \| 10.3 | 3 \| 6 | 1 in total |

Totals over both lifetimes: pet-caused bumps 32 / 48 / 9; the person walked into
the pet 40 / 52 / 48 times (the Habitat person never avoids the pet, often
approaching from behind, outside its 150-degree view); pets received 25 / 16 / 15.

Who bumped whom (sim/habitat_bridge/bump_analysis.py): in the control step in
which contact began, the pet counts as the one that bumped only if it was
moving toward the person (> 0.02 m/s) at least as fast as the person was moving
toward it. An independent review (2026-09-26) found the first version read the
pet's speed two steps before contact; the numbers above use the corrected rule
(before: 36 / 50 / 8).

Statistics, with the caveat that matters: there are only TWO lifetimes per
condition, and days within a lifetime share learned state (map, critic,
mushroom body), so days are not independent samples. Even treating the 20 days
as independent, pet-caused bumps per day do not differ significantly between
manners and v0 (Mann-Whitney p = 0.17; bumps come in bursts on a few days; the
p = 0.036 reported before the review used the misaligned rule). What the data
do show: in both lifetimes, manners cut the bumps the pet caused 6-7 fold
(3 vs 21, 6 vs 27) while reaching the bowl as often (8 vs 9, 8 vs 7 days).
Eating time did not differ clearly (10.6 vs 14.3, 10.3 vs 10.2 s/day); this is
not evidence of "no cost". More seeds are needed for a real test.

Other caveats:
- The client's contact counter (summary.txt, per contact onset in Habitat) and
  the who-bumped-whom list (one entry per 0.1 s control step with an onset)
  differ by 1-4 per condition when two onsets fall in one step.
- The speed limit's effect is not measured here: comparing with the run of
  2026-09-25 (no limit) is not valid, as trajectories diverge between runs. In
  Habitat contact happens at 0.6-0.8 m between centres (Spot-sized body), where
  the limit allows 0.1-0.16 m/s, not the 0.08 m/s "contact crawl".
