# Battery pet, first runs, 2026-09-26: a negative result

    OUT=... SEEDS="1 2" CONDS="pet petreflex" SAFE_SPEED=1 BATTERY=0.7 sim/habitat_bridge/run_cortex_life.sh

The bowl became a charging dock and hunger the battery charge (robot/battery.py,
sim/habitat_bridge/home.py battery mode); "pet" = cortex with manners and naps,
dock taste/odour scaled by hunger; "petreflex" = the same with the dock sensed at
full strength whatever the charge.

Run A (battery_pet_fastdrain, stopped after 2 days): drain rates 3x too high
(a full battery lasted < 2 days of activity) and a low pet kept choosing its
person over food; fixed (slower drain, food urgency above half hungry).

Run B (10 days): 3 of 4 lifetimes never found the dock and went flat on day 3
(pet seed 1 and 2, petreflex seed 1); petreflex seed 2 docked on 6 days and
never fell below 40%. Cause: the cortex only knows places it has been; on day 0
the pet was not hungry (charge 0.7), so nothing drew it to the dock, and when it
became hungry it could only find the dock by smell, in a large house. (Pets in
the non-battery runs started every day hungry.) Also: a flat pet stayed flat
(the overnight rescue was added after this run started), and naps almost never
happened (the nap condition requires not hungry and not wanting company).

Conclusion: the hunger-scaled-senses vs reflex comparison is inconclusive; the
design needs the dock to be known from the start, as for any real robot (it
starts on its dock), and a navigation-layer emergency return at critical charge
(counted as a failure). Next run implements both.

## Run C (10 days, 3 seeds; dock known from "birth", emergency return < 10%)

The first attempt was cut short by a power loss at days 3-4 and restarted from
scratch (partial output kept on the box as battery_pet_runC_interrupted).

| (3 seeds x 10 days) | pet (dock senses scale with hunger) | petreflex (always-on reflex) |
|---|---|---|
| flat batteries / emergency returns | 0 / 0 | 0 / 0 |
| charging while already full (charge >= 83%) | 0 s | 150 s |
| end-of-day charge, mean (lowest) | 0.45 (0.19) | 0.75 (0.47) |
| naps | 0 s | 40 s |
| near person / pets received | 474 s / 32 | 390 s / 22 |
| pet ran into person | 40 | 38 |

Hunger-scaled senses do make a charged pet ignore its dock (0 s vs 150 s of
charging when full). But hunger then rose from 90% down and there was no meal
state, so the pet charged in small sips, lived at ~45% charge, felt half-hungry
all day and never napped (naps require not hungry). Next: hunger onset below
60% and meals that continue to 90% (robot/battery.py).

## Run D (final of the day): meals, boredom naps, dock approach

Changes after run C, each found by tracing single hungry pets (one-day smoke
tests with the charge starting at 45%):
1. hunger starts below 60% charge; a meal (charging) continues to 90%, and only
   starts when hungry (robot/battery.py);
2. the neocortex stays hungry during a meal ("meal mode": other goals wait, it
   steers back to the dock after a drift, pays its person little attention),
   and settles at the bowl between feeding bursts (rest drive 0.6);
3. a meal survives short wanders (15 s off the dock);
4. robot-level slow final approach within 1.2 m of the dock (0.15 m/s) and
   pivot-in-place when the dock is > 60 deg to the side (the pet orbited it);
5. boredom naps (alone > 20 s, nothing worth doing, not hungry) besides sleepy naps.
Smoke test after 1-4: 3/3 hungry pets charged 0.40 -> 0.90 in one meal.

| (3 seeds x 10 days) | pet | petreflex (always-on dock reflex) |
|---|---|---|
| complete meals (to 90%) | 12 | 5 |
| end-of-day charge, mean +- sd (lowest) | 0.65 +- 0.16 (0.27) | 0.75 +- 0.19 (0.30) |
| flat / rescued / emergency returns | 0 / 0 / 0 | 0 / 0 / 0 |
| naps (sleepy / bored) | 6 / 0 (83 s) | 2 / 0 (38 s) |
| near person / pets received | 449 s / 23 | 380 s / 20 |
| pet ran into person | 45 | 40 |

Energy is now managed like a pet's: meals when hungry, a healthy charge, no
flat batteries in 60 pet-days. Open: no boredom naps occurred (the person is
seen often in the small house, or unexplored cells keep "something to do");
"charging while full" was not measured cleanly (the counter starts at 83%,
inside a meal's intended range) and needs a >= 90% counter.
