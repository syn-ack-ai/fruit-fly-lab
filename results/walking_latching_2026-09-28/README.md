# Why the complete brain walked so little, and the latching fix, 2026-09-28

## 1. Walking: a body constant measured on FAFB

With the same robot layers, the complete male brain walked ~3x less than
FlyWire FAFB in Habitat: 7-9 vs 20-28 m/day (`results/unstuck_2026-09-28/`).

**What the logs showed.** Over all steps, the body's forward-walking command
DNg100 averaged 5.6 Hz (FAFB 23 Hz). The backward-walking command MDN averaged
9.2 Hz (FAFB 0.3), and 13% of the time was spent walking backward (FAFB 0%).
Mean speed was 0.07 vs 0.21 m/s.

**Two causes.**

1. **The walking-speed readout used FAFB's resting DNg100 rate.** The body
   (`fly/body/foraging_body.py`) walks at a fly's typical pace (12 mm/s) when
   DNg100 fires at `DNG100_REST_HZ`, and scales with it. That constant, 14.6
   Hz, was measured on FAFB. The male brain rests at ~3 Hz, so it walked at
   ~1/4 of a fly's pace.

   **Fix.** Each dataset's own resting rate is measured the same way
   (`cognition/calibrate_body.py`: resting receptor input, calibrated
   dynamics, 12 seeds x 2 s after a 1 s warm-up). Merged: **3.10 +- 0.32 Hz**,
   written to `data/metadata/body_readout_merged.json` with the dynamics'
   fingerprint; the body warns if they have changed since. FAFB keeps 14.6.

2. **Goals also drove the pursuit neurons, and in the male brain those walk
   backward.** The neocortex steered with two top-down channels: a goal
   through the central complex (FC2 -> PFL3 -> DNa02) and a copy of it into
   the pursuit neurons LC10a. On FAFB both drive forward walking. In the male
   CNS, LC10a is the courtship-pursuit pathway, normally gated by P1 arousal,
   and the phantom drive made the robot walk BACKWARD. That runs through
   LAL/GNG neurons -> DNpe023 (781 synapses onto MDN) -> MDN, 3 -> 24 Hz,
   while DNg100 fell.

   | (male brain, closed loop) | DNg100 | MDN | speed |
   |---|---|---|---|
   | nothing | 5.3 | 8.0 | 2.0 mm/s |
   | goal channel (central complex) | 7.3 | 2.5 | 6.3, forward |
   | pursuit channel (LC10a) | 8.7 | 18.3 | 2.5, backward |
   | both (the old routing) | 6.3 | 13.5 | 3.1 |

   The goal channel alone steers the male brain correctly (turn bias -0.11 for
   a goal 60 deg left, +0.07 right; FAFB -0.07 / +0.10).

   **Fix.** On male brains, navigation goals go through the central complex
   only (`TopDown.attend_from_goal`, `FLY_ATTEND_FROM_GOAL`; FAFB unchanged).
   Explicit attention -- the orienting and unstick reflexes -- still uses
   LC10a.

**Tuning seeds** (11-13, 3 days; walking readout still at the pre-warm-up
4.25 Hz): walked 7-9 -> 12-19 m/day; total 81-83% -> 86.7% (goal routing);
heading reversals 2.6 -> 0.6-1.7 per active minute.

## 2. Latching: brain-wide short-term depression

The exam's `no_latching` test failed on some seed sets whatever the
calibration. It measures whether activity dies away after an odour ends.

- In every latching case, a brain <-> nerve-cord loop stayed on: DNg33 <->
  AN09A005 <-> IN09A005, joined by FR1, optic-lobe Mi18/Lawf2 or DN pairs.
- No calibration passed on all of three seed sets.
- Depressing only the descending and ascending neurons (1-5% per spike) did
  not fix it either: it passed on at most 3 of 4 seed sets, and 5% broke the
  startle to sound.

**Fix.** Short-term depression of every non-sensory neuron's output synapses,
0.5% per spike, recovering with the ORN depression's 893 ms
(`loop_depression` in `data/metadata/dynamics_calibrated_merged.json`).

- **Rate-dependent:** at 130 Hz a neuron transmits at ~2/3 strength; at
  5-20 Hz, 92-98%.
- **The robot's readout is untouched:** the body reads DN spikes, which
  depression does not change.
- **No other setting changed.**

| | before | brain-wide depression |
|---|---|---|
| no_latching, seed sets 0 / +100 / +200 / +300 | 2 of 4 | **4 of 4** |
| exam (exam seeds / +200) | 32/32 / 30/32 | 31/32 / 30/32 (only odour steering fails) |
| held-out | 21/21 | 21/21 |
| robustness | 0.85 | **0.89** |

The CPU engine decayed every depressed neuron each 0.1 ms step: 151k neurons
instead of 2.6k ORNs made it 2.3x slower (review). The decay is now lazy
(`native/lif_native.c`: brought up to date when a neuron spikes). It is 3x
faster and statistically equivalent to the per-step decay over 6 seeds; it is
not bit-identical, because float rounding differs.

## 3. Final validation

**Fly exam** (`exam_seeds0.txt`, `exam_seeds200.txt`, with the lazy engine):
31/32 on the exam's seeds and 30/32 on +200. Only the odour-steering tests
fail; held-out 21/21, robustness 0.89.

**Habitat, held-out seeds 41-43, 10 days, the final configuration**
(`score_final_heldout.txt`, `stats_fafb_vs_ours_final.txt`). FAFB was run with
the same robot. "Ours before" = this morning's robot (`results/unstuck_2026-09-28`).
Pairs are lidar / lidar steering:

| | FAFB | **ours, final** | ours before |
|---|---|---|---|
| walked m/day | 19 / 29 | **8.9 / 13.3** | 7.4 / 8.7 |
| 90th-percentile speed m/s | 0.42 / 0.48 | 0.24 / 0.29 | 0.14 / 0.16 |
| aliveness | 91% / 89% | **93% / 93%** | 90% / 90% |
| heading reversals / active min | 1.6 / 2.5 | **0.6 / 1.7** | 3.5 / 3.6 |
| turns toward the person | 74% / 71% | 77% / **82%** | 78% / 79% |
| safety-layer brake s/day | 58 / 34 | 19 / 11 | 1.3 / 0.7 |
| blocked (pressing) s/day | 6.1 / 4.2 | 7.5 / 5.7 | 1.3 / 0.4 |
| stuck s/day | 28 / 0.2 | 43 / 2.8 | 15 / 2.3 |
| total | **80.8%** | 78.8% | 79.0% |

**What it means.**

- **More walking, not all of the gap.** The fixes raised walking by 20-50%
  and made the robot the most lifelike of the three (highest aliveness, least
  twitching). It still walks about half as far as FAFB.
- **The safety drop is from moving at all.** The "before" robot barely moved
  (90th-percentile speed 0.14-0.16 m/s), which is why its brake and pinned
  times were so low. At its new speed its pressing time is FAFB's, and it
  still needs the lidar brake about 3x less than FAFB.
- **Totals are unchanged** (78.8% vs 79.0%) and a little below FAFB (80.8%):
  the "life" score gained, the safety score lost.
- **Open:**
  - with lidar alone (no steering, so no unstick reflex) stuck time rose to
    43 s/day;
  - why DNg100 runs lower in the closed loop than at rest (P1 excitement
    slows walking; the goal and touch inputs).
