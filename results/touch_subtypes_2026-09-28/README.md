# Which head bristles are "touch"? The male brain's 69 BM cells, 2026-09-28

## The problem

With the complete male brain, lidar "touch" made Milo back up. Over held-out
runs, the backward-walking command MDN averaged 12.9 Hz while an obstacle was
within touch range and 5.5 Hz otherwise (`heldout_walk`, lidar alone). FAFB's
touch drives MDN not at all.

On FAFB, touch drives the antennal and vibrissal bristles (BM_Ant, BM_Vib).
The male CNS types its 69 head-bristle neurons only as "BM". Their FlyWire
type is ambiguous across seven bristle subtypes, and all 69 enter through the
antennal nerve, so neither annotation separates them. Touch drove all 69.

## What the 69 cells are (`brain/sensory/bm_subtypes.py`)

Each male BM cell is assigned the FAFB subtype whose mean output profile
(fraction of its synapses onto each postsynaptic cell type) is closest, by
cosine similarity.

- **Validation on FAFB:** leave-one-out, 72 of 81 cells are assigned their
  own subtype (BM_Ant 38 of 41).
- **Result for the male (`data/metadata/bm_subtypes_malecns.csv`):**

  | subtype | left | right |
  |---|---|---|
  | antennal (BM_Ant) | 16 | 3 |
  | frontal (BM_Fr) | 5 | 7 |
  | fronto-orbital (BM_FrOr) | 12 | 10 |
  | inter-ocular (BM_InOc) | 2 | 6 |
  | orbital (BM_Or) | 4 | 4 |

- **Caveat:** margins are small (median 0.06 between the best and second-best
  subtype). The side imbalance of BM_Ant is probably partly real, partly
  classification. The merge mirrored "BM" as one group, ignoring subtypes.

**Driven alone** (40 Hz on the resting input, calibrated dynamics):

| assigned group | cells | MDN (backward) | MN9 (proboscis) |
|---|---|---|---|
| BM_Ant | 19 | 4.4 | 0.8 |
| BM_Fr | 12 | 4.0 | 2.3 |
| **BM_FrOr** | 22 | **23.3** | **37.0** |
| BM_InOc | 8 | 5.3 | 0.8 |
| BM_Or | 8 | 12.7 | 0.3 |
| BM_Vib | 22 | 5.1 | 1.0 |

The cells wired like fronto-orbital bristles, and partly the orbital ones,
make the male fly back up and extend its proboscis. That matches head-bristle
grooming and avoidance responses, but for a robot it means freezing and
backing at every wall.

## The touch set (`experiments/touch_subtypes_test.py`, `touch_sets_*.txt`)

| male set | cells L/R | MDN at 12 Hz (adapted) | at 80 Hz (burst) | MN9 at 12 / 80 Hz |
|---|---|---|---|---|
| all 69 BM + BM_Vib (before) | 51 / 40 | 24.6 | 36.8 | 7.0 / 35.2 |
| **BM_Ant + Fr + InOc + BM_Vib (now)** | 35 / 26 | **7.4** | **14.2** | **1.0 / 0.7** |
| BM_Ant + BM_Vib | 28 / 13 | 6.6 | 14.0 | 0.0 / 1.8 |
| BM_Vib only | 12 / 10 | 3.4 | 22.3 | 0.8 / 10.3 |

(Rest: MDN 2.9. FAFB BM_Ant + BM_Vib: MDN 0.0; walking DNg100 rises.)

Touch now drives the antennal, frontal and inter-ocular cells plus BM_Vib
(`robot/lidar.LidarTouch`; `FLY_TOUCH_BM=all` restores the old mapping). This
set is a choice about the robot's behaviour, not a claim about which bristles
a lidar "is". Steering stays balanced: DNa02 right minus left is about 0.

## The unstick reflex's channel

The review found that the unstick reflex also drove the pursuit neurons
LC10a, which make the male brain walk backward. On male brains it now pulls
through the central complex only (`cortex/topdown.reflex_channel`,
`FLY_UNSTICK_CHANNEL`).

The orienting reflex keeps LC10a on every brain. Through the goal channel,
Milo turned toward a person who appeared 67% / 52% of the time instead of 86%
/ 77% (tuning seeds, 2026-09-28), and that reflex lasts only 2 s.

## Tuning seeds 11-13, 3 days (`score_tuning_touch_unstick.txt`)

A = before, B = the touch remap, C = B + unstick through the central complex.
Pairs are lidar / lidar steering:

| | A | B | C |
|---|---|---|---|
| walked m/day | 12.9 / 19.9 | 14.4 / 18.5 | 14.4 / 18.5 |
| pinned/pressing s/day | 4.6 / 8.1 | 2.4 / 5.2 | 2.4 / 4.1 |
| stuck s/day | 38.7 / 2.0 | 34.0 / 0.0 | 34.0 / 0.0 |
| aliveness | 99% / 93% | 96% / 94% | 96% / **97%** |
| total | 83.1% | **86.8%** | 86.2% |

**Unstick with lidar alone** (`FLY_UNSTICK_LIDAR=1`,
`score_tuning_unstick_lidar_alone.txt`) made things worse:

- stuck time 34 -> 43 s/day;
- aliveness 96 -> 91%;
- no dock visits.

Without the steering layer, pulling the goal toward open space mostly drives
Milo into the lidar brake. The reflex stays with lidar steering (answering
the ROADMAP question).

## Held-out seeds 41-43, 10 days

Files: `score_heldout.txt`, `stats_before_vs_final.txt`,
`stats_fafb_vs_final.txt`. "Before" = `results/walking_latching_2026-09-28/`.
FAFB was re-run with today's code. Pairs are lidar / lidar steering:

| | FAFB | before | **final** |
|---|---|---|---|
| MDN during touch, Hz | 0.2 / 0.3 | 12.9 / 13.8 | **5.1 / 5.3** |
| MDN overall, Hz | 0.3 / 0.3 | 8.6 / 6.8 | 5.0 / 5.2 |
| DNg100 overall, Hz | 22.2 / 24.0 | 3.4 / 4.2 | 3.6 / 4.0 |
| walked m/day | 19.5 / 29.0 | 8.9 / 13.3 | 10.2 / 15.8 |
| safety-layer brake s/day | 58 / 34 | 19 / 11 | 24 / 12 |
| pinned/pressing s/day | 6.1 / 7.1 | 7.5 / 7.1 | 9.3 / 6.6 |
| pet-caused person bumps/day | 0.20 / 0.43 | 0.00 / 0.33 | 0.37 / 0.20 |
| turns toward the person | 74% / 70% | 77% / 82% | 78% / 69% |
| aliveness | 91% / 88% | 93% / 93% | 93% / 89% |
| total | 80.5% / 81.9% | 80.6% / 77.0% | 75.4% / 81.0% |
| **total, both** | **81.2%** | **78.8%** | **78.2%** |

**What it means.**

- **The fix works in the brain.** Touch no longer drives backward walking in
  the closed loop: MDN during touch fell from ~13 Hz to its no-touch level.
- **Behaviour did not change detectably.** No held-out difference between
  "before" and "final" survives the statistics; the tuning seeds' gain
  (83 -> 87%) did not replicate. Walking rose 15-20% (n.s.).
- **The walking gap is not backing any more.** In the closed loop the male
  brain's forward command DNg100 runs at about 1.2x its resting rate (3.6-4.0
  vs 3.1 Hz); FAFB's at about 1.6x (22-24 vs 14.6 Hz). The male brain's sensory
  and goal inputs drive walking less above rest. That is the next question
  (ROADMAP).
  *(2026-09-29: FAFB's 14.6 Hz was stale; measured the same way it rests at
  18.5 Hz, so its closed-loop DNg100 is also ~1.2-1.3x rest. The gap was in
  the body's readouts: `results/body_readout_2026-09-29/`.)*
- The first held-out run of the final configuration had a bug the review
  found: the unstick pull let the orienting reflex's LC10a attention through.
  It was re-run with the fix; lidar-alone results are identical, as they must
  be.
