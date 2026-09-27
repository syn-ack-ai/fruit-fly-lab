# The complete brain: the male CNS, gap-filled and balanced, 2026-09-27

Since 2026-09-27 the project's default brain (`FLY_DATASET=merged`) is the
Janelia MaleCNS v1.0 -- brain, optic lobes and ventral nerve cord of one male
fly -- with its demonstrated reconstruction gaps filled and made left/right
symmetric, and with its dynamics refitted against the fly exam. FlyWire FAFB
(`FLY_DATASET=fafb`) remains available and serves as the reference.

Build (after `brain/connectivity/build_malecns.py`,
`brain/connectivity/malecns_annotations.py` and `brain/sensory/crossmap.py`):

    python -m brain.connectivity.merge
    FLY_DATASET=merged python -m cognition.exam --dynamics calibrated --workers 6

## 1. What is a gap? (evidence before filling)

Two connectomes differ for three reasons -- reconstruction gaps, individual
variation, and sex -- and only the first should be filled. Measured on
connection groups ((pre type, side) -> (post type, side), synapses per
postsynaptic cell):

| check | result | meaning |
|---|---|---|
| male left/right asymmetry beyond FAFB's 99th percentile, per strength bin | 0.1-1% of groups | the male is as symmetric as FAFB overall |
| groups > 4x lopsided in the male where FAFB is symmetric | 3,640 | ... |
| ... the reverse (FAFB lopsided, male symmetric) | 4,135 | equal rates: individual variation, not gaps |
| type pairs > 4x weaker in the male than FAFB (after the 1.24x synapse scale) | 37,864 | ... |
| ... the reverse | 77,044 | both ways at scale: no blanket female fill |
| olfactory receptor neurons by wiring, left / right | male 994 / 1,637; FAFB 1,117 / 1,132 | the male's **left antenna** is under-reconstructed |
| connected Johnston's-organ A/B neurons, left / right | male 75 / 9; FAFB 207 / 169 (published ~480 JO per antenna) | hearing input missing, mostly on the right |
| connected head-bristle neurons, left / right | male 333 / 238; FAFB 618 / 617 | touch input short on both sides |

"JO-A 0 left / 24 right" looked like a gap by subtype but is a typing
difference (JO family 349 / 324); the male's "JO-unclear" neurons have almost
no outputs (fragments). Sex differences are concentrated in higher brain
centres, the sensory and motor periphery largely isomorphic (Berg et al.
2026) -- which is what makes FAFB a fair reference for sensory neuron counts.

## 2. What was filled (`brain/connectivity/merge.py`)

1. **Left-antenna ORNs** (17 types lopsided beyond FAFB's type-level variation
   inside a lopsided ORN family): the complete right side is the template; the
   deficient side's connection groups become its mirror image; groups found
   only on the deficient side are dropped (939 groups, 25,317 synapses).
2. **Johnston's organ and head bristles**: each side's total output raised to
   FAFB's (x the 1.24 synapse scale), at most 8x:
   JO-A/B 2.10x left, 8.0x right (capped); JO-C/E 2.06x right; other JO 1.49x /
   2.44x; head bristles 1.85x / 1.65x.
3. **Left/right symmetrisation** so the robot cannot favour a side: 1,115,226
   paired groups set equal per postsynaptic cell (exact to half a synapse per
   group), 487,616 one-sided groups mirrored; left as they are: types with
   neurons on one side only (0.66% of synapses) and neurons without a side
   (1.9%).
Male-specific / dimorphic types are never filled. Total change: 88,816,375 ->
89,376,657 synapses (+0.63%). Logs: `fill_log_population.csv.gz`,
`fill_log_groups.csv.gz`, `data/metadata/build_manifest_merged.json`.

What did NOT work and was replaced (kept here because it matters): a first
version filled every group more than 1.5x lopsided (231k groups, +16% synapses)
-- that was natural variation; "raise the weaker side to the stronger" copied
an artefact (the few left VA1v ORNs are mostly wired to the RIGHT PNs) onto both
sides and created a backwards odour channel.

## 3. Dynamics refitted for this brain

`cognition/calibrate_merged.py`: random search over all interacting settings
against the whole exam, every finalist re-tested on two held-out seed sets
(the first 32/32 candidate turned out to pass odour lateralisation by seed luck;
since then a candidate must pass on all three seed sets).
`data/metadata/dynamics_calibrated_merged.json` + `calibration_merged.json`:

| setting | FAFB v3 | merged |
|---|---|---|
| synaptic gain | 1.0 | **0.974** |
| ORN depression release_f / full strength | 0.96 / 1.0 | 0.966 / 0.55 |
| ORN->PN ipsi/contra release ratio | 1.4 | 2.29 |
| adaptation (all / extra AL local neurons) | 0.2 / 1.0 mV | 0.314 / 1.83 mV |
| excitatory AL-LN -> PN gain | 0.1 | 0.098 |
| Giant Fibre threshold correction | 0.7 | 0.584 |
| **new:** nerve-cord adaptation (VNC + ascending neurons) | - | 1.58 mV |
| **new:** ORN->PN input normalisation (Tobin et al. 2017, uniglomerular, 0.5-2x) | - | exponent 0.111 |

Nerve-cord adaptation: without it, the flight (DLM/DVM) and abdominal motor
circuits kept firing after a stimulus (the exam's latching test) -- activity a
real fly's sensory feedback and neuromodulation would stop, which the wiring
diagram lacks. FAFB has no nerve cord, so this never arose before.

## 4. Results

**Fly exam** (`exam_merged_seeds{0,100,200}.txt`):

| seed set | passed | held-out | constraints | failed |
|---|---|---|---|---|
| exam seeds | 31/32 | 21/21 | 3/3 | symmetry_odour |
| +100 | 31/32 | 21/21 | 3/3 | symmetry_odour |
| +200 | 30/32 | 21/21 | 3/3 | odour_lateralization, symmetry_odour |

Robustness (exam seeds; FAFB v3 0.95): mean AUC 0.86 -- w_syn 0.85, edge drop
0.93, NT flip 0.80, lesion 0.93, sensor drop 0.87, sensor gain 0.80.

**Left/right bias with symmetric input** (`symmetry_*.txt`, 8 paired seeds,
steering DNs L-R): FAFB has a real bias -- DNa01 +3.5 Hz at rest, +5.3 Hz while
walking (p < 0.01). The complete brain: none significant (all p >= 0.09),
including the legs' turn index. A robot on FAFB would drift.

**Tests**: `pytest tests` (default = merged) 162 passed; `FLY_DATASET=fafb`
184 passed. `tests/test_merge.py` guards the balance, the protected types, the
exam gain and the held-out seed offset.

## 5. Known limitations (open)

- **Odour steering** toward an odour on one antenna is weak and not reliable on
  fresh seeds (turn bias 0.004, p = 0.5; FAFB 0.023, p = 0.003). The side signal
  is clear in the antennal lobe and lateral horn but barely reaches the steering
  DNs: DNa02 sits near threshold, and the male's enlarged VA1v pheromone channel
  -- which fruit odour suppresses (Hallem & Carlson 2006: all six fruit
  odorants lower Or47b) -- pushes the other way via DNb05. The rover has no
  nose and does not use odour steering (`--real-senses`); an independent
  review found no left/right convention bug.
- **Nerve-cord leg readout**: with the final calibration, DNa02 moves the leg
  coxa turn index weakly and in the opposite direction to this morning's first
  test (`vnc_turn_merged.txt`; that test ran at a wrong gain of 1.0 and before
  nerve-cord adaptation). The rover steers from the steering DNs, which the
  exam shows working (pursuit_sign, goal_steering_sign); a leg-level readout
  is future work.
- The raw male data (`FLY_DATASET=malecns`) keeps FAFB's dynamics and is not
  calibrated (sugar -> MN9 6 Hz); it is kept only as the starting point.

## 6. Code review findings fixed today (two independent reviews)

Critical: the exam, the calibration and several experiments built engines
without the dataset's synaptic gain (the male-based brain was examined at 1.0
while the pet ran at 0.62) -- every engine now goes through
`session.apply_calibrated_gain`. Also: recovery counts accumulated across days,
stale avoidance state reused by the pivot, a lidar test that could not fail,
male PEN names missing from the central-complex ring mask, symmetrisation not
exact under integer rounding, unbounded input normalisation (up to 327x; now
uniglomerular only, clipped), stale calibration notes. The second review
checked every left/right convention on the odour path (sides of receptors,
eyes, descending neurons, the merge's mirror copies) and found them correct.
