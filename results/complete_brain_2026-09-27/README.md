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
   deficient side's connection groups become its mirror image.
2. **Senses short on both sides** -- hearing (the sound modality: JO-A/B),
   wind (JO-C/D/E), the rest of Johnston's organ, head bristles: each is
   mirrored as a WHOLE group (their subtypes are typed unevenly per side), the
   side with more real output is the template, raised to the target = the
   larger over the sides of max(its own output, FAFB's x the 1.24 synapse
   scale): hearing 2.10x, wind 1.17x, other JO 1.0x, head bristles 1.85x;
   no side is ever lowered.
3. **Left/right symmetrisation** of everything else so the robot cannot
   favour a side: 1,113,774 paired groups set equal per postsynaptic cell
   (exact to half a synapse per group), 486,422 one-sided groups mirrored
   (copies transmit with their own neuron's sign; 1,008 copies from neurons with
   no known transmitter skipped). Left as they are: cell types with neurons on
   one side only (0.62% of synapses), neurons without a side (1.9%).
Groups found only on a deficient side, not confirmed by the template: 631
groups / 13,710 synapses dropped (ORN 9,925; other JO 1,641; head bristles
1,569; wind 532; hearing 43). Male-specific / dimorphic types are never
filled. Mirror copies are mapped by rank; where that would wire a neuron to
itself (an autapse the data does not have), the copy is moved to the next cell
of the type (`move_autapses`, added the same evening after a review: 255
created, 85 moved). Where the mirror side has a single cell of the type (170
groups), a within-type connection cannot exist there, so the group is left
unmirrored at its raw strength, like other one-sided types (second review). The
brain keeps the raw data's 26 autapses. Total: 88,816,375 -> 89,414,640 synapses
(+0.67%). Logs:
`fill_log_population.csv.gz`, `fill_log_groups.csv.gz`,
`data/metadata/build_manifest_merged.json`.

Balance checks (`tests/test_merge.py`): every connection group with >= 50
synapses matches its mirror per postsynaptic cell within 10%; for every sense
with >= 10 neurons the input each receiving cell gets from the left organ
matches its mirror cell's from the right within 5% (heat, 7 neurons with a
one-sided subtype, is 13% off and not used by the robot); the Giant Fibre's
Johnston's-organ input is equal left and right.

What did NOT work and was replaced (kept because it matters): filling every
group more than 1.5x lopsided (231k groups, +16% synapses: natural
variation); "raise the weaker side to the stronger" (copied an artefact --
the few left VA1v ORNs are mostly wired to the RIGHT PNs -- onto both sides);
mirroring Johnston's organ by subtype (left "other JO" 2.8x lopsided).

## 3. Dynamics refitted for this brain

`cognition/calibrate_merged.py`: random search over all interacting settings.
Protocol (after a review found the first version selected on held-out
material): candidates are ranked ONLY on the exam's fit and constraint tests,
on the exam's seeds and one more seed set (+100); the 21 held-out tests and
the seed set +200 were never used to choose, and are reported below. The two
odour-steering tests were exempted from ranking (a documented limitation, §5).
`data/metadata/dynamics_calibrated_merged.json` + `calibration_merged.json`:

| setting | FAFB v3 | merged (first fit) | **merged (current)** |
|---|---|---|---|
| synaptic gain | 1.0 | 0.976 | **0.962** |
| ORN depression release_f / full strength | 0.96 / 1.0 | 0.968 / 0.656 | 0.96 / 0.611 |
| ORN->PN ipsi/contra release ratio | 1.4 | 2.11 | 1.82 |
| adaptation (all / extra AL local neurons) | 0.2 / 1.0 mV | 0.368 / 1.72 mV | 0.357 / 1.64 mV |
| excitatory AL-LN -> PN gain | 0.1 | 0.097 | 0.08 |
| Giant Fibre threshold correction | 0.7 | 0.40 | 0.507 |
| **new:** nerve-cord adaptation (nerve-cord + ascending neurons) | - | 1.8 mV | 1.21 mV |
| **new:** ORN->PN input normalisation (Tobin et al. 2017; uniglomerular, 0.5-2x) | - | exponent 0.21 | 0.106 |
| **new:** descending-neuron adaptation | - | - | 0.017 mV (searched 0-0.5) |

**Why it was refitted.** The autapse fix changed only a few hundred
connections, but it moved the exam's `no_latching` test (0.019 -> 0.28 on the
+200 seeds).
Rebuilding showed why: after some odour mixtures, loops of mutually exciting
descending neurons (DNg33, DNg12, DNge019, DNge027), the brain <-> nerve-cord
loop and optic-lobe circuits either settle or lock on, depending on small
details. The first fit sat near that tipping point, so its pass was partly luck.

A new random search (48 candidates around the first fit, same protocol,
descending-neuron adaptation added as a free setting) chose the values above.
The search left descending adaptation near zero; the other settings moved
instead (weaker ORN depression and excitatory LNs, a higher Giant Fibre
threshold). A sweep of descending adaptation alone (0.1-0.3 mV) did not make
`no_latching` reliable either way.

A second review then restored 170 single-cell within-type groups to raw
strength (§2). That changed 288 connections, and `no_latching` failed again on
the +200 seeds (0.127). A third search chose on THREE seed sets (the exam's,
+100 and +300; `--choose-offsets 100,300`): none of its top six candidates
passed `no_latching` on all three.

So latching is structural, not a tuning problem. In every latching case the
DNg33 <-> AN09A005 <-> IN09A005 brain <-> nerve-cord loop stays on (DNg33 ->
AN09A005: ~2,900 synapses; the two DNg33 excite each other through ~750 each
way, FAFB ~140), joined by other sites depending on the case: FR1 ring neurons
of the fan-shaped body, optic-lobe Mi18 / Lawf2, the DNg12 / DNge019 pairs.
The applied calibration stays the second search's (above). The fix is a
mechanism the model lacks -- most likely short-term depression at high-rate
synapses -- and is open (ROADMAP.md).

Nerve-cord adaptation: without it, the flight (DLM/DVM) and abdominal motor
circuits kept firing after a stimulus -- activity a real fly's sensory
feedback and neuromodulation would stop, which the wiring diagram lacks. FAFB
has no nerve cord, so this never arose before.

## 4. Results

**Fly exam** (`exam_merged_seeds{0,200}.txt`):

| seed set | passed | held-out | constraints | fit | failed |
|---|---|---|---|---|---|
| +200 (never used for choosing) | **30/32** | **21/21** | 3/3 | 6/8 | no_latching (0.127; §3, §5), symmetry_odour |
| exam seeds (used for choosing) | **32/32** | 21/21 | 3/3 | 8/8 | none |

Robustness (exam seeds): mean AUC **0.85**. The first fit scored 0.92 and
FAFB v3 scores 0.95. The weakest perturbations are synaptic gain x1.3,
flipped transmitter signs (13% of neurons) and 80% sensor loss (0.6 each at
the strongest level). Since the second review, the lesion test also spares
the male brain's sensory, motor and descending neurons, as it always did for
FAFB (`registry.canonical_super_class`). The first fit's exam files are kept
as `exam_merged_seeds*_before_autapse_fix.txt` (31/32 on both seed sets).

**Left/right bias with symmetric input** (`symmetry_*.txt`, 8 paired seeds,
steering DNs L-R): FAFB has a real bias -- DNa01 +3.5 Hz at rest, +5.3 Hz
while walking (p < 0.01). The complete brain: none significant (current fit:
all p >= 0.07, including the legs; first fit, `symmetry_merged_before_refit.txt`:
all p >= 0.10). A robot on FAFB would drift.

**Tests** (current): `pytest tests` (default = merged) 197 passed;
`FLY_DATASET=fafb` 218 passed.

## 5. Known limitations (open)

- **Latching after a stimulus** on some seed sets (above): a brain <-> nerve-cord
  loop (DNg33, AN09A005, IN09A005) can stay on after an odour ends. The robot
  shows no symptom of it in Habitat that we have found (stray startles came
  from a readout, now fixed), but it is a fragility of the model.
- **Odour steering** toward an odour on one antenna is weak (passes
  odour_lateralization on some seed sets, symmetry_odour on none reliably;
  on 12 fresh seeds an earlier calibration gave turn bias 0.004, p = 0.5 vs
  FAFB 0.023, p = 0.003). The side signal is clear in the antennal lobe and
  lateral horn but barely reaches the steering DNs: DNa02 sits near
  threshold, and the male's enlarged VA1v pheromone channel -- which fruit
  odour suppresses (Hallem & Carlson 2006: all six fruit odorants lower
  Or47b) -- pushes the other way via DNb05. An independent review found no
  left/right convention bug. The rover has no nose (`--real-senses`).
- **Giant Fibre at rest**: 1-3 stray spikes per 6 s on some seed sets,
  driven by a brain <-> nerve-cord loop (DNg33 at ~155 Hz with no input,
  ascending AN09A005 ~43 Hz). Adapting descending neurons too silences it
  but weakens steering (`vnc.include_descending`, off).
- **Nerve-cord leg readout**: DNa02 now moves the legs' coxa turn index
  consistently (left DNa02 -0.09 on 8/8 seeds, right +0.06 on 7/8;
  `vnc_turn_merged.txt`), but summed coxa-motor-neuron rate is not stride
  length (DNa02 turns the fly by shortening ipsilateral strides; Yang et al.
  2024), so the sign cannot be read as a turn direction without a leg model.
  The rover steers from the brain's steering DNs (pursuit_sign and
  goal_steering_sign pass).
- The raw male data (`FLY_DATASET=malecns`) keeps FAFB's dynamics and is not
  calibrated (sugar -> MN9 6 Hz); it is kept only as the starting point.

## 6. Code review findings fixed today (two independent reviews)

Review 1 -- critical: the exam, the calibration and several experiments built engines
without the dataset's synaptic gain (the male-based brain was examined at 1.0
while the pet ran at 0.62) -- every engine now goes through
`session.apply_calibrated_gain`. Also: recovery counts accumulated across days,
stale avoidance state reused by the pivot, a lidar test that could not fail,
male PEN names missing from the central-complex ring mask, symmetrisation not
exact under integer rounding, unbounded input normalisation (up to 327x; now
uniglomerular only, clipped), stale calibration notes. The second review
checked every left/right convention on the odour path (sides of receptors,
eyes, descending neurons, the merge's mirror copies) and found them correct.

Review 3 (full): the Habitat camera filter used prefixed sensor names and so
removed every camera (now: none by default, `--video` keeps the head camera);
the calibration had selected on held-out tests and seeds (now: §3 protocol);
one-sided Johnston's-organ subtypes left the hearing/wind groups lopsided and
mirror copies used the source neuron's sign (now: §2); remote-mode startup read
stale "server ready" lines; the pivot could act during the emergency return;
fixed-seed exam tests ignored the seed offset; plus reporting and test gaps.
