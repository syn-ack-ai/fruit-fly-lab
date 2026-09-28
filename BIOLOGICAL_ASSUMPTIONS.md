# BIOLOGICAL_ASSUMPTIONS.md

What in this project is real, what is a published model, what is an
approximation, and what is just our code.

Read this before drawing any biological conclusion from the simulation.

---

## Provenance categories

Every component in the codebase is tagged with one of these, in its module
docstring and in the UI's provenance panel.

| Tag | Meaning |
|---|---|
| **A — Real data** | Comes from FlyWire FAFB v783 or the Janelia MaleCNS v1.0 unmodified (the default gap-filled, symmetrised male brain is category C where it was changed: §11). Neuron identities, cell types, connectivity, synapse counts, neurotransmitter predictions, 3D positions, column assignments, community labels. |
| **B — Published model** | An assumption or measurement taken from a peer-reviewed paper, cited in place. The LIF equations and constants, the excitatory/inhibitory sign convention, sensory tuning properties, descending-neuron → behaviour assignments. |
| **C — Our approximation** | A modelling choice we made because the data does not determine it. Always documented, and where it matters, counted. |
| **D — Our engineering** | Code with no biological content: data loading, the simulation loop, the web server, the drawing routines. |

---

## 1. What the simulation actually is

(Sections 1-8 describe the original model on FlyWire FAFB, `FLY_DATASET=fafb`,
with the published dynamics. The default complete male brain and the calibrated
dynamics are in §11.)

```
looming object (exact geometry)                      A/D
  → LC4 + LPLC2 firing rates                          B + C  ← the only modelled step
  → Poisson drive onto 314 real FlyWire neurons       B
  → 139,255-neuron LIF network,                       A + B
    3,732,460 real connections, 50,666,648 synapses
  → real descending neurons (DNp01 …)                 A
  → motor channel activation                          B + C
  → kinematic body                                    C/D
```

The middle of that chain — by far the largest part — is real connectome and
published model. The two ends are where we had to model something.

**There is no rule anywhere that says "if looming then escape."** The proof is
in `experiments/02_escape_controls.py`: silencing the 104 real LC4 and 210 real
LPLC2 neurons drops the Giant Fibre from 37 spikes to **exactly zero** while the
stimulus is unchanged and LC4/LPLC2 themselves keep firing. The response travels
through the connectome or it does not happen at all.

---

## 2. The neuron model (category B)

Verbatim from Shiu et al. 2024 (`brain/neuron_models/lif.py`):

```
dv/dt = (v_0 - v + g) / t_mbr     (unless refractory)
dg/dt = -g / tau                  (unless refractory)
spike when v > v_th ; then v ← v_rst, g ← 0
presynaptic spike, after t_dly:  g ← g + w
w = sign(presynaptic neurotransmitter) × synapse_count × w_syn
```

with v_0 = v_rst = −52 mV, v_th = −45 mV, t_mbr = 20 ms, tau = 5 ms,
t_rfc = 2.2 ms, t_dly = 1.8 ms, w_syn = 0.275 mV.

**What this model does not have:**

- **No dendrites or compartments.** Every neuron is a point. A real *Drosophila*
  neuron performs substantial computation in its neurites; a Kenyon cell, an
  LPLC2 and a Giant Fibre are treated as electrically identical apart from their
  connections.
- **No synaptic plasticity, facilitation, depression, or adaptation** in the
  published model. Firing rates do not decline with sustained input the way real
  sensory neurons do. (The calibrated dynamics add spike-frequency adaptation
  and ORN depression, see `data/metadata/dynamics_calibrated*.json`, and
  `brain/plasticity/` adds dopamine-gated mushroom-body learning.)
- **No neuromodulation.** Dopamine, serotonin and octopamine are treated as fast
  excitatory transmitters. In the real animal they act on slow metabotropic
  receptors and change the state of circuits rather than driving spikes. This
  affects 1,677 neurons (584 DA, 1,021 5-HT, 72 OA).
- **No gap junctions.** The dataset contains chemical synapses only. This
  matters acutely for escape: the Giant Fibre's output onto the tergotrochanteral
  motor neuron and the peripherally synapsing interneuron is largely
  **electrical**, and is therefore absent from the model.
- **No spontaneous activity** in the published model (the calibrated dynamics
  add resting receptor input). With no stimulus, the network is completely
  silent (`tests/test_circuits.py::test_an_unstimulated_brain_is_silent`). Real
  brains have ongoing background firing and a balance of excitation and
  inhibition that this model lacks. A consequence is visible in the escape
  experiment: the Giant Fibre fires occasional spikes at small angular sizes
  where a real fly, sitting in a tonically inhibited state, would not.
- **A single global synaptic weight.** `w_syn = 0.275 mV` is one free parameter
  fitted across the whole brain. Real synaptic strengths vary by orders of
  magnitude between cell types.

---

## 3. Excitatory / inhibitory assignment (category B, with a C fallback)

Following Shiu et al.: **ACh, dopamine, octopamine, serotonin → excitatory;
GABA, glutamate → inhibitory.** Glutamate is inhibitory in *Drosophila* via the
GluClα chloride channel, which is the opposite of the vertebrate convention.

Neurotransmitters are **predicted from electron-microscopy images** (Eckstein et
al. 2024), not measured. They are wrong for some neurons. Two consequences we
observed directly:

- **Histamine is not in FlyWire's vocabulary.** The classifier predicts exactly
  six transmitters (ACh, GABA, Glu, DA, 5-HT, OA). Photoreceptors are
  histaminergic and their synapse onto lamina monopolar cells is **inhibitory
  and sign-inverting**. In this dataset R1-6 are labelled ACh, i.e. excitatory.
  **Driving photoreceptors would invert the sign of the entire early visual
  pathway, so we disabled it** rather than produce a plausible-looking wrong
  answer. The UI shows "Not currently modeled" with this reason.
- **DNp01 is annotated inconsistently.** The left Giant Fibre is predicted ACh,
  the right one GLUT. The Giant Fibre is cholinergic. One of these is a
  prediction error, which means the right Giant Fibre's chemical output has the
  wrong sign in the model. Its *inputs* — which is what our escape experiment
  measures — are unaffected.

**Our fallback (C):** 19,658 of 139,255 neurons (14.1%) have no neuron-level
neurotransmitter prediction. For those we take the synapse-count-weighted
majority transmitter across the neuron's own outgoing connections. This resolves
18,032 of them. The remaining **1,626 neurons (1.2%) have no sign and therefore
produce no output** in the simulation (FAFB; the MaleCNS has 3,275 unsigned
neurons, and treats histamine as inhibitory). All three counts are recorded in
`data/metadata/build_manifest.json` and shown in the UI provenance panel.

---

## 4. Connectivity (category A, with one C choice)

- We use `connections_princeton.csv.gz`, the **≥5-synapse thresholded** table.
  That is Codex's default and matches the reference model. Weaker connections
  are dominated by synapse-detection false positives. The unthresholded table is
  present on disk and deliberately unused.
- **We sum synapses across neuropils (C).** A neuron pair connected in two
  neuropils becomes one edge with the combined count. The model is a point-neuron
  model, so it cannot use the spatial separation anyway.
- **Proofreading is not uniform.** FlyWire's central brain is proofread to a high
  standard; parts of the optic lobes are less complete. Weakly connected or
  poorly reconstructed neurons will be under-represented.
- **This is one fly.** A single adult female. Individual variability, sexual
  dimorphism, and developmental variation are not represented. FAFB was also
  fixed and imaged, so this is a snapshot of one animal's wiring, not a species
  average.

---

## 5. Sensory encoding (category B + C) — the honest weak point

The connectome is a wiring diagram. It contains no phototransduction, no
odorant-receptor binding, no mechanotransduction. **Something has to turn a
physical stimulus into spikes, and that something is not the connectome.**

Shiu et al. handled this by driving chosen neurons with Poisson input at a fixed
rate — effectively simulating optogenetic activation. We do the same, and for
looming we additionally modulate the rate by published tuning and a receptive
field.

### Looming (`brain/sensory/encoders.py`)

| Element | Category | Notes |
|---|---|---|
| Which neurons (104 LC4, 210 LPLC2) | **A** | Real FlyWire cell types |
| Receptive-field centres and radii | **A→C** | Computed as the synapse-weighted mean visual direction of each cell's *column-assigned presynaptic partners*. The column assignments are real data; treating their centroid as a receptive-field centre is our inference. Mean radius came out at 14–15°, consistent with published LC/LPLC receptive fields, and lateralisation with a frontal binocular overlap fell out correctly — but these are **not measured receptive fields**. |
| Hex-lattice → visual angle mapping | **A→C** | Axis orientation was *measured* against real 3D anatomy (u = p + q/2 tracks the dorsoventral axis, R² = 0.98; the orthogonal axis tracks the anteroposterior axis, partial r = 0.95). The scaling onto a 175° × 160° field of view is a linear approximation; the real interommatidial angle varies from ~4.5° frontally to ~8° laterally. |
| LC4 ∝ angular velocity, LPLC2 ∝ angular size | **B** | von Reyn et al. 2017 |
| LPLC2 gated by outward motion | **B** | Klapoetke et al. 2017 |
| Saturating (Naka-Rushton) tuning shape and its half-maximum constants | **C** | The *shape* is a standard choice; the constants (300°/s for LC4, 25° for LPLC2) are **fitted by us to reproduce published qualitative behaviour, not measured**. Changing them changes when the escape triggers. |

### All other modalities (`brain/sensory/modalities.py`)

Intensity maps **linearly** to firing rate up to 150 Hz, with no receptor
adaptation and no spatial structure. Which neurons each stimulus drives is real
data plus a cited experimental identification; how hard it drives them is a
placeholder.

---

## 6. Motor output (category A + B + C)

**On FAFB the simulation ends at the descending neurons, and that is a hard limit
of the dataset, not a shortcut.** FlyWire FAFB is a brain connectome (the male
CNS includes the nerve cord: §11). The motor neurons
that move legs and wings are in the ventral nerve cord. The 110 neurons FlyWire
labels `motor` innervate head structures. This is asserted by
`tests/test_circuits.py::test_no_leg_or_wing_motor_neurons_in_this_brain_dataset`.

One genuine exception: **proboscis motor neurons are in the brain**, so the
sugar-feeding experiment reaches a real motor neuron.

The descending-neuron → behaviour table in `brain/motor/descending.py` is
**category B**: it comes from optogenetic activation and silencing experiments,
each entry carrying its citation. It is not derivable from the connectome. 20
descending cell types have assignments (plus 3 more pooled in the population
readout); **the other ~450 descending types in the dataset have none, and the
UI reports them as "Not currently modeled."** The Giant Fibre is read per cell
(one command neuron); the long-mode takeoff is read as the pooled population of
its looming DNs (DNp02, DNp04, DNp11; von Reyn et al. 2014, Ache et al. 2019).

Mapping a firing rate to a 0–1 "command strength" with a half-maximum of 60 Hz,
and the thresholds at which the body acts, are **category C** — our choices.

---

## 7. What is not modelled at all

| Not modelled | Why |
|---|---|
| Touch on thorax, abdomen, legs | Those mechanosensory neurons project to the ventral nerve cord: absent from FAFB v783; present in the MaleCNS but not yet mapped to a stimulus. |
| Light / vision from photoreceptors | Histamine is missing from the neurotransmitter vocabulary, so the photoreceptor→lamina sign would be inverted (§3). |
| Leg and wing motor neurons, muscles | FAFB: ventral nerve cord (§6). MaleCNS: present, but no leg or muscle model reads them yet (§11). |
| Gap junctions, including the Giant Fibre's output synapses | Not in the dataset (§2). |
| Neuromodulatory state, hunger, arousal, circadian phase | No model of internal state. A real fly's response to food depends heavily on satiety. |
| Learning and memory | The published model has no plasticity; `brain/plasticity/mushroom_body.py` adds dopamine-gated KC->MBON learning (graded APL inhibition), used by the robot. |
| Flight aerodynamics, leg biomechanics | The body is kinematic (§8). |
| The ocelli, and most of the ~465 unassigned descending neuron types | No behavioural assignment we would stand behind. |

---

## 8. The body (category C/D)

`fly/body/fly_body.py` is **not** part of the neural simulation. It takes motor
channel activations and nothing else — it never sees the stimulus. If the neural
simulation produces no descending activity, the body does nothing
(`tests/test_sensory_and_body.py::test_body_does_nothing_without_descending_activity`).

Published kinematics are used where they exist: short-mode (Giant-Fibre) takeoff
~5 ms after the command with no preparatory wing raising, long-mode takeoff after
~200 ms of wing raising (Card & Dickinson 2008; von Reyn et al. 2014). Everything
else — walking speed, turn rate, the ballistic jump — is a plausible kinematic
approximation, not a biomechanical model.

---

## 9. Known quantitative discrepancies

Places where the simulation visibly departs from the real animal:

1. **Giant Fibre firing rate.** The real Giant Fibre is essentially all-or-none:
   it fires one or two spikes and triggers takeoff. Our model produces graded
   rates up to ~250 Hz. We read the escape command off a rate threshold, which
   is a reinterpretation, not a reproduction.
2. **Giant Fibre activity at small angular sizes.** Occasional Giant Fibre spikes
   occur at θ ≈ 3–10°, where a real fly would not escape. This follows from the
   absence of background inhibitory tone (§2).
3. **No escape-probability behaviour.** Real flies escape probabilistically and
   choose between short and long mode depending on stimulus dynamics
   (von Reyn et al. 2017). Our model is deterministic given a seed.
4. **Sensory populations are driven synchronously.** Every cell in a modality
   gets independent Poisson input at the same rate; real populations have
   heterogeneous thresholds and correlated noise.

---

## 11. The default brain since 2026-09-27: the complete male CNS ("merged")

`FLY_DATASET=merged` (the default) runs the Janelia MaleCNS -- brain AND
ventral nerve cord of one male (DATA_SOURCES.md §1b) -- with the following
additions. Everything here is category C unless a source is given; the numbers
and the evidence are in `results/complete_brain_2026-09-27/README.md`.

What changes against §7: the nerve cord is present, so leg and wing motor
neurons, leg sensory neurons and the descending -> nerve cord -> motor neuron
path exist. A leg-level turning readout is NOT yet usable: DNa02 moves the
legs' coxa turn index consistently in sign (left DNa02 -0.09 on 8/8 seeds,
right +0.06 on 7/8; `experiments/vnc_turn_test.py`), but summed coxa-motor-
neuron rate is not stride length (DNa02 turns the fly by shortening ipsilateral
strides; Yang et al. 2024), so it cannot be read as a turn without a leg model.
The robot's body is driven from the brain's steering descending neurons. The fly is male; sex differences
are concentrated in higher brain centres and the sensory/motor periphery is
largely isomorphic (Berg et al. 2026), which the gap filling relies on.

**Gap filling (`brain/connectivity/merge.py`)**, only where the data show a
reconstruction gap, never where the two animals merely differ:
1. *Left-antenna olfactory receptor neurons*: 994 left vs 1,637 right by wiring
   (FAFB 1,117 / 1,132). The deficient side's connection groups become the
   mirror image of the complete side ("template"); groups found only on the
   deficient side are dropped (631 groups, 13,710 synapses, in all the filled
   families).
2. *Hearing, wind, other Johnston's organ, head bristles*, short on both sides
   against FAFB (connected JO-A/B 75 left / 9 right vs 207 / 169; published
   ~480 JO neurons per antenna, Kamikouchi et al. 2006): each is mirrored as a
   whole group from its better side, raised to max(its own output, FAFB's x
   the synapse ratio 1.24). Fewer, stronger synapses stand in for the missing
   neurons.
3. *Left/right symmetrisation* of every connection group (per postsynaptic
   cell), so the robot cannot drift or favour a side. Unmirrorable groups
   (cell types present on one side only): 0.62% of synapses; neurons without
   a side: 1.9%. Mirror copies are mapped by rank; where that would wire a
   neuron to itself (an autapse the data does not have), the copy moves to the
   next cell of the type (255 created, 85 moved); where the mirror side has a
   single cell of the type, the within-type group is left unmirrored at raw
   strength (170 groups). The merged brain keeps the raw data's 26 autapses.
Male-specific and (potentially) dimorphic cell types are never filled.

**Dynamics refitted for this brain** (`data/metadata/dynamics_calibrated_merged.json`,
gain in `calibration_merged.json`, fitted with `cognition/calibrate_merged.py`
on the exam's fit and constraint tests only, held-out tests and a held-out seed
set reported, never optimised), including two new mechanisms:
- *nerve-cord adaptation*: extra spike-frequency adaptation in nerve-cord and
  ascending neurons. Without it the flight and abdominal motor circuits
  sustain activity after a stimulus -- in the real fly they are held by sensory
  feedback and neuromodulation the wiring diagram does not contain.
- *descending-neuron adaptation* (`vnc.descending_adapt_mV_per_spike`): the male
  CNS's DN-DN and brain <-> nerve-cord loops (e.g. the two DNg33 exciting each
  other through ~750 synapses each way, FAFB ~140) can latch after a stimulus
  and leak stray spikes into the Giant Fibre at rest; see the calibration notes
  in `results/complete_brain_2026-09-27/README.md`.
- *proboscis readout*: the male brains read proboscis extension from MN9, the
  proboscis-extension motor neuron (FlyWire's proboscis label group does not
  exist in the MaleCNS).
- *ORN -> PN input normalisation* (Tobin, Wilson & Lee 2017, generalised across
  glomeruli, clipped 0.5-2x, uniglomerular PNs only): PNs with more receptor
  synapses have lower input resistance. Needed for the male-enlarged VA1v
  glomerulus (111 receptor neurons per side, Or47b 47 Hz spontaneous).

**Known limitation -- odour steering.** Odour on one antenna reaches the
antennal lobe and lateral horn with a clear side signal, but steering toward
it is weak and not statistically reliable on fresh seeds (turn bias ~0.004 vs
FAFB 0.023): the steering neuron DNa02 sits near threshold, and the male's
enlarged VA1v pheromone channel -- which fruit odour SUPPRESSES (Hallem &
Carlson 2006) -- pushes the opposite way. The robot has no nose, so it does not
use odour steering (`--real-senses`); fixing it is an open biology task.

---

## 10. What this simulation is not

It is a **circuit-level simulation of one connectome under one published neuron
model**. It reproduces some experimentally established input–output
relationships — LC4/LPLC2 → Giant Fibre, sugar → proboscis extension, bitter →
no proboscis extension — and it lets you cut those pathways and watch the
behaviour disappear.

It does **not** reproduce, and makes no claim to reproduce, the subjective
experience, awareness, consciousness, or inner life of a fruit fly. It is a
network of differential equations wired according to a photograph of one
animal's synapses. Nothing here bears on what it is like to be a fly, and
nothing here should be described that way.
