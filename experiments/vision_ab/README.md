# Vision through the optic lobes: option A vs option B

The robot's camera (and, later, its lidar) used to reach the fly brain only
through hand-made encoders on ~600 central visual neurons (LC4, LPLC2,
LC10a: `brain/sensory/encoders.py`, `robot/head.py`). The ~100,000 optic-lobe
neurons got no input. Two ways to send vision through them, compared on the
same stimuli and read-outs (`harness.py`):

- **A, the connectome's own eye** (`robot/retina.py`): the visual field drives
  the photoreceptors (A1) or the lamina's L1-L3 (A2), optionally with a
  resting drive to the medulla's columnar neurons (A3); the whole-brain
  spiking model computes everything after that through the real wiring.
- **B, the published eye model** (`robot/flyvis_eye.py`): flyvis
  (Lappalainen et al. 2024, Nature) computes the optic lobe from the visual
  field; its cell types' activity drives the matching connectome neurons.

## Option A (2026-10-01, Mac, CPU engine, merged brain, calibrated dynamics)

| Variant | Optic lobe, blank -> moving stripes | T4/T5 (motion detectors) | Direction selective? | LPLC2 / HS |
|---|---|---|---|---|
| A1 photoreceptors (rest 10, gain 120 Hz) | 0.02 -> 0.03 Hz | 0 | no | 0 |
| A2 lamina (30, 60) | 1.8 -> 2.6 Hz (L2 30 -> 38, Tm1 4.6 -> 7.6 Hz) | ~0.02 Hz | no | 0 |
| A3 lamina + medulla resting drive (30, 60, 8 Hz) | 3.5 -> 4.2 Hz | ~0.01 Hz | no | 0 |
| A3, strong (150, 300, 37) | 15.8 Hz (Tm1 43 Hz) | ~0.07 Hz | no (left = right) | ~0.03 Hz |

Why A1 fails: photoreceptors inhibit L1 / L2 (histamine), and in the spiking
model those cells rest silent, so more inhibition changes nothing (in the fly
they are graded and respond to the release of inhibition in the dark).
Why A2 / A3 stop at the medulla: T4 / T5 sum many weak inputs from several
medulla types, and direction selectivity needs fast and slow inputs to meet
at the right time; the whole-brain model's uniform, spiking parameters give
neither (the real cells are graded, with type-specific time constants).
Making A compute motion would mean calibrating the optic lobe itself
(weights, thresholds, time constants per type): a research project.

## Option B (2026-10-01, box RTX 3080 Ti, CUDA engine, merged brain, calibrated dynamics)

flyvis 1.2.0, model `flow/0000/000` (details: `option_b_flyvis.md`). Its
cell types drive 60,910 connectome neurons in 51 types; photoreceptors are
not driven (flyvis does that stage).

| | A3, strong | B (harness: 10 ms, gain 100) | B (the robot: 20 ms, gain 50, 2 s baseline) |
|---|---|---|---|
| Optic lobe, blank -> moving stripes | 16.2 -> 15.8 Hz | 0.06 -> 14 Hz | 0.12 -> 6.2 Hz |
| T4a / T4b / T4c DSI (right eye) | ~0 | 0.47 / 0.48 / 0.48 | 0.45 / 0.48 / 0.47 |
| T5a / T5c / T5d DSI | ~0 | 0.28 / 0.51 / 0.21 | 0.21 / 0.48 / 0.23 |
| HS front-to-back (left / right) | silent | 0.81 / 0.74 | 0.63 / 0.69 |
| Looming (LPLC2), small dot (LC10a) | no | no | no |

T4d and T5b are not direction selective in this flyvis model. The harness's
lead-in now matches each stimulus's background (loom and dot sit on 0.8): a
grey-to-bright step at onset had kept the ON pathway and HS (~130 Hz) busy.
B's input adapts to each eye's mean brightness (photoreceptor adaptation):
without it a bright room drives HS ~100 Hz all the time.

**Decision: B.** It computes what the fly's optic lobe computes (motion, by
direction) and the connectome downstream follows it (HS); A would need the
optic lobe itself calibrated (a research project). Neither gives looming or
small-object responses yet.

**On the robot** (`robot/eye.py`, `--eye`; Jetson Orin, live room, camera
and lidar): at the harness's gain (100 Hz), the lidar's flicker and a person
moving at the desk kept LC4 at ~2 Hz and the looming escape neurons
DNp02 / DNp04 at 10-20 Hz: Milo was in "flight" most of the time. Steadied
lidar, a 2 s adapting baseline per neuron and gain 50 bring them to ~1 Hz
(without the eye 0.4 / 0 Hz), mostly walking again; HS still follows motion
(table, last column).
