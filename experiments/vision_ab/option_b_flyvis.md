# Option B: flyvis drives the connectome's optic lobe

`robot/flyvis_eye.py` (FlyvisEncoder). flyvis 1.2.0 (MIT; Lappalainen et al. 2024),
pretrained model `flow/0000/000`, data in `~/milo/models/flyvis` (`FLYVIS_ROOT_DIR`).

Run (box):

    PYTHONPATH=. FLY_DATASET=merged FLY_DYNAMICS=calibrated FLY_DT=0.1 \
      FLY_NATIVE_LIB=$PWD/native/liblif_cuda.so FLYVIS_ROOT_DIR=~/milo/models/flyvis \
      .venv/bin/python -m experiments.vision_ab.harness --kinds B --out results/vision_ab

Mapping: both eyes as a batch of two through flyvis (10 ms Euler steps); flyvis hex
(u, v) -> direction: x = 5.8 v, y = 5.8 (u + v/2); right eye az = 55 - x, left eye
az = -55 + x, elevation = -y (orientation measured from T4/T5 preferences). Each
connectome neuron of a flyvis type (R1-R8 excluded; CT1(Lo1) -> CT1, Am -> Am1,
TmY9 -> TmY9a/b; Mi3, Mi11, Mi12, Tm28 absent from the merged brain) takes its
type's activity at the nearest flyvis column of its own eye; direction from the
column map, else the synapse-weighted mean of its column-assigned inputs; > 1.5
columns from the lattice -> no drive. Rate = 100 Hz x clip((a - a_grey) / s_type, 0, 1.5),
s_type = 99th percentile of the type's response above grey to 0.6 s of calibration
gratings. 60,910 neurons driven, 51 types.

Results (2026-10-01, box RTX 3080 Ti): results/vision_ab/B.json.
flyvis's own T4/T5 DSIs on the harness gratings (model 000): T4a/b/c ~0.47,
T5c 0.46, T5a 0.26, T5d 0.20, T4d and T5b ~0 (this model's T4d/T5b are not
selective; other ensemble members may be).
Caveat: loom and dot stimuli sit on a 0.8 background (grey = 0.5): the brighter
sustained background drives ON-pathway types above their grey level, and HS fires
~130 Hz on all of them; loom/recede and LC10a scores are not meaningful under B yet
(LPLC2 < 0.1 Hz, LC10a 0 Hz).
Cost: 0.72 s wall per simulated s (no-vision baseline 0.105); flyvis stepping
0.47 ms/step on GPU (~0.05 s per sim s), rates_hz incl. panorama sampling 2.9 ms
per 10 ms; flyvis on CPU (i9-9900K) 8.7 ms/step, no gain from threads.
Since then (robot/eye.py, for the Jetson): sampling as a sparse matrix (0.16 ms),
the input built on the GPU in one copy (7.8 -> 0.4 ms on the Orin), the step as
a CUDA graph (Orin: 1.6 ms of GPU), light adaptation, an optional adapting
baseline (FLY_EYE_BASELINE_S), the step and gain as settings (FLY_EYE_DT,
FLY_EYE_RATE_MAX). The harness's defaults are unchanged (10 ms, gain 100, no
baseline); the loom / dot caveat above is fixed in the harness (lead-in at the
stimulus's background).
