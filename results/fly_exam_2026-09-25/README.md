# Fly exam, 2026-09-25 (RTX 3080 Ti, CUDA engine, dt 0.1 ms)

    python -m cognition.exam --dynamics <published|calibrated_v2|calibrated> --controls --workers 6

Final run after an independent code review (fixes in commit b356400 and later).

| dynamics | passed | held-out tests | constraints | fit tests | shuffled wiring | robustness AUC |
|---|---|---|---|---|---|---|
| published (Shiu et al. 2024) | 16/32 | 10/21 | 3/3 | 3/8 | 9/32 | 0.97 |
| calibrated v2 | 26/32 | 19/21 | 3/3 | 4/8 | 10/32 | 0.87 |
| calibrated v3 (default) | 32/32 | 21/21 | 3/3 | 8/8 | 9/32 | 0.95 |

Test roles (cognition/exam/tests.py ROLES): "fit" = used to set or choose a
calibration value; "constraint" = calibration had to keep it passing;
"held_out" = never used in calibration. Files: exam_<published|calibrated_v2|
calibrated_v3>_real.txt (report) and .json (every test, control, perturbation).

History: a first run the same day gave v3 32/32 before the review; the review
found that two left/right corrections compounded, that the compass test passed
by construction, and that the odour-steering tests (4 trials) measured noise.
After the fixes v3 scored 30/32; the one fitted value (ORN depression) was refit
by the same criterion (the measured ORN->PN transform) and the odour tests were
given 16 trials for every model, giving the table above.

## dt 0.2 ms (the Pi's time step), 2026-09-26

    FLY_DT=0.2 python -m cognition.exam --dynamics calibrated --workers 6

Calibrated v3 with the integration step doubled (the setting the Pi 5 uses to
run near real time): 32/32 (held-out 21/21, fit 8/8, constraints 3/3),
robustness AUC 0.94 (0.95 at dt 0.1; the difference is the w_syn x1.15 / x1.3
levels, 0.8 instead of 1.0). Nothing was re-tuned for dt 0.2. Files:
dt02/exam_calibrated_real.txt / .json (no shuffled-wiring controls).

## ORN depression constants (verified 2026-09-26)

Nagel, Hong & Wilson 2015 (Nat Neurosci 18:56, PMC4289142), Fig. 1c legend:
the single-component depression model fitted to 10 Hz antennal-nerve trains in
DM6/VM2 PNs has f = 0.78 (fraction of resource left after a spike, the same
meaning as our release_f) and tau = 893 ms. The paper itself notes this model
makes PN odour responses too transient; in vivo, a slow component (r = 0.0073
per spike, tau 33 s) and presynaptic GABA inhibition keep responses sustained.
Our v3 keeps tau = 893 ms and uses f = 0.96, FITTED by us to the Olsen 2010
ORN->PN transform (with 0.78, PN odour responses collapse, as the paper's
caveat predicts). 0.96 is an effective value, not a literature value.
