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
