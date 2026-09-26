# Fly exam, 2026-09-25 (RTX 3080 Ti, CUDA engine, dt 0.1 ms)

    python -m cognition.exam --dynamics <published|calibrated_v2|calibrated> --controls --workers 6

| dynamics | passed | shuffled wiring | robustness AUC |
|---|---|---|---|
| published (Shiu et al. 2024) | 17/32 | 10/32 | 0.97 |
| calibrated v2 | 27/32 | 11/32 | 0.87 |
| calibrated v3 (current default) | 32/32 | 10/32 | 0.95 |

Files: `exam_<published|calibrated_v2|calibrated_v3>_real.txt` (report) and `.json` (every test, control and
perturbation). The v2 run used `--dynamics calibrated` before v3 became the
default; its settings are in `data/metadata/dynamics_calibrated_v2.json`.
Narrow passes in v3: symmetry_odour 0.616 (limit 0.6), odour_lateralization
0.036 (limit 0.01). See data/metadata/dynamics_calibrated.json for what each
change is and its source.
