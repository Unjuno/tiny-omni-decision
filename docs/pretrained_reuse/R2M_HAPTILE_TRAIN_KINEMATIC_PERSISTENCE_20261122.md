# HapTile train-only kinematic-persistence diagnostic — 2026-11-22

## Result and scope

A fixed, parameter-free rule predicted the next coarse end-effector direction
from only the final two XYZ samples in each causal state sequence. On the
1,112 frozen TRAIN examples it matched 858 targets (77.16% sample Accuracy;
76.34% macro Accuracy across the 24 train task groups). Always predicting
`hold` scored 36.24% sample Accuracy and 36.56% task-macro Accuracy. The fixed
rule beat always-hold in 22 of 24 tasks.

This is an **in-sample TRAIN-only task-shape diagnostic**, not held-out
generalization, an OOF result, evidence of a model's quality, or evidence that
video/tactile/text adds value. No parameter or threshold was fitted. The
historical full-readout validation result (63.1% macro Accuracy over five
different task groups) is not directly comparable to these training rows.

## Frozen rule

1. Read the final two causal state observations and their actual timestamps.
2. Extrapolate the XYZ displacement over the fixed 0.5-second difference
   between the protocol's current and target window centers.
3. Apply the existing 5 mm Euclidean deadband and dominant signed-axis label
   function, with x/y/z tie order.

The rule consumes no video, tactile, text or learned representation. Its result
shows that this target can often be predicted by short-term motion persistence
in the observed state. It does not show whether any model or other modality can
generalize that cue to held-out tasks or environments.

## Per-task result

| Task group | n | Persistence Accuracy | Always-hold Accuracy |
|---|---:|---:|---:|
| insert_peg | 50 | 68.00% | 40.00% |
| move_cable | 55 | 94.55% | 89.09% |
| move_plush_toy | 70 | 82.86% | 62.86% |
| move_rubiks_cube | 30 | 80.00% | 63.33% |
| move_toy_car | 30 | 76.67% | 56.67% |
| move_water_bottle | 49 | 81.63% | 20.41% |
| put_apple | 63 | 73.02% | 41.27% |
| put_banana | 42 | 73.81% | 28.57% |
| put_baseball | 29 | 82.76% | 86.21% |
| put_fork | 50 | 72.00% | 20.00% |
| put_golf_ball | 50 | 82.00% | 36.00% |
| put_lego | 60 | 70.00% | 38.33% |
| put_orange | 47 | 57.45% | 31.91% |
| put_spatula | 49 | 65.31% | 16.33% |
| put_stack_glass_cups | 26 | 65.38% | 53.85% |
| remove_cloth | 30 | 73.33% | 16.67% |
| remove_laundry_pod | 60 | 98.33% | 98.33% |
| remove_screwdriver | 50 | 68.00% | 12.00% |
| remove_sugar_bag | 61 | 81.97% | 8.20% |
| stack_bowls | 50 | 74.00% | 4.00% |
| stack_disposable_cup | 51 | 80.39% | 0.00% |
| stack_glass_cups | 30 | 80.00% | 26.67% |
| turn_can | 30 | 56.67% | 26.67% |
| wipe_whiteboard | 50 | 94.00% | 0.00% |

## Reproduction and integrity

- Repository source commit before this diagnostic: `c2df5f48ea633e3dc2da28fb00909abd0e625a8f`.
- Command: `py -3.11 scripts/pretrained_reuse/evaluate_haptile_kinematic_persistence_train.py --train-cache C:\CodexLocalArtifacts\pretrained-reuse-r2\haptile-task-v1\train-joined-v1\decision-features.npz --output C:\CodexLocalArtifacts\pretrained-reuse-r2\haptile-task-v1\train-kinematic-persistence-20261122-attempt-02\result.json`.
- Input cache SHA-256: `4d0b8bfc4d9d279ec04e76811f6a9fc6a8410ceb3d498a33cca9cc0112ba42c2`.
- Input receipt SHA-256: `b0292b9396c8dfa338f915228749aedca8ea76ca8690faf2bb3fdeb41b30ecf4`.
- Executed script SHA-256: `fe19accbc7b1c7b3eae2c24c9ee0c1eb1f52434ff6bea20e20b7139ff50a64b7`.
- Output SHA-256: `290c4ec94d06d401d8be2d30a193368aef23da202f7a1a6998eca91cb78a8a70`.
- Started `2026-11-22T04:04:26.807409+00:00`; finished `2026-11-22T04:04:27.028615+00:00`; elapsed 0.221 seconds.
- Python 3.11.9, NumPy 2.4.6, CPU; CUDA unused; no model was loaded; no weights or corpus were changed.
- Regression coverage: focused test module `7 passed`; full CPU suite `3,271 passed, 11 skipped`; `ruff check .` passed; the new JSON receipt parsed successfully.
- Only arrays for state values, state timestamps, targets, task groups and sample IDs were accessed. Validation, calibration and sealed-audit caches/labels were not loaded.
- Per-example output remains outside Git at `C:\CodexLocalArtifacts\pretrained-reuse-r2\haptile-task-v1\train-kinematic-persistence-20261122-attempt-02\result.json`.

The source remains `REVIEW_NOT_ADMITTED` due to unresolved release-count and
operator-mapping questions. This diagnostic does not change that status, pass
R2-M, or justify quantization or product promotion. Do not continue tuning on
the same exposed validation. Any further use of this source must first resolve
its admission gate and use predeclared task-grouped TRAIN-only controls; a
separate admitted source is still required for broader multimodal evidence.
