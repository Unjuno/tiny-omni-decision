# HapTile fixed kinematic-persistence validation diagnostic — 2026-11-22

## Result and scope

Applied the already fixed, parameter-free constant-velocity rule to the frozen
169-example HapTile validation cache. The rule used only the last two causal
XYZ positions, projected by 0.5 seconds, with the existing 5 mm deadband and
direction-label rule. It predicted 122/169 examples correctly: 72.19% example
Accuracy and 72.29% macro Accuracy across the five held-out task groups. An
always-`hold` baseline scored 37.87% example Accuracy and 40.60% task-macro
Accuracy; the persistence rule beat it in all five groups.

For context, the already selected full video+tactile+state+text readout scored
61.54% example Accuracy and 63.13% task-macro Accuracy on these same groups,
with validation macro NLL 1.120 at epoch 19. The fixed kinematic rule exceeded
its macro Accuracy by 9.16 percentage points. Per-task macro Accuracy was:

| Validation task group | n | Fixed persistence | Selected full readout |
|---|---:|---:|---:|
| `fold_tshirt` | 30 | 76.67% | 53.33% |
| `move_can` | 30 | 86.67% | 83.33% |
| `put_spoon` | 49 | 71.43% | 48.98% |
| `put_strawberry` | 30 | 63.33% | 86.67% |
| `turn_cleanser_bottle` | 30 | 63.33% | 43.33% |

This is a post-hoc diagnostic on development validation that had already
participated in arm/checkpoint selection. It is not an independent estimate,
blind test, or basis for selecting another checkpoint. The fixed rule was
derived from the prior TRAIN-only diagnostic and fits no parameter on this
validation split. Its strong result indicates that this particular target is
often predictable from short-term robot motion alone; it weakens the case that
the task is a clean test of learned cross-modal complementarity. The model's
previous validation-selected gains and bootstrap intervals remain
selection-affected. No paired confidence interval is claimed for this
post-hoc comparison.

HapTile has no audio. The source remains `REVIEW_NOT_ADMITTED` because source
release-count and operator-mapping questions remain unresolved. This diagnostic
does not pass R2-M, establish product quality, or change source admission.
Sealed-audit data were not accessed. No fitting, calibration, checkpoint
selection, quantization, or GPU work was performed.

## Reproduction and integrity

- Repository: `Unjuno/tiny-omni-decision`
- Source commit before the diagnostic: `c0766f519755ef9383d39fe683783414fef95153`
- Command:
  `python -m scripts.pretrained_reuse.evaluate_haptile_kinematic_persistence_validation --validation-cache C:\CodexLocalArtifacts\pretrained-reuse-r2\haptile-task-v1\validation-joined-v1\decision-features.npz --output C:\CodexLocalArtifacts\pretrained-reuse-r2\haptile-task-v1\validation-kinematic-persistence-20261122-attempt-01\result.json`
- Input cache SHA-256: `ed874a01274e23e6ac8da4645f5bc2a46aca1422df7ca3145c8ddb33c797fa44`.
- Executed validation runner SHA-256:
  `9d4d375a1c7bd70e8713c3a34b611fb04234cbf2403322c6be5765f85076f5cd`.
- Imported fixed-rule helper SHA-256:
  `fe19accbc7b1c7b3eae2c24c9ee0c1eb1f52434ff6bea20e20b7139ff50a64b7`.
- Output SHA-256:
  `533220d65f61cc6db68519abb9f1c6cd2efdc1b49c49d45fde19ed1153454339`.
- Output remains outside Git at
  `C:\CodexLocalArtifacts\pretrained-reuse-r2\haptile-task-v1\validation-kinematic-persistence-20261122-attempt-01\result.json`.
- Python 3.11.9, NumPy 2.4.6, CPU; 0.044 seconds; no GPU or paid compute.
- Focused CPU tests cover split-path rejection, pinned validation hash/split
  requirements, sealed-audit provenance rejection, and fixed-rule behavior.

## Next implication

Do not spend more capacity on this same exposed validation. For a useful
multimodal Decision task, freeze a new task where a simple causal state/history
rule is an explicit baseline and where the target genuinely requires the
additional stream. Resolve source identity/admission first, then evaluate on
new task groups with predeclared unimodal and paired-input controls.
