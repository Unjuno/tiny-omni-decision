# Fixed role-supervised supplied-option QA comparison

The initial standalone [cover-point protocol](R2_COVER_POINT_PROTOCOL_20261014.md)
failed closed before training: one train clip's cover track begins at8, not0.
Further schema checks found four validation clips without the exact grounded
cover role;14 flags disagree with grounded roles. Do not substitute flags or
invent missing boxes. Preserve all attempts and all original QA examples.

## Fixed before the new training attempt

Train all99 existing videos/198 original questions; evaluate all257 videos/514
original questions, same frozen256×512 TAPIR banks,1536-d candidate text features,
pair/media/encoder revisions and supplied three options. Grounded role labels
never select model inputs. Primary metrics are the original QA Accuracy/NLL/
Brier/ECE15 and condition-pair results. No sealed/test/audit, cloud, quantization,
new base or encoder training. This is observed development, not blind evaluation.

Two arms differ **only** in auxiliary training loss weight0 versus1. Both use
the same shared512→64 point/context encoder, scalar cover membership head,
1536→64 candidate query,64→64 value; attention adds the predicted cover logit.
QA loss is original three-option CE. Available Boolean point-in-frame0-cover-bbox
labels add clip/class-balanced BCE to the shared encoder only in weight1 arm.
Missing/ambiguous roles or frame0 boxes disable that clip's auxiliary loss,
not its QA loss.98 train clips supply auxiliary labels,99 still train QA.
Validation auxiliary membership metrics use253 labeled clips only, explicitly
reported as supplemental coverage diagnosis; all257 remain in primary QA metrics.
Ignore `is_occluder` as a replacement for missing role targets.

Fresh seed17 per arm, identical initial state and sampler. AdamW0.001,
weight_decay0.01, batch8 videos,50 epochs, CPU4/deterministic. Selector is earliest
minimum mean original-question validation NLL; complete all50 epochs and save
best/final separately. No seeds, prompts, options, datasets or hyperparameter sweep.
Compute QA NLL from target log-softmax, preserving extreme errors without flooring
probabilities. Save QA log probabilities with every epoch and control. Supplemental
binary membership NLL uses probabilities clipped1e-12 for reporting only; it never
selects a QA checkpoint. Best/final replay compares against the original saved
epoch predictions on train and validation, not merely two loaded copies.
No runtime source boxes/IDs, GT-match seeding or actor-coordinate heuristic.
This tests whether supervision helps; it does not assume latent target binding
or temporal/viewpoint semantics are solved. Initial-only later-appearance
limitations remain, and no score here qualifies the whole Omni product.

Save per-epoch train/validation QA and supplemental cover metrics, all predictions,
source/module/config/hash/order metadata, untrained/best/final weights and optimizer/
sampler states. Best/final output replay must be exact. Fixed selected controls:
zero state, cyclic other-video state, temporal columns4:128 zero, shuffled options
(equivariance test, not an option-count change). Report full denominators,
train/validation gap, paired video bootstrap10,000/seed17/percentile95% and selection
bias. Official script×participant grouping does not prove participant-disjointness.

First execute a separate-process real-data two-update train-only smoke. Only after
it passes run both fixed50-epoch arms. Reject product adoption without genuine
QA improvement/controls/uncertainty and all other four-modality requirements.

## Pre-run verification

Real-data separate-process smoke03 passed two updates per arm; exact best/final
original-epoch replay and candidate permutation checks passed. Initial states and
sampler orders are identical. Smoke01/02 preserve earlier successful plumbing;
neither measured validation quality. Standalone missing-box failure and two aborted
source-schema inventory attempts remain preserved outside Git. Read-only complete
inventory03 records all356 cases and does not invent labels.

Fresh-context review found one Important NLL flooring issue and two Minor replay/
helper-provenance gaps. Regression tests failed before fixes and passed after;
re-review found no remaining issues. No legacy metric implementation or old result
was changed. Reviewer did not train or inspect sealed data. Full CPU suite/Ruff and
manifest checks are required before the committed-source full run.
