# R2 X-Capture Path A frozen-feature result — 2026-11-28

## Decision

The CPU Path A run completed 1,000 optimizer updates and its saved best
checkpoint reloaded with identical validation predictions. It does **not** show
useful generalization for this retrieval task: the selected 100-way validation
accuracy is 2.33% (uniform chance 1%), and the score deteriorates sharply after
step 20 while train retrieval accuracy approaches 99%. This is a clear
train/validation generalization failure for this setup, not evidence that a
multimodal Decision model is impossible or that training merely needs more
steps. Do not promote this readout or claim general-purpose multimodal quality.

The result is valuable as a data-and-task diagnostic. X-Capture provides
repeated image, impact-audio, and tactile observations for physical objects,
but this Path A adaptation asks a model trained to associate *seen object
identities* to retrieve *unseen object identities* across capture points. The
observed gap suggests the current features, label construction, or available
cross-object supervision do not teach a transferable correspondence. It does
not test a text task, video, or a user-defined decision benchmark. BERT is
therefore not a remedy for this experiment; BERT-family work belongs on a
separate, admitted text task with source-disjoint labels and a fixed validation
protocol.

## Frozen data and provenance

- Dataset: X-Capture at pinned revision
  `b4468bf999aac250f34ef1b02459d902492a068f`; the screened source is allowed
  for this narrow cross-sensory object-identity retrieval study.
- Split: seed-17 object groups; 400 train objects / 800 capture points and 100
  validation objects / 200 capture points. Each object has exactly two
  distinct points. Train/validation object identity overlap is zero.
- Evaluation: 100 evaluation objects were excluded during the split plan;
  no evaluation media or records were loaded by the feature or fit scripts.
- Acquisition: 4,000 files for the 1,000 train/validation points, rehashed
  after retrieval. The full archive and sealed/evaluation media were not
  needed for this run.
- Deterministic point-selection plan SHA-256:
  `89ed14c08929e09c93f5e7af7d83cf65777645bda40f577caba15a6fcdc3ef3e`.
- Acquired sample receipt SHA-256:
  `8432848b8889dae896e376008a7cd804e8d23815f3d8e86e8dee7b9e946d5c2f`.
- Frozen feature receipt SHA-256:
  `74ca28bcb460563be27c801af3aaf6295734624cfdd6372c5f23599f73a6b403`.
- Feature arrays are 1,000 × 512 image, 1,000 × 384 audio, and 1,000 × 512
  tactile vectors; all were finite. Image and tactile use the same frozen
  TinyCLIP visual tower; audio uses frozen EfficientAT. The encoders were not
  trained.
- Feature array SHA-256 values: image
  `8d5f85be1232e944eb432007bf7ba9d064c7114e2ef72543bea69dcb1e56f7a0`, audio
  `38687cca848cba6a99fdc2b7185212b9995c4fc0dcfe1322a76deac48c33ee53`, tactile
  `830eed436ab7e7ea4cb1dd39b97dd5f84726a22801817bd9b35ae5b5119cbfa9`.
- Mask review: all 1,000 masks were nonempty and binary. Coverage ranged
  0.4108%–97.171% (median 16.788%); reviewed low/high coverage examples were
  plausible localized contact regions or objects filling the frame. Explicit
  crop thresholds were 0.3%–98%; no point was dropped. At least 5/20 crops in
  the earlier bounded visual screen contained readable package/brand marks.
  This is a documented shortcut risk; no claim of corpus-wide shortcut-free
  data is made.
- Source commit reported by the run: `f31213566165c62bb4ed73c11e1f95308b25f2bb`.
- Fit script SHA-256 at execution:
  `723dbd51730a8cfe905c310e29ae32dcd6fbd5ad4b1af9079be5fde7f6334376`.
  (The fit script was a worktree addition at run time; the source commit alone
  does not contain it.)

## Model and protocol

- Path A is a trainable three-modality projection lens over fixed features;
  there is no LoRA. The pretrained encoders and source media remain frozen.
- Trainable parameters: 180,225 (three bias-free projections to 128 dimensions
  plus one learned logit scale).
- Seed 17; AdamW, learning rate 0.001, weight decay 0.0001, global gradient
  clipping at 1.0; 1,000 optimizer updates; validation every 20 updates.
- Training objective: symmetric cross-point InfoNCE over all three unordered
  modality pairs and both point directions, with 400 train-object identities
  in the contrastive batch.
- Selector: lowest validation macro NLL over six ordered cross-modal retrieval
  directions; earliest tie wins. It selected step 20. The final step is
  preserved separately and was not substituted for the selected checkpoint.
- Evaluation: six ordered 100-way retrieval directions (200 queries per
  direction, two point directions), plus three two-modality fused-query probes.
  Per-query probabilities and IDs are saved. Candidate-permutation and
  uniform-probability controls are included. These are development diagnostics,
  not an independently held-out benchmark.
- Runtime: Python 3.11.9, PyTorch CPU execution, 174.44 seconds, cloud cost $0.
  GPU remained outside this run; existing Orbitune work was not interrupted.

## Learning curve

Macro metrics average the six ordered cross-modal retrieval directions. Each
validation direction has 200 queries and a 100-object candidate gallery.

| Step | Train loss | Train macro accuracy | Validation macro accuracy | Validation macro NLL | Validation Brier | Validation ECE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.0845 | 0.21% | 0.92% | 4.7074 | 0.9927 | 0.0218 |
| 20 (selected) | 5.5083 | 4.56% | 2.33% | 4.6149 | 0.9908 | 0.0341 |
| 100 | 2.9288 | 56.60% | 2.00% | 5.0714 | 1.0165 | 0.1087 |
| 200 | 1.7675 | 81.06% | 2.17% | 5.2666 | 1.0251 | 0.1245 |
| 400 | 1.0158 | 91.44% | 2.17% | 5.6129 | 1.0436 | 0.1600 |
| 600 | 0.5417 | 97.15% | 1.92% | 5.7659 | 1.0548 | 0.1828 |
| 800 | 0.3378 | 98.15% | 1.50% | 6.0219 | 1.0756 | 0.2149 |
| 1,000 | 0.1944 | 98.83% | 2.00% | 6.2753 | 1.0957 | 0.2423 |

This is strong evidence of overfitting under the chosen object-identity split:
training retrieval improves continuously, while validation NLL, Brier, and ECE
degrade after the early selected step. The selected checkpoint is only modestly
above chance; several individual directions remain at or near chance. The
validation curve was used for selection, so these metrics are not blind test
estimates.

Selected-step validation accuracy by direction:

| Direction | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Image → audio | 4.00% | 4.4654 | 0.9856 | 0.0232 |
| Image → tactile | 3.00% | 4.5815 | 0.9899 | 0.0276 |
| Audio → image | 3.00% | 4.5296 | 0.9897 | 0.0364 |
| Audio → tactile | 0.50% | 4.7447 | 0.9938 | 0.0400 |
| Tactile → image | 1.50% | 4.6425 | 0.9932 | 0.0579 |
| Tactile → audio | 2.00% | 4.7256 | 0.9928 | 0.0193 |

Fused-query validation Accuracy was 1.50% for image+tactile → audio, 1.00% for
image+audio → tactile, and 5.00% for audio+tactile → image. The fixed candidate
permutation control was 0.50%–1.50% across directions. The tiny selected
improvement above chance does not establish robust cross-modal grounding.

## Artifacts and checks

The run used a new, external output directory and kept the failed earlier
attempt unchanged:

`C:\CodexArtifacts\pretrained-reuse-r2-local\xcapture-path-a-seed17-20261128-attempt-02`

It contains `best-validation.pt`, `last.pt`, `config.json`,
`training-history.json`, `result.json`, and `result.sha256`. The selected
checkpoint SHA-256 is
`9f4fbf5fcfd7130b44e6d4dec88e4fc3f9a7698b03c9a8ce74ea390b9d5645c0`; the final
checkpoint SHA-256 is
`5794208b60866ac9797c2000c1b1ee843295b9400e603ef7ea56e82c53bc07b5`.
The result JSON SHA-256 is
`c1e7fb9b144717e2b82c3db8291907169d70d83a522e96cb32266a8f7673c629`.
Best-checkpoint reload reproduced the selected NLL and all saved validation
predictions exactly. Trainable count and output tensors were checked; the
evaluation partition remained unopened.

Focused tests: 18 passed. Ruff and `git diff --check` passed before report
finalization. Full repository tests and manifest validation remain to be run
before commit/push.

## What the result says about data and next steps

The source media and split controls were successfully verified, but the model
did not learn transferable object correspondences from this formulation. The
large train/validation gap could arise from limited distinct objects, weak
semantic alignment between modalities, imperfect object crops or labels, or a
task whose unseen-object identity target is not supported by the available
supervision. These hypotheses are not distinguished by this one run. More
updates are not justified as the next action because validation deteriorated
for nearly the entire run.

Before another fit, define a task that represents a useful decision over
attributes shared across modalities, then audit that each label is observable
in the intended inputs and that split groups exclude duplicate objects,
captures, and source-level leakage. Add a blinded human data-quality review and
explicit shortcut controls (including text/mark occlusion) before selecting a
readout. Do not broaden to a BERT sweep here: this corpus has no text question
or language label, and BERT cannot fix audio/tactile correspondence. A
separate, rights-cleared text task can compare a compact BERT-family model
against existing frozen text encoders; existing Japanese BERT results remain
task-specific reference evidence, not a substitute for this missing task.

This experiment does not meet the product target (≤200 MB quantized package,
95–98% task-specific quality across modalities), and it provides no evidence
for video or text. R2 and the overall Pretrained World Representation Reuse
goal remain open.
