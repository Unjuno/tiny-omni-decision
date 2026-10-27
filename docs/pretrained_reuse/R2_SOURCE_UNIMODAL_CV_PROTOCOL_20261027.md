# Source-candidate matched unimodal baselines: train-only protocol

Use only the already frozen 2,593 official-train candidate records / 320 video
groups and cached VideoPrism and EfficientAT features. Do not access the old
development labels/features, official validation predictions, test or sealed
audit. Three folds: sort video IDs by SHA-256 of `source-cv-v1:17:<video_id>`
(UTF-8), assign sorted IDs round-robin to folds. All records from a video stay
together. This is a new internal development diagnostic, not an independent
test: all source videos are from one upstream dataset and group identity does
not prove participant or perceptual independence.

Fit three freshly initialized, equally sized readouts per fold: audio only,
video only and joint. The same input masks apply in training and evaluation.
Each readout uses trainable audio/video projections and a shared candidate
scorer, but a **separate learned none-visible scorer** receives the audio and
mean visual state. In audio-only mode all visible candidates have the same
score, reflecting that no candidate-specific visual information is supplied;
the separate none score can still differ. In video-only mode audio is zeroed
for the entire fit. No identifiers or labels enter features. Masked inputs and
parameters are not evidence of both-input necessity.

Freeze seed 17, 7 full epochs (the historical best epoch, chosen on old
development data), batch 32, AdamW LR 0.001, weight decay 0.0001, gradient
clip 5. Same fold, initialization seed, and epoch permutations for all arms.
Use the final 7th-epoch weights; no new checkpoint selection, threshold or
hyperparameter tuning. Save per-fold metrics and predictions, parameter counts,
input/code hashes and separate checkpoints outside Git. Evaluate all folds and
report group-level paired uncertainty and visible/none strata if feasible.
No model promotion, independent quality claim, or 95% claim follows.
