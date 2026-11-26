# Training-only slot centroid recoverability

Question: does the stored 128-dimensional slot representation retain linearly
accessible spatial information despite not explicitly appending mask coordinates?
This is a diagnostic of representation, not another Decision model candidate.

Used the existing first eight train videos, taking only exactly intersecting
mask/feature frame indices: 18 frames, 126 slots. Seven videos fit each fold and
the eighth is held out. No question targets, development set, or audit was loaded.
Checkpoint, loader, seed, media manifest and source hashes agree between extraction
protocols. Exact frame and slot indices are matched. Separate extraction outputs
have not been verified together in a joint forward: retain that limitation.

## Fixed method and result

For each nonnegative finite 64x64 soft mask with positive mass, compute x/y
centroids on a [0,1] coordinate grid, weighted by mask mass. Use the 128 feature
coordinates (exclude time). Center predictors and targets on the seven fit videos.
Solve `W = solve(Xc.T @ Xc + 0.001 I, Xc.T @ Yc)`, with unpenalized intercept.
Predict the held-out video's centroids without clipping. No normalization search,
coefficient tuning, seed search or checkpoint selection. Compare to the fit-target
mean. Score mean squared error over both coordinates, then equal-weight videos.

| Diagnostic | Equal-video MSE |
|---|---:|
| Linear ridge readout | .023801654 |
| Fit-only mean centroid | .050247350 |

Ridge improves all eight video folds. Adjacent JSON preserves every predicted and
target coordinate, per-video scores, intersecting frame IDs and input hashes.
Local output: `C:/CodexArtifacts/pretrained-reuse-r2/slot-centroid-recoverability-01`.
CPU NumPy float64 solve; no encoder inference or checkpoint modification.

## Interpretation

This supports some cross-video linear accessibility of the model's own spatial
mask information. It contradicts treating omitted explicit coordinates as proof
that the slots contain no position information. It does not demonstrate ground
truth localization, instance identity, occluded-object tracking, or Decision
generalization. Sparse intersections, only eight known training videos and model
generated targets limit the conclusion. Do not promote an architecture or launch
another full fit from this result. The useful next distinction is correspondence
and condition readout versus complete loss of spatial information; the latter is
not supported by this diagnostic. Existing failed Decision results remain valid.

## Slot-index confound control

Same saved targets and leave-one-video-out folds; predict each slot's centroid
using that slot index's mean over frames in the other seven videos. No test-video
coordinates enter the mean. This tests whether consistent slot-index location
bias alone explains the feature readout result (the initialization seed was fixed).
Equal-video MSE is .054323103, compared with .023801654 for the feature readout.
Features improve every fold over this index-only control. Full predictions and
source-result hash are in `R2_SLOT_CENTROID_INDEX_CONTROL_20261013.json`.

Reproduce by reshaping each fold's saved targets to [frames,7,2], concatenating
the other seven folds on the frame axis, averaging that axis and broadcasting
the resulting [7,2] predictions to the held-out frames. Average squared error
over frames, slots and coordinates, then average the eight video errors.
This is a post-hoc training diagnostic, not an independent validation result.
It weakens this specific fixed-index confound; it does not resolve separate-run
correspondence, object identity, ground-truth localization or Decision quality.

## Actual decoder correspondence check

Loaded the pinned 93,530,201-parameter checkpoint strictly on CPU FP32 and decoded
all 18 intersecting frames directly from their saved 128-dimensional slots.
Reshaped each decoder mask to 37x37 and bilinearly resized to 64x64 with
align_corners=False, matching the original mask cache. All 18 frames pass
atol=1e-4, rtol=1e-4 against the separate CUDA extraction. Maximum absolute
mask difference is 3.755093e-6. All checkpoint tensors remain exactly unchanged.
This resolves the separate-extraction correspondence concern for these 18 frames;
it does not verify the entire corpus or turn model masks into ground truth.

Decode/check time .892491 CPU seconds excludes model loading. No GPU job launched.
Result: `R2_SLOT_MASK_CORRESPONDENCE_20261013.json`; local directory
`C:/CodexArtifacts/pretrained-reuse-r1/videosaur-slot-mask-correspondence-01`.
Reproduction harness:
`C:/CodexArtifacts/pretrained-reuse-r1/videosaur-source-audit-01/check_slot_mask_correspondence.py`,
SHA256 da4e2d46aab3af6bf60dfc228dca2689126f3d6d592318fd7823ca697213db97.
It checks the pinned loader hash, uses a fresh output directory, loads all weights
strictly, and reports every frame including failures. Torch2.6.0+cu124, NumPy2.4.6.
The fixed slot input plus pretrained decoder can recover its masks without
re-encoding raw video; any future geometric readout must still account for decoder
parameters and demonstrate task quality before adoption.
