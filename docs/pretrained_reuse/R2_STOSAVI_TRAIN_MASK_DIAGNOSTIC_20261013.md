# StoSAVi natural-video train diagnostic — 2026-10-13

**Completed; do not expand this checkpoint into a full Decision fit.** Strict
runtime admission did not establish useful natural-video object identity.
This bounded diagnostic supplies unfavorable evidence for direct reuse of the
CLEVRER-trained masks on the existing natural-video task. It does not establish
a ceiling for all object-centric models, latent readouts or additional adaptation.

## Fixed experiment

First eight entries of the immutable existing train media manifest; no video
removed. Seed17 reset per video, all consecutive frames before its exclusive
cut (full clip if cut=-1), chunks of six carrying posterior slots forward without
reset. CPU FP32/four threads, same loaded frozen StoSAVi, same corrected tensor
normalization/resize as [runtime admission](R1_STOSAVI_RUNTIME_20261013.md).
No additional training, validation, audit set, question answers or cloud compute.

Decode masks only at exact unmasked marked-object annotation frames. Pixel-center
rasterization of normalized boxes; argmax over the seven decoder masks. Select
highest mask/box IoU at frame0 (first slot breaks ties), then keep that slot ID.
Controls: per-frame best slot using the correct box (oracle upper comparator),
and the unchanged initial annotation box. All conditions use identical frames.

These are **oracle-initialized box geometry diagnostics**, not Decision accuracy,
segmentation-mask ground truth or a fair tracker benchmark. Box overlap penalizes
masks that capture only part of an object; the static box has privileged geometry.
Nevertheless, oracle reassignment remains weak on this sample, so identity swaps
alone are not an adequate explanation. Reconstruction-mask weakness does not
prove that every possible latent readout lacks information.

## Measured results

| Train video | Later annotation points | Fixed-slot IoU | Oracle-rematched IoU | Static-box IoU |
| --- | ---: | ---: | ---: | ---: |
| video_10103 |15|.0000|.0464|.5927|
| video_10186 |21|.0634|.1992|.1717|
| video_10212 |35|.0440|.1961|.0278|
| video_10387 |18|.0119|.0249|.2642|
| video_1087 |19|.1457|.2808|.5223|
| video_10886 |20|.0494|.2487|.2637|
| video_10894 |18|.1555|.2529|.2759|
| video_10903 |18|.0227|.0858|.2451|
| Pooled points |164|.061294|.174399|.260975|

8 videos, 4,979 frames;124.456seconds summed sequential video processing, including
media hash/decode, inference and masks, excluding initial model loading. One CPU
process; no GPU work. No claim of population significance from eight ordered
train examples, and no seed search. Complete weight state remained unchanged.

## Reproduction and safeguards

Output: `C:/CodexArtifacts/pretrained-reuse-r1/slotformer-stosavi-train-masks-01/`.
Contains protocol written before extraction, per-video compressed masks/frame IDs,
progress, result and exact replay. Imported source/config/checkpoint hashes match
runtime admission. Extraction harness is
`C:/CodexArtifacts/pretrained-reuse-r1/slotformer-source-audit-01/train_masks_diagnostic.py`;
its hash is in protocol. Harness runs via `runpy` from repository root, uses
exclusive-create output and the previously hashed isolated runtime loader.

Protocol SHA256 `63d824fa43918a63bbcb01ccb61fe47116ff89483334b51443634cf0badbff86`.
Result SHA256 `ec68f75532392c8b6cad573324cf5d301333f4e8db2bbbdd8b06bb3e07db94ab`.
Small replay summary is included beside this report. Raw masks and weights stay
outside Git. Parent source commit70d3cd7 plus the new hashed metric helper are
recorded here; this commit records that helper and its regression tests.

```powershell
python -m scripts.replay_slot_mask_diagnostic --run C:/CodexArtifacts/pretrained-reuse-r1/slotformer-stosavi-train-masks-01 --annotations C:/CodexArtifacts/pretrained-reuse-r2/perception-test-source-audit-01/train_annotations.zip --output C:/CodexArtifacts/pretrained-reuse-r1/slotformer-stosavi-train-masks-01/replay.json
```

Replay verified exact per-frame results with unchanged annotation, mask and helper
hashes; requires a fresh output filename if repeated. Unit tests distinguish fixed
identity from oracle reassignment, pixel-center geometry, missing initial boxes
and nonfinite rejection. Ruff passed;670 CPU tests passed,1 skipped; CI-equivalent
model/dataset manifest and catalog commands passed. Manifest validity is not new
license approval; checkpoint rights remain REVIEW.

## Decision

Do not spend a full feature-cache/readout training run on this checkpoint now.
The CLEVRER object extractor's synthetic-to-natural transfer has not earned that
cost. A useful next object-centric candidate needs pretrained natural-video
coverage and verified rights/runtime, or a separately justified identity mechanism
using the already-tested natural-video tracker. Do not claim the SlotFormer
forecasting module repairs these observations: it has not been tested, and consumes
slots from this extractor. Existing artifacts and broader task quality gates remain
unchanged; R2 quality and the full R0–R8 objective are unfinished.
