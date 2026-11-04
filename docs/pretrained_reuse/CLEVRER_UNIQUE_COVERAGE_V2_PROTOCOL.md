# CLEVRER unique-scene coverage experiment v2

This protocol corrects the first preflight before any readout fitting. The
previous 512-train/96-validation acquisition attempt used an undersized
exclusion list. Integrity checks found that the proposed training sample
reused prior scenes and that 77 of 96 validation scenes overlapped the
existing validation sample. No model fitting occurred. The partial media
download is preserved and its scene identities are excluded from this run.

## Question and design

At a fixed 2,048 AdamW update budget, does adding 512 previously unused
CLEVRER training scenes to the existing 256-scene pool improve a frozen
VideoPrism + MiniLM Decision readout on 256 unused validation scenes?

Both arms use the same new ordered validation IDs, seed-17 initialization,
frozen feature encoders and readout architecture. The existing 256-scene
training pool is read-only. The comparison is a single-seed development
experiment; validation is used for checkpoint selection and is not a blind
audit.

## Split integrity

- CLEVRER source revision: `98b842082ba4f7c18b6b9e3f39145871782a65ef`.
- Video scene identity is the split group. Train, baseline validation, and
  new validation must be disjoint by scene and SHA-256 media identity.
- The comprehensive historical exclusion manifest is
  `manifests/pretrained_reuse/clevrer-heldout-video-excluded-scenes-20261011.json`.
  The new artifact-root exclusion file additionally excludes every scene in
  the preserved partial download and every validation scene selected in the
  rejected preflight.
- CLEVRER repeats question templates across independent scenes. Exact
  normalized question-template overlap is therefore explicitly permitted for
  this development comparison and counted in the fetch and feature-cache
  reports. No claim of text-content independence is made.
- Every candidate example retains its original scene ID. The official test
  and sealed audit are not loaded.

The first rejected attempt selected 512 training scenes against an exclusion
file containing only 50 train and 12 validation scene IDs. All proposed
additional training scenes overlapped the existing 256-scene training pool;
77/96 proposed validation scenes overlapped the existing 200-scene validation
pool. It downloaded 409 clips before integrity review stopped acquisition.
Those bytes remain preserved in the `...-96val-v1` artifact directory and are
not reused in v2.

## Frozen model and training controls

- VideoPrism `google/videoprism-base-f16r288`, pinned revision and checkpoint
  hash in `configs/pretrained_reuse/clevrer_unique_coverage_v2.json`; 8 frames
  at 288px, frozen mean-spatiotemporal-token features.
- MiniLM-L3, pinned revision and checkpoint hash in the config; frozen text
  features.
- Only the existing shared CandidateScorer is fitted.
- Existing-pool baseline: 256 train scenes / 512 questions.
- Expanded arm: those same scenes plus 512 new train scenes / 1,536 questions.
- New validation: 256 new scenes / 512 questions, 256 per descriptive subtype.
- Both arms: seed 17, AdamW, LR 0.001, weight decay 0.001, batch 8, 2,048
  updates, constant LR, evaluation every 128 updates, no performance early
  stopping. Best and final checkpoints are distinct.
- Select by minimum unweighted macro NLL over static and temporal descriptive
  questions, earliest tie, `min_delta=1e-5`.

The sampler shuffles each full unique question pool per pass and records
consumed counts. The arms have the same update budget, not the same number of
passes or per-example exposures.

## Reporting and limits

Report macro and per-type Accuracy, NLL, Brier, and 15-bin ECE; paired
scene-cluster bootstrap intervals with 10,000 resamples and seed 17; source,
sample, feature and effective-config hashes; unique scene/media coverage;
template overlap; compute time and peak VRAM; per-example predictions; and
save/reload parity. A positive result supports only the tested training-pool
change for this scorer and development validation. It does not establish
general video reasoning, multimodal product readiness, or the 95–98% objective.

No prior corpus, feature cache, model checkpoint, or run directory is
modified. New artifacts are written under
`C:/CodexArtifacts/pretrained-reuse-r2/clevrer-coverage-512train-256val-v2`
for acquisition/features and the separate
`...-attempt-05` directory for the completed training run. The initial
selector/config and bootstrap defects, retained failed attempts, complete
metrics, and interpretation are documented in
[`CLEVRER_UNIQUE_COVERAGE_V2_RESULTS.md`](CLEVRER_UNIQUE_COVERAGE_V2_RESULTS.md).
