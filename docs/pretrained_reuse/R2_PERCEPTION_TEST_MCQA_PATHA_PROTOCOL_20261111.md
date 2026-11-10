# Perception Test MCQA Path A protocol — 2026-11-11

## Scope and decision

This is a development-only, task-specific **video-plus-text multiple-choice
Decision** experiment. It asks whether a small learned reader can reuse a frozen
video representation together with a frozen text representation. It does not
measure audio/video complementarity, participant generalization, or a general
Omni Decision capability. The official Perception Test test split remains
sealed.

The selected public-validation candidate pool contains 1,518 video groups and
2,886 questions. The corrected artifact-path audit found no direct media,
feature, prediction, or selected-cohort path match for these groups; exact
normalized question-plus-three-choice overlap with train was removed. This is
stronger evidence than a source-index mention, but it does **not prove** that a
person or model has never seen these clips or that scripts are independent.
Participant and script-variation IDs are absent. Results must be reported as
development-candidate validation, not blind or independently untouched
generalization. The pool and its selection hash are frozen; do not select a new
cohort after inspecting predictions.

## Frozen data

- Source: `google-deepmind/perception_test` at
  `3938d2f1ba3a6b502025741cea4cd73c7b3bdfaf`; annotations and media declared
  CC BY 4.0.
- Train: 7,392 decisions / 1,955 video groups.
- Validation candidate: 2,886 decisions / 1,518 video groups.
- Every question from a video stays in that video's source split. The official
  train/validation video-ID overlap is zero. Validation exact normalized
  question/choice overlap with train is zero.
- Frozen annotation archive hashes and record-order hashes are in the
  external corpus manifest. The exposure report SHA-256 is
  `8da8b59847ddad68da09e029735a248661fb1d68ac95a8442aa2c10b452e076b`;
  final candidate video-ID list SHA-256 is
  `af673f6282598366058d42df62b43a522d17d13766aca6e13e4d8a3a94c08382`.
- Reasoning mix is descriptive-heavy: descriptive 2,281; explanatory 295;
  predictive 93; counterfactual 217. Report each type separately.
- Media identity and semantic/template-level duplicates across unrelated
  projects are not fully provable from source IDs. Do not claim those gates
  passed beyond the recorded checks.

## Frozen encoders and reader

Use the locally verified OpenGVLab VideoMAE V2 Small checkpoint because the
previously selected VideoPrism TensorFlow/JAX path is now blocked by Windows
Code Integrity (`_pywrap_tf2.pyd`, Policy ID
`0283ac0f-fff1-49ae-ada1-8a933130cad6`). Do not alter the pinned environment or
work around that policy. VideoMAE is a separate already-audited candidate, not a
claim that it outperforms VideoPrism.

- VideoMAE V2 Small: 22,153,286 loaded parameters, checkpoint SHA-256
  `24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172`, Hub
  revision `706cc172d65ebd4dedbee3f9c0183a93df9fa125`, source commit
  `29eab1e8a588d1b3ec0cdec7b03a86cca491b74b`, Apache-2.0.
- Use 16 uniformly spaced endpoint-inclusive RGB frames, 224-pixel full resize,
  `/255`, then channel mean/std `0.5`, matching the existing real-video runtime
  audit. Preserve final-block tokens as `[8, 14, 14, 384]` using the already
  verified final-block hook; do not collapse them before the query reader.
- Frozen text encoder: existing local MiniLM-L3, revision
  `4ca70771034acceecb2e72475f72050fcdde4ddc`, weight SHA-256
  `cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433`,
  17,389,824 parameters. Encode question and each supplied option separately.
- Train only a small query-conditioned video-token reader and shared
  candidate scorer. Encoders stay frozen. Record exact parameter counts,
  adapter/readout bytes, package bytes, VRAM, and merge/package implications.

The encoder subtotal is 39,543,110 parameters before the learned reader. This
is a task-specific component allocation, not the complete multimodal product
count.

## Fit and controls

1. Keep the publisher's three options and answer index. All choices for a video
   remain in one split.
2. Encode exactly `Question: {question}\nCandidate answer: {option text}` for
   each choice using MiniLM masked mean pooling (max length 256). Remove only
   the publisher's `A:`, `B:`, or `C:` prefix from the choice text before
   encoding; do not expose choice position to the encoder. Preserve publisher
   option IDs/order and target index in the Decision data.
3. The readout is fixed before public-validation evaluation: project each
   384-D candidate query and each 384-D VideoMAE token into 192 dimensions,
   apply four-head cross-attention from each candidate to the full 1,568-token
   video grid, then score `[query, attended, product, absolute difference]`
   with a shared MLP. It has **444,673 trainable parameters** and contains no
   option-position embedding. CPU tests verify finite scoring and exact
   candidate permutation equivariance within numerical tolerance.
4. From the fixed 7,392-row source train split, sort unique video IDs, permute
   with NumPy `default_rng(17)`, and allocate `ceil(15% of video groups)` to
   internal selection. All questions for a video remain together. Train only
   on the remaining groups. Use seed 17, AdamW (`lr=3e-4`, weight decay
   `1e-4`), batch size 16 questions, maximum 30 epochs, and evaluate once per
   epoch. Stop after six epochs without a new minimum selection NLL; treat
   NLL changes smaller than `1e-5` as ties and prefer lower Brier. Save each
   epoch's train/selection metrics, the best-selected state, and the final
   state separately. No temperature fitting or post-hoc calibration is used.
5. The internal selection subset is used only to choose the readout checkpoint
   by NLL, then Brier as a tie-break. The public validation candidate pool is
   not used for fitting, early stopping, or checkpoint selection.
6. Freeze the trained checkpoint and all preprocessing before opening the
   public-validation candidate labels for the one final comparison.
7. Compare the same candidate scorer and train/selection schedule under:
   - correct VideoMAE tokens plus question/option MiniLM features;
   - text-only features with the video memory replaced by a fixed zero tensor;
   - the trained joint reader with a fixed, deterministic, different-video
     donor for each validation video.
8. A video-only condition is not meaningful for this source because answering
   the supplied QA requires the question and options. Option-order robustness
   is checked by a fixed permutation and restoring probabilities to publisher
   option IDs.
9. Do not tune preprocessing, reader width, epochs, or sampling after
   validation predictions are observed. If the frozen comparison fails, report
   it and design a new task/cohort separately.

## Evaluation and reporting

Save every example ID, video ID, target, prediction, option probabilities,
reasoning type, and donor mapping. Report Accuracy, NLL, Brier, and ECE overall
and by reasoning type, plus uniform-choice baseline. Compare correct-video,
text-only, and mismatched-video predictions paired by question; calculate
uncertainty by video-group bootstrap (seed 17, 2,000 replicates, percentile
95% interval). State that validation candidate exposure is not proven blind and
that checkpoint selection uses only the train-group internal selection subset.

No 95% threshold is redefined or claimed from this one dataset. The quality
target remains a project-wide goal across text, image, audio, and video. This
experiment can only add a bounded video-plus-text evidence point.

## Current gates

- The frozen corpus generator, metadata preflight, and receipt-based media
  acquisition are implemented and tested. Full media acquisition is running
  under a separate D: attempt; do not modify it while active.
- The VideoPrism candidate extractor has unit coverage but cannot currently
  execute under the active Windows Code Integrity policy.
- VideoMAE's earlier real-video CUDA runtime and source/license audit passed.
  The proposed extractor also passed a new train-only smoke on `video_10001`:
  16 frames decoded in 1.808s, frozen forward in 0.452s, finite token grid
  `[8,14,14,384]`, RTX 3080 peak allocated memory 240,685,056 bytes. The media
  SHA matched its acquisition receipt. The report is
  `D:\CodexArtifacts\pretrained-reuse-r2\perception-test-videomae-smoke-20261111-attempt-02\smoke-report.json`
  (SHA-256 `51f4f2d5e30ebdb2a1f6f6cfe5a317608156bd450773c94feca981e917d3a4b8`);
  its feature shard SHA-256 is
  `3fe0a89cbd43fa65bd5c978edaf73073ac695864c17a08beddea0c9eb3bb10c6`.
  This is a runtime smoke only; no question, label, validation, or test data was
  loaded and no quality metric was computed.
- No feature cache for this MCQA cohort, readout fit, or quality metric exists
  yet. Do not report this protocol as an evaluation result.
