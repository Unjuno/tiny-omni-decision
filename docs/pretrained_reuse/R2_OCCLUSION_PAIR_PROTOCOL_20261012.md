# Prospective initial/final position decision family

## Fixed construction, before any model scoring

Use the two exact train-observed occlusion-game questions asking the hidden
object's initial and final position from the person's point of view. Keep the
upstream three answer strings unchanged, with canonical lexical ordering and
target remapping by answer text. Keep unchanged-target examples as well as
changed-target examples. Both questions remain in the same upstream video split.
This is a narrow temporal condition-following diagnostic, not the full product
battery and not a synchronized audio/video task merely because sound is present.

`tiny_omni_decision.occlusion_pairs.build_pairs` was executed on the pinned
official train and development-valid archives. No test/sealed archive was opened.

| Split | Complete video pairs | Changed targets | Unchanged targets | Held |
|---|---:|---:|---:|---:|
| train |99|53|46|17|
| valid |257|127|130|42|

Held records remain saved with reasons for missing/duplicate exact questions or
incompatible option sets. They cannot be silently scored as complete pairs or
replaced; report paired-family coverage separately from the broader source.
All complete pairs are included irrespective of correctness/difficulty. Video-ID
overlap is zero, but this does not prove participant or content independence.
Official validation has already been used for other development tasks; no blind
claim is permitted.

Artifacts under `C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-pairs-01`:
train.json SHA256 `8869bb903591af3a0995eeca0eeae1b50b990c8b08752a6e5d4d549f8a356fd7`;
valid.json SHA256 `30846aaa87e2292e97ee3187c5f2f986ba3c3d57d46a249b361e5a14c0cc8ec2`.
result.json preserves counts and exact upstream archive hashes. Tests verify
option remapping, unchanged-target retention, incomplete-pair retention and
rejection of test split input. Construction does not run models or train weights.

## Required evaluation contract

Report per-condition Accuracy/NLL/Brier/ECE, both-questions-correct per video,
changed-target and unchanged-target strata with counts, plus video-cluster
uncertainty. A change in argmax alone is not a correct condition-following result.
Controls must preserve options and compare no-video, mismatched-video and temporal
order evidence with the fixed scorer. Do not tune prompts or select checkpoints
on those controls. An actor-perspective left/right mapping must be verified rather
than inferred from viewer-left pixels.

Input admission remains open: original reveal cuts, video hashes, sufficient
trajectory coverage and resource bounds must be verified across the selected
family. The separate4fps window smoke demonstrates local feasibility only.
It does not admit all356 videos, establish a model/head design, or authorize
claiming quality from a subset of easy/evidence-visible cases. No new fitting
has begun. Existing eight-frame correspondence run and all legacy artifacts stay
unchanged. Broader text/image/audio/video and fusion targets remain in force.

## Acquisition and option semantics checks

The pinned ZIP-index plan covers all356 complete-pair videos.21 already acquired
files pass SHA checks;335 require4,940,073,790 additional bytes. Observed free
space22,067,601,408bytes leaves the8GB reserve.87/99train and236/257valid videos
have upstream reveal cuts. Plan SHA256
`a0631ff6e69a1ba709ee9c8c502d7ea67896f04863f73a737ab46b7406fbb186`.
Source461dc77 launched separate train/valid acquisition processes into
`perception-occlusion-train-media-01` / `perception-occlusion-valid-media-01`.
This is launch, not completed media verification. Both processes remain active.

An attempted common-three-label prior failed its option-equality assertion before
producing metrics. This revealed three option sets in each split: train82 lateral
position,16 near/far/middle,1 four-object-distance subset; valid236 lateral,
20near/far/middle,1 previously unseen fourth/second/third-left subset. Preserve all
sets. A fixed three-class spatial head cannot silently treat these meanings as
interchangeable; supplied candidate semantics must remain part of the task.

A no-media baseline then used **train-only counts per exact option set and
condition**, Laplace+1 smoothing, uniform probabilities for unseen sets and
lowest-index tie breaks. On all257 development videos:

| Condition | Accuracy | NLL | Brier | ECE15 |
|---|---:|---:|---:|---:|
| initial |0.396887|1.093395|0.662681|0.026099|
| final |0.381323|1.089626|0.663716|0.083357|

Both questions correct:0.206226. One unseen option set uses the uniform fallback.
Artifact `perception-occlusion-pairs-01/train-prior-baseline.json` records the
source hashes and exact estimator. No pretrained model, media evidence,
temperature fitting or model selection was used. This measured prior is a
necessary comparison; exceeding33.3% alone is not sufficient evidence of useful
video decisions. No sample was removed because of the unexpected option set.

## Completed train acquisition and decode check (2026-10-13)

All99 planned training videos finished acquisition with exit0. A separate read-only
pass rehashed every file and matched IDs, byte sizes and cut indices to the pinned
plan. Acquisition manifest SHA256:
`469299ad240ff85c47ba44abb8920aae9994b0e3ec15a3fca1373c49b2fa6ee3`.
The separate PyAV18.1 decode process then completed all99 videos /69,192 frames in
302.204 seconds, checking strictly increasing presentation timestamps and valid
cut indices (87 videos have a cut). It exited0. Decode report:
`C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-train-decode-01.json`,
SHA256 `f017c71e877d00e41fa3d812a9d4dd6e4236aae2e69070f7bd8905a8960dc9d2`.
Integrity report: `perception-occlusion-train-media-integrity-01.json` in the same
artifact root. These checks establish decode/clock/cut validity only, not dense
feature admission, semantic evidence sufficiency, or decision quality. Validation
acquisition is still running; no fitting has started for this family.

## Completed validation acquisition and decode check (2026-10-13)

Validation acquisition finished257/257 with exit0. A separate all356-file pass
matched byte sizes, SHA256, ordered video IDs and cut indices against the pinned
plan; cross-split video-ID and exact-media overlap are both0. Report:
`C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-all-media-integrity-01.json`.
Validation acquisition manifest SHA256:
`213bd33994fb80d79ee05ee0580af2cc9915dfc61094aed00c847c2c67f1d546`.
The separate PyAV18.1 process decoded all257 videos /182,087 frames in715.792
seconds, checking strictly increasing PTS and cut-index bounds (236 cuts), exit0.
Report `perception-occlusion-valid-decode-01.json` in the same artifact root has
SHA256 `1319430044bc766ca1be00726fcf2ac1c21e5a6c2db9042cec45ec9b5572a8d5`.
All356 videos now pass these media/clock checks. Dense-window feature extraction,
prepared-content overlap checks and conditional decision fitting remain pending;
these results establish no model quality or temporal reasoning performance.

## Dense feature extraction launched (2026-10-13)

Source `e8fb6d7` runs `scripts.extract_occlusion_dense_features` with output
`C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-dense-features-01`.
This is a new CPU process using the same pinned VideoPrism checkpoint/source,
4fps and eight-frame windows tested in the earlier smoke. All356 complete pairs
remain scheduled; no labels are passed to the encoder. Each video retains its
window sequence, exclusive reveal cut and existing common-PTS interval policy.
Media and pair manifests are pinned. Prepared video/window hashes reject exact
cross-split duplication; this does not prove participant or semantic independence.
Per-video feature hashes, sampling indices, timestamps and timings are saved.

The first video completed (eight windows), confirming the full runner reaches
real inference. Extraction is still in progress; no complete-feature admission or
new decision training result is claimed. Ruff and four focused CPU tests passed;
both GitHub Actions runs for `e8fb6d7` succeeded. Prior artifacts are unchanged.

## Prospective fixed readout comparison (before fitting)

Compare exactly two seed17 heads over the same admitted dense features: masked
window mean versus candidate-conditioned attention with relative temporal position.
Both use frozen VideoPrism features and the existing frozen MiniLM question/option
features `[q, option, abs(q-option), q*option]`; options keep their actual text.
The shared head width is64. Video windows are768-dimensional, candidate features
1536-dimensional. This is a variable-length adaptation of the existing token
readout, not base-model fine-tuning. No new encoder or checkpoint search is added.

Each head starts from the same seed17 initialization, AdamW lr0.001,
weight_decay0.01, fifty epochs, eight video pairs per batch (last partial batch
retained), both questions per pair, train-only seeded epoch shuffles, cross-entropy
with equal question weights. No augmentation or class reweighting. All99 training
pairs and257 development pairs remain included. Both heads retain final and earliest
minimum validation mean-question NLL checkpoints. Record every epoch, full-run and
selected-checkpoint presentations separately, fixed train and validation metrics,
parameter counts, artifact hashes, CPU runtime and environment. The mean model's
key/position parameters are unused and must be reported separately from effective
trainable parameters; its representational capacity is lower by design.

After selection, evaluate each fixed selected head on normal inputs, zero-video,
video-ID cyclic mismatch and reversed window order. These controls cannot select
checkpoints. Check option permutation and saved-checkpoint reload parity. Report
initial/final Accuracy/NLL/Brier/ECE15, both-correct rate, changed/unchanged strata,
and10000 video-cluster bootstrap draws, seed17, percentile95% intervals. Compare
with the already frozen train-only prior. No claim of improvement without direct
paired evidence; no blind-audit claim. Preserve all saved predictions.

The100% finite-input and full356 coverage gates must pass before fitting. Smoke
runs use separate processes and outputs and do not select hyperparameters. This
comparison tests whether temporal window readout uses video evidence for this
narrow family; it cannot establish the complete Omni product objective.

## Readout execution preparation

Frozen text extraction completed all712 questions in a separate CPU process,
using the existing MiniLM revision and unchanged mean-pooling/feature formula.
Every feature was repeated with exact equality. Artifacts:
`C:/CodexArtifacts/pretrained-reuse-r2/perception-occlusion-text-features-01`.
The report pins model/tokenizer files, pair manifests, identities and feature files.

`train_occlusion_readout.py` implements the predeclared two-head fit and controls.
It rejects an incomplete video extraction, changed feature hashes or routing,
nonfinite values and cross-split prepared-window duplicates. Six focused CPU tests
passed, including padding/permutation, temporal order, paired aggregation, control
routing and rejection of incomplete extraction. No new fit has run yet: the full
video feature process remains active. This is preparation, not a quality result.
CI for295e2f8 passed both runs after the optional-Torch collection fix.

## Completed execution (2026-10-13)

The preparation above is historical. All356 features and both50-epoch runs
completed; see [results](R2_OCCLUSION_RESULTS_20261013.md). Neither head is adopted.
