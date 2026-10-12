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
