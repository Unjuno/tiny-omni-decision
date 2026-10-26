# Full DROID selected-shard protocol — 2026-10-26

## Frozen source selection

This is a small, reproducible development sample from the official DROID
training split. It is **not** the source's official validation/test split, a
sealed audit, or evidence of general-purpose Decision quality.

The official `dataset_info.json` at GCS generation `1710518471693866` describes
version `1.4.0`, 92,233 train episodes, and 2,048 shards. Before reading
selected episodes, 12 shard IDs were chosen as
`sorted(random.Random(17).sample(range(2048), 12))`. Their IDs and expected
episode counts are fixed in
[`droid-selected-shards-seed17-20261026.json`](../../manifests/pretrained_reuse/droid-selected-shards-seed17-20261026.json).
The chosen IDs are 110, 451, 715, 1019, 1033, 1139, 1186, 1242, 1497, 1573,
1696, and 1718, totaling 539 expected episodes. Metadata lists 10,724,824,602
bytes across the 12 shards and three pinned release metadata objects. Each
object is pinned by GCS generation, exact size, and MD5.

Dataset-info SHA-256 is
`8cbd8f70428943515a67e2a571267c4de2eb71f33957c0269d8fcd5608951e4f`, the
feature-schema SHA-256 is
`4c556e2e38e179e41089e07cf54273b1540d67d3c78a420898370329813c9a25`, and the
CC BY 4.0 notice is generation `1710518471643551` with SHA-256
`7e7170e3cebf88a9f60c7b8421418323c09304da1af4d5e90f4da1dc1c8a2661`.
The DROID paper states that the full data release is CC BY 4.0. Preserve its
attribution as specified in the [source screen](R2_DROID100_SOURCE_TASK_SCREEN_20261026.md).

## Split policy, before model work

First audit the selected records' source-file and recording-folder identities
against one another and the already-screened DROID-100 debug subset. The debug
subset is not to be mixed into this experiment. Exclude any selected full
record whose recording-folder or source-file identity overlaps the debug
subset. Use exact normalized instruction overlap only to keep duplicate text
content in one partition; report near-synonymous tasks separately.

If at least 400 independent recording groups remain, reserve at least 200
groups for a development validation partition using this deterministic rule:

1. Union records that share either recording-folder identity or normalized
   instruction text.
2. Sort the resulting components by
   `SHA256("droid-r2-seed17:" + sorted(component recording-folder IDs))`.
3. Add whole components to validation until it contains at least 200 unique
   recording groups; use all remaining components for training.
4. Fail closed if validation has fewer than 200 groups or training has fewer
   than 200. Do not weaken the group rule or inspect model metrics to change the
   allocation.

This is a development train/validation split from a source whose official
split is `train`. It is not a final blind audit. Keep any later final audit in
unused assets outside these selected shards. The source sample is only 539
episodes; if group/text collisions leave fewer than the predeclared minimum,
stop before model fitting and report the exact shortfall.

## Acquisition and split audit results

The fixed 15-object acquisition completed successfully in 324.84 seconds.
Receipt SHA-256:
`2bfb6322e9c0c4649f875d8ad37fb3e990f786324fe2fdbd9cd6b2872af50667`.
The metadata-only group audit JSON SHA-256 is
`541595a78e565613b9e1929a2be0fa73dc8f78bb863a71b32a76e81af9cfb7ab`; the
development split SHA-256 is
`d8d2e1ef353d2ea6388dd7e7574623e3b4d6518474afed14f042fc1838df7e43`.
All receipts and split files are outside Git under
`C:\CodexArtifacts\pretrained-reuse-r2\droid-selected-shards-seed17-20261026-attempt-01\`.

| Audit | Result |
| --- | ---: |
| Downloaded objects and bytes | 15 objects; 10,724,824,602 bytes |
| Episodes / unique source files / unique recording groups | 539 / 539 / 539 |
| Per-shard episode counts matching pinned TFDS metadata | 12/12 |
| Success / failure folder labels | 444 / 95 |
| Unique normalized instruction strings / repeats | 669 / 45 |
| Source-file overlap with DROID-100 debug subset | 0 |
| Recording-group overlap with DROID-100 debug subset | 0 |
| Normalized instruction overlap with DROID-100 debug subset | 6 |
| Debug-overlap episodes excluded | 0 |
| Development train / validation episodes and groups | 339 / 200 |
| Train/validation source, group, normalized-instruction overlap | 0 / 0 / 0 |
| Sealed audit from this sample | None |

The 6 matching instruction strings did not cause data rows to be imported from
the debug subset. Within the selected full-data sample, every repeated
normalized instruction was assigned wholly to one side by the union-group
split rule. These results establish deterministic development partitioning,
not task quality or external generalization.

## Temporal-field and language-coverage audit

A second pass used the same pinned bytes and same split rule to count serialized
step fields without decoding JPEG pixels or fitting a model. The enriched audit
JSON is external at
`C:\CodexArtifacts\pretrained-reuse-r2\droid-selected-shards-seed17-20261026-attempt-01\group-audit-03.json`
with SHA-256
`d60f850f88725a45f918a992685ab1174b38f086ea14029cac1ab665a504bc3b`.
Regenerating the split produces the same SHA-256 as the original:
`d8d2e1ef353d2ea6388dd7e7574623e3b4d6518474afed14f042fc1838df7e43`.

| Coverage check | Result |
| --- | ---: |
| Episode steps (min / median / p90 / max) | 7 / 227 / 534 / 2,238 |
| Mean episode steps | 289.38 |
| Episodes under 8 steps | 1 |
| Instruction present / absent | 302 / 237 |
| Train instructions present / absent | 181 / 158 |
| Validation instructions present / absent | 121 / 79 |
| Episodes where all 3 camera counts equal step count | 539 / 539 |
| Episodes where 7D joint position, 6D Cartesian position, gripper state, and 6D action position/velocity match steps | 539 / 539 |
| Episodes where 7D joint velocity and gripper action count match steps | 539 / 539 |

Thus text conditioning is available for only 56.0% of this sample. Instruction-
conditioned scores must use the text-present subset and also report text-missing
behavior separately; treating missing instructions as ordinary text examples
would misstate the task. The one 7-step episode is not enough for an 8-step
history and must be explicitly excluded from that windowed analysis. These
checks establish serialized field counts and alignment, not timestamp units,
value semantics, image content, model inputs, or action-prediction quality.
The pinned feature schema places cameras, state, and action in a shared step
sequence, but exposes no explicit per-step timestamp field. Only step-index
alignment can be claimed here; absolute timing, sampling jitter, and clock
offset are unknown and must not be fabricated.

## Candidate Decision task

The first action-choice proxy considered was: observe eight step indices, then
choose whether the gripper command one step later is open or closed, using the
fixed threshold `0.5`. The official implementation maps gripper state to
`1 - width / max_width` and clips commanded position to `[0,1]`; this makes the
binary names interpretable but does not turn them into a rich manipulation
decision ([official implementation](https://github.com/droid-dataset/droid/blob/main/droid/franka/robot.py)).
On the fixed dev split, previous-command persistence alone reaches 98.80% on
train and 99.04% on validation (12,543 / 6,652 windows); the current gripper
state reaches 92.08% / 92.99%. This is not model performance. The persistence
result is saved at
`C:\CodexArtifacts\pretrained-reuse-r2\droid-selected-shards-seed17-20261026-attempt-01\gripper-action-feasibility-02.json`
with SHA-256
`dd7261a5168cae8bf307db09c769587f8ba72354188ce73185ae9ba5be0c9c5f` and is
reproduced by `scripts/audit_droid_gripper_action_persistence.py`. Because the
target is almost always the previous command, this definition is retained only
as a low-compute persistence control, not the main video-sensor fusion task.

The current task candidate is **early episode outcome selection**: from the
first 25% of a DROID episode, choose `[success, failure]` using three
step-aligned camera views and robot-state history. The label is derived from
the exact `success` / `failure` folder segment in the episode source path; that
path is not an input. This is a provisional weak target, not a separately
verified human annotation. Text and audio are excluded, so this candidate
tests the roadmap's video + sensor M2-VS slice only. The pinned data schema
provides common step order but no explicit timestamps; no seconds or sampling
rate are claimed.

The frozen config is
[`droid_early_outcome_video_sensor_v0.json`](../../configs/pretrained_reuse/droid_early_outcome_video_sensor_v0.json),
SHA-256 `d3061257e19afe3960ed136f63811255bae43728318d371ee77486dcd1da3ce9`.
The deterministic metadata-only task manifest was prepared at
`C:\CodexArtifacts\pretrained-reuse-r2\droid-early-outcome-v0-attempt-02\`
with SHA-256 `b882b66f8be2dd38aa7d69badd3d00b93f58769a9daa2be5f38088fbe084ec80`;
train and validation JSONL hashes are
`7f56ee2fb525fc5e40e694c23566aa8b306b0de3444e0664e94531d3da6391dd` and
`6b014d506cb2f0ff2179a6f6394504759233b5a688765a639a617310beeb44a6`.
After requiring eight unique observations inside the first quarter, train has
337 episodes (274 success / 63 failure; two short failure episodes recorded
as excluded) and development validation has 200 (170 / 30), still with one
recording group per episode and no group overlap. The validation majority-class
baseline is 85%; 30 failure groups are a limited basis for uncertainty
estimates, so balanced and per-class results are mandatory. `run-metadata.json`
records `PREPARED_NOT_TRAINED`; no model features, model predictions, checkpoint,
or sealed audit were created.

A CPU-only raw-input smoke then decoded every selected early-window JPEG and
read each corresponding sensor vector. It passed for all 537 task episodes:
12,888 RGB images decoded at 320×180, totaling 293,728,684 selected JPEG
bytes; every view had eight ordered frames and every sensor window had shape
8×(7 joint-position + 6 Cartesian-position + 1 gripper-state) finite values.
The per-episode ordered JPEG/RGB/state hashes and split counts are saved in
`C:\CodexArtifacts\pretrained-reuse-r2\droid-early-outcome-v0-attempt-02\raw-input-audit.json`,
SHA-256
`23fe0d90ed2ee0d905d27ca2d76b606aed8277bedb677fa1236b339194375a87`.
This smoke did not apply VideoPrism preprocessing, compute embeddings, train,
or run inference. The three views and state arrays align by step index only;
true elapsed time remains unknown. The CPU parser and smoke are reproducible
with `scripts/audit_droid_early_outcome_inputs.py` and the hashes in the frozen
config/task manifests.

The next model comparison should use frozen VideoPrism-B features for the
ordered 8×3 view window and a small temporal sensor/readout head. Compare
video-only, sensor-only, and video+sensor on identical episode IDs/options,
then test shuffled/replaced video, shuffled/replaced state, and a one-step
temporal offset. Include a majority predictor and report accuracy, balanced
accuracy, per-class recall, NLL, Brier, and ECE. Keep the development split for
iteration only; no sealed audit exists in this sample. Proceed to a model
smoke only after the raw-input pipeline proves the fixed windows and passes
the local HotUGPU resource guard.

For any later recorded-action task, freeze and test the sampling rate,
camera-view policy, temporal window, action option construction, hard
negatives, target mapping, and choice shuffle. Then compare identical
examples/options with video-only, state-only, and video+state models. Require
missing, shuffled-state, stale-state, temporal-offset, and contradictory-input
controls. Fusion is useful only if the actual paired inputs improve on the
best unimodal result and replacing either input causes the expected error
increase. High aggregate accuracy alone does not close R2 M-2.

A reasonable frozen-representation candidate from the existing R1 inventory
is VideoPrism-B (114.37M parameters), English MiniLM-L3 (17.39M), and a small
trained sensor-time LSTM plus gated Decision readout. Whisper Tiny (37.76M)
is the already-audited English speech candidate if a later task includes
audio. These are candidate components, not an adopted architecture or a
quality result. Measure the complete resident parameter count and package
bytes before making any product-size claim.

## Execution boundary

Download only the 12 fixed shards and their release metadata to a new directory
outside the repository. The downloader verifies generation-pinned HTTPS
responses, GCS MD5/size, metadata SHA-256, the 1.4.0 release, and each shard's
episode count. The group auditor reads serialized metadata, instructions, and
protobuf list element counts; it does not decode or display images, extract
model features, or open a sealed audit. No model training or GPU work has
started.
Model work still waits for a tested causal-window/action-option task builder
and matched baseline/control definitions.

The 100-episode debug-source screen is recorded in
[`R2_DROID100_SOURCE_TASK_SCREEN_20261026.md`](R2_DROID100_SOURCE_TASK_SCREEN_20261026.md).
Official sources: [DROID dataset docs](https://droid-dataset.github.io/droid/the-droid-dataset)
and [DROID RSS 2024 paper](https://www.roboticsproceedings.org/rss20/p120.pdf).

## 2026-10-26 frozen VideoPrism real-input CPU smoke

A fixed first training episode from shard 1242 was passed through the pinned
VideoPrism-B encoder using its 8 selected early-window steps from each of the
three camera views. The smoke produced finite `[3,2048,768]` tokens and finite
`[3,8,768]` time-preserving features after mean spatial-token pooling. It used
the upstream resize-short-side/center-crop preprocessing, VideoPrism revision
`be719a406d563b66f0ac969e7c94bab8e997c81a`, source commit
`c6eba0ece3dd49a24ffc44582966997a4a59f1d8`, and checkpoint SHA-256
`61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
The actual backend was CPU (`TFRT_CPU_0`), with 11.25 s for the encoder forward
and 4,259,770,368 bytes peak process working set. This is one-input runtime
and representation-shape evidence only; it is not fitting or a quality score.
The result JSON is outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\droid-videoprism-early-window-cpu-smoke-02\smoke.json`,
SHA-256 `a370a3aea69cd78cc2aa7f6715eb3346702ac18e7dab40199b2798f232edc639`.
The reproducible one-off driver is under the separate attempt-03 directory, `run_smoke.py` SHA-256 `0f90986e1138ca4451c37f95a9fe5565e3ac9233a237627ff8f41c475bf24cb8`;
attempts 01 and 02 preserve the Pillow-missing and rank-5 resize failures.

The RTX 3080 was idle at the last read-only snapshot, but the existing project
Python environment cannot import its CUDA PyTorch DLL: Windows Application
Control returns `WinError 4551`. The installed VideoPrism JAX/TensorFlow runtime
is CPU-only, so no GPU extraction was attempted. The full repository pytest
collection consequently stopped with 100 errors at optional Torch imports;
this is an environment gate, not a model/test-quality result. The 16 focused
DROID tests and Ruff checks passed using the read-only `python -S` test entry.
No environment package, GPU process, dataset, or existing checkpoint was
changed.

## 2026-10-26 complete frozen features and early-outcome readout

The one-episode smoke was followed by full feature extraction for the already
fixed 337 train and 200 validation task rows. All 537 IDs were found exactly
once in the pinned selected shards. VideoPrism features have shape
`[537, 3, 8, 768]`; sensor windows have shape `[537, 8, 14]`. Both arrays are
finite, and each temporal video feature row is hashed in the extraction
progress ledger. Checkpoint, task manifest, shard manifest, arrays, labels,
and progress are hashed in the feature attempt directory outside Git.
Extraction used CPU JAX and took 2,068.2 seconds; no GPU or cloud compute was
used. Full-run peak working set was not captured.

Three fixed L2 logistic heads were fit with train-only standardization on the
same IDs: video temporal pooling, sensor sequence, and their concatenation.
The results and per-example predictions are saved under
`C:\CodexArtifacts\pretrained-reuse-r2\droid-early-outcome-readouts-repo-attempt-02\`.
The correct probability outputs were generated by the repository evaluator;
the first external one-off attempt mistakenly wrote hard labels in its
`p_success` column and is preserved but excluded. The majority baseline uses
the train-set success prevalence rather than validation prevalence.

Validation Accuracy was 76.5% video, 83.0% sensors, and 79.5% fused, against
85.0% for the train-prevalence majority predictor. The fused readout had 23.3%
failure recall, and its NLL/Brier were worse than sensor-only. Paired
class-stratified intervals do not establish a calibrated complementary fusion
benefit. This weak folder-derived task is **not useful for model promotion**;
it does not close R2 M-2 or replace the broader task battery. Full metrics,
hashes, fitted parameter counts, runtime, and limitations are in
[`R2_DROID_EARLY_OUTCOME_RESULTS_20261026.md`](R2_DROID_EARLY_OUTCOME_RESULTS_20261026.md).

The portable evaluator and readout/metric functions are
`scripts/evaluate_droid_early_outcome_readouts.py` and
`src/tiny_omni_decision/droid_early_outcome_readout.py`. Focused tests pass
with SciPy available (20 passed); the default system Python lacks SciPy, so
the optional optimizer smoke skips there. Existing Teacher, sealed-audit,
legacy run, and checkpoint artifacts were not changed.
