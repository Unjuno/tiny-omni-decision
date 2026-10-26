# DROID-100 source and task screen — 2026-10-26

## Disposition

**Admitted as an exploratory video + robot-state + instruction task source; not
admitted as an independent quality benchmark and not sufficient to close
R2 M-2.** The official `droid_100` release is a 100-episode debugging subset
of the train split. The actual pinned release contains 100 distinct
`recording_folderpath` groups, but it is too small and is not a held-out test.
This screen did not decode video frames, extract model features, train a model,
or measure decision quality.

A later fixed 539-episode full-train shard sample and temporal-field audit is
recorded in the [selected-shard protocol](R2_DROID_SELECTED_SHARDS_PROTOCOL_20261026.md).
Its one-step gripper-choice proxy was rejected as the main fusion task because
previous-command persistence reaches 99% on development validation. The current
development candidate uses early episode outcome labels with video + robot
state; it remains weakly labeled. Its frozen-feature readout was measured on
337 train / 200 development-validation episodes and performed below the
train-prevalence baseline; see the
[results](R2_DROID_EARLY_OUTCOME_RESULTS_20261026.md).

DROID is a plausible setting for a decision model because it pairs three
synchronized RGB camera views with proprioceptive state, a natural-language
task instruction, and a recorded action sequence. The `droid_100` schema has
180×320 RGB images, 6D Cartesian and 7D joint positions, and a 7D `action`
vector documented as six joint velocities plus a gripper position. The schema
does not identify units for every field; this project will not invent them.
There is **no audio stream** in this release. These properties make it a
candidate for the roadmap's video/sensor branch, not a four-modality task.

## Source, rights, and pinned files

- Official DROID docs describe `droid_100` as an example 100-episode RLDS
  subset for debugging (about 2 GB), and document the per-episode recording
  folder, original source file, step instructions, robot state, actions, and
  three RGB cameras: [dataset docs](https://droid-dataset.github.io/droid/the-droid-dataset).
- The DROID paper states the full release is CC BY 4.0:
  [RSS 2024 paper](https://www.roboticsproceedings.org/rss20/p120.pdf), pp. 1 and 10.
- The paper's license statement is the basis for this research use. Preserve
  attribution to Khazatsky et al., *DROID: A Large-Scale In-The-Wild Robot
  Manipulation Dataset*, RSS 2024, DOI `10.15607/RSS.2024.XX.120`.
- Download inventory, per-object GCS generation, size, and MD5 are pinned in
  [`droid100-pinned-objects-20261026.json`](../../manifests/pretrained_reuse/droid100-pinned-objects-20261026.json).
  The inventory contains the two official metadata files and all 31 face-blur
  TFRecord shards for `robotics/droid_100/1.0.0/`. Manifest SHA-256:
  `3203e55037185a054f456d57cb759b38ddee4e7930a7f68c3ed930a43d17c528`.
- Every object was downloaded over HTTPS from its pinned generation and
  verified against the inventory's exact size and GCS MD5. The 33 objects sum
  to `2,192,615,094` bytes. No bypass or unpinned fallback was used.

## Metadata audit

The downloaded object inventory is outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2\droid100-download-20261026-attempt-01\`.
The downloader receipt SHA-256 is
`e7dbac63780d24522c8802b0a48c0da9d7f35153ff9f64a999d686be941cb9eb`.
The TFRecord metadata audit JSON SHA-256 is
`9adb67c3aee29ce7d0cecad14dbecedb2f506b9077b316fd74da80a6c09157d`.

The audit read episode metadata from TFRecord `Example` messages only and confirmed:

| Check | Result |
| --- | ---: |
| Episodes | 100 |
| Unique original source files | 100 |
| Unique recording-folder groups | 100 |
| TFRecord shard episode counts matching `dataset_info.json` | 31/31 |
| Success / failure folder labels | 81 / 19 |
| Instruction strings after whitespace normalization | 106 unique; no exact normalized duplicates |
| Video frames decoded or displayed | 0 |
| Audio fields | None in the schema |

The 106 instruction strings can include alternate instructions within an
episode. Exact normalized uniqueness does not establish semantic uniqueness:
near-synonyms, repeated object/task concepts, and shared collection sites still
need to be grouped or reported in a future split protocol. Episode windows must
never cross recording-folder groups. Because this release has only the
official train split, any held-out partition cut from these 100 episodes is
development-only and cannot be presented as an independent benchmark.

Reproduce the metadata audit after download with:

```powershell
python scripts/download_droid100_pinned.py `
  --output-dir C:\CodexArtifacts\pretrained-reuse-r2\droid100-download-20261026-attempt-01

python scripts/audit_droid100_groups.py `
  --data-dir C:\CodexArtifacts\pretrained-reuse-r2\droid100-download-20261026-attempt-01\robotics\droid_100\1.0.0 `
  --output C:\CodexArtifacts\pretrained-reuse-r2\droid100-download-20261026-attempt-01\group-audit.json
```

The auditor validates TFRecord framing, the exact per-shard episode count in
`dataset_info.json`, 100 total episodes, non-empty recording/source metadata,
and source-file uniqueness. It does not decode image or numeric tensors.

## Candidate task and controls to pre-register before fitting

Candidate: **select the next recorded action from a small set of matched
alternatives, given the task instruction, a short timestamped camera window,
and the synchronized robot-state history.** This is closer to a typed
Decision task than free-form policy generation. It can exercise visual context
and temporal sensor context together. The label/options must be constructed
from recorded actions only after the recording-group split is frozen; option
wording or ordering must not reveal the answer.

Before any model fitting, a separate protocol must define and unit-test:

1. Recording-folder-group train/validation/test partition, with all windows
   from a recording held in one partition. Since the source itself is
   train-only, label the partitions exploratory.
2. Timestamp alignment and a causal cutoff: no future frame, state, action, or
   instruction annotation may leak into the input for the target action.
3. Action-value normalization and hard-negative construction. Verify the
   action component semantics from the pinned official schema/code before
   assigning units or physical meanings.
4. Identical option sets, choice counts, calibration metrics, and per-task
   reporting across video-only, state-only, instruction-only, video+state,
   and full-input baselines.
5. Missing-video/state/instruction, shuffled state, stale state, temporal
   offset, and contradictory-input controls. R2 M-2 requires evidence that
   more than one input is necessary; an accuracy increase after concatenation
   alone is insufficient.

The present 100-episode debug subset is useful for checking data plumbing,
causal windowing, and whether the task/controls are implementable. It is below
the roadmap's independent-group target and cannot establish the requested
95–98% general-purpose quality. Do not tune a product checkpoint or claim
M-2 success from it. Do not download the full 1.7 TB corpus as an implicit
next step.

## Code and execution record

- Generation-pinned HTTPS downloader:
  [`download_droid100_pinned.py`](../../scripts/download_droid100_pinned.py).
- Metadata-only TFRecord auditor:
  [`audit_droid100_groups.py`](../../scripts/audit_droid100_groups.py).
- CPU tests cover manifest integrity, traversal rejection, checksums, TFRecord
  framing, metadata extraction, and duplicate group accounting.
- No model weights, corpus shards, extracted frames, model features, or
  checkpoints were added to Git. No GPU, cloud, or paid compute was used.
- The existing Open Images/TinyCLIP and PE-Core artifacts were not modified.

