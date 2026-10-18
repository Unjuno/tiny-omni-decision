# Greatest Hits audio/video correspondence Decision probe — fixed protocol

## Purpose and boundary

This is a new R2 task candidate for a narrow, real audio/video joint decision:
given an audio recording and a video recording, return JSON probabilities for
whether both came from the same recorded material event. This tests relation
information carried jointly across modalities. It does not claim general AV
reasoning, physical sync accuracy, or product quality.

The official Greatest Hits release is CC BY 4.0 per the project source audit.
Use only the previously inventoried official TRAIN partition; exclude all
source groups used in the 21-event manual source review. Do not open official
validation, quarantine, or test media for this experiment. The existing R1/R2
checkpoints, Path A package, corpora, feature caches, and other run directories
remain read-only. All new corpus/features/checkpoints go to a fresh attempt
directory outside the repository. No sealed audit is loaded.

## Ground truth and hard-negative construction

Each source parent has a corresponding `*_denoised.wav` and
`*_denoised.mp4`; their common source ID defines a positive match. Hard-negative
pairs use two distinct source parents with the same majority material/action
annotation. Annotation signatures are used only to choose hard negatives and
are never model inputs. A source recording and all same-timestamp-prefix
siblings are one group and must stay in a single split.

For each split and material/action signature, source groups are paired in a
deterministic seed-17 order. Each two-source cluster emits four questions:

| Audio | Video | Target |
|---|---|---|
| A | A | match |
| B | B | match |
| A | B | mismatch |
| B | A | mismatch |

This makes every individual audio and every individual video appear once with
each target within its cluster. A model that uses only one modality has a
balanced target marginal and cannot solve the task by source prevalence. All
options are fixed and ordered `[mismatch, match]`; the runtime returns
`{"options":[...],"probabilities":[...]}` in that order.

## Split and exclusions

- Source split: official TRAIN only; split again by normalized source parent.
- Group ID: remove a trailing numeric clip suffix so same timestamp-prefix
  siblings cannot cross splits; select one parent per group by sorted ID.
- Exclude all 21 reviewed source prefixes listed in the frozen source-review
  plan.
- Annotation signature: deterministic mode of known material/action labels.
- Keep signature strata with at least 10 remaining source groups. Drop one
  deterministic group from odd-sized strata before assigning splits, so every
  group can participate in a balanced match/mismatch pair.
- Within each signature, stable SHA-256 ordering with seed 17; allocate 60% to
  train, 20% calibration, 20% validation, rounding each split to an even group
  count and requiring at least two groups per split.
- Every group and pair cluster is assigned once. Save group IDs and ordered
  sample-ID hashes. Reject cross-split group/media/normalized-content overlap.

## Model comparison and metrics

Use frozen, rights-cleared EfficientAT DyMN-0.4 audio and VideoPrism-B video
features, with no encoder/projector updates. Compare fresh small readouts with
the same seed, training data, optimizer budget and checkpoint selector:
audio-only, video-only, and paired AV. The AV head may use an explicit
cross-modal interaction; it must not receive source IDs, file names, class
signatures, date, or split metadata. The validation set selects one checkpoint
under the predeclared minimum validation NLL rule; calibration fits only the
separate calibration partition.

Report Accuracy, NLL, Brier, ECE, mean confidence, and counts for each modality
condition and material/action signature. Use pair-cluster bootstrap, seed 17,
10,000 replicates, percentile 95% intervals; label uncertainty low-power if
fewer than 200 independent clusters. Report accuracy as well as calibration;
do not infer joint benefit from an output change alone. Save every example's
ID, source-group pair, target, prediction and probability vector.

Before feature extraction, the builder must verify pinned partition,
annotations, prior source-review plan and archive-directory-index hashes; media
members must exist in the official archive index. After fetch, verify member
CRC, local SHA-256, duration/decoder output, and exact media identity. After
feature extraction, verify exact source and feature hashes, finite tensors,
ordered IDs and fresh-process replay. Run GPU work only through the established
HotUGPU wrapper; its CUDA handoff guard must be established before the wrapper
waits for Orbitune's cooperative yield. The child may launch only with no other
compute PID, low utilization, `<14,000 MiB` free and `>=8,000 MiB` free after
reservation; otherwise it does not start. Do not stop or signal Orbitune.

## Metadata-only preflight completed

The seed-17 builder used only the official TRAIN source parents, selected one
parent per timestamp group, and excluded all 21 source groups in the prior
manual review. Nine material/action strata met the minimum count. The frozen
corpus has 126 train groups / 63 pair clusters / 252 rows, 38 calibration
groups / 19 clusters / 76 rows, and 56 validation groups / 28 clusters / 112
rows. Source-group overlap is zero; pair-cluster and audio/video target
marginals are balanced. Independent validation clusters are below the
predeclared 200-cluster reporting threshold, so any interval and task result
must be labeled low-power.

The three ordered JSONL SHA-256 digests are:

| Split | SHA-256 |
|---|---|
| Train | `5684d3a07186f8b6db1951ddb1131b1bb90cd74de57307217fe39c05b9185859` |
| Calibration | `4fea517675f3f85d948c8e7a9733bec84b787a0ada680a387589aa21fbe84ee9` |
| Validation | `60fd0ff39a7d164e4082688749def3c224fad613881b932b75e587df9208ce83` |

The selection references 440 unique raw files (220 audio and 220 video):
933,852,565 compressed bytes and 1,910,230,808 expanded bytes. The largest
member is 35,381,804 expanded bytes, above the older general-purpose 32 MiB
reader cap. A candidate-specific 40 MiB per-member cap and 2 GiB aggregate
expanded / 1 GiB compressed caps are now checked by the builder; all selected
members pass. This preflight read only the pinned ZIP directory index. No raw
media, model weights, sealed audit, or evaluation-only source was opened; no
encoder, optimizer, GPU, or quality metric was used. The output is a feasible
but low-power experiment, not evidence of task quality. Before fitting, raw
media still needs exact CRC/SHA validation, decoded-content overlap checks,
duration/stream validation, and feature replay.

The frozen files are in
`C:\CodexArtifacts\pretrained-reuse-r2\gh-av-match-preflight-02`; the
metadata-only builder refuses to overwrite an existing output directory.
