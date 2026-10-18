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

## TRAIN media acquisition and decoder smoke

All 440 predeclared assets were acquired from the pinned archive using exact
HTTPS range responses with the pinned ETag. Local ZIP member names, decompressed
sizes and CRC-32 passed; the acquisition manifest records a SHA-256 for every
asset and a source-group/split assignment. The complete media manifest SHA-256
is `2949ee916c40bed3d0495409075d0122ca471b1da8b7bc343fe16e8171f0b120`.
Acquisition took 738.172 seconds and requested 933,897,889 range bytes; 440
expanded media files total 1,910,230,808 bytes. No official validation,
quarantine, test, or sealed-audit media was acquired. Exact raw-file hash overlap
across splits is zero.

A CPU-only decoder smoke on the first fixed TRAIN group succeeded: the WAV
decoder observed 96 kHz and produced 815,214 finite 24 kHz mono samples; the
video decoder observed 1,018 increasing-PTS frames and selected eight endpoint-
inclusive indices `[0, 145, 291, 436, 581, 726, 872, 1017]`. This verifies the
actual sample rate handling path; it does not represent every asset. The full
220-group decode and normalized-content overlap audit have since completed.
No model weights, optimizer, GPU, or quality metric was used during this stage.

## Full input preparation and independent reload audit

The pinned TRAIN-only source groups were fully decoded on CPU under source
commit `e4eb77a5e5881c54ec3b7c30de09e6df7ddb4701`. All 220 paired groups
produced one normalized mono float32 24 kHz audio array and eight uint8
288x288 RGB frames. The decoder processed 205,402,615 normalized audio samples
and saved 914,919,543 bytes of NPZ inputs. The `groups.jsonl` SHA-256 is
`dfc6f4130a4de547a89ed47b3f58cb009361662f74f1be384002093788e1d724`; the
input-preparation receipt SHA-256 is
`f5a784f206298c0b357e291e9bbf36c25bd6eb8375f209f6d2e06b2b5588745c`.

A fresh-process audit reloaded all 220 NPZ files, checked their SHA-256,
shape/dtype/finiteness, and reconstructed expected source groups from the
frozen train/calibration/validation Decision JSONL. It found 220 distinct
normalized audio and 220 distinct full decoded-video contents, zero normalized
cross-split overlaps, zero raw-media cross-split hash overlaps, and exact split
counts `126/38/56`. Audit receipt SHA-256:
`1c6ddbb77888bf149e6994f86e4f3d0ed890e5f8a9e3a241b56eb53abf56c540`.
This rules out exact byte/decoded-content leakage for the selected media, not
near-duplicate scenes or shared recording environments. The input arrays are
frozen in `C:\CodexArtifacts\pretrained-reuse-r2\gh-av-match-inputs-01`.

The frozen feature runner is committed at `8874103c2124cb8f3ff2cbf22024f56bc6cffe51`.
It uses the pinned EfficientAT DyMN-0.4 and VideoPrism-B checkpoints read-only,
repeats every forward, and verifies each input/feature hash. VideoPrism remains
CPU-pinned. EfficientAT CUDA extraction was attempted through the HotUGPU
cooperative handoff, but Orbitune resumed while the child was active; the child
alone yielded and the final audio extraction completed on CPU. The readout fit
also ran on CPU.

## Frozen features and readout result

All 220 audio and 220 video source-group features are now frozen and pass fresh-
process replay; both replay maximum absolute deltas are exactly zero. CUDA audio
extraction was not used in the final path because Orbitune resumed while our
child was active; the child alone yielded and audio extraction was completed on
CPU. Orbitune was not stopped or signaled. A fixed CPU-only seed-17 comparison
then completed fresh audio-only, video-only, and paired AV readouts. It chose
checkpoints by earliest minimum uncalibrated validation NLL, and fit temperature
only on the separate calibration split. Joint AV selected epoch 1, reached
100% train accuracy at epoch 100, and validation NLL rose to 2.6145 with 55.36%
accuracy. The calibrated selected joint checkpoint was 53.57% validation
accuracy and 0.69305 NLL; its paired 28-cluster bootstrap intervals include no
improvement. Audio/video-only validation remained at chance as expected from
the balanced marginal construction. This is a low-power overfit result, not
general AV quality or product evidence. Full metrics, hashes, failures, runtime,
and the per-example prediction record are documented in
[`R2_GREATEST_HITS_AV_MATCH_READOUT_RESULTS_20261018.md`](R2_GREATEST_HITS_AV_MATCH_READOUT_RESULTS_20261018.md).
An independent CPU process reloaded all three selected checkpoints; all 336
validation probability rows matched the saved predictions exactly (maximum
absolute difference 0.0), with exact argmax parity. Its verifier and receipt
hashes are in the result report.

## HotUGPU handoff race and guard

The earlier wrapper's first attempt waited for Orbitune PID 9412 to yield. After
the process exited, its five-second quiet interval left a restart race: another
GPU process appeared at the final reservation check. The wrapper rejected the
launch and released its small guard; the audio feature child never started, and
no process was stopped or signaled. A separate focused implementation,
`scripts/run_with_hotgpu_guard.py`, reserves VRAM immediately after the first
idle sample, checks for external PIDs/utilization/free-memory again, and waits
again if a race occurs. It launches only when no external compute PID remains,
utilization is below 5%, and free memory is in `[8,000, 14,000)` MiB after the
reservation. It is covered by CPU tests; the real feature job is its first
end-to-end use. No Orbitune process or files are modified.
