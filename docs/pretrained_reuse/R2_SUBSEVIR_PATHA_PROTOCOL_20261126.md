# sub-SEVIR frozen-feature Path A protocol — 2026-11-26

Status: completed for the pre-registered frozen-feature Path A conditions.
Feature extraction, three readout fits, controls, saved-checkpoint reload
parity, and split-integrity audit are recorded in
[`R2_SUBSEVIR_PATHA_RESULTS_20261126.md`](R2_SUBSEVIR_PATHA_RESULTS_20261126.md).
This is a narrow sensor/weather Decision experiment, not an Omni quality claim.

## Task and frozen split

- Source: official `sub-sevir` Zenodo record `10.5281/zenodo.7011372`, v0.1,
  CC BY 4.0. The verified archive and extracted official **train** Zarr are
  outside Git under `C:\CodexArtifacts\pretrained-reuse-r2-active`.
- Only the publisher's training Zarr is used. Publisher validation and test
  archive members have not been extracted or read.
- Task adaptation: given eight consecutive 5-minute frames from a storm event,
  predict whether `label_1d_class` is positive on the immediately following
  frame. This is a project-created five-minute forecast, not the publisher's
  original per-frame classification task.
- A candidate target is retained only when all eight inputs and the target are
  exactly five minutes apart. This excludes irregular-cadence windows without
  looking at labels.
- Split seed is 17. A SHA-256 threshold assigns each whole event ID to train or
  validation; labels are not used for assignment. The frozen manifest is
  `C:\CodexArtifacts\pretrained-reuse-r2-active\subsevir-patha-manifest-seed17-v1.json`,
  SHA-256 `3adcaf1713a562772ff44ba0bbf15533c411eb187699b96d0764139745e46b4b`.
- Manifest: train 3,969 event groups / 15,828 windows; validation 998 event
  groups / 3,988 windows. There is no event-ID overlap. The 998 validation
  groups exceed the policy's 200-group minimum; windows from a storm are
  treated as clustered observations, never as independent bootstrap units.
- The manifest fixes sample IDs, input row indices, targets, split IDs and
  source-array hashes. No validation score has been observed.

## Frozen representation candidate

The initial pinned VideoPrism-B runtime was confirmed on real train data, but
its native JAX backend is CPU-only on this Windows runtime. Batch 8 took 15.23 s
for satellite and 9.89 s for radar, after the separate one-time compile cost.
Extrapolating these measured batch times to all 19,816 windows would take about
17.2 hours for the two streams, before Zarr and image-preprocessing overhead.
Bulk VideoPrism extraction is therefore deferred on runtime cost grounds; this
is not a negative quality result.

The bounded Path A candidate is pinned DINOv2-small:

- `facebook/dinov2-small@ed25f3a31f01632728cabb09d1542f84ab7b0056`;
- Apache-2.0 per the exact pinned model card;
- local `model.safetensors` SHA-256
  `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`;
- 22,056,576 frozen FP32 parameters; strict load on RTX 3080 Laptop passed.
- Before any fitting, one actual train window produced finite `[8,768]` frame
  sequences for both sensor streams. The combined 16-frame forward took
  0.205 s and peaked at 194,804,224 allocated VRAM bytes. This is a smoke, not
  a throughput or task-quality result.

The four source channels are mapped to pseudo-RGB as satellite
`[ir069, ir107, vis]` and radar `[vil, vil, vil]`. Per-channel p01/p99 ranges
are estimated only from deterministic every-97th-pixel samples of training
events, then clipped to `[0,1]` and passed through the pinned DINOv2 image
processor. This mapping is a domain-transfer hypothesis; satellite and radar
quality must be measured independently. Validation pixels never set the
normalization ranges.

For each frame, the readout input is concatenated DINOv2 CLS and mean patch
tokens (768 dimensions). A small temporal LSTM processes the ordered eight
features. The encoder stays frozen. No validation features or labels are used
to fit the encoder or normalization.

## Readout comparison plan

Before opening validation metrics, the implementation will freeze these
conditions for seed 17:

- satellite-only, radar-only, and two-stream LSTM readouts use the same fixed
  event split, training windows, batch size 128, cross-entropy loss, AdamW
  (`lr=1e-3`, `weight_decay=1e-4`), gradient norm cap 1.0, and at most 60
  epochs;
- each stream uses a one-layer LSTM with hidden size 64; the joint candidate
  uses one such LSTM per stream, concatenates final states, then a 32-unit GELU
  scorer; each candidate outputs two logits;
- evaluate each epoch on the fixed validation IDs; choose its own checkpoint
  by minimum validation NLL, `min_delta=1e-4`, patience 10. No other config or
  threshold is selected from validation; all attempted curves are retained;
- after selection, report Accuracy, NLL, Brier, ECE-15, mean confidence and
  confusion counts, with a 10,000-replicate event-group bootstrap (seed 17);
- run input-zero, radar event-shuffle, and time-reversal controls once on the
  selected joint checkpoint. No refit follows these controls;
- compare joint against each unimodal baseline. A joint-input benefit remains
  unproven unless Accuracy and NLL both improve under paired event-group
  uncertainty.

This task can establish only five-minute lightning prediction from these
GOES/NEXRAD-derived fields. It cannot establish natural-image, general video,
audio, language, synchronized cross-modal reasoning, or 95–98% product quality.
The source's label distribution is near balanced, so majority baseline and
proper scores remain necessary.

## Artifacts and next gate

All frozen manifests, feature caches, checkpoints, metrics and model outputs
stay outside Git in the dedicated active artifact directory. The completed
Path A result does not close R2: a rights-admitted product-aligned task battery,
multi-input-required evidence, and remaining modality quality gates are still
needed. VideoPrism bulk extraction was deferred based on measured local runtime
before fitting; this was not a quality comparison. The v3 split audit reports
zero event, raw-frame, and full two-stream normalized-sequence overlap, but
repeated normalized radar-only encodings remain a limitation for interpreting
the radar unimodal condition. Sealed audit, publisher validation/test payloads,
and legacy worktrees remain untouched.
