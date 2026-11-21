# DextraVerse video+tactile contact Decision pilot — 2026-11-22

## Candidate and scope

This is a bounded R2-M candidate for testing whether a small Decision readout
can reuse frozen video features and causal tactile history to choose a likely
near-future fingertip-contact outcome. It is **not** a four-modality benchmark
or evidence for the product accuracy target. The dataset release is public,
ungated, and marked CC BY 4.0; attribution is required. The pinned archive is
being fetched to an external attempt directory; no member data have yet been
parsed, decompressed, or used for fitting.

The candidate uses only the real-robot `pick_cube_place_box` task. Its 261
episodes contain 138,382 frames from one fixed cube-and-box task, recorded on
one robot setup. The README describes SpaceMouse arm teleoperation and a
learned grasp policy. It does not establish participant identities or an
official split. Episodes are the proposed indivisible groups; this supports
within-task, held-out-episode diagnostics only. It cannot establish unseen
operator, object, task, or embodiment generalization.

## Pinned source and resource preflight

- Dataset: [official Hugging Face repository](https://huggingface.co/datasets/ntumars/DextraVerse)
- Revision: `257731928e25cd068b02d80d18d8f3cb77075a14`
- Official card license: CC BY 4.0; repository is public and `gated=false`.
- Pinned API metadata response: 4,528 bytes; SHA-256
  `6cf0084a1f4d589a2e92ce7bb3e3d4af4bd4f4494626c8179f9c678e546a0e95`.
- Pinned file-tree response: 11,817 bytes; SHA-256
  `9a7b6e6b897ca089f22855a943e6fe17eddc5cfe587c116ff28fb81e1118c4d2`.
- `DextraVerse/pick_cube_place_box.zarr.zip`: 44,197,744,857 bytes; HF blob
  OID `7456e5175751abb6fd833e242c412ea62412def4`; publisher SHA-256
  `9f9d8d0e6a6eebac12f1bf63fae59bea46e9531db2b9921440aa553289d07aa4` from
  the 509-byte `SHA256SUMS.txt` (SHA-256
  `9f35f048b691eb0220c6cbecde11b94d38aed8b7d71cc9903e5c082075430968`).
- Planned frozen video encoder, only if the array preflight passes: Google
  VideoPrism-B, Hub revision
  `be719a406d563b66f0ac969e7c94bab8e997c81a`, upstream source commit
  `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`, 114,365,184 parameters,
  checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`.
  It will stay frozen; source-faithful 288-pixel resize/crop and model
  preprocessing will be recorded. The existing checkpoint and upstream tree
  are read-only.
- A ranged read of the ZIP64 central directory (1,647,106 bytes, 11,740
  members) reports 44,194,715,939 uncompressed bytes (41.16 GiB). Prefer a
  read-only Zarr `ZipStore` to avoid a second 41.16 GiB extracted copy. If the
  installed Zarr reader cannot safely read this pinned ZIP64 archive, archive
  plus full extraction would use about 82.3 GiB. C: had 201.8 GiB free at
  preflight; recheck before extraction or cache writes and preserve the fixed
  2 GiB reserve. Keep all payloads and caches outside Git.

## Proposed fixed task

Question: **Which fingertip contact outcome will occur next?** Given recent
head-camera video and the past force series from both hands, choose among 12
fixed options: `left_thumb`, `left_index`, `left_middle`, `left_ring`,
`left_little`, `right_thumb`, `right_index`, `right_middle`, `right_ring`,
`right_little`, `simultaneous`, or `no_new_contact_within_1s`.

Proposed deterministic target transform, to be applied identically in all
folds:

1. Concatenate left then right fingertip forces at every timestamp; do not
   infer an active side. Use the head/egocentric camera for every episode, so
   side-specific camera selection is not required. Retain single-hand,
   bimanual, and never-grasp episodes. A bimanual target event can produce the
   `simultaneous` class.
2. Form intended query times on an episode-relative 0.5-second grid, beginning
   1.0 second after start and ending 1.0 second before end. Use the nearest
   tactile timestamp only when it is within 25 ms; that matched timestamp is
   the actual query time.
3. Compute per-fingertip force magnitude as the Euclidean norm of the three
   calibrated force components (newtons), separately for all ten fingertips.
   At query time, mark each fingertip
   in-contact at magnitude >= 0.5 N. A target event is a below-to-at-least
   0.5 N rising crossing in `(query_time, query_time + 1.0 s]`, found from the
   full episode force series. Already-contacting fingertips are not new events.
   The causal input remains only the past 1.0 s of force history. Crossings on
   distinct fingertips on either hand within one tactile sample interval of
   the earliest crossing are `simultaneous`; otherwise the label identifies
   the hand and fingertip. No crossing is
   `no_new_contact_within_1s`. Queries with invalid force at query time or no
   valid head-camera frame are ineligible, not silently assigned a negative
   label. Any non-finite force value or timestamp gap above 200 ms in
   the past-input or future-target window makes that query ineligible.
4. Use only the head camera. Sample eight frames nearest to
   equally spaced target times in the 1.0-second interval ending 0.25 seconds
   before query time; require each selected frame timestamp to be within 50 ms
   of its target. Use only causal sensor samples at or before query time.
   Reject windows that cross a timestamp gap above 200 ms; do not reject whole
   episodes because of local stalls.

These rules were frozen before payload retrieval. Verify the 0.5 N threshold,
camera timestamp semantics, label support, and resulting class balance from
the pinned arrays before any fit. If fewer than 1,000 valid queries or 200
episode groups remain, or any of the 12 classes has fewer than 50 queries from
at least 15 distinct episodes, stop and report insufficient support rather
than collapsing classes or relaxing the rules.

## Split, arms, and evaluation

- No publisher train/validation/test split is documented. For each episode
  index `i`, compute the lowercase hex digest of UTF-8 bytes for
  `dextraverse-contact-decision-v1|17|outer|i`; sort by digest then integer
  index, and assign sorted position modulo five as the held-out fold. Each
  valid query receives one OOF prediction. No episode may cross folds.
- In each outer training partition, hash
  `dextraverse-contact-decision-v1|17|inner|<outer_fold>|i` the same way and
  sort. Reserve the first `ceil(0.10*n)` groups for calibration, the next
  `ceil(0.10*n)` for checkpoint selection, and the rest for fitting. If fewer
  than 10 episode groups remain or any subset is empty, stop rather than fall
  back to query-level splitting. The outer held-out fold is never used for
  either calibration or checkpoint selection.
- Fit feature normalization on fitting episodes only. Choose each checkpoint
  by the lowest equal-episode mean NLL on the reserved selection episodes
  (earliest epoch on ties); fit one scalar temperature on calibration episodes
  only. Score the outer-fold predictions once with these frozen choices.
- Compare frozen-video-only, causal-tactile-only, and paired video+tactile
  readouts on identical query IDs and 12 options. The existing causal
  `SensorTemporalLSTMStem` and `VideoSensorTemporalDecisionScorer` are the
  starting integration; no encoder is unfrozen.
- Use the existing scorer defaults with 768 video features, 30 force channels,
  12 one-hot options, 64-wide sensor LSTM, 96-wide/four-head fusion, and a
  128-wide candidate scorer: CPU instantiation measures 261,059 trainable
  readout parameters. No encoder parameters are included in that count.
- Encode the 12 fixed outcome types as a fixed one-hot typed option schema;
  keep option IDs and their order in the result JSON. Do not vary option count
  or labels after inspecting model scores.
- Before fitting, measure a train-fold class-prior/option-only baseline and an
  episode-time-only baseline; these receive no video or force content and use
  the identical outer folds. Report any shortcut performance directly.
- Required controls: no tactile, episode-shuffled tactile, a fixed 500 ms
  tactile offset, no video, and option-order permutation. Add a
  parameter-matched null-tactile arm by replacing every tactile window with an
  identical all-zero 1-second sequence on a fixed 50-ms time grid, retaining
  the sensor branch. This lets its full 261,059 readout parameters train while
  withholding tactile information; report it separately from the strict
  missing-tactile arm.
- Report all-example and macro-class Accuracy, equal-episode macro Accuracy
  and NLL, Brier, ECE-15, calibration/coverage, per-class counts, per-episode
  results, and 10,000 episode-cluster bootstrap replicates (seed 17). Report the actual
  number of groups and examples; episode OOF does not imply unseen-person or
  object generalization.
- Train only the small readout, with a fixed maximum budget and nested
  calibration-based selection. Save best and final separately. No sealed audit
  exists for this source; do not label OOF output a blind test.
- Call paired-input complementarity supported only if paired OOF predictions
  beat the best unimodal arm on both equal-episode macro Accuracy and NLL:
  the paired-minus-unimodal Accuracy 95% interval must be entirely above zero,
  and its NLL interval entirely below zero. For each no-video, no-tactile,
  offset, and shuffled-tactile intervention, the intervention-minus-paired
  NLL interval must be entirely above zero. Otherwise record uncertainty or a
  negative result; this is a preregistered development criterion, not the
  product-quality gate.

## Stop gates and limitations

Stop before training if the archive hash fails, episode boundaries or force
arrays do not match the pinned README, timestamps are invalid, the target
transform yields inadequate support, or there are fewer than 200 usable
episode groups / 1,000 queries. Stop a fit on NaN/Inf, frozen-weight mutation,
OOM, or storage below the reserve. Do not search thresholds or revise the
query grid after seeing validation/OOF scores.

The dataset uses one task, one fixed purple-cube/orange-box setup, few camera
views, and no disclosed participant IDs. Exact camera exposure timing is not
documented in the release notes, so temporal claims must remain limited to the
declared coarse 250 ms video guard and the timing-offset control. Even a
positive result would be a narrow sensor-fusion mechanism result, not a
product-quality pass.

The deterministic contact-window/episode-split helper is implemented in
`scripts/pretrained_reuse/dextraverse_contact.py`; 12 CPU tests cover synthetic
crossings, simultaneity, causal cutoffs, gaps, missing values and grouped split
determinism. These tests establish code behavior only. No real source rows,
feature cache, model fit, or quality metrics exist yet. A synthetic Zarr v2
`ZipStore` smoke passed end-to-end through the read-only source preflight and
returned 26 valid fixture queries; it does not verify the downloaded archive.
The real-data preflight script will write stable episode/fold/sample IDs,
target labels, causal frame indices and sensor index ranges to the external
attempt report, alongside per-class and duplicate-media gates.

## Evidence links

- [Hugging Face dataset card and license](https://huggingface.co/datasets/ntumars/DextraVerse)
- [Real-robot array schema and known recording issues](https://huggingface.co/datasets/ntumars/DextraVerse/blob/257731928e25cd068b02d80d18d8f3cb77075a14/DextraVerse/README_real_robot.md)
- [Project dataset page](https://ntumars.github.io/DextraVerse/dataset/)
- [Project tactile-policy results](https://ntumars.github.io/DextraVerse/tactile/)
