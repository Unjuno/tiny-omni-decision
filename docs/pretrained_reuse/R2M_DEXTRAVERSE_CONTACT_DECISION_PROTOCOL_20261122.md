# DextraVerse video+tactile contact Decision pilot — 2026-11-22

## Candidate and scope

This is a bounded R2-M candidate for testing whether a small Decision readout
can reuse frozen video features and causal tactile history to choose a likely
near-future fingertip-contact outcome. It is **not** a four-modality benchmark
or evidence for the product accuracy target. The dataset release is public,
ungated, and marked CC BY 4.0; attribution is required. The pinned archive was
downloaded outside Git and passed its exact byte-count and publisher SHA-256
checks. A read-only array preflight completed but failed the frozen per-class
support gate; no features were extracted and no model was fit.

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
- The locally retrieved pinned `README_real_robot.md` is 3,449 bytes and
  SHA-256 `8cb64a0519db1fa60f4963eaf99af98f3c098ffb8f90c24e14a3fa80923db20c`,
  matching the source manifest. It documents episode-relative seconds,
  approximately 19.5 Hz sampling, Newton-valued fingertip forces in each
  sensor's own frame, and a learned grasp policy triggered by operator
  command. Since most episodes use one hand and the behavior is policy-driven,
  task-phase or hand-side cues may predict contacts without genuine video/tactile
  complementarity; unimodal and input-intervention controls are necessary.
  These source conventions were subsequently checked against the pinned
  archive arrays in the read-only preflight below.
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

### Completed read-only archive preflight

The archive reached the pinned 44,197,744,857 bytes and SHA-256
`9f9d8d0e6a6eebac12f1bf63fae59bea46e9531db2b9921440aa553289d07aa4`. Its Zarr
group is nested at `pick_cube_place_box.zarr/`; the initial root-only open
failed with `GroupNotFoundError` before reading arrays. The preflight and
feature extractor now resolve one unique top-level Zarr group, with tests for
root, wrapped, and ambiguous archive layouts. The corrected reader reported
261 episode boundaries, 138,382 timestamps, both `[138382, 5, 3]` force arrays,
and head RGB `[138382, 240, 320, 3]`.

The frozen target transform produced 10,983 eligible windows across all 261
episodes. Every timestamp sequence was strictly increasing. No duplicate
head-video hashes or cross-fold media identities were found. The one failing
gate is `left_index`: 38 queries from 18 episodes, below the fixed minimum of
50 queries (the episode minimum of 15 passes). The other 11 classes pass both
support thresholds. The fold episode counts are 53/52/52/52/52. Per the
predeclared no-class-collapse rule, stop this source candidate before feature
extraction, Decision fitting, or quality evaluation. The full 8,476,768-byte
preflight report is retained outside Git at
`C:/CodexLocalArtifacts/pretrained-reuse-r2/dextraverse-contact-decision-attempt-01/real-data-preflight-02/report.json`
(SHA-256 `f7806087ef47405c3e270445842250770d589970a7b40f8971032d82fb24f014`).

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
views, and no disclosed participant IDs. Arm motion is teleoperated and hand
closure uses a learned policy after an operator command, so apparent
anticipation may reflect task phase or the policy's regularity rather than
cross-modal evidence. Exact camera exposure timing is not documented in the
release notes, so temporal claims must remain limited to the declared coarse
250 ms video guard and timing-offset control. Even a positive result would be a
narrow sensor-fusion mechanism result, not a product-quality pass.

The deterministic contact-window/episode-split helper is implemented in
`scripts/pretrained_reuse/dextraverse_contact.py`; 12 CPU tests cover synthetic
crossings, simultaneity, causal cutoffs, gaps, missing values and grouped split
determinism. `scripts/pretrained_reuse/extract_dextraverse_contact_features.py`
now provides a separate hash-gated, read-only ZIP64 `ZipStore` path that writes
temporal VideoPrism features and query-relative tactile sequences to an
external, non-overwriting output directory. It preserves the exact preflight
sample order and causal timestamps and records individual array hashes. Four
CPU tests cover temporal pooling, left/right sensor ordering and causal
timestamps; two more verify binary framing between the isolated Zarr and
VideoPrism environments; three cover CLI/runtime forwarding and archive-root
resolution. All 21 targeted CPU tests passed. A synthetic Zarr v2 `ZipStore`
smoke returned 26 fixture queries, and a metadata-only open of the real ZIP
confirmed its wrapped root and array shapes. The full read-only preflight
measured the real source and stopped on insufficient `left_index` support.
No frozen-video feature cache, model fit, or quality metric was produced. The
failed root-only attempt and corrected run are documented; the full report and
all source data remain outside Git.

## Runtime preflight

The existing native VideoPrism environment has Python 3.11.16, JAX 0.6.2 and
TensorFlow 2.18.1. A GPU request failed with JAX `Unknown backend: gpu`; the
environment enumerates only `TFRT_CPU_0`. The native RTX 3080 is visible to
Windows and WSL2, but the inspected Ubuntu 24.04 has no installed JAX, PyTorch,
TensorFlow, Zarr or Numcodecs runtime. No package was installed or upgraded.
One cold one-example CPU pipe smoke passed in 17.09 seconds including model
startup/first compilation; it produced finite `[1, 8, 768]` features. That cold
time is not multiplied across the corpus. The prior pinned-model runtime study
reports 1.178 seconds per item for warmed batch-8 inference and 4.15 GB peak
RSS on the same VideoPrism revision and eight-frame model input; its separate
rough estimate was 6.2 hours for 13,688 clips including media preprocessing.
Treat this as planning evidence only: DextraVerse failed the fixed class-support
gate, so no feature extraction was run. The extractor defaults
to batch 8 and reads each episode's RGB span once when it fits within a 1 GiB
cache cap. A separate batch-parity smoke compared one synthetic clip alone and
as the first item of a batch of eight: maximum absolute feature difference was
`1.91e-6`, within the predeclared `atol=rtol=1e-5`; the batch-8 first-shape
compile took 21.87 seconds. This validates batch invariance, not throughput. A
regression test verifies that the required VideoPrism Python path is forwarded
by the CLI. See the prior [runtime smoke](R2_AVATAR_ENCODER_RUNTIME_SMOKES_20261101.md).
Local receipts are under
`C:/CodexLocalArtifacts/pretrained-reuse-r2/dextraverse-contact-decision-attempt-01/`:
`gpu-backend-probe-01/probe.json`, `gpu-worker-smoke-01/smoke-report.json`, and
`cpu-batch8-parity-smoke-01/batch-parity-report.json`.

## Evidence links

- [Hugging Face dataset card and license](https://huggingface.co/datasets/ntumars/DextraVerse)
- [Real-robot array schema and known recording issues](https://huggingface.co/datasets/ntumars/DextraVerse/blob/257731928e25cd068b02d80d18d8f3cb77075a14/DextraVerse/README_real_robot.md)
- [Project dataset page](https://ntumars.github.io/DextraVerse/dataset/)
- [Project tactile-policy results](https://ntumars.github.io/DextraVerse/tactile/)
