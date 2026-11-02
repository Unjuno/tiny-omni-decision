# RoboMNIST audiovisual match preflight — 2026-11-03

## Frozen split and task

The new development manifest is
`C:\CodexArtifacts\pretrained-reuse-r2\robomnist-av-match-split-20261103-attempt-03\manifest.json`,
SHA-256 `3f4575ba41d944310e1de83cbdc68385390581723262245b97dc0d2c71db660f`.
It selects 780 training and 1,000 validation repetition groups, arranged as
390 / 500 independent paired units. The resulting four-example binary Decision
format has 1,560 train and 2,000 validation rows. Every media group in a pair
stays in one split. This is a development split inside the previously exposed
non-audit metadata pool, not a blind or scene-held-out benchmark. Historical
sealed-audit manifests and data were not opened.

For each pair A/B, the task is to distinguish the true same-repetition AV pair
from a cross-repetition pairing, using only the episode's Rx1 audio and video.
The two repetitions are from the same recorded condition. Group IDs, filenames,
repetition labels, CSI, and PositionGT are metadata/supervision only and are
never encoder inputs. Whole-episode association is used; audio/frame-clock
alignment is not claimed.

## Pinned feature smoke

Successful run:
`C:\CodexArtifacts\pretrained-reuse-r2\robomnist-av-match-feature-smoke-20261103-attempt-06`.
It used source commit `b0c020bb3f269ad299ccd1945cbd4870a0251a73` and the exact
attempt-03 manifest hash above.

- VideoPrism-B: 114,365,184 parameters, checkpoint SHA-256
  `61fa62eedbe4f8d4ba76a99a8b79abd0137426a4931c48204ab70446b6411f29`, upstream
  source commit `c6eba0ece3dd49a24ffc44582966997a4a59f1d8`.
- EfficientAT DyMN-0.4: 1,961,239 parameters, checkpoint SHA-256
  `49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`, upstream
  source commit `a425fdce92572e602a1d5634799bd9f1f2efa806`.
- Runtime: Python 3.11.16, NumPy 1.26.4, JAX 0.6.2 CPU, TensorFlow 2.18.1
  CPU-only, PyTorch 2.6.0+cu124 with CUDA uninitialized, Torchaudio 2.6.0+cu124.
- Both groups produced finite feature shapes Video `[8,768]` and Audio
  `[8,384]`; the frozen audio state hash matched before and after inference.
- 48.13 seconds wall time, 4.71 GB peak process working set, 8,509,914 Range
  bytes for one uncached group, and a 68,863-byte compressed feature file with
  SHA-256 `4d0e5a305cb65e39ebd441fe9149b1bf2eb782e722a981c7def42c650df2f453`.
- No training, validation metric, model-quality evaluation, raw-media output,
  GPU, or sealed-audit access occurred.

Attempts 01–05 are retained as small `INCOMPLETE` execution receipts, each with
the stopping error; none wrote raw media or a feature artifact. Failures found
and fixed were module import path, generated upstream build metadata in an
otherwise tracked-clean checkout, adjacent-PTS comparison, VideoPrism batch
shape, PCM bit depth, and a nonexistent manifest field. The pinned upstream
source revisions and checkpoint hashes were unchanged. Generated build,
egg-info, and Python bytecode paths in the VideoPrism checkout are recorded in
the successful result; they were not removed.

## Scale decision

The exact selected audio/video compressed-member total is 14,631,647,490 bytes
(13.63 GiB), which exceeds current C: free space (6.48 GiB). Keep media in
memory only while extracting; do not create a raw-media mirror. Float32 features
for all 1,780 groups require about 62.55 MiB, with a two-group compression-ratio
projection near 61.2 MB. The smoke's 48 seconds includes both encoder startup
and network transfer and is not a standalone throughput benchmark. Older
source retrieval rates span roughly 0.134–0.492 MiB/s, so total acquisition
time is material and uncertain.

## Full-cache extraction in progress

The per-group writer is implemented in
`scripts/extract_robomnist_av_match_feature_cache.py`. It checks the pinned
manifest and encoder hashes, writes one feature+receipt pair via an atomic
directory commit, validates existing hashes on resume, leaves partial attempts
in place, and stops before C: falls below 2 GiB free. It stores no raw media.
The fixed contract, split group hashes, encoder pins, and source-code hashes are
captured in the run metadata.

The full CPU-only run started at
`C:\CodexArtifacts\pretrained-reuse-r2\robomnist-av-match-feature-cache-20261103-attempt-01`.
At the 20-group snapshot it had extracted **20 / 1,780 groups** in 8.6 minutes,
transferred 158.45 MiB, and retained 6.47 GiB free. The first 13 completed
receipts averaged 25.27 seconds/group and 0.366 MiB/s for the Range requests;
the linear remaining-time estimate was about 12.4 hours at that point. This is
a changing throughput projection, not a completion guarantee. RTX 3080 usage
was 0 MiB / 0%; all encoder work was CPU-only.

The one-shot readout implementation is prepared: a 32-rank bilinear AV match
model (73,728 trainable parameters) and separate audio-only/video-only logistic
controls. It has a fixed 400-update train-only schedule, no validation
checkpoint selection or temperature fitting, option-ID probability output,
paired bootstrap at the 500-pair-unit level, missing/shuffled-input controls,
and save/reload parity checking. CPU tests cover grouping, metrics, the
bootstrap floor, and a two-step synthetic optimizer smoke. No readout has been
fitted and no validation quality result exists yet.

## Remaining gate

Wait for the frozen feature cache to complete, then fit and evaluate the one
predeclared readout/control suite on its fixed train/validation data. Do not
infer accuracy, calibration, cross-modal necessity, or general-purpose Omni
quality before those measured results exist.
