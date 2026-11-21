# R2-M audio-vibration temporal-bandwidth audit (2026-11-21)

## Question and boundary

Before considering another readout or training-budget change, does the current
frozen R2-M feature path preserve the time-varying vibration signal at a useful
temporal bandwidth?

This was a deterministic, read-only signal audit. It selected one trial per
available activity × TRAIN participant from the already frozen participant
split v2 and pair-index metadata, using SHA-256 ranking with seed 17. It read
133 trials from TRAIN participants only. It did not read validation or sealed
signals, the quarantined participant, labels beyond the allowlisted activity
identity, or model outputs. No model was loaded or fitted, and no existing
cache, checkpoint, split, or run was changed.

The selected archive files were rehashed against the existing verified Zenodo
receipt before HDF5 access. For each selected HDF5 row, the script matched
participant/trial identifiers first and accessed vibration arrays only for a
selected TRAIN row. The run receipt is outside Git at
`C:\CodexArtifacts\pretrained-reuse-r2m\temporal-bandwidth-audit-v1-attempt-01`.

## Source and implementation facts

The dataset paper identifies the vibration sensor as the ST LIS3DHH, publishes
integer readings corresponding to ±2.5 g, and reports a per-record sampling
rate near 1.085 kHz. The paper also explains that audio did not carry sample
timestamps: the publisher aligned audio and vibration with low-pass
cross-correlation when their signals had sufficient shared events. See
[Pfitzinger et al., 2024, Tables 3–4 and Sections 3.3–3.6](https://www.mdpi.com/2306-5729/9/12/144).
The HDF5 vibration datasets in the inspected payload have no unit or calibration
attributes; the g conversion below uses the paper's published integer range.

The frozen extractor currently maps the entire variable-length sensor record
to exactly 128 points with linear interpolation across normalized time `[0,1]`.
The temporal scorer then timestamps these points evenly across the activity
duration. For the endpoint-inclusive 128-point grid, the effective sample rate
is `127 / duration_seconds`, and its Nyquist limit is `63.5 / duration_seconds`.
This is a varying, duration-dependent representation: short and long trials do
not share one sensor bandwidth.

## Measured TRAIN-only results

The deterministic sample contained 133 trials, spanning the available
activity-participant combinations across the seven TRAIN participants. Source
vibration rates were 1,076.1–1,084.4 Hz (median 1,079.8 Hz). Trial durations
were 2.262–213.714 seconds (median 4.303 seconds).

| Quantity | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Effective 128-token sample rate | 0.594 Hz | 29.516 Hz | 56.143 Hz |
| Effective Nyquist limit | 0.297 Hz | 14.758 Hz | 28.072 Hz |
| Raw source rate / effective token rate | 19.20× | 36.55× | 1,817.24× |

After removing each axis's mean for this diagnostic only, the median fraction
of raw spectral energy above the effective token-grid Nyquist limit was
**98.21%** across the 399 sampled trial-axis signals (10th–90th percentile:
95.70%–99.81%). This is a spectral exposure diagnostic. It is **not** a direct
measurement of what the linear interpolator retains, a model-quality estimate,
or proof that high-frequency content is task-relevant. Linear interpolation
does attenuate some content and may alias the remainder; the current features
do not explicitly compute an anti-aliased pooled envelope or spectrum.

Using the paper's ±2.5 g mapping, the median norm of the three per-trial mean
axes was 1.0058 g. This is consistent with a large static gravity component in
the installed sensor frame; it does not indicate that the signed integer data
were malformed. The dynamic per-axis standard deviation had a median of 7.63
counts, so the dynamic signal is small relative to the static baseline.

## Interpretation and next controlled step

The previous same-trial matching model and simple correlation baseline were
near chance. This audit identifies a plausible feature-path limitation: a
roughly 1.1 kHz signal is represented as 128 point samples over durations that
range from seconds to over three minutes. It does not establish that this
limitation caused the prior result or that a new representation will help.

The next bounded experiment is preregistered as a **separate feature-cache and
run**: replace point interpolation with 128 equal-duration bins, each storing
per-axis RMS of the mean-centered raw vibration within that bin. This preserves
high-frequency vibration energy as a temporal envelope while keeping the
existing `[128,3]` scorer interface and model architecture fixed. Reuse the
same pinned audio features, pair index, seven TRAIN participants, seed 17,
fold plan, optimizer schedule, and calibration-only checkpoint selector.
Compare audio-only and audio+sensor using the same grouped OOF rows; report
participant-level paired effects and proper scores. Do not use existing
calibration or validation participants for tuning, and do not open sealed
audit. A result from these seven training participants remains development
evidence only. If fusion does not improve on audio-only under this fixed
representation, retire this source's fusion path rather than expanding model
capacity or increasing epochs.

The RMS-bin cache implementation and its exact cache hash must be frozen before
the controlled fit. Existing linear-interpolation cache and historical runs
remain unchanged.

## Reproduction and hashes

Base source commit: `f005d7c0ea9880e17d4581ae9e23569439e54f36`; exact audit script
SHA-256: `66ee1b60c04ca6870c16c27ae0926d4678f519583abd1d49a4831ce86e1a334`.
The pure bandwidth helper SHA-256 is
`ee862352f400afc3bdb0a298f8011c0a3c053b636a18fbd877de5f1883e73d12`, and its
CPU test SHA-256 is
`83ec29f8b6f48135ff3b9574cf165dda8497cbb780568ba2a6ee8d8a52dce704`.

- Participant split v2: `0f1539f3ca4fa59fb854f62def1c538e1f5851576444b678af4aa050a393a6f0`.
- Pair-index internal digest: `d1e1c74a1114fc0d071f71a50876b852e8d3fce8919fde9696368d4ebf365276`.
- Allowed metadata SHA-256: `faee790bfaa53016e7c92f0ac545f0a7dfe23cd126fa7e1287b10d8d8b0282c0`.
- Source download-index SHA-256: `715dd57fdf1967767f28f2b2246c57232dfba3e74ea548d46c83e68b882402d4`.
- Selected TRAIN rows JSONL SHA-256: `94e308a46eafee1a5e22261b6569fdd31d94aebb905a0c96faa4cc656bc4cf2d`.
- Audit summary SHA-256: `9c024c840936971d261358a4e5d2a4a616e8ad9fd849dc0f5a01d3fdc8dbccfc`.

Focused tests: **4 passed**; Ruff: **passed**. Compute: local CPU only, no GPU,
no cloud cost. Sealed-audit and validation signals remained unopened.
