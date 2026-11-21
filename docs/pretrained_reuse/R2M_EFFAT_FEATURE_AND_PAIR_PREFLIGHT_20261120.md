# R2-M frozen audio feature and pair preflight (2026-11-20)

> Status update: the pair index and full allowed feature cache were subsequently
> built, and the fixed-budget pilot completed. The final result supersedes this
> report's intermediate “Next gate” language; see
> [the full pilot report](R2M_AUDIO_VIBRATION_MATCH_PILOT_20261120.md).

## Frozen pretrained audio representation

Using one permitted train record only (`01_walk.zip`, participant 2, trial
1649), the pinned EfficientAT DyMN-0.4 AudioSet checkpoint loaded strictly and
ran in CPU FP32 eval mode. The checkpoint SHA-256 is
`49b0e63577021994257b3de2ac5f1686d71554b93c8149f9d2568ad5760b904e`; the
upstream revision is `a425fdce92572e602a1d5634799bd9f1f2efa806`. The encoder has
1,961,239 parameters, all frozen. Official mel preprocessing at 32 kHz produced
35 temporally ordered feature tokens of width 384; the tensor was finite. The
feature-byte SHA-256 is
`34528184e6cac158fdfa69b7beaa5ca03d7a1c3fd830a4417da848c5e4e08695`.

This was an inference-only representation smoke, not model fitting or a quality
result. The waveform and feature tensor were not persisted. Runtime record:
`D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-pretrained-feature-smoke-v1-attempt-02\result.json`.

The committed extractor was separately run twice on the same train sample with
one CPU thread. Both executions produced `[35,384]` and byte-identical feature
SHA-256 `e662c72ef734c18ca22ee65cc0431743786e33651ea22e788b1ae9591f64349b`;
the vibration resampler returned finite `[128,3]`. This makes the current
extractor reproducible under its recorded environment. Its hash does **not**
match the earlier smoke hash above. That earlier run did not record the exact
runtime/thread configuration or tensor artifact, so the cause is UNKNOWN and
the two runs are not claimed equivalent. Both outputs and the discrepancy are
preserved under separate `r2m-feature-extractor-script-smoke-v1-attempt-*`
directories. The extraction pipeline now pins one CPU thread and its own
repeat hash before caching the full feature set.

## Deterministic pair-index implementation

Added `tiny_omni_decision.r2m_pairs.build_audio_vibration_pairs`. It creates
same-activity, different-trial duration-matched negatives, prefers a different
participant, minimizes negative reuse, and fails closed on sealed/quarantined
rows or participant split crossings. The immutable v2 participant split is
used; the superseded v1 splitter is not used for this operation.

A first feasibility pass used only the existing bounded layout preflight (364
records: 255 train, 44 calibration, 65 validation; at most two records per
participant/activity). At a 10% audio-duration tolerance it produced 221 train,
19 calibration, and 41 validation anchors. Validation had 11
cross-participant and 30 within-participant anchors; train had 207 and 14,
respectively. This is a small schema/preflight subset, **not** the complete
frozen corpus and not suitable for a quality claim or final fitting. Its
pair-index SHA-256 is
`43d3073d38d56d93eaf87941f08b9d01bbef0a318f4fb744af15b921b83e082a`; the
artifact is kept outside Git at
`D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-pair-index-preflight-v1-attempt-01\pair-index.json`.

Focused tests cover deterministic output, split isolation, duration gating,
negative relation preference, and duplicate rejection (4 passed); the existing
participant-split tests also pass (13 total), and Ruff passes for all added
R2-M code.

## Bounded subset limitation

The first 364-record feasibility pass was only a schema subset and was too
small for quality evaluation. It has been superseded by the full frozen
train/calibration/validation pair inventory below. No model fit or product
admission was claimed from either index.

## Full frozen pair-index result

The follow-up indexer rechecked all 24 active source archives against the
publisher receipt and local SHA-256 before reading `multisensor-1`. It read
participant/trial identifiers to select rows and dereferenced audio/vibration
arrays only for the frozen train/calibration/validation participants. Sealed
participants 1 and 16 and quarantined participant 0 were skipped before signal
access. It emitted only lengths, sample rates, source IDs, activity, and split;
no waveform or vibration values were persisted. All synchronized durations
were within 0.5%.

| Split | Allowed trials | Eligible matching anchors | Unpaired trials | Cross-participant negatives | Within-participant negatives |
|---|---:|---:|---:|---:|---:|
| Train | 5,464 | 5,448 | 16 | 5,386 | 62 |
| Calibration | 779 | 759 | 20 | 0 | 759 |
| Validation | 775 | 761 | 14 | 395 | 366 |

The duration tolerance was fixed at 10%; negatives use the same activity and a
different trial, prefer a different participant where possible, and minimize
reuse. Validation anchors must be stratified by negative relation because
calibration has only within-participant negatives. The pair-index payload
digest is `d1e1c74a1114fc0d071f71a50876b852e8d3fce8919fde9696368d4ebf365276`;
file SHA-256 is
`33664b32205626fe1ec2102961e82163d2f55586f6c0567c9f1c80c0f4d3e7bf`.
Metadata file SHA-256 is
`faee790bfaa53016e7c92f0ac545f0a7dfe23cd126fa7e1287b10d8d8b0282c0`.
Both files are outside Git under
`D:\CodexArchiveRelocated\CodexArtifacts\pretrained-reuse-r2\zenodo-audio-vibration-v1\r2m-full-pair-index-v1-attempt-01`.

This establishes that the matching task has enough grouped rows to proceed to
feature caching and a train/calibration/validation pilot. It does not establish
that the modalities contain trial-match information or that the 95–98% product
target is plausible. Next, build immutable frozen temporal features only for
the indexed allowed trial IDs; then compare a fixed temporal baseline with the
candidate-conditioned gated model, using calibration only for model selection
and validation only for final pilot measurement. Include missing, shuffled,
stale, and time-offset controls. No model was fit and the sealed audit remains
unopened.
