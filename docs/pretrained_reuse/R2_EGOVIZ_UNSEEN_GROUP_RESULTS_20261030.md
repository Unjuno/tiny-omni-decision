# EgoViz unseen-session temporal correspondence — 2026-10-30

## Protocol recorded before fitting

This executes the [frozen 44-session plan](R2_EGOVIZ_UNSEEN_GROUP_TEMPORAL_PLAN_20261029.md)
after the [two-session smoke](R2_EGOVIZ_UNSEEN_GROUP_TEMPORAL_SMOKE_20261029.md).
Train/development session selection, seed, timestamps, four candidates, frozen
VideoPrism/MOMENT encoders and the existing small readout remain unchanged.
The 20 sealed sessions are metadata only and are not acquired or evaluated.
The old 16-session pilot is retained and excluded from the new split. Since
development sessions change, comparison with that pilot is a reference rather
than a controlled measurement of the effect of adding training sessions.

Feature acquisition runs sequentially on CPU from source commit
`3fefdb2a6b8d170d08bce44120711a2191264f3a`. Each newly downloaded MCAP is
hash/CRC checked, converted, encoded and independently validated before its
temporary raw file is released. Source receipts, source SHA-256, synchronized
window corpus, features and validation records remain available. The two
existing smoke sessions are reused without changing their original artifacts.
The download gate preserves 3,000,000,000 bytes beyond the next payload.
No existing checkpoint is deleted and no active GPU process is interrupted.

The fixed CPU fit uses seed 17, AdamW, LR 0.003, weight decay 0.01, batch 32,
at most 100 epochs, and best development NLL with patience 12 / minimum
improvement 1e-6. The joint scorer has 81,920 parameters; the sensor-only
control has 513. This executes the previously specified early-stopping rule;
it is not a long-budget or grokking test. A separate process must reproduce
saved best-checkpoint predictions before results are accepted.

## Prospective inference controls

Before any new fit or development metric, the following controls were fixed
in an external plan at `2026-10-30T04:22:48.530519+00:00`:

1. Unaltered features.
2. Reverse the eight video feature positions.
3. Reverse the sixteen sensor feature positions.
4. Reverse both streams.
5. Circularly shift sensor features by two patches (one video slot).
6. Missing video features (zero input).
7. Missing sensor features (zero input).

All use the same selected weights, development IDs, four candidates and targets.
No fitting, calibration, checkpoint selection or raw-media re-encoding occurs.
The script checks input/checkpoint hashes, prediction IDs/targets, saved-output
parity and unchanged weights. Paired session bootstrap uses 2,000 replicates,
seed 17, and percentile 95% intervals; it remains a diagnostic on development
data that already selected the checkpoint.

These are **cached feature-position controls**, not tests of upstream encoders
on reversed media. Reversing both streams preserves this scorer algebraically;
that is not learned robustness. Missing-input logits are tied, so the report
separates actual first-index tie breaking from expected random-tie Accuracy
(25%). Normalized dot products bound logits to [-1, 1], limiting four-choice
confidence to about 71.1%; this is not an Accuracy ceiling. The fixed fit is
not changed to remove this limitation.

External execution root:
`C:\CodexArtifacts\egoviz-unseen-group-acquisition-20261030-attempt-02`

- Frozen split SHA-256: `8f19a087ab0d63031ce28045def2767acfe03290951da0f4b2abfc327428672c`
- Prospective `postfit-control-plan.json` SHA-256:
  `c6c6a743e07a1729e03c718767e9998a22a90d88a5bb09078d0de0193f41c9cc`
- Diagnostic runner SHA-256 at freeze:
  `9e55abc25ce013d9accb50041664460c8d3608c360da3b76cbe29b1c06e8d1b0`

The added tests use synthetic arrays and small randomly initialized CPU
fixtures; they do not load media, pretrained weights, sealed data or a GPU.
The only existing readout change replaces a hard-coded eight-group limitation
with the actual development group count. Model math and optimization are
unchanged. Results will be appended after actual execution and reload checks.
