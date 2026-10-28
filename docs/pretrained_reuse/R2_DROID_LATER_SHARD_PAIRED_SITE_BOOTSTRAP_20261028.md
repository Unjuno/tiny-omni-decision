# DROID later-shard paired site bootstrap (2026-10-28)

## Scope

This is a post-hoc uncertainty analysis of the **already saved** later-shard
DROID predictions. It performs no model inference, training, calibration,
checkpoint selection, data decoding, or sealed-audit access. It is not an
independent or blind evaluation. Its purpose is to quantify uncertainty in the
matched VideoPrism+sensor versus sensor-only comparison already reported in
[`R2_DROID_LATER_SHARD_FROZEN_TRANSFER_RESULTS_20261028.md`](R2_DROID_LATER_SHARD_FROZEN_TRANSFER_RESULTS_20261028.md).

## Frozen input and method

- Prediction file: `C:\CodexArtifacts\pretrained-reuse-r2\droid-later-shard-transfer-evaluation-attempt-01\predictions.jsonl`.
- SHA-256: `8a358bb980d10e8a323e5547b56e5a77c2494e7cf9520a52d45194a0184f6496`, matching the saved evaluation receipt.
- Rows: 137 unique episodes/groups across 14 collection sites; all seven original options retained.
- Contrast: `video_sensor - sensor` for Accuracy, NLL, and multiclass Brier.
- Bootstrap: 10,000 paired cluster resamples, seed 17. Resample 14 sites with replacement and retain all prediction rows in each sampled site; calculate pooled-row metrics per draw. Percentile 95% intervals use the 2.5th and 97.5th quantiles.
- Analysis code: `scripts/audit_droid_later_shard_paired_site_bootstrap.py`.
- Analysis code SHA-256: `f2ef7d32e506b88204152afb70c3f1573222e84f0820093ce5f8baed1ed50b9d`.
- Output receipt: `C:\CodexArtifacts\pretrained-reuse-r2\droid-later-shard-transfer-evaluation-site-bootstrap-attempt-01\result.json`, SHA-256 `dbdfa672e4f84e60e180272169efc48e471e38b091b56b19371289b51c0b502`.

## Results

| Metric | Joint − sensor | 95% site-bootstrap interval |
|---|---:|---:|
| Accuracy | −2.92 percentage points | −8.08 to +0.97 pp |
| NLL | +0.2822 | +0.1598 to +0.4871 |
| Brier | +0.0887 | +0.0448 to +0.1595 |

Accuracy uncertainty includes no difference. The NLL and Brier intervals
indicate worse probability quality for the joint model on this sample. This
does not establish a population-level effect: only 14 sites are represented,
and the input is already observed official-train development data. Combined
with the lower point Accuracy and worse calibration in the original report,
the analysis provides no support for another fit on these IDs or for promoting
this DROID video+sensor readout.

## Verification and reproduction

The prediction hash was checked before analysis. The standalone CPU tests pass
(3 tests), the full branch CPU suite passes (2,918 passed, 11 skipped), and
Ruff passes. The script refuses invalid probability vectors, duplicate episode
groups, fewer than two sites, and output-path overwrites.
It requires the expected input SHA-256 and creates the result with exclusive
file creation.

~~~powershell
py -3.11 scripts\audit_droid_later_shard_paired_site_bootstrap.py `
  --predictions C:\CodexArtifacts\pretrained-reuse-r2\droid-later-shard-transfer-evaluation-attempt-01\predictions.jsonl `
  --predictions-sha256 8a358bb980d10e8a323e5547b56e5a77c2494e7cf9520a52d45194a0184f6496 `
  --output C:\CodexArtifacts\pretrained-reuse-r2\droid-later-shard-transfer-evaluation-site-bootstrap-attempt-01\result.json `
  --repetitions 10000 --seed 17
~~~

No existing report, predictions, source media, feature cache, readout, or
checkpoint was modified.
