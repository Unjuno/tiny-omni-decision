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
unchanged. This protocol and the diagnostic code were committed as
`982eafd4863dd67d1f013c96694cddd96cf4c674` before fitting. Executed results follow.

## Executed result

**The fixed CPU experiment, fresh-process reload and seven controls completed.**
All 44 planned train/development sessions passed the full aggregator. The result
supports temporal correspondence signal in the reused representations, while
showing substantial generalization and calibration gaps. It does not close R2-M,
establish a useful four-modality product, or justify compression or promotion.

| Data | Train | Development |
|---|---:|---:|
| Publisher sessions / source clips | 24 | 20 |
| Unique two-second query windows | 332 | 307 |
| Categories represented | 12 | 12 |

There are 639 unique IMU windows, zero session overlap, and no overlap with the
old pilot's 16 sessions. Adjacent windows overlap in time within each session;
they are not 639 independent recordings. The bootstrap therefore resamples
sessions. Cross-session participant/site independence remains unknown.
Neither publisher-generated action labels nor sealed media were loaded.

### Same-development-set comparison

| Condition | Accuracy | NLL | Brier | ECE | Mean confidence |
|---|---:|---:|---:|---:|---:|
| Selected joint temporal scorer | 55.70% | 1.2190 | 0.6607 | 0.2235 | 0.3335 |
| Selected sensor-only control | 28.66% | 1.3873 | 0.7505 | 0.0282 | 0.2585 |
| Joint scorer, wrong video within session | 22.80% | 1.4341 | 0.7735 | 0.0948 | 0.3228 |
| Video-only uniform expectation | 25.00% | 1.3863 | 0.7500 | 0.0000 | 0.2500 |

The joint-minus-sensor Accuracy difference is **+27.04 percentage points**,
paired 20-session bootstrap 95% interval **[+22.51, +32.00]**. The joint-minus-
mismatched-video difference is **+32.90 points [ +25.77, +39.93 ]**. Both use
2,000 replicates, seed 17, percentile intervals. These intervals describe the
observed development comparison; they do not remove checkpoint-selection bias
or establish participant-independent performance. The video-only result is
an expected uniform baseline, not a separately fitted network.

Option permutation `[2, 0, 3, 1]` preserved logits up to tolerance and retained
55.70% Accuracy after target remapping. Fresh-process reload reproduced joint
and mismatched-video probabilities with **maximum absolute difference 0.0**;
sensor-only predicted options also matched exactly.

### Learning curve and exposure accounting

| Joint epoch | Rolling train CE | Development Accuracy | Development NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| 1 | 1.3910 | 25.41% | 1.3860 | 0.7498 | 0.0056 |
| 5 | 1.1874 | 52.44% | 1.2563 | 0.6815 | 0.2078 |
| 10 | 1.0402 | 52.77% | 1.2342 | 0.6697 | 0.1977 |
| 15 | 1.0124 | 57.65% | 1.2260 | 0.6649 | 0.2450 |
| **19, selected by NLL** | **0.9644** | **55.70%** | **1.2190** | **0.6607** | **0.2235** |
| 20 | 0.9634 | 54.40% | 1.2221 | 0.6621 | 0.2092 |
| 25 | 0.9455 | 52.77% | 1.2255 | 0.6645 | 0.1924 |
| 31, stopped | 0.9306 | 51.79% | 1.2354 | 0.6701 | 0.1881 |

The selected joint checkpoint's fixed-weight training metrics are Accuracy
92.77%, NLL 0.9599, Brier 0.5121, ECE 0.5356. Its Accuracy gap to development
is **37.07 points**. After epoch 19, rolling training CE continued falling
while development NLL and Brier worsened. This is an overfit signal under the
predefined budget and selector. Rolling CE is not used as a fixed-example
generalization-gap estimate. The experiment does not test late grokking or
establish an irreducible capacity ceiling.

The predefined NLL rule selected epoch 19 even though epoch 15 had higher
Accuracy. Patience 12 stopped the joint run at epoch 31. Sensor-only selected
epoch 1 and stopped at 13. There was no restart, extra seed or changed selector.

| Model | Unique train examples / assets | Entire fit exposures / updates | Selected checkpoint exposures / updates |
|---|---:|---:|---:|
| Joint, 81,920 parameters | 332 / 24 | 10,292 / 341 | 6,308 / 209 |
| Sensor-only, 513 parameters | 332 / 24 | 4,316 / 143 | 332 / 11 |

Exposures and updates follow full passes over 332 examples, batch 32, with no
skips. They count repetitions separately from unique examples. Every epoch's
metrics are preserved in the [machine-readable result](../../manifests/pretrained_reuse/egoviz-unseen-session-temporal-results-20261030.json)
and the external `learning-curves.csv` (SHA-256
`7cf422a5feb44e9d4dc65dd46823e4a38f0559646c39cffab60cbcc7694d1659`).

The old pilot's 47.27% development Accuracy / 1.2773 NLL used 110 examples in
eight different sessions and only 106 training examples. The new 55.70% result
is higher as a reference observation; it is **not a paired causal estimate of
data scaling**, because both training and development sessions changed.

### Fixed feature-order and missing-input controls

| Control | Accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|
| Unaltered | 55.70% | 1.2190 | 0.6607 | 0.2235 |
| Reverse video feature positions | 26.38% | 1.4065 | 0.7584 | 0.0786 |
| Reverse sensor feature positions | 26.38% | 1.4065 | 0.7584 | 0.0786 |
| Reverse both streams | 55.70% | 1.2190 | 0.6607 | 0.2235 |
| Shift sensor by one video slot | 43.97% | 1.2875 | 0.6984 | 0.1178 |
| Missing either stream, first-index tie break | 26.71% | 1.3863 | 0.7500 | 0.0171 |
| Missing either stream, expected random tie | 25.00% | 1.3863 | 0.7500 | 0.0000 |

Unaltered-minus-single-stream-reversal Accuracy is +29.32 points, session CI
[+22.74, +36.00]; unaltered-minus-one-slot-shift is +11.73 points
[+6.40, +17.24]. This supports dependence on relative alignment of the cached
streams. Reversing both leaves the mathematical correspondence unchanged.
The result does not establish understanding of raw-media reversal, conditional
semantics, sensor conflicts or arbitrary missing-input robustness.

The joint scorer is underconfident on this development set (mean confidence
33.35% versus 55.70% Accuracy). Its unscaled cosine logits constrain probability
sharpness. No temperature or alternate readout was fitted after observing this
result. Calibration and ranking/generalization must be evaluated separately in
any future preregistered experiment.

## Execution, resources and retained artifacts

New acquisition began at **2026-10-30 12:54:27 JST**. The first three sessions
finished extraction at 13:00:48. The remaining 39-session controller ran from
13:07:44 to **13:49:07**. The two prior smoke sessions were reused. New transfers
totaled **9,216,715,592 bytes** across 42 files and **746.58 seconds** of recorded
download time. New VideoPrism feature extraction totaled **1,287.98 seconds**;
MOMENT totaled **28.45 seconds**. These component timers omit some process
startup, hashing, conversion and aggregation work, so they do not sum to the
full wall interval. Maximum recorded VideoPrism end-of-process RSS was
8,327,610,368 bytes; this is an end sample, **not a measured peak**.

Fit/reload/controls ran at **13:49:29–13:49:48 JST** from source commit
`982eafd4863dd67d1f013c96694cddd96cf4c674`, on an i7-12700H (14 cores / 20
logical processors), Windows 11, Python 3.11.9, PyTorch 2.6.0+cu124 and NumPy
2.4.6. Despite the CUDA-enabled package, all work executed on CPU with
`CUDA_VISIBLE_DEVICES=-1`. No packages were upgraded and Orbitune was untouched.

| Stage | Process wall time | Sampled/OS peak working set |
|---|---:|---:|
| Both fixed readout fits plus reporting | 9.01 s | 1,505,521,664 bytes |
| Fresh-process reload | 4.03 s | 1,031,426,048 bytes |
| Seven inference-only controls | 6.01 s | 1,031,020,544 bytes |

The internal joint fit timer was 2.77 seconds; sensor-only was 0.46 seconds.
No new GPU allocation or cloud compute was used; cloud cost was zero. Minimum
sampled free disk during these stages was 5,126,152,192 bytes.

Cache cleanup had earlier recovered **5,291,851,776 bytes (4.93 GiB)**. All 42
new temporary MCAPs were released only after their packages validated; none
remain under the new execution root. The retained execution root was about
270 MB before summary publication. Existing teacher/model checkpoints were
preserved. A separate proposed cleanup of 68 old CLEVR range parts was blocked
by automatic approval review and remains unperformed; it was not needed for
this streaming run.

Artifacts under the execution root include:

- `sessions/<chunk>/`: source receipts, media SHA-256, corpus, features, validation logs.
- `full-aggregate/`: combined corpus, features and integrity report.
- `readout-results-cpu-01/`: selected weights, all epoch histories and per-example predictions.
- `readout-reload-cpu-01/`: hash and probability-parity verification.
- `temporal-controls-cpu-01/`: all seven metrics, paired intervals and per-example probabilities.
- `readout-execution.json`: exact argv, environment, timestamps, resource samples and exit codes.
- `execute-fixed-readouts.py` / `summarize-execution.py`: retained orchestration/report scripts.

The joint state file is 329,254 bytes; sensor-only is 3,604 bytes. These are
readout-only sizes and exclude the frozen encoders. VideoPrism's participating
module has 114,365,184 parameters and MOMENT's embedding module 35,337,408;
neither this arithmetic nor separate feature extraction establishes the
complete four-modality runtime budget. No large artifact is added to Git.

| Artifact | SHA-256 |
|---|---|
| Train examples | `95b99c26e9c31835fe898526069d23dbecdb8f6f63a2d9b2d8551914f12d7e6e` |
| Development examples | `55fe808c824825762da63d2c0f376dde30d61986a8fb6481c0768f661aa9d232` |
| Video features | `531e4a59b91b288695bb838a0bf30fccd888942a45f9f690ec948372d2b49629` |
| Sensor features | `c4c05ba6e404830e875779f6d867e02ec522030e82df2dc5050ed8d60057fe46` |
| Joint selected checkpoint | `b62c1cac0c38c459263dbc2c5ffd8eddc76b8654a2254a5f4d15fac8fe95a862` |
| Sensor-only selected checkpoint | `2a8ca8eaa2539f551618739ad61456c6f4607077ef5a3856556fa1c4c7cd0b58` |
| Readout result | `c9a18c74535b75763d7fe5b0306524341a2a2ccfb7fe33da8225d43c6017f851` |
| Reload verification | `5abf86a815ffb14198135d75469da93b20204cac662e05884195f8fcfe78b8a9` |
| Feature-control result | `14718706030ef90e7df24ce8c0f466641dabda08f733202946d00f82d873c1a5` |

The complete manifest includes all source/cache hashes, group/category metrics,
commands and full learning curves. After recreating the hash-matched aggregate,
the three model commands are:

```powershell
$root = 'C:\CodexArtifacts\egoviz-unseen-group-acquisition-20261030-attempt-02'
$corpus = Join-Path $root 'full-aggregate'
$py = 'C:\Users\junny\AppData\Local\Programs\Python\Python311\python.exe'
$env:CUDA_VISIBLE_DEVICES = '-1'
$env:HF_HUB_OFFLINE = '1'
# Exact original argv; these output directories now exist and reject overwriting.
& $py -m scripts.train_egoviz_vs_sync_readout --corpus $corpus --video-cache "$corpus\video-features.npz" --sensor-cache "$corpus\sensor-features.npz" --output "$root\readout-results-cpu-01"
& $py -m scripts.verify_egoviz_vs_sync_readout_reload --corpus $corpus --video-cache "$corpus\video-features.npz" --sensor-cache "$corpus\sensor-features.npz" --result-dir "$root\readout-results-cpu-01" --output-dir "$root\readout-reload-cpu-01"
& $py -m scripts.diagnose_egoviz_temporal_controls --corpus $corpus --video-cache "$corpus\video-features.npz" --sensor-cache "$corpus\sensor-features.npz" --result-dir "$root\readout-results-cpu-01" --output "$root\temporal-controls-cpu-01"
```

## Verification and next decision

Ruff passed. The normal CPU suite passed **1,999 tests, 163 skipped**; the eight
targeted synthetic control/report tests also passed in the existing CPU Torch
environment. All 22 dataset manifests, four catalogs, model manifest and example
manifest audit passed. An older unrelated CI virtualenv failed during Python
site initialization because its editable Japanese path was decoded as cp932;
it was preserved and an existing working ASCII-path environment was used.

Retain this as a completed correspondence baseline. Before adding recurrent
capacity or a fusion gate, separate cross-session generalization, channel/time
pooling and calibration limitations with a new fixed protocol. The present
controls establish relative feature alignment use, so a new module should have
to improve a defined failure under matched conditions. No additional training,
sealed evaluation, product promotion or quantization followed this result.
